import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms, models

class RCCarDatasetTemporal(Dataset):
    def __init__(self, video_dir, control_data_path, transform=None, 
                 sequence_length=3, training_lag_ms=150, inference_lag_ms=150):
        """
        Temporal dataset using sequences of frames to capture motion information
        
        Args:
            sequence_length: Number of consecutive frames to use (e.g., 3 frames = ~100ms at 30fps)
            Other args same as predictive dataset
        """
        self.video_dir = video_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.training_lag_seconds = training_lag_ms / 1000.0
        self.inference_lag_seconds = inference_lag_ms / 1000.0
        self.total_lag_compensation = self.training_lag_seconds + self.inference_lag_seconds
        
        print(f"Temporal Model Configuration:")
        print(f"  Sequence length: {sequence_length} frames")
        print(f"  Temporal window: ~{sequence_length * 33:.0f}ms at 30fps") 
        print(f"  Total lag compensation: {self.total_lag_compensation:.3f}s")
        print(f"  Strategy: Frame sequence at time T → Control for time T+{self.total_lag_compensation:.3f}s")
        
        # Load control data
        self.control_data = np.load(control_data_path)
        
        # Process videos to create temporal sequences
        self.frame_sequences = []
        self.controls = []
        
        self._process_data()
    
    def _process_data(self):
        """Process videos to create frame sequences with predictive control"""
        video_files = sorted([f for f in os.listdir(self.video_dir)
                             if f.endswith(('.mp4', '.avi')) and not f.startswith('._')])
        
        total_sequences = 0
        valid_sequences = 0
        
        for video_file in tqdm(video_files, desc="Processing videos (temporal sequences)"):
            video_path = os.path.join(self.video_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                continue
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                cap.release()
                continue
            
            try:
                video_timestamp = float(os.path.splitext(video_file)[0])
            except ValueError:
                cap.release()
                continue
            
            # Read all frames from this video first
            frames = []
            frame_timestamps = []
            frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_timestamp = video_timestamp + (frame_idx / fps)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frames.append(frame)
                frame_timestamps.append(frame_timestamp)
                frame_idx += 1
            
            cap.release()
            
            # Create temporal sequences
            for i in range(self.sequence_length - 1, len(frames)):
                total_sequences += 1
                
                # Get sequence of frames (oldest to newest)
                frame_sequence = frames[i - self.sequence_length + 1:i + 1]
                current_timestamp = frame_timestamps[i]  # timestamp of most recent frame
                
                # Find future control signal
                future_control_time = current_timestamp + self.total_lag_compensation
                
                # Find closest control signal
                time_diffs = np.abs(self.control_data[:, 0] - future_control_time)
                closest_idx = np.argmin(time_diffs)
                
                # Check alignment quality
                if time_diffs[closest_idx] > 0.2:
                    continue
                    
                control_signal = self.control_data[closest_idx, 1:3]
                
                # Skip idle sequences
                if np.all(np.abs(control_signal) < 1e-3):
                    continue
                
                valid_sequences += 1
                self.frame_sequences.append(frame_sequence)
                self.controls.append(control_signal)
        
        print(f"Temporal sequence results:")
        print(f"  Total sequences processed: {total_sequences}")
        print(f"  Valid temporal sequences: {valid_sequences}")
        print(f"  Retention rate: {100 * valid_sequences / total_sequences:.1f}%")
    
    def __len__(self):
        return len(self.frame_sequences)
    
    def __getitem__(self, idx):
        frame_sequence = self.frame_sequences[idx]
        control = np.array(self.controls[idx], dtype=np.float32)
        
        # Transform each frame in the sequence
        if self.transform:
            transformed_frames = []
            for frame in frame_sequence:
                transformed_frame = self.transform(frame)
                transformed_frames.append(transformed_frame)
            # Stack frames along new dimension: [sequence_length, channels, height, width]
            frame_sequence = torch.stack(transformed_frames, dim=0)
        
        return frame_sequence, control

class TemporalRCCarModel(nn.Module):
    def __init__(self, sequence_length=3):
        super(TemporalRCCarModel, self).__init__()
        
        self.sequence_length = sequence_length
        
        # CNN feature extractor for individual frames
        self.cnn_backbone = models.resnet18(pretrained=True)
        self.cnn_backbone.fc = nn.Identity()  # Remove final layer
        
        # Temporal fusion layer (simple approach: LSTM)
        self.temporal_fusion = nn.LSTM(
            input_size=512,  # ResNet18 output
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Control prediction head
        self.control_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # steering, throttle
        )
    
    def forward(self, x):
        # x shape: [batch_size, sequence_length, channels, height, width]
        batch_size, seq_len, channels, height, width = x.shape
        
        # Process each frame through CNN
        # Reshape to process all frames at once
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Extract features
        features = self.cnn_backbone(x)  # [batch_size * seq_len, 512]
        
        # Reshape back for temporal processing
        features = features.view(batch_size, seq_len, -1)  # [batch_size, seq_len, 512]
        
        # Process temporal sequence
        lstm_out, (hidden, cell) = self.temporal_fusion(features)
        
        # Use the final hidden state for prediction
        final_features = lstm_out[:, -1, :]  # [batch_size, 256]
        
        # Predict control
        control_output = self.control_head(final_features)
        
        return control_output

def train_temporal_model(model, train_loader, val_loader, num_epochs=250, lr=0.001, steering_weight=10.0):
    """Train the temporal model with weighted loss for steering"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Use weighted MSE loss - give much higher weight to steering
    def weighted_mse_loss(outputs, targets, steering_weight=steering_weight):
        steering_loss = nn.MSELoss()(outputs[:, 0], targets[:, 0]) * steering_weight
        throttle_loss = nn.MSELoss()(outputs[:, 1], targets[:, 1])
        return steering_loss + throttle_loss
    
    criterion = weighted_mse_loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.7)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 15
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for sequences, controls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
            sequences = sequences.to(device)
            controls = controls.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, controls)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for sequences, controls in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
                sequences = sequences.to(device)
                controls = controls.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, controls)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_model_temporal.pth')
        else:
            patience_counter += 1
            
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    if patience_counter > 0:
        print("Loading best model from training...")
        model.load_state_dict(torch.load('models/best_model_temporal.pth'))
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Temporal Model - Best Val Loss: {best_val_loss:.4f}')
    plt.savefig('training_curve_temporal.png')
    plt.close()
    
    return model

def evaluate_model(model, test_loader):
    """Evaluate the temporal model with error distribution plots"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    test_loss = 0.0
    steering_errors = []
    throttle_errors = []
    
    with torch.no_grad():
        for sequences, controls in tqdm(test_loader, desc="Evaluating"):
            sequences = sequences.to(device)
            controls = controls.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, controls)
            test_loss += loss.item()
            
            steering_error = torch.abs(outputs[:, 0] - controls[:, 0])
            throttle_error = torch.abs(outputs[:, 1] - controls[:, 1])
            
            steering_errors.extend(steering_error.cpu().numpy())
            throttle_errors.extend(throttle_error.cpu().numpy())
    
    test_loss /= len(test_loader)
    mean_steering_error = np.mean(steering_errors)
    mean_throttle_error = np.mean(throttle_errors)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Mean Steering Error: {mean_steering_error:.4f}")
    print(f"Mean Throttle Error: {mean_throttle_error:.4f}")
    
    # Plot error distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(steering_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title('Steering Error Distribution (Temporal Model)')
    ax1.set_xlabel('Absolute Error')
    ax1.set_ylabel('Frequency')
    ax1.axvline(mean_steering_error, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_steering_error:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(throttle_errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title('Throttle Error Distribution (Temporal Model)')
    ax2.set_xlabel('Absolute Error')
    ax2.set_ylabel('Frequency')
    ax2.axvline(mean_throttle_error, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_throttle_error:.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_distributions_temporal.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Error distribution plot saved as 'error_distributions_temporal.png'")
    
    return test_loss, mean_steering_error, mean_throttle_error

def main():
    print("Training Temporal RC Car Model (Motion-Aware)")
    
    video_dir = "data/videos"
    control_data_path = "data/control_signals.npy"
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create temporal dataset (3 frames = ~100ms motion window)
    dataset = RCCarDatasetTemporal(
        video_dir, control_data_path, transform=transform, 
        sequence_length=3, training_lag_ms=150, inference_lag_ms=150
    )
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])
    
    # Smaller batch size due to temporal sequences
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Create and train temporal model with higher steering weight
    model = TemporalRCCarModel(sequence_length=3)
    trained_model = train_temporal_model(model, train_loader, val_loader, num_epochs=300, steering_weight=15.0)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(trained_model.state_dict(), 'models/rc_car_model_temporal.pth')
    
    # Evaluate
    test_loss, steering_error, throttle_error = evaluate_model(trained_model, test_loader)
    
    print("\n" + "="*50)
    print("TEMPORAL MODEL SUMMARY")
    print("="*50)
    print("Key Features:")
    print("  - Uses 3-frame sequences (~100ms motion window)")
    print("  - LSTM for temporal fusion")
    print("  - Motion-aware predictions")
    print("  - Predictive control (300ms lookahead)")
    print("="*50)

if __name__ == "__main__":
    main() 