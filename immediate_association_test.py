import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms, models

class ImmediateAssociationDataset(Dataset):
    def __init__(self, video_dir, control_data_path, transform=None, 
                 sequence_length=3, balance_steering=True):
        """Dataset with IMMEDIATE temporal association (no lag compensation)"""
        
        self.video_dir = video_dir
        self.transform = transform
        self.sequence_length = sequence_length
        
        print(f"🔬 IMMEDIATE ASSOCIATION Dataset Configuration:")
        print(f"  Sequence length: {sequence_length} frames")
        print(f"  NO LAG COMPENSATION - immediate frame-to-control association")
        print(f"  Steering balancing: {balance_steering}")
        
        # Load control data
        self.control_data = np.load(control_data_path)
        
        # Process videos to create temporal sequences
        self.frame_sequences = []
        self.controls = []
        self.steering_weights = []
        
        self._process_data()
        
        if balance_steering:
            self._create_steering_weights()
    
    def _process_data(self):
        """Process videos with IMMEDIATE temporal association (no lag compensation)"""
        video_files = sorted([f for f in os.listdir(self.video_dir)
                             if f.endswith(('.mp4', '.avi')) and not f.startswith('._')])
        
        for video_file in tqdm(video_files, desc="Processing videos (immediate association)"):
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
            
            # Create temporal sequences with IMMEDIATE association
            for i in range(self.sequence_length - 1, len(frames)):
                frame_sequence = frames[i - self.sequence_length + 1:i + 1]
                current_timestamp = frame_timestamps[i]
                
                # 🔬 KEY CHANGE: Use IMMEDIATE temporal association
                # Instead of: future_control_time = current_timestamp + lag_compensation
                immediate_control_time = current_timestamp
                
                # Find closest control signal to the CURRENT frame time
                time_diffs = np.abs(self.control_data[:, 0] - immediate_control_time)
                closest_idx = np.argmin(time_diffs)
                
                # Check alignment quality (same threshold)
                if time_diffs[closest_idx] > 0.2:
                    continue
                    
                control_signal = self.control_data[closest_idx, 1:3]
                
                # Skip idle sequences
                if np.all(np.abs(control_signal) < 1e-3):
                    continue
                
                self.frame_sequences.append(frame_sequence)
                self.controls.append(control_signal)
        
        print(f"Created {len(self.frame_sequences)} temporal sequences (immediate association)")
    
    def _create_steering_weights(self):
        """Create sampling weights to balance steering distribution"""
        steering_values = np.array([control[0] for control in self.controls])
        abs_steering = np.abs(steering_values)
        
        # Define steering categories
        straight = abs_steering < 0.1
        light_turn = (abs_steering >= 0.1) & (abs_steering < 0.3)
        sharp_turn = abs_steering >= 0.3
        
        print(f"\nSteering Distribution (Immediate Association):")
        print(f"  Straight driving: {np.sum(straight)} samples ({100*np.sum(straight)/len(steering_values):.1f}%)")
        print(f"  Light turns: {np.sum(light_turn)} samples ({100*np.sum(light_turn)/len(steering_values):.1f}%)")
        print(f"  Sharp turns: {np.sum(sharp_turn)} samples ({100*np.sum(sharp_turn)/len(steering_values):.1f}%)")
        
        weights = np.ones(len(steering_values))
        
        # Weight inversely proportional to category frequency
        if np.sum(straight) > 0:
            weights[straight] = 1.0
        if np.sum(light_turn) > 0:
            weights[light_turn] = 3.0
        if np.sum(sharp_turn) > 0:
            weights[sharp_turn] = 10.0
            
        self.steering_weights = weights
        print(f"Applied steering-aware sampling weights")
    
    def get_sampler(self):
        """Return weighted sampler for balanced training"""
        if hasattr(self, 'steering_weights'):
            return WeightedRandomSampler(self.steering_weights, len(self.steering_weights))
        return None
    
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
            frame_sequence = torch.stack(transformed_frames, dim=0)
        
        return frame_sequence, control

class SteeringAwareLoss(nn.Module):
    def __init__(self, steering_weight_multiplier=5.0):
        super().__init__()
        self.steering_weight_multiplier = steering_weight_multiplier
        
    def forward(self, predictions, targets):
        steering_pred, throttle_pred = predictions[:, 0], predictions[:, 1]
        steering_true, throttle_true = targets[:, 0], targets[:, 1]
        
        # Standard MSE for throttle
        throttle_loss = torch.mean((throttle_pred - throttle_true)**2)
        
        # Steering-magnitude-aware loss
        abs_steering_true = torch.abs(steering_true)
        
        # Higher weights for larger steering angles
        steering_weights = torch.where(
            abs_steering_true < 0.1, 1.0,  # Normal weight for straight
            torch.where(
                abs_steering_true < 0.3, 3.0,  # 3x weight for light turns
                8.0  # 8x weight for sharp turns
            )
        )
        
        steering_loss = torch.mean(steering_weights * (steering_pred - steering_true)**2)
        
        # Combine losses
        total_loss = self.steering_weight_multiplier * steering_loss + throttle_loss
        
        return total_loss, steering_loss, throttle_loss

def train_immediate_association_model():
    """Train a model with IMMEDIATE temporal association (no lag compensation)"""
    
    print("🔬 TRAINING IMMEDIATE ASSOCIATION MODEL")
    print("=" * 60)
    print("Testing: Does immediate frame-to-control association work better?")
    print("=" * 60)
    
    # Data setup
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create immediate association dataset
    dataset = ImmediateAssociationDataset(
        'data/videos', 'data/control_signals.npy', transform=transform,
        sequence_length=3, balance_steering=True
    )
    
    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])
    
    # Create data loaders with balanced sampling
    sampler = dataset.get_sampler()
    if sampler:
        # Use weighted sampler for training
        train_indices = train_dataset.indices
        train_weights = [dataset.steering_weights[i] for i in train_indices]
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, num_workers=4)
    else:
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Model setup (identical to steering-fixed model)
    class ImmediateAssociationModel(nn.Module):
        def __init__(self, sequence_length=3):
            super().__init__()
            self.sequence_length = sequence_length
            
            # CNN feature extractor
            self.cnn_backbone = models.resnet18(pretrained=True)
            self.cnn_backbone.fc = nn.Identity()
            
            # Temporal fusion
            self.temporal_fusion = nn.LSTM(
                input_size=512,
                hidden_size=256,
                num_layers=2,
                batch_first=True,
                dropout=0.2  # Moderate dropout for LSTM
            )
            
            # Control prediction head with progressive dropout
            self.control_head = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.3),  # Moderate dropout early layers
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),  # Less dropout as we get more specific
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),  # Minimal dropout near output
                nn.Linear(64, 2)  # steering, throttle
            )
        
        def forward(self, x):
            batch_size, seq_len, channels, height, width = x.shape
            
            # Process frames through CNN
            x = x.view(batch_size * seq_len, channels, height, width)
            features = self.cnn_backbone(x)
            features = features.view(batch_size, seq_len, -1)
            
            # Temporal processing
            lstm_out, _ = self.temporal_fusion(features)
            final_features = lstm_out[:, -1, :]
            
            # Control prediction
            control_output = self.control_head(final_features)
            
            return control_output
    
    # Use GPU 1 (second 3090) to avoid interfering with other training
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Using device: {device}")
    model = ImmediateAssociationModel(sequence_length=3).to(device)
    
    # Training setup (identical parameters)
    criterion = SteeringAwareLoss(steering_weight_multiplier=3.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.7)
    
    # Training loop
    num_epochs = 300
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20
    
    train_losses = []
    val_losses = []
    train_steering_losses = []
    train_throttle_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0.0
        steering_loss_sum = 0.0
        throttle_loss_sum = 0.0
        
        for sequences, controls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
            sequences = sequences.to(device)
            controls = controls.to(device)
            
            outputs = model(sequences)
            loss, steering_loss, throttle_loss = criterion(outputs, controls)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steering_loss_sum += steering_loss.item()
            throttle_loss_sum += throttle_loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_steering_loss = steering_loss_sum / len(train_loader)
        avg_throttle_loss = throttle_loss_sum / len(train_loader)
        train_losses.append(avg_train_loss)
        train_steering_losses.append(avg_steering_loss)
        train_throttle_losses.append(avg_throttle_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for sequences, controls in val_loader:
                sequences = sequences.to(device)
                controls = controls.to(device)
                
                outputs = model(sequences)
                loss, _, _ = criterion(outputs, controls)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} (Steering: {avg_steering_loss:.4f}, Throttle: {avg_throttle_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/immediate_association_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('models/immediate_association_model.pth'))
    
    # Plot and save comprehensive training curves
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss curves
    ax1.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Overall Training Progress (Immediate Association)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Steering vs Throttle loss breakdown
    ax2.plot(train_steering_losses, label='Steering Loss', color='orange', linewidth=2)
    ax2.plot(train_throttle_losses, label='Throttle Loss', color='green', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Component Loss')
    ax2.set_title('Steering vs Throttle Loss Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Loss convergence (zoomed)
    epochs = range(1, len(train_losses) + 1)
    ax3.plot(epochs, train_losses, label='Total Training', color='blue', alpha=0.8)
    ax3.plot(epochs, val_losses, label='Validation', color='red', alpha=0.8)
    ax3.axhline(y=best_val_loss, color='red', linestyle='--', alpha=0.7, 
               label=f'Best Val: {best_val_loss:.4f}')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Convergence Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Training stability (loss ratio)
    if len(train_steering_losses) > 10:
        steering_ratio = np.array(train_steering_losses) / (np.array(train_steering_losses) + np.array(train_throttle_losses))
        ax4.plot(steering_ratio, label='Steering/Total Ratio', color='purple', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Steering Loss Ratio')
        ax4.set_title('Steering Loss Dominance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('training_curve_immediate_association.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Training curves saved as: training_curve_immediate_association.png")
    
    print(f"\n🔬 EVALUATING IMMEDIATE ASSOCIATION MODEL")
    print("=" * 50)
    
    # Test the model and analyze steering predictions
    model.eval()
    predicted_steering = []
    true_steering = []
    predicted_throttle = []
    true_throttle = []
    
    with torch.no_grad():
        for sequences, controls in test_loader:
            sequences = sequences.to(device)
            controls = controls.to(device)
            
            outputs = model(sequences)
            
            predicted_steering.extend(outputs[:, 0].cpu().numpy())
            true_steering.extend(controls[:, 0].cpu().numpy())
            predicted_throttle.extend(outputs[:, 1].cpu().numpy())
            true_throttle.extend(controls[:, 1].cpu().numpy())
    
    predicted_steering = np.array(predicted_steering)
    true_steering = np.array(true_steering)
    predicted_throttle = np.array(predicted_throttle)
    true_throttle = np.array(true_throttle)
    
    # Calculate errors
    steering_errors = np.abs(predicted_steering - true_steering)
    throttle_errors = np.abs(predicted_throttle - true_throttle)
    
    print(f"STEERING ANALYSIS (Immediate Association):")
    print(f"  Mean error: {np.mean(steering_errors):.4f}")
    print(f"  Predicted std: {predicted_steering.std():.4f}")
    print(f"  True std: {true_steering.std():.4f}")
    print(f"  Prediction range: {predicted_steering.min():.3f} to {predicted_steering.max():.3f}")
    print(f"  True range: {true_steering.min():.3f} to {true_steering.max():.3f}")
    
    print(f"\nTHROTTLE ANALYSIS:")
    print(f"  Mean error: {np.mean(throttle_errors):.4f}")
    
    # Check if steering collapse is fixed
    steering_variation_ratio = predicted_steering.std() / true_steering.std()
    print(f"\nSTEERING COLLAPSE CHECK (Immediate Association):")
    if steering_variation_ratio > 0.7:
        print(f"  ✅ STEERING FIXED! Variation ratio: {steering_variation_ratio:.3f}")
        print(f"  Model shows good steering diversity")
    elif steering_variation_ratio > 0.4:
        print(f"  ⚠️  PARTIAL FIX: Variation ratio: {steering_variation_ratio:.3f}")
        print(f"  Better but still some steering suppression")
    else:
        print(f"  ❌ STILL COLLAPSED: Variation ratio: {steering_variation_ratio:.3f}")
        print(f"  Steering predictions too conservative")
    
    # Generate comprehensive error distribution plots
    print(f"\n📊 GENERATING ERROR DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    # Create enhanced error distribution plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Steering error histogram with steering magnitude analysis
    ax1.hist(steering_errors, bins=50, alpha=0.7, color='blue', edgecolor='black', density=True)
    ax1.set_title('Steering Error Distribution (Immediate Association)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Absolute Steering Error')
    ax1.set_ylabel('Density')
    ax1.axvline(np.mean(steering_errors), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(steering_errors):.4f}')
    ax1.axvline(np.median(steering_errors), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(steering_errors):.4f}')
    ax1.axvline(np.percentile(steering_errors, 95), color='purple', linestyle='--', linewidth=1,
               label=f'95th %ile: {np.percentile(steering_errors, 95):.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Throttle error histogram
    ax2.hist(throttle_errors, bins=50, alpha=0.7, color='green', edgecolor='black', density=True)
    ax2.set_title('Throttle Error Distribution (Immediate Association)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Absolute Throttle Error')
    ax2.set_ylabel('Density')
    ax2.axvline(np.mean(throttle_errors), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(throttle_errors):.4f}')
    ax2.axvline(np.median(throttle_errors), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(throttle_errors):.4f}')
    ax2.axvline(np.percentile(throttle_errors, 95), color='purple', linestyle='--', linewidth=1,
               label=f'95th %ile: {np.percentile(throttle_errors, 95):.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Steering prediction vs true scatter plot (key diagnostic!)
    ax3.scatter(true_steering, predicted_steering, alpha=0.6, s=10, color='blue')
    ax3.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('True Steering')
    ax3.set_ylabel('Predicted Steering')
    ax3.set_title('Steering Prediction Quality (Immediate Association)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    
    # Add correlation coefficient
    correlation = np.corrcoef(true_steering, predicted_steering)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax3.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Error vs steering magnitude (most important plot!)
    abs_true_steering = np.abs(true_steering)
    ax4.scatter(abs_true_steering, steering_errors, alpha=0.6, s=10, color='red')
    ax4.set_xlabel('True Steering Magnitude |steering|')
    ax4.set_ylabel('Steering Error')
    ax4.set_title('Error vs Steering Magnitude (Immediate Association)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(abs_true_steering, steering_errors, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(0, 1, 100)
    ax4.plot(x_trend, p(x_trend), "purple", linestyle='-', linewidth=2, 
             label=f'Trend: slope={z[0]:.3f}')
    ax4.legend()
    
    # Color code by steering magnitude for better visualization
    straight_mask = abs_true_steering < 0.1
    light_mask = (abs_true_steering >= 0.1) & (abs_true_steering < 0.3)
    sharp_mask = abs_true_steering >= 0.3
    
    if np.any(straight_mask):
        mean_error_straight = np.mean(steering_errors[straight_mask])
        ax4.axhline(y=mean_error_straight, color='green', linestyle=':', alpha=0.7,
                   label=f'Straight error: {mean_error_straight:.3f}')
    if np.any(sharp_mask):
        mean_error_sharp = np.mean(steering_errors[sharp_mask])
        ax4.axhline(y=mean_error_sharp, color='red', linestyle=':', alpha=0.7,
                   label=f'Sharp turn error: {mean_error_sharp:.3f}')
    
    plt.tight_layout()
    plt.savefig('error_distributions_immediate_association.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print detailed steering analysis by category
    print(f"\n🎯 STEERING PERFORMANCE BY CATEGORY (Immediate Association):")
    print(f"=" * 60)
    
    if np.any(straight_mask):
        straight_errors = steering_errors[straight_mask]
        print(f"STRAIGHT DRIVING (|steering| < 0.1):")
        print(f"  Samples: {np.sum(straight_mask)}")
        print(f"  Mean error: {np.mean(straight_errors):.4f}")
        print(f"  Std error: {np.std(straight_errors):.4f}")
        
    if np.any(light_mask):
        light_errors = steering_errors[light_mask]
        print(f"\nLIGHT TURNS (0.1 ≤ |steering| < 0.3):")
        print(f"  Samples: {np.sum(light_mask)}")
        print(f"  Mean error: {np.mean(light_errors):.4f}")
        print(f"  Std error: {np.std(light_errors):.4f}")
        
    if np.any(sharp_mask):
        sharp_errors = steering_errors[sharp_mask]
        print(f"\nSHARP TURNS (|steering| ≥ 0.3):")
        print(f"  Samples: {np.sum(sharp_mask)}")
        print(f"  Mean error: {np.mean(sharp_errors):.4f}")
        print(f"  Std error: {np.std(sharp_errors):.4f}")
    
    print(f"\n📈 Error distribution plots saved as: error_distributions_immediate_association.png")
    print(f"🏁 Model saved as: models/immediate_association_model.pth")
    
    print(f"\n🔬 EXPERIMENT COMPLETE!")
    print(f"Compare this model's performance with the lag-compensated model")
    print(f"Key metrics to compare:")
    print(f"  1. Steering variation ratio: {steering_variation_ratio:.3f}")
    print(f"  2. Mean steering error: {np.mean(steering_errors):.4f}")
    print(f"  3. Correlation coefficient: {correlation:.3f}")
    print(f"  4. Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_immediate_association_model() 