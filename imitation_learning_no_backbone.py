import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# Use the same dataset class from the original file
class RCCarDataset(Dataset):
    def __init__(self, video_dir, control_data_path, transform=None):
        """
        Dataset for RC car imitation learning
        
        Args:
            video_dir: Directory containing video files
            control_data_path: Path to numpy file with control signals
            transform: Optional transform to be applied on images
        """
        self.video_dir = video_dir
        self.transform = transform
        
        # Load control data (columns: timestamp, steering, throttle)
        self.control_data = np.load(control_data_path)
        
        # Get list of video files
        video_files = []
        for filename in os.listdir(video_dir):
            if filename.endswith(('.mp4', '.avi')):
                # Extract timestamp from filename (without extension)
                timestamp = float(os.path.splitext(filename)[0])
                video_files.append((timestamp, filename))
        
        # Sort by timestamp
        video_files.sort()
        self.video_files = video_files
        
        # Build index of all frames with corresponding control signals
        self.frame_index = []
        
        for video_timestamp, video_filename in self.video_files:
            video_path = os.path.join(video_dir, video_filename)
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}")
                continue
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            for frame_idx in range(frame_count):
                # Calculate timestamp for this frame
                frame_time = video_timestamp + (frame_idx / fps)
                
                # Find closest control signal
                time_diffs = np.abs(self.control_data[:, 0] - frame_time)
                closest_idx = np.argmin(time_diffs)
                
                # Only include if time difference is reasonable (< 0.1 seconds)
                if time_diffs[closest_idx] < 0.1:
                    steering = self.control_data[closest_idx, 1]
                    throttle = self.control_data[closest_idx, 2]
                    
                    self.frame_index.append({
                        'video_path': video_path,
                        'frame_idx': frame_idx,
                        'steering': steering,
                        'throttle': throttle
                    })
            
            cap.release()
        
        print(f"Dataset initialized with {len(self.frame_index)} frames")
    
    def __len__(self):
        return len(self.frame_index)
    
    def __getitem__(self, idx):
        item = self.frame_index[idx]
        
        # Read specific frame from video
        cap = cv2.VideoCapture(item['video_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, item['frame_idx'])
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError(
                f"Could not read frame {item['frame_idx']} from {item['video_path']}"
            )
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            frame = self.transform(frame)
        
        control = [item['steering'], item['throttle']]
        return frame, torch.tensor(control, dtype=torch.float32)


class SimpleCNNModel(nn.Module):
    def __init__(self):
        """Simple CNN model without any backbone for predicting control signals from images"""
        super(SimpleCNNModel, self).__init__()
        
        # Simple CNN feature extractor
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Control prediction head
        self.control_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # 2 outputs: steering, throttle
        )
    
    def forward(self, x):
        features = self.features(x)
        control = self.control_head(features)
        return control


def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    """Train the imitation learning model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        train_desc = f"Epoch {epoch+1}/{num_epochs} (Training)"
        for images, controls in tqdm(train_loader, desc=train_desc):
            images = images.to(device)
            controls = controls.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, controls)
            
            # Backward and optimize
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
            val_desc = f"Epoch {epoch+1}/{num_epochs} (Validation)"
            for images, controls in tqdm(val_loader, desc=val_desc):
                images = images.to(device)
                controls = controls.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, controls)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Curves - Simple CNN (No Backbone)')
    plt.savefig('training_curve_no_backbone.png')
    plt.close()
    
    return model


def evaluate_model(model, test_loader):
    """Evaluate the model on test data"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    test_loss = 0.0
    
    steering_errors = []
    throttle_errors = []
    
    with torch.no_grad():
        for images, controls in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            controls = controls.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, controls)
            test_loss += loss.item()
            
            # Calculate individual errors for steering and throttle
            errors = torch.abs(outputs - controls)
            steering_errors.extend(errors[:, 0].cpu().numpy())
            throttle_errors.extend(errors[:, 1].cpu().numpy())
    
    test_loss /= len(test_loader)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Mean Steering Error: {np.mean(steering_errors):.4f}")
    print(f"Mean Throttle Error: {np.mean(throttle_errors):.4f}")
    
    # Plot error distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(steering_errors, bins=30, alpha=0.7)
    ax1.set_title('Steering Error Distribution (No Backbone)')
    ax1.set_xlabel('Absolute Error')
    
    ax2.hist(throttle_errors, bins=30, alpha=0.7)
    ax2.set_title('Throttle Error Distribution (No Backbone)')
    ax2.set_xlabel('Absolute Error')
    
    plt.tight_layout()
    plt.savefig('error_distributions_no_backbone.png')
    plt.close()


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("Training RC Car Model - Simple CNN (No Backbone)")
    
    # Example usage
    video_dir = "data/videos"
    control_data_path = "data/control_signals.npy"
    
    # Create transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and split into train/val/test
    dataset = RCCarDataset(video_dir, control_data_path, transform=transform)
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Create and train model
    model = SimpleCNNModel()
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    
    trained_model = train_model(model, train_loader, val_loader, num_epochs=20)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(trained_model.state_dict(), 'models/rc_car_model_no_backbone.pth')
    
    # Evaluate model
    evaluate_model(trained_model, test_loader)
    
    print("Training and evaluation complete. Model saved to models/rc_car_model_no_backbone.pth")


if __name__ == "__main__":
    main() 