import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt


# PEP8: two blank lines above top-level definitions

class RCCarDataset(Dataset):
    def __init__(self, video_dir, control_data_path, transform=None):
        """
        Dataset for RC car imitation learning
        
        Args:
            video_dir: Directory containing video files
            control_data_path: Path to control signal data file
            transform: Optional transforms to apply to images
        """
        self.video_dir = video_dir
        self.transform = transform
        self.last_control_idx = 0
        
        # Load control data (assuming format: timestamp, steering, throttle)
        self.control_data = np.load(control_data_path)
        
        # Process video files to extract frames and match with control signals
        self.frames = []
        self.controls = []
        
        self._process_data()
    
    def _process_data(self):
        """Process video and match frames with control signals"""
        # List video files
        video_files = sorted([f for f in os.listdir(self.video_dir)
                             if f.endswith(('.mp4', '.avi'))])
        
        for video_file in tqdm(video_files, desc="Processing videos"):
            video_path = os.path.join(self.video_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            
            # Extract timestamp from filename (assuming format: timestamp.mp4)
            video_timestamp = float(os.path.splitext(video_file)[0])
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate timestamp for this frame
                frame_timestamp = video_timestamp + (frame_idx / cap.get(cv2.CAP_PROP_FPS))
                idx = self.last_control_idx
                if idx < len(self.control_data) and self.control_data[idx, 0] > frame_timestamp:
                    while idx > 0 and self.control_data[idx, 0] > frame_timestamp:
                        idx -= 1
                else:
                    while idx < len(self.control_data) - 1 and self.control_data[idx + 1, 0] <= frame_timestamp:
                        idx += 1
                
                self.last_control_idx = idx
                control_signal = self.control_data[idx, 1:3]

                # ------------------------------------------------------------
                # Skip idle frames (no steering & no throttle)
                # Empirically, values exactly equal to 0.0 indicate periods
                # before/after active driving when the operator is not sending
                # commands.  Including these overwhelms the dataset and teaches
                # the model to predict zeros.  We therefore discard frames
                # whose absolute steering *and* throttle are below a small
                # threshold.
                # ------------------------------------------------------------
                if np.all(np.abs(control_signal) < 1e-3):
                    frame_idx += 1
                    continue  # Skip this pair and move to next frame

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                self.frames.append(frame)
                # Store as plain list to avoid 2-D slices holding reference
                self.controls.append(control_signal)
                
                frame_idx += 1
            
            cap.release()
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        control = self.controls[idx]
        
        if self.transform:
            frame = self.transform(frame)
        
        return frame, torch.tensor(control, dtype=torch.float32)

class RCCarModel(nn.Module):
    def __init__(self):
        """CNN model for predicting control signals from images"""
        super(RCCarModel, self).__init__()
        
        # Use ResNet18 as backbone. The older `pretrained` flag is deprecated
        # and triggers a weight download that can fail without proper SSL
        # certificates or an internet connection.
        #
        # We therefore default to `weights=None` (random initialization).

        if os.getenv("IML_PRETRAINED") == "1":
            from torchvision.models import ResNet18_Weights

            resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            resnet = models.resnet18(weights=None)
        
        # Replace final FC layer
        num_features = resnet.fc.in_features
        resnet.fc = nn.Identity()  # Remove classification layer
        
        self.backbone = resnet
        self.control_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2 outputs: steering, throttle
        )
    
    def forward(self, x):
        features = self.backbone(x)
        control = self.control_head(features)
        return control


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------


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
        
        for images, controls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
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
            for images, controls in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
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
    plt.savefig('training_curve.png')
    
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
    ax1.hist(steering_errors, bins=30)
    ax1.set_title('Steering Error Distribution')
    ax1.set_xlabel('Absolute Error')
    
    ax2.hist(throttle_errors, bins=30)
    ax2.set_title('Throttle Error Distribution')
    ax2.set_xlabel('Absolute Error')
    
    plt.tight_layout()
    plt.savefig('error_distributions.png')

class RealTimeInference:
    def __init__(self, model_path, camera_id=0):
        """
        Real-time inference for RC car control
        
        Args:
            model_path: Path to saved model weights
            camera_id: Camera device ID
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = RCCarModel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Setup camera
        self.cap = cv2.VideoCapture(camera_id)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def run(self):
        """Run real-time inference loop"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                control = self.model(input_tensor).cpu().numpy()[0]
            
            steering, throttle = control
            
            # Display prediction
            cv2.putText(frame, f"Steering: {steering:.4f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Throttle: {throttle:.4f}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # ------------------------------------------------------------------
            # Visual signal bars
            # Map steering (−1..1) to bar across screen width (red = left, blue = right)
            h, w = frame.shape[:2]
            bar_y = h - 40  # vertical position of the bar
            # Background bar
            cv2.rectangle(frame, (0, bar_y - 10), (w, bar_y + 10), (50, 50, 50), -1)

            # Steering bar (center = 0)
            steer_pos = int((steering + 1) / 2 * w)
            bar_color = (0, 0, 255) if steering < 0 else (255, 0, 0)
            cv2.rectangle(frame,
                          (min(steer_pos, w // 2), bar_y - 10),
                          (max(steer_pos, w // 2), bar_y + 10),
                          bar_color, -1)

            # Throttle bar (height proportional)
            throttle_height = int(abs(throttle) * 100)
            throt_x = w - 60
            cv2.rectangle(frame, (throt_x, h - 10 - throttle_height),
                          (throt_x + 40, h - 10),
                          (0, 255, 0) if throttle >= 0 else (0, 128, 255), -1)

            cv2.imshow('RC Car Imitation Learning', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

def main():
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
    model = RCCarModel()
    trained_model = train_model(model, train_loader, val_loader, num_epochs=20)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(trained_model.state_dict(), 'models/rc_car_model.pth')
    
    # Evaluate model
    evaluate_model(trained_model, test_loader)
    
    print("Training and evaluation complete. Model saved to models/rc_car_model.pth")
    
    # For real-time inference:
    # inference = RealTimeInference('models/rc_car_model.pth')
    # inference.run()

if __name__ == "__main__":
    main()