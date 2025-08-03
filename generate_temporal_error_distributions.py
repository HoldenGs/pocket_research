import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from imitation_learning_temporal import RCCarDatasetTemporal, TemporalRCCarModel
from tqdm import tqdm

def generate_error_distributions():
    """Generate error distribution plots for the trained temporal model"""
    
    print("Loading temporal model and generating error distributions...")
    
    # Set up data loading
    video_dir = "data/videos"
    control_data_path = "data/control_signals.npy"
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = RCCarDatasetTemporal(
        video_dir, control_data_path, transform=transform, 
        sequence_length=3, training_lag_ms=150, inference_lag_ms=150
    )
    
    # Create test split (same as training)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalRCCarModel(sequence_length=3)
    
    # Try to load the best model first, fall back to regular model
    try:
        model.load_state_dict(torch.load('models/best_model_temporal.pth'))
        print("Loaded best temporal model")
    except:
        try:
            model.load_state_dict(torch.load('models/rc_car_model_temporal.pth'))
            print("Loaded temporal model")
        except:
            print("Error: Could not find trained temporal model!")
            print("Please train the model first using: python imitation_learning_temporal.py")
            return
    
    model = model.to(device)
    model.eval()
    
    # Evaluate and collect errors
    steering_errors = []
    throttle_errors = []
    test_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for sequences, controls in tqdm(test_loader, desc="Generating error distributions"):
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
    
    # Print results
    print(f"\nTemporal Model Performance:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Mean Steering Error: {mean_steering_error:.4f} ({100*mean_steering_error:.2f}%)")
    print(f"Mean Throttle Error: {mean_throttle_error:.4f} ({100*mean_throttle_error:.2f}%)")
    
    # Create enhanced error distribution plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Steering error histogram
    ax1.hist(steering_errors, bins=40, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title('Steering Error Distribution (Temporal Model)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Absolute Error')
    ax1.set_ylabel('Frequency')
    ax1.axvline(mean_steering_error, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_steering_error:.4f}')
    ax1.axvline(np.median(steering_errors), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(steering_errors):.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Throttle error histogram
    ax2.hist(throttle_errors, bins=40, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title('Throttle Error Distribution (Temporal Model)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Absolute Error')
    ax2.set_ylabel('Frequency')
    ax2.axvline(mean_throttle_error, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_throttle_error:.4f}')
    ax2.axvline(np.median(throttle_errors), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(throttle_errors):.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Cumulative distribution for steering
    sorted_steering = np.sort(steering_errors)
    ax3.plot(sorted_steering, np.arange(1, len(sorted_steering) + 1) / len(sorted_steering), 
             color='blue', linewidth=2)
    ax3.set_title('Steering Error Cumulative Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Absolute Error')
    ax3.set_ylabel('Cumulative Probability')
    ax3.axvline(np.percentile(steering_errors, 90), color='red', linestyle='--',
               label=f'90th percentile: {np.percentile(steering_errors, 90):.4f}')
    ax3.axvline(np.percentile(steering_errors, 95), color='orange', linestyle='--',
               label=f'95th percentile: {np.percentile(steering_errors, 95):.4f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Cumulative distribution for throttle
    sorted_throttle = np.sort(throttle_errors)
    ax4.plot(sorted_throttle, np.arange(1, len(sorted_throttle) + 1) / len(sorted_throttle), 
             color='green', linewidth=2)
    ax4.set_title('Throttle Error Cumulative Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Absolute Error')
    ax4.set_ylabel('Cumulative Probability')
    ax4.axvline(np.percentile(throttle_errors, 90), color='red', linestyle='--',
               label=f'90th percentile: {np.percentile(throttle_errors, 90):.4f}')
    ax4.axvline(np.percentile(throttle_errors, 95), color='orange', linestyle='--',
               label=f'95th percentile: {np.percentile(throttle_errors, 95):.4f}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_distributions_temporal.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a comparison summary
    print(f"\n📊 DETAILED PERFORMANCE ANALYSIS:")
    print(f"=====================================")
    print(f"Steering Error Statistics:")
    print(f"  Mean: {mean_steering_error:.4f} ({100*mean_steering_error:.2f}%)")
    print(f"  Median: {np.median(steering_errors):.4f}")
    print(f"  Std Dev: {np.std(steering_errors):.4f}")
    print(f"  90th percentile: {np.percentile(steering_errors, 90):.4f}")
    print(f"  95th percentile: {np.percentile(steering_errors, 95):.4f}")
    print(f"  Max error: {np.max(steering_errors):.4f}")
    
    print(f"\nThrottle Error Statistics:")
    print(f"  Mean: {mean_throttle_error:.4f} ({100*mean_throttle_error:.2f}%)")
    print(f"  Median: {np.median(throttle_errors):.4f}")
    print(f"  Std Dev: {np.std(throttle_errors):.4f}")
    print(f"  90th percentile: {np.percentile(throttle_errors, 90):.4f}")
    print(f"  95th percentile: {np.percentile(throttle_errors, 95):.4f}")
    print(f"  Max error: {np.max(throttle_errors):.4f}")
    
    print(f"\n✅ Error distribution plots saved as 'error_distributions_temporal.png'")
    print(f"📈 High-resolution plots with detailed statistics generated!")

if __name__ == "__main__":
    generate_error_distributions() 