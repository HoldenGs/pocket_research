#!/usr/bin/env python3
"""
Retrain temporal model with balanced dataset and weighted loss to fix steering issues
"""
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from balanced_temporal_dataset import BalancedTemporalDataset
from imitation_learning_temporal import TemporalRCCarModel, train_temporal_model, evaluate_model


def main():
    print("Retraining Temporal Model with Balanced Dataset")
    print("=" * 50)
    
    # Configuration
    video_dir = "data/videos"
    control_data_path = "data/control_signals.npy"
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create balanced temporal dataset
    print("\nCreating balanced dataset...")
    dataset = BalancedTemporalDataset(
        video_dir, 
        control_data_path, 
        transform=transform, 
        sequence_length=3, 
        training_lag_ms=150, 
        inference_lag_ms=150,
        min_steering_threshold=0.03,  # Keep sequences with |steering| > 0.03
        steering_sample_ratio=0.6     # 60% high-steering, 40% low-steering
    )
    
    if len(dataset) == 0:
        print("ERROR: No data found! Check your video_dir and control_data_path")
        return
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=2)
    
    print(f"\nDataset splits:")
    print(f"  Training: {len(train_dataset)} sequences")
    print(f"  Validation: {len(val_dataset)} sequences")
    print(f"  Test: {len(test_dataset)} sequences")
    
    # Create model
    print("\nCreating temporal model...")
    model = TemporalRCCarModel(sequence_length=3)
    
    # Train with weighted loss (steering gets 20x weight)
    print("\nTraining with steering-weighted loss...")
    trained_model = train_temporal_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=300,
        lr=0.0005,  # Slightly lower learning rate
        steering_weight=20.0  # High weight for steering
    )
    
    # Save the improved model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/rc_car_model_temporal_balanced.pth'
    torch.save(trained_model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Evaluate
    print("\nEvaluating model...")
    test_loss, steering_error, throttle_error = evaluate_model(trained_model, test_loader)
    
    print("\n" + "=" * 60)
    print("BALANCED TEMPORAL MODEL RESULTS")
    print("=" * 60)
    print("Improvements:")
    print("  ✓ Balanced dataset (more steering diversity)")
    print("  ✓ Weighted loss (20x steering weight)")
    print("  ✓ Lower learning rate for stability")
    print(f"\nFinal errors:")
    print(f"  Steering error: {steering_error:.4f}")
    print(f"  Throttle error: {throttle_error:.4f}")
    print("=" * 60)
    print(f"\nTo use this model:")
    print(f"python inference_controller.py <racer_ip> {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main() 