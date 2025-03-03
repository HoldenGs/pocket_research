# RC Car Imitation Learning Pipeline

This repository contains an imitation learning pipeline for training an RC car to mimic human driving behavior using video data and control signals.

## Overview

The system uses:
- Video data from the RC car's camera
- Control signal data (steering and throttle)
- A deep learning model to predict control signals from video frames

## Directory Structure

```
imitation_learning/
├── data/                    # Data directory
│   ├── videos/              # Video files
│   └── control_signals.npy  # Control signal data
├── models/                  # Saved models
├── imitation_learning.py    # Main implementation
└── README.md                # This file
```

## Data Format

### Video Data
- Video files should be stored in the `data/videos/` directory
- File naming convention: `<timestamp>.mp4` or `<timestamp>.avi`

### Control Signal Data
- Control signals should be stored in `data/control_signals.npy`
- Format: Numpy array with shape (N, 3)
  - Column 0: Timestamp (seconds)
  - Column 1: Steering value (normalized between -1 and 1)
  - Column 2: Throttle value (should be normalized between 0 and 1, but currently has a zero value around 0.5 and a max positive throttle of 1.5)

You should tune the values for yourself as they may not be perfectly normalized, and could have different values for each racer.

## Usage

### 1. Data Collection

Collect video data and control signals:
- Record video feed from the RC car
- Record corresponding control signals with timestamps
- Ensure synchronization between video and control data

### 2. Training

Run the main script:

```bash
python imitation_learning.py
```

The script will:
1. Process the video files and match frames with control signals
2. Split the dataset into training, validation, and test sets
3. Train the model and save it to `models/rc_car_model.pth`
4. Evaluate the model and save performance graphs

### 3. Real-time Inference

For deploying the trained model on the RC car:

```python
from imitation_learning import RealTimeInference

# Create inference object
inference = RealTimeInference('models/rc_car_model.pth')

# Run inference loop
inference.run()
```

## Model Architecture

The model uses a ResNet18 backbone with a custom regression head:
- Feature extraction: Pre-trained ResNet18
- Control head: MLP with 2 outputs (steering and throttle)

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- numpy
- OpenCV (cv2)
- matplotlib
- tqdm

Install requirements:

```bash
pip install torch torchvision numpy opencv-python matplotlib tqdm
```

## Future Improvements

- Data augmentation for improved generalization
- Model distillation for faster inference
- Integration with ROS for robotic control
- Support for action branching (different scenarios)
- Online learning capabilities
