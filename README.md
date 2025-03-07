# DeepRacer Imitation Learning

This project contains the necessary components to run the DeepRacer, record data, and train a model to mimic human behavior on the DeepRacer using imitation learning. Currently, the imitation learning pipeline has not been completed, but the controller is in a working state.

## Overview

This project provides a unified control interface for DeepRacer that combines:

1. **Unified Controller** - Combined servo control and video streaming
2. **Interactive UI** - Real-time visualization of control inputs and video feed
3. **Data Collection** - Automated recording of driving data
4. **Imitation Learning Trainer** - Trainer for imitation learning model on collected data

The controller system consists of two main components:

- **controller_local.py**: Runs on your computer, handles keyboard inputs and video display
- **controller_racer.py**: Runs on the DeepRacer, handles motor control and video streaming

## Getting Started

### Prerequisites

- Python 3.6+
- OpenCV
- NumPy
- pynput (for keyboard control)
- AWS DeepRacer with ROS2 setup

### Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/pocket_racers.git
cd pocket_racers/imitation_learning
```

#### Virtual Environment Setup (Recommended)

It's recommended to use a virtual environment to avoid conflicts with other Python packages:

```bash
# Create a virtual environment
python -m venv pocket_research_env

# Activate the virtual environment
# On Windows:
pocket_research_env\Scripts\activate
# On macOS/Linux:
source ./pocket_research_env/bin/activate
```

Once the virtual environment is activated, install the dependencies:

```bash
# Install dependencies
pip install -r requirements.txt
```

Your terminal prompt should change to indicate that the virtual environment is active. When you're done working on the project, you can deactivate the virtual environment:

```bash
deactivate
```

### Running the Controller


1. **On the DeepRacer (Racer):**
   When running on the DeepRacer, ensure you're logged in as root and source the following files:
   ```bash
   source /opt/ros/foxy/setup.bash
   source ~/deepracer_ws/aws-deepracer-interfaces-pkg/install/setup.bash
   ```
   If you don't have aws-deepracer-interfaces-pkg installed, follow the instructions on the package's GitHub page: https://github.com/aws-deepracer/aws-deepracer-interfaces-pkg

   By default, the DeepRacer also runs a node that takes control of the USB video camera. We want to kill that process by finding it with lsof:
   ```bash
   lsof /dev/video0 # or video1 sometimes
   ```
   ```bash
   kill <pid of offending process>
   ```
   Now, you're ready to start the controller:
   ```bash
   python controller_racer.py
   ```

3. **On your computer (Local):**
   ```bash
   python controller_local.py <racer_ip_address>
   ```
   Replace `<racer_ip_address>` with the IP address of your DeepRacer.

## Controller Usage

| Key | Function |
|-----|----------|
| W | Forward throttle (up to limit) |
| S | Reverse throttle (up to limit) |
| A | Left steering (up to limit) |
| D | Right steering (up to limit) |
| UP | Increase throttle limit (+0.2) |
| DOWN | Decrease throttle limit (-0.2) |
| LEFT | Decrease steering limit (-0.2) |
| RIGHT | Increase steering limit (+0.2) |
| V | Toggle video display |
| R | Toggle data recording |
| ESC | Exit |

## Data Collection

### Recording Sessions

Press the `R` key to toggle recording on and off. Each recording session is stored in a timestamped directory:

```
data/session_YYYYMMDD_HHMMSS/
```

### Data Format

The data is stored in an efficient video format, along with synchronized control signals:

```
session_YYYYMMDD_HHMMSS/
├── video.mp4            # Video recording of frames
├── control_data.json    # Control signals with timestamps and frame indices
├── metadata.json        # Recording session metadata
└── dataset.json         # Prepared dataset for training
```

### Preparing Training Data

The system includes tools to prepare the data for training:

```python
from data_collector import prepare_training_data

# Basic usage - references frames in the video file
prepare_training_data("data/session_YYYYMMDD_HHMMSS")

# Extract frames as individual images
prepare_training_data("data/session_YYYYMMDD_HHMMSS", extract_frames=True)

# Limit to a maximum number of samples
prepare_training_data("data/session_YYYYMMDD_HHMMSS", max_samples=1000)

# Save extracted frames to a custom directory
prepare_training_data("data/session_YYYYMMDD_HHMMSS", 
                      extract_frames=True, 
                      output_dir="my_training_data")
```

## Imitation Learning Integration

The collected data is specifically formatted for an imitation learning trainer:

1. Each control signal is precisely matched with its corresponding video frame
2. The dataset includes timestamps, frame indices, throttle, and steering values
3. Frames can be referenced directly in the video file or extracted as needed

For frameworks that require individual image files, use `extract_frames=True` when preparing the data.

## Advanced Configuration

### Video Settings

To customize video recording, modify the `DataCollector` initialization:

```python
from data_collector import DataCollector

# Custom settings
collector = DataCollector(
    base_dir="data",      # Base directory for storing data
    fps=30,               # Video frame rate
    video_format="mp4",   # Video format (mp4, avi)
    video_codec="avc1"    # Video codec (avc1, XVID, MJPG)
)
```

## Troubleshooting

### Video Display Issues

- On macOS, OpenCV UI must run in the main thread
- If video appears blocky, try reducing resolution on the racer side
- If high latency is persistent, consider modifying the DeepRacer to run on your phone's hotspot by using the USB configuration method

### Controller Connection Issues

- Verify the IP address is correct
- Check that ports 9999 (control) and 5005 (video) are not blocked
- Press 'C' to check connection status

## Future Improvements

- Data augmentation for improved generalization
- Model distillation for faster inference
- Integration with ROS for robotic control
- Support for action branching (different scenarios)
- Online learning capabilities


## License

[MIT License](LICENSE)
