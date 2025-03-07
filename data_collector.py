#!/usr/bin/env python3
import os
import time
import json
import cv2
import numpy as np
import threading
import queue
from datetime import datetime

class DataCollector:
    def __init__(self, base_dir="data", fps=30, video_format="mp4", video_codec="avc1"):
        """Initialize the data collector.
        
        Args:
            base_dir: Base directory to store collected data
            fps: Frame rate for recorded video
            video_format: Format for video recording (mp4, avi)
            video_codec: Video codec to use (avc1, XVID, etc)
        """
        self.base_dir = base_dir
        self.fps = fps
        self.video_format = video_format
        self.video_codec = video_codec
        self.recording = False
        self.session_dir = None
        self.frame_count = 0
        self.control_queue = queue.Queue()
        self.frame_queue = queue.Queue(maxsize=30)  # Limit queue size to prevent memory issues
        self.worker_thread = None
        self.control_data = []
        self.video_writer = None
        self.start_time = None
        
    def toggle_recording(self):
        """Toggle recording state."""
        if not self.recording:
            self.start_recording()
            return True
        else:
            self.stop_recording()
            return False
    
    def start_recording(self):
        """Start a new recording session."""
        if self.recording:
            print("Already recording!")
            return
        
        # Create a new session directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.base_dir, f"session_{timestamp}")
        
        # Create directory structure
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Reset state
        self.frame_count = 0
        self.control_data = []
        self.recording = True
        self.start_time = time.time()
        
        # Start worker thread to process queues
        self.worker_thread = threading.Thread(target=self.process_queues)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        print(f"Started recording session in {self.session_dir}")
    
    def stop_recording(self):
        """Stop the current recording session and save metadata."""
        if not self.recording:
            print("Not recording!")
            return
        
        self.recording = False
        
        # Wait for queues to be processed
        if not self.frame_queue.empty() or not self.control_queue.empty():
            print("Waiting for remaining data to be saved...")
            time.sleep(1.0)  # Give time for the worker to process remaining items
        
        # Save control data as JSON
        control_file = os.path.join(self.session_dir, "control_data.json")
        with open(control_file, 'w') as f:
            json.dump(self.control_data, f, indent=2)
            
        # Save metadata
        metadata = {
            "frame_count": self.frame_count,
            "fps": self.fps,
            "duration": time.time() - self.start_time,
            "start_time": self.start_time,
            "video_format": self.video_format,
            "video_codec": self.video_codec
        }
        
        # Save session metadata
        metadata_file = os.path.join(self.session_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Close video writer if it exists
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            
        print(f"Recording stopped. Saved {self.frame_count} frames and control data.")
        print(f"Data saved to {self.session_dir}")
    
    def add_frame(self, frame):
        """Add a video frame to the recording queue.
        
        Args:
            frame: OpenCV image frame
        """
        if not self.recording:
            return
        
        try:
            timestamp = time.time()
            # Add to queue non-blocking (drop frames if queue is full)
            self.frame_queue.put((timestamp, frame.copy()), block=False)
        except queue.Full:
            print("Frame queue full, dropping frame")
    
    def add_control_signal(self, throttle, steering):
        """Add control signal to the recording queue.
        
        Args:
            throttle: Throttle value (-1.0 to 1.0)
            steering: Steering value (-1.0 to 1.0)
        """
        if not self.recording:
            return
        
        try:
            timestamp = time.time()
            self.control_queue.put((timestamp, throttle, steering))
        except Exception as e:
            print(f"Error adding control signal: {e}")
    
    def process_queues(self):
        """Worker thread to process frame and control queues."""
        while self.recording or not self.frame_queue.empty() or not self.control_queue.empty():
            # Process frames
            try:
                while not self.frame_queue.empty():
                    timestamp, frame = self.frame_queue.get(block=False)
                    self.save_frame(timestamp, frame)
                    self.frame_queue.task_done()
            except queue.Empty:
                pass
            
            # Process control signals
            try:
                while not self.control_queue.empty():
                    timestamp, throttle, steering = self.control_queue.get(block=False)
                    self.save_control(timestamp, throttle, steering)
                    self.control_queue.task_done()
            except queue.Empty:
                pass
            
            # Sleep to prevent CPU overuse
            time.sleep(0.01)
    
    def save_frame(self, timestamp, frame):
        """Save a frame to the video file.
        
        Args:
            timestamp: Frame timestamp
            frame: OpenCV image frame
        """
        # Initialize video writer if needed
        if self.video_writer is None:
            height, width = frame.shape[:2]
            video_path = os.path.join(self.session_dir, f"video.{self.video_format}")
            
            # Determine the FourCC code for the codec
            if self.video_codec == "avc1":  # H.264
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
            elif self.video_codec == "XVID":  # XVID (more compatible)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            elif self.video_codec == "MJPG":  # Motion JPEG
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            else:
                # Default to a common codec
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
            
            self.video_writer = cv2.VideoWriter(
                video_path, fourcc, self.fps, (width, height)
            )
            
            if not self.video_writer.isOpened():
                print(f"Failed to open video writer with codec {self.video_codec}. Trying XVID...")
                # Fallback to XVID if the first codec fails
                self.video_writer = cv2.VideoWriter(
                    video_path, cv2.VideoWriter_fourcc(*'XVID'), self.fps, (width, height)
                )
                if not self.video_writer.isOpened():
                    print("Failed to open video writer with XVID too. Using MJPG...")
                    # Last resort - try MJPG
                    self.video_writer = cv2.VideoWriter(
                        video_path, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (width, height)
                    )
        
        # Write the frame to video
        if self.video_writer and self.video_writer.isOpened():
            self.video_writer.write(frame)
            self.frame_count += 1
        else:
            print("Error: Video writer not initialized or not opened!")
    
    def save_control(self, timestamp, throttle, steering):
        """Save control signal to memory (will be written to JSON later).
        
        Args:
            timestamp: Control signal timestamp
            throttle: Throttle value
            steering: Steering value
        """
        self.control_data.append({
            "timestamp": timestamp,
            "throttle": float(throttle),
            "steering": float(steering),
            "frame_idx": self.frame_count - 1  # Index of the last saved frame
        })
    
    def get_status(self):
        """Get current recording status.
        
        Returns:
            Dictionary with recording status and frame count
        """
        return {
            "recording": self.recording,
            "frame_count": self.frame_count,
            "session": os.path.basename(self.session_dir) if self.session_dir else None
        }

# Utility function to extract frames from video for training
def extract_frame_from_video(video_path, frame_idx):
    """Extract a specific frame from a video file.
    
    Args:
        video_path: Path to the video file
        frame_idx: The index of the frame to extract (0-based)
        
    Returns:
        The extracted frame as a NumPy array, or None if extraction failed
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return None
        
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file: {video_path}")
        return None
    
    # Get frame count to validate the requested index
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx >= frame_count or frame_idx < 0:
        print(f"Invalid frame index: {frame_idx}. Video has {frame_count} frames.")
        cap.release()
        return None
    
    # Set the position and read the requested frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Failed to read frame {frame_idx} from video.")
        return None
        
    return frame

# Utility function to format data for training
def prepare_training_data(session_dir, extract_frames=False, max_samples=None, output_dir=None):
    """Prepare collected data for training by creating a dataset with video frames and control signals.
    
    Args:
        session_dir: Path to session directory
        extract_frames: Whether to extract frames to individual files (default: False)
        max_samples: Maximum number of samples to extract (default: all)
        output_dir: Directory to save extracted frames (default: frames subdirectory)
        
    Returns:
        Path to the prepared dataset
    """
    print(f"Preparing training data from {session_dir}...")
    
    # Load metadata
    metadata_file = os.path.join(session_dir, "metadata.json")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        
    # Load control data
    control_file = os.path.join(session_dir, "control_data.json")
    with open(control_file, 'r') as f:
        control_data = json.load(f)
    
    # Limit number of samples if requested
    if max_samples and len(control_data) > max_samples:
        print(f"Limiting to {max_samples} samples from {len(control_data)} available")
        # Evenly sample from the data
        step = len(control_data) / max_samples
        indices = [int(i * step) for i in range(max_samples)]
        control_data = [control_data[i] for i in indices]
    
    # Create dataset with frame references and control signals
    dataset = []
    video_path = os.path.join(session_dir, f"video.{metadata['video_format']}")
    
    # Prepare directory for extracted frames if needed
    if extract_frames:
        if output_dir is None:
            output_dir = os.path.join(session_dir, "frames")
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each control data point
    for i, control in enumerate(control_data):
        frame_idx = control["frame_idx"]
        
        if extract_frames:
            # Extract and save the frame
            frame = extract_frame_from_video(video_path, frame_idx)
            if frame is not None:
                frame_file = f"frame_{frame_idx:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_file)
                cv2.imwrite(frame_path, frame)
                
                # Add to dataset with relative path
                rel_frame_path = os.path.join("frames", frame_file) if output_dir == os.path.join(session_dir, "frames") else frame_file
            else:
                print(f"Warning: Could not extract frame {frame_idx}")
                continue
        else:
            # Just reference the frame in the video
            rel_frame_path = f"video.{metadata['video_format']}#{frame_idx}"
        
        # Add to dataset
        dataset.append({
            "frame": rel_frame_path,
            "frame_idx": frame_idx,
            "throttle": control["throttle"],
            "steering": control["steering"],
            "timestamp": control["timestamp"]
        })
    
    # Save dataset file
    dataset_file = os.path.join(session_dir, "dataset.json")
    with open(dataset_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Dataset prepared: {len(dataset)} samples")
    print(f"Dataset saved to {dataset_file}")
    
    return dataset_file


if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    # Toggle recording on
    collector.toggle_recording()
    
    # Example of collecting data (in real usage, this would come from the controller)
    for i in range(100):
        # Create a dummy frame (black image with frame number)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add frame and random control signal
        collector.add_frame(frame)
        collector.add_control_signal(throttle=np.random.uniform(-1.0, 1.0), 
                                    steering=np.random.uniform(-1.0, 1.0))
        time.sleep(0.1)
    
    # Toggle recording off
    collector.toggle_recording()
    
    # Prepare data for training, extracting frames
    if collector.session_dir:
        prepare_training_data(collector.session_dir, extract_frames=True) 