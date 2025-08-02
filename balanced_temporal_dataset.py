import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
from tqdm import tqdm
from imitation_learning_temporal import TemporalRCCarModel


class BalancedTemporalDataset(Dataset):
    def __init__(self, video_dir, control_data_path, transform=None, 
                 sequence_length=3, training_lag_ms=150, inference_lag_ms=150,
                 min_steering_threshold=0.05, steering_sample_ratio=0.7):
        """
        Balanced temporal dataset that keeps more frames with significant steering
        
        Args:
            min_steering_threshold: Minimum absolute steering value to always keep
            steering_sample_ratio: Ratio of high-steering samples to low-steering samples
        """
        self.video_dir = video_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.training_lag_seconds = training_lag_ms / 1000.0
        self.inference_lag_seconds = inference_lag_ms / 1000.0
        self.total_lag_compensation = self.training_lag_seconds + self.inference_lag_seconds
        self.min_steering_threshold = min_steering_threshold
        self.steering_sample_ratio = steering_sample_ratio
        
        print(f"Balanced Temporal Dataset Configuration:")
        print(f"  Sequence length: {sequence_length} frames")
        print(f"  Min steering threshold: {min_steering_threshold}")
        print(f"  Steering sample ratio: {steering_sample_ratio}")
        print(f"  Total lag compensation: {self.total_lag_compensation:.3f}s")
        
        # Load control data
        self.control_data = np.load(control_data_path)
        
        # Process videos to create balanced temporal sequences
        self.frame_sequences = []
        self.controls = []
        
        self._process_data_balanced()
    
    def _process_data_balanced(self):
        """Process videos with balanced sampling for steering diversity"""
        video_files = sorted([f for f in os.listdir(self.video_dir)
                             if f.endswith(('.mp4', '.avi')) and not f.startswith('._')])
        
        high_steering_sequences = []
        low_steering_sequences = []
        total_sequences = 0
        
        for video_file in tqdm(video_files, desc="Processing videos (balanced)"):
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
            
            # Create temporal sequences and categorize by steering magnitude
            for i in range(self.sequence_length - 1, len(frames)):
                total_sequences += 1
                
                # Get sequence of frames (oldest to newest)
                frame_sequence = frames[i - self.sequence_length + 1:i + 1]
                current_timestamp = frame_timestamps[i]
                
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
                
                # Categorize by steering magnitude
                steering_magnitude = abs(control_signal[0])
                sequence_data = (frame_sequence, control_signal)
                
                if steering_magnitude >= self.min_steering_threshold:
                    high_steering_sequences.append(sequence_data)
                else:
                    low_steering_sequences.append(sequence_data)
        
        # Balance the dataset
        num_high_steering = len(high_steering_sequences)
        num_low_steering_to_keep = int(num_high_steering / self.steering_sample_ratio)
        
        # Randomly sample low steering sequences
        if len(low_steering_sequences) > num_low_steering_to_keep:
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(low_steering_sequences), 
                                     num_low_steering_to_keep, replace=False)
            low_steering_sequences = [low_steering_sequences[i] for i in indices]
        
        # Combine sequences
        all_sequences = high_steering_sequences + low_steering_sequences
        
        # Shuffle the combined dataset
        np.random.seed(42)
        np.random.shuffle(all_sequences)
        
        # Extract frame sequences and controls
        self.frame_sequences = [seq[0] for seq in all_sequences]
        self.controls = [seq[1] for seq in all_sequences]
        
        print(f"Balanced dataset results:")
        print(f"  Total sequences processed: {total_sequences}")
        print(f"  High steering sequences: {num_high_steering}")
        print(f"  Low steering sequences kept: {len(low_steering_sequences)}")
        print(f"  Final dataset size: {len(self.frame_sequences)}")
        print(f"  High steering ratio: {100 * num_high_steering / len(self.frame_sequences):.1f}%")
    
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