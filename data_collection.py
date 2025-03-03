import os
import time
import numpy as np
import cv2
import argparse
import threading
import queue
import json
from datetime import datetime

class DataCollector:
    """
    Collects and synchronizes video and control signal data for RC car imitation learning.
    """
    def __init__(self, config_path='config.json'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create data directories
        os.makedirs(self.config['video_dir'], exist_ok=True)
        control_dir = os.path.dirname(self.config['control_data_path'])
        os.makedirs(control_dir, exist_ok=True)
        
        # Initialize control signal storage
        self.control_signals = []
        self.start_time = None
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.config['camera_id'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['frame_width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['frame_height'])
        self.cap.set(cv2.CAP_PROP_FPS, self.config['fps'])
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera.")
        
        # Initialize video writer
        self.video_writer = None
        self.current_video_path = None
        self.frame_count = 0
        self.max_frames_per_video = self.config['max_seconds_per_video'] * self.config['fps']
        
        # Initialize control input
        # This should be modified to use your actual control input method
        # (e.g., game controller, custom hardware, etc.)
        self.initialize_control_input()
        
        # Threading
        self.is_running = False
        self.video_queue = queue.Queue(maxsize=30)  # Buffer a few frames
        self.control_queue = queue.Queue(maxsize=100)  # Buffer control signals
    
    def initialize_control_input(self):
        """
        Initialize the control input method.
        This is a placeholder - implement based on your actual control input device.
        """
        # Example: Initialize joystick if using pygame
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"Initialized joystick: {self.joystick.get_name()}")
            else:
                print("No joystick detected. Using keyboard fallback.")
                self.joystick = None
        except ImportError:
            print("Pygame not available. Using keyboard fallback.")
            self.joystick = None
    
    def read_control_input(self):
        """
        Read control signals from input device.
        This is a placeholder - implement based on your actual control input device.
        
        Returns:
            tuple: (steering, throttle) where both values are normalized
        """
        if self.joystick:
            # Example using pygame joystick
            import pygame
            pygame.event.pump()
            
            # Read steering (left-right) from axis 0
            steering = self.joystick.get_axis(0)  # -1 (left) to 1 (right)
            
            # Read throttle (forward-backward) from axis 1 or triggers
            # Depending on your controller, you might need to adjust this
            throttle = -self.joystick.get_axis(1)  # -1 (backward) to 1 (forward)
            throttle = (throttle + 1) / 2  # Normalize to 0-1
            
            return steering, throttle
        else:
            # Keyboard fallback
            key = cv2.waitKey(1) & 0xFF
            steering, throttle = 0.0, 0.0
            
            # Simple keyboard controls
            if key == ord('a'):  # left
                steering = -1.0
            elif key == ord('d'):  # right
                steering = 1.0
            
            if key == ord('w'):  # forward
                throttle = 1.0
            elif key == ord('s'):  # backward
                throttle = 0.0
            
            return steering, throttle
    
    def start_new_video(self):
        """Start a new video file"""
        timestamp = time.time()
        filename = f"{timestamp:.6f}.mp4"
        video_path = os.path.join(self.config['video_dir'], filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            video_path,
            fourcc,
            self.config['fps'],
            (self.config['frame_width'], self.config['frame_height'])
        )
        
        self.current_video_path = video_path
        self.frame_count = 0
        print(f"Started new video: {video_path}")
        
        return timestamp
    
    def video_capture_thread(self):
        """Thread function for capturing video frames"""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame")
                time.sleep(0.01)
                continue
            
            # Put frame in queue with timestamp
            frame_time = time.time() - self.start_time
            try:
                self.video_queue.put((frame_time, frame), block=False)
            except queue.Full:
                # If queue is full, drop the oldest frame
                try:
                    self.video_queue.get_nowait()
                    self.video_queue.put((frame_time, frame), block=False)
                except:
                    pass
    
    def control_capture_thread(self):
        """Thread function for capturing control signals"""
        while self.is_running:
            # Read control input
            steering, throttle = self.read_control_input()
            
            # Record timestamp
            control_time = time.time() - self.start_time
            
            # Add to queue
            try:
                self.control_queue.put((control_time, steering, throttle), block=False)
            except queue.Full:
                # If queue is full, drop the oldest control signal
                try:
                    self.control_queue.get_nowait()
                    self.control_queue.put((control_time, steering, throttle), block=False)
                except:
                    pass
            
            # Sleep to maintain control sampling rate
            time.sleep(1.0 / self.config['control_hz'])
    
    def processing_thread(self):
        """Thread function for processing and saving data"""
        video_timestamp = self.start_new_video()
        
        while self.is_running:
            # Process video frames
            try:
                frame_time, frame = self.video_queue.get(timeout=0.1)
                
                # Start a new video if needed
                if self.frame_count >= self.max_frames_per_video:
                    self.video_writer.release()
                    video_timestamp = self.start_new_video()
                
                # Write frame to video
                self.video_writer.write(frame)
                self.frame_count += 1
                
                # Display frame with overlay
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Time: {frame_time:.2f}s", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Get latest control signal for display
                if not self.control_queue.empty():
                    control_time, steering, throttle = self.control_signals[-1] if self.control_signals else (0, 0, 0)
                    cv2.putText(display_frame, f"Steering: {steering:.2f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Throttle: {throttle:.2f}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('RC Car Data Collection', display_frame)
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.is_running = False
                    break
                
            except queue.Empty:
                pass
            
            # Process control signals
            while not self.control_queue.empty():
                control_data = self.control_queue.get()
                self.control_signals.append(control_data)
    
    def start(self):
        """Start data collection"""
        print("Starting data collection...")
        print("Press 'q' to stop recording")
        
        self.is_running = True
        self.start_time = time.time()
        
        # Start threads
        self.video_thread = threading.Thread(target=self.video_capture_thread)
        self.control_thread = threading.Thread(target=self.control_capture_thread)
        self.process_thread = threading.Thread(target=self.processing_thread)
        
        self.video_thread.start()
        self.control_thread.start()
        self.process_thread.start()
        
        # Wait for threads to finish
        self.video_thread.join()
        self.control_thread.join()
        self.process_thread.join()
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources and save control data"""
        print("Stopping data collection...")
        
        # Release video writer and camera
        if self.video_writer:
            self.video_writer.release()
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Save control signals to numpy file
        if self.control_signals:
            control_array = np.array(self.control_signals)
            np.save(self.config['control_data_path'], control_array)
            print(f"Saved {len(self.control_signals)} control signals to {self.config['control_data_path']}")
        
        print("Data collection complete")

def main():
    parser = argparse.ArgumentParser(description='RC Car Data Collection')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # If config doesn't exist, create a default one
    if not os.path.exists(args.config):
        default_config = {
            "camera_id": 0,
            "frame_width": 640,
            "frame_height": 480,
            "fps": 30,
            "control_hz": 50,
            "max_seconds_per_video": 60,
            "video_dir": "data/videos",
            "control_data_path": "data/control_signals.npy"
        }
        
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        print(f"Created default configuration at {args.config}")
    
    # Create and start data collector
    data_collector = DataCollector(args.config)
    data_collector.start()

if __name__ == "__main__":
    main()