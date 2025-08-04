#!/usr/bin/env python3
import os
import socket
import json
import sys
import threading
import time
import signal
import cv2
import numpy as np
import queue
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from imitation_learning import RCCarModel
from imitation_learning_temporal import TemporalRCCarModel

# Constants (same as controller_local.py)
VIDEO_RECEIVE_PORT = 5005
UI_WIDTH = 1300
UI_HEIGHT = 820
TELEMETRY_HEIGHT = 300

# UI Colors (BGR format)
COLOR_BG = (50, 50, 50)
COLOR_TEXT = (240, 240, 240)
COLOR_CONNECTED = (0, 230, 0)
COLOR_DISCONNECTED = (0, 0, 230)
COLOR_THROTTLE_FORWARD = (0, 240, 120)
COLOR_THROTTLE_REVERSE = (0, 120, 240)
COLOR_STEERING = (240, 180, 0)
COLOR_GRID = (100, 100, 100)
COLOR_PREDICTION = (0, 255, 255)  # Yellow for AI predictions


class SteeringFixedTemporalModel(nn.Module):
    """Steering-aware temporal model from fix_steering_problem.py"""
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


class InferenceController:
    def __init__(self, server_ip, model_path, server_port=9999):
        self.server_ip = server_ip
        self.server_port = server_port
        self.model_path = model_path
        
        # Current predicted control values
        self.predicted_angle = 0.0
        self.predicted_throttle = 0.0
        
        # Connection tracking
        self.last_successful_send = 0
        self.connection_ok = False
        
        # Video variables
        self.frame_queue = queue.Queue(maxsize=10)
        self.ui_frame = self.create_base_ui_frame()
        self.last_video_frame = None
        
        # Model and inference setup
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self.is_temporal_model = "temporal" in model_path.lower()
        self.is_steering_fixed_model = "steering_fixed" in model_path.lower()
        self.sequence_length = 3  # For temporal models
        self.frame_buffer = []  # Store recent frames for temporal inference
        self.load_model()
        
        # UDP client setup for control commands
        self.setup_udp_client()
        
        # Control flags
        self.running = True
        self.inference_enabled = True
        
        # Start response receiver thread
        self.receiver_thread = threading.Thread(target=self.response_receiver)
        self.receiver_thread.daemon = True
        
        # Start video receiver thread
        self.video_thread = threading.Thread(target=self.video_receiver)
        self.video_thread.daemon = True
        
        # Start inference and control sender thread
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.daemon = True
        
    def load_model(self):
        """Load the trained model and setup preprocessing."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model from {self.model_path}")
        
        # Choose model class based on model type
        if self.is_steering_fixed_model:
            print("Loading steering-fixed temporal model...")
            self.model = SteeringFixedTemporalModel(
                sequence_length=self.sequence_length)
            self.is_temporal_model = True  # Steering-fixed is always temporal
        elif self.is_temporal_model:
            print("Loading temporal model...")
            self.model = TemporalRCCarModel(
                sequence_length=self.sequence_length)
        else:
            print("Loading regular model...")
            self.model = RCCarModel()
            
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms (same as in imitation_learning.py)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if self.is_steering_fixed_model:
            model_type = "steering-fixed temporal"
        elif self.is_temporal_model:
            model_type = "temporal"
        else:
            model_type = "regular"
        print(f"{model_type.capitalize()} model loaded successfully "
              f"on device: {self.device}")
    
    def create_base_ui_frame(self):
        """Create the base UI frame with fixed elements."""
        frame = np.zeros((UI_HEIGHT, UI_WIDTH, 3), dtype=np.uint8)
        frame[:] = COLOR_BG
        
        # Draw separator line between video and telemetry areas
        cv2.line(frame, (0, UI_HEIGHT - TELEMETRY_HEIGHT), 
                 (UI_WIDTH, UI_HEIGHT - TELEMETRY_HEIGHT), COLOR_GRID, 2)
        
        # Draw title
        title_y = UI_HEIGHT - TELEMETRY_HEIGHT + 25
        cv2.putText(frame, "AI Inference Controller", (20, title_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT, 2)
        
        return frame
    
    def draw_telemetry(self, frame, video_frame=None):
        """Draw telemetry information on the UI frame."""
        if frame is None:
            frame = self.create_base_ui_frame()
        
        # Add the video frame if available
        if video_frame is not None:
            self.last_video_frame = video_frame
        
        # Display video frame
        if self.last_video_frame is not None:
            # Resize the video frame to fit the UI
            video_height = UI_HEIGHT - TELEMETRY_HEIGHT
            video_shape = self.last_video_frame.shape
            video_width = int(video_shape[1] * video_height / video_shape[0])
            
            # Center the video if it's smaller than the UI width
            if video_width < UI_WIDTH:
                x_offset = (UI_WIDTH - video_width) // 2
                resized_video = cv2.resize(
                    self.last_video_frame, (video_width, video_height))
                frame[0:video_height, 
                      x_offset:x_offset+video_width] = resized_video
            else:
                # If video is wider, resize to fit width and crop height
                video_width = UI_WIDTH
                video_shape = self.last_video_frame.shape
                video_height_scaled = int(
                    video_shape[0] * video_width / video_shape[1])
                resized_video = cv2.resize(
                    self.last_video_frame, (video_width, video_height_scaled))
                # Crop to fit
                crop_y = ((video_height_scaled - video_height) // 2 
                          if video_height_scaled > video_height else 0)
                end_y = crop_y + video_height
                frame[0:video_height, 0:video_width] = resized_video[
                    crop_y:end_y, 0:video_width]
            

        
        # Draw telemetry panel
        panel_start_y = UI_HEIGHT - TELEMETRY_HEIGHT + 60
        
        # Connection status
        left_col_x = 50
        status_y = panel_start_y
        cv2.putText(frame, "Connection:", (left_col_x, status_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 1)
        
        status_color = COLOR_CONNECTED if self.connection_ok else COLOR_DISCONNECTED
        status_text = "Connected" if self.connection_ok else "Disconnected"
        cv2.circle(frame, (left_col_x + 180, status_y - 5), 10, status_color, -1)
        cv2.putText(frame, status_text, (left_col_x + 200, status_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 1)
        
        # Server info
        server_y = status_y + 40
        cv2.putText(frame, f"DeepRacer: {self.server_ip}:{self.server_port}", 
                  (left_col_x, server_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 1)
        
        # Model info
        model_y = server_y + 30
        model_text = f"Model: {os.path.basename(self.model_path)}"
        cv2.putText(frame, model_text, (left_col_x, model_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 1)
        
        # Inference status
        inference_y = model_y + 30
        inference_color = COLOR_CONNECTED if self.inference_enabled else COLOR_DISCONNECTED
        inference_text = "AI Enabled" if self.inference_enabled else "AI Disabled"
        cv2.putText(frame, f"Status: {inference_text}", (left_col_x, inference_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, inference_color, 1)
        
        # Controls section
        controls_y = inference_y + 50
        cv2.putText(frame, "Controls:", (left_col_x, controls_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 1)
        
        line_height = 30
        cv2.putText(frame, "SPACE: Toggle AI", 
                   (left_col_x, controls_y + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
        cv2.putText(frame, "ESC: Exit", 
                   (left_col_x, controls_y + line_height*2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
        
        # Predicted control values (center)
        center_col_x = UI_WIDTH // 3 + 50
        bar_width = 200
        
        # Throttle visualization
        throttle_y = panel_start_y
        cv2.putText(frame, "AI Throttle:", (center_col_x, throttle_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_PREDICTION, 1)
        
        # Draw throttle bar
        self.draw_control_bar(frame, center_col_x, throttle_y + 20, bar_width, 
                            self.predicted_throttle, COLOR_THROTTLE_FORWARD, COLOR_THROTTLE_REVERSE)
        
        cv2.putText(frame, f"{self.predicted_throttle:.3f}", 
                  (center_col_x + bar_width + 20, throttle_y + 35), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_PREDICTION, 1)
        
        # Steering visualization (right)
        right_col_x = UI_WIDTH * 2 // 3 + 50
        
        steering_y = panel_start_y
        cv2.putText(frame, "AI Steering:", (right_col_x, steering_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_PREDICTION, 1)
        
        # Draw steering bar
        self.draw_control_bar(frame, right_col_x, steering_y + 20, bar_width, 
                            self.predicted_angle, COLOR_STEERING, COLOR_STEERING)
        
        cv2.putText(frame, f"{self.predicted_angle:.3f}", 
                  (right_col_x + bar_width + 20, steering_y + 35), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_PREDICTION, 1)
        
        return frame
    
    def draw_control_bar(self, frame, x, y, width, value, pos_color, neg_color):
        """Draw a control value bar."""
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + 30), COLOR_GRID, -1)
        
        # Value bar
        bar_width = int(abs(value) * (width / 2))
        if value >= 0:
            bar_x = x + width // 2
            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_width, y + 30), pos_color, -1)
        else:
            bar_x = x + width // 2 - bar_width
            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_width, y + 30), neg_color, -1)
        
        # Center line
        cv2.line(frame, (x + width // 2, y), (x + width // 2, y + 30), (200, 200, 200), 2)
    
    def draw_prediction_overlay(self, frame, video_height, video_width):
        """Draw prediction overlay on the video area."""
        if not self.inference_enabled:
            return
        
        # Calculate overlay position (bottom of video area)
        overlay_y = video_height - 60
        center_x = UI_WIDTH // 2
        
        # Steering bar
        bar_width = 300
        bar_x = center_x - bar_width // 2
        
        # Background for steering bar
        cv2.rectangle(frame, (bar_x - 10, overlay_y - 10), 
                     (bar_x + bar_width + 10, overlay_y + 30), (0, 0, 0, 128), -1)
        
        # Steering bar
        cv2.rectangle(frame, (bar_x, overlay_y), 
                     (bar_x + bar_width, overlay_y + 20), COLOR_GRID, -1)
        
        # Steering value
        steer_bar_width = int(abs(self.predicted_angle) * (bar_width / 2))
        if self.predicted_angle >= 0:
            # Right steering
            steer_x = bar_x + bar_width // 2
            cv2.rectangle(frame, (steer_x, overlay_y), 
                         (steer_x + steer_bar_width, overlay_y + 20), COLOR_STEERING, -1)
        else:
            # Left steering
            steer_x = bar_x + bar_width // 2 - steer_bar_width
            cv2.rectangle(frame, (steer_x, overlay_y), 
                         (steer_x + steer_bar_width, overlay_y + 20), COLOR_STEERING, -1)
        
        # Center line
        cv2.line(frame, (center_x, overlay_y), (center_x, overlay_y + 20), (255, 255, 255), 2)
        
        # Throttle bar (vertical, right side)
        throttle_x = bar_x + bar_width + 30
        throttle_height = 100
        throttle_y = overlay_y - throttle_height // 2
        
        # Throttle background
        cv2.rectangle(frame, (throttle_x, throttle_y), 
                     (throttle_x + 20, throttle_y + throttle_height), COLOR_GRID, -1)
        
        # Throttle value
        throttle_bar_height = int(abs(self.predicted_throttle) * (throttle_height / 2))
        throttle_color = COLOR_THROTTLE_FORWARD if self.predicted_throttle >= 0 else COLOR_THROTTLE_REVERSE
        
        if self.predicted_throttle >= 0:
            # Forward throttle
            cv2.rectangle(frame, (throttle_x, throttle_y + throttle_height // 2 - throttle_bar_height), 
                         (throttle_x + 20, throttle_y + throttle_height // 2), throttle_color, -1)
        else:
            # Reverse throttle
            cv2.rectangle(frame, (throttle_x, throttle_y + throttle_height // 2), 
                         (throttle_x + 20, throttle_y + throttle_height // 2 + throttle_bar_height), throttle_color, -1)
        
        # Center line for throttle
        cv2.line(frame, (throttle_x, throttle_y + throttle_height // 2), 
                (throttle_x + 20, throttle_y + throttle_height // 2), (255, 255, 255), 2)
    
    def setup_udp_client(self):
        """Set up UDP client socket for control commands."""
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.settimeout(1.0)
    
    def setup_video_socket(self):
        """Set up UDP socket for video reception."""
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.video_socket.bind(("0.0.0.0", VIDEO_RECEIVE_PORT))
        print(f"Video receiver listening on port {VIDEO_RECEIVE_PORT}")
    
    def predict_control(self, frame):
        """Run inference on a frame to predict control signals."""
        if not self.inference_enabled or self.model is None:
            return 0.0, 0.0
        
        try:
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.is_temporal_model:
                # Add frame to buffer for temporal processing
                processed_frame = self.transform(frame_rgb)
                self.frame_buffer.append(processed_frame)
                
                # Keep only the required sequence length
                if len(self.frame_buffer) > self.sequence_length:
                    self.frame_buffer.pop(0)
                
                # Need enough frames for temporal prediction
                if len(self.frame_buffer) < self.sequence_length:
                    # Pad with current frame if we don't have enough history
                    while len(self.frame_buffer) < self.sequence_length:
                        self.frame_buffer.insert(0, processed_frame)
                
                # Stack frames into sequence: [sequence_length, channels, height, width]
                frame_sequence = torch.stack(self.frame_buffer, dim=0)
                # Add batch dimension: [1, sequence_length, channels, height, width]
                input_tensor = frame_sequence.unsqueeze(0).to(self.device)
            else:
                # Regular single-frame processing
                input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                control = self.model(input_tensor).cpu().numpy()[0]
            
            steering, throttle = control
            return float(steering), float(throttle)
        except Exception as e:
            print(f"Error during inference: {e}")
            return 0.0, 0.0
    
    def send_command(self, angle, throttle):
        """Send control command to the DeepRacer via UDP."""
        try:
            command = {
                'angle': float(angle),
                'throttle': float(throttle)
            }
            
            json_data = json.dumps(command).encode('utf-8')
            self.udp_socket.sendto(json_data, (self.server_ip, self.server_port))
            
            self.last_successful_send = time.time()
            return True
        except Exception as e:
            if self.connection_ok:  # Only print on status change
                print(f"Error sending command: {e}")
            self.connection_ok = False
            return False
    
    def ping_server(self):
        """Send a ping message to test server connection."""
        try:
            ping_msg = json.dumps({"ping": True}).encode('utf-8')
            self.udp_socket.sendto(ping_msg, (self.server_ip, self.server_port))
            print(f"Ping sent to {self.server_ip}:{self.server_port}")
        except Exception as e:
            print(f"Error pinging server: {e}")
    
    def response_receiver(self):
        """Thread that listens for server responses."""
        while self.running:
            try:
                data, addr = self.udp_socket.recvfrom(1024)
                response = json.loads(data.decode('utf-8'))
                
                if not self.connection_ok:
                    print(f"Connection established with DeepRacer at {addr}")
                    self.connection_ok = True
            except socket.timeout:
                pass
            except Exception as e:
                if self.running:
                    print(f"Error receiving response: {e}")
            time.sleep(0.01)
    
    def video_receiver(self):
        """Thread that receives video frames and puts them in a queue."""
        print("Starting video receiver...")
        
        while self.running:
            try:
                data, addr = self.video_socket.recvfrom(65535)
                
                # Decode the frame
                np_data = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Non-blocking put - drop frame if queue is full
                    try:
                        self.frame_queue.put(frame, block=False)
                    except queue.Full:
                        pass
            except Exception as e:
                if self.running:
                    print(f"Error receiving video: {e}")
                time.sleep(0.1)
    
    def control_loop(self):
        """Thread that runs inference and sends control commands."""
        send_interval = 0.1  # Send commands at 10Hz
        last_frame = None
        
        while self.running:
            current_time = time.time()
            
            # Get latest frame
            try:
                while not self.frame_queue.empty():
                    last_frame = self.frame_queue.get(block=False)
            except queue.Empty:
                pass
            
            # Run inference if we have a frame
            if last_frame is not None:
                self.predicted_angle, self.predicted_throttle = self.predict_control(last_frame)
                
                # Send control commands if inference is enabled
                if self.inference_enabled:
                    success = self.send_command(self.predicted_angle, self.predicted_throttle)
                    if success and (abs(self.predicted_throttle) > 0.01 or abs(self.predicted_angle) > 0.01):
                        print(f"AI Control: steering={self.predicted_angle:.3f}, throttle={self.predicted_throttle:.3f}")
                else:
                    # Send idle command when inference is disabled
                    self.send_command(0.0, 0.0)
            
            time.sleep(send_interval)
    
    def start(self):
        """Start the inference controller."""
        print(f"Starting AI Inference Controller")
        print(f"DeepRacer: {self.server_ip}:{self.server_port}")
        print(f"Model: {self.model_path}")
        print("Controls:")
        print("  SPACE: Toggle AI inference")
        print("  ESC: Exit")
        
        # Send initial ping
        self.ping_server()
        
        # Set up video reception
        self.setup_video_socket()
        
        # Start all threads
        self.receiver_thread.start()
        self.video_thread.start()
        self.control_thread.start()
        
        # Main loop - handle UI display
        try:
            cv2.namedWindow("AI Inference Controller", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("AI Inference Controller", UI_WIDTH, UI_HEIGHT)
            
            last_ui_update = time.time()
            ui_update_interval = 0.05  # 20Hz UI updates
            
            while self.running:
                current_time = time.time()
                
                # Get latest frame for display
                new_frame_received = False
                try:
                    video_frame = self.frame_queue.get(block=False)
                    new_frame_received = True
                except queue.Empty:
                    video_frame = None
                
                # Update UI
                if new_frame_received or (current_time - last_ui_update > ui_update_interval):
                    ui_frame = self.draw_telemetry(None, video_frame if new_frame_received else None)
                    cv2.imshow("AI Inference Controller", ui_frame)
                    last_ui_update = current_time
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    self.inference_enabled = not self.inference_enabled
                    status = "enabled" if self.inference_enabled else "disabled"
                    print(f"AI inference {status}")
                elif key == 27:  # ESC key
                    print("ESC pressed, stopping...")
                    break
                
                # Check connection status
                if time.time() - self.last_successful_send > 5.0:
                    if self.connection_ok:
                        print("Warning: No recent successful communication with DeepRacer")
                        self.connection_ok = False
                        self.ping_server()
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the controller and clean up resources."""
        print("Stopping inference controller...")
        self.running = False
        
        if hasattr(self, 'udp_socket'):
            self.udp_socket.close()
        if hasattr(self, 'video_socket'):
            self.video_socket.close()
        cv2.destroyAllWindows()

def main():
    if len(sys.argv) < 3:
        print("Usage: python inference_controller.py <server_ip> <model_path>")
        print("Example: python inference_controller.py 192.168.1.100 models/rc_car_model.pth")
        sys.exit(1)
    
    server_ip = sys.argv[1]
    model_path = sys.argv[2]
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    controller = InferenceController(server_ip, model_path)
    
    # Setup signal handler for clean shutdown
    def signal_handler(sig, frame):
        print("Ctrl+C pressed, shutting down...")
        controller.running = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    controller.start()

if __name__ == '__main__':
    main() 