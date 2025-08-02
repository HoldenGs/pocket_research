#!/usr/bin/env python3
import socket
import json
import sys
import threading
import time
import signal
from pynput import keyboard
import cv2
import numpy as np
import queue
from data_collector import DataCollector  # Import our data collector

# Constants
IDLE_THROTTLE = 0.0
VIDEO_RECEIVE_PORT = 5005
UI_WIDTH = 1300
UI_HEIGHT = 820
TELEMETRY_HEIGHT = 300  # Increased from 240 to 300 for more vertical space
CONNECTION_CHECK_INTERVAL = 5  # seconds

# UI Colors (BGR format)
COLOR_BG = (50, 50, 50)
COLOR_TEXT = (240, 240, 240)
COLOR_CONNECTED = (0, 230, 0)
COLOR_DISCONNECTED = (0, 0, 230)
COLOR_THROTTLE_FORWARD = (0, 240, 120)
COLOR_THROTTLE_REVERSE = (0, 120, 240)
COLOR_STEERING = (240, 180, 0)
COLOR_GRID = (100, 100, 100)
COLOR_RECORDING = (0, 0, 240)  # Red color for recording indicator

class CombinedLocalController:
    def __init__(self, server_ip, server_port=9999):
        self.server_ip = server_ip
        self.server_port = server_port
        
        # Control values
        self.angle = 0.0
        self.throttle = IDLE_THROTTLE
        self.throttle_limit = 1.0  # Maximum throttle magnitude (can be adjusted with UP/DOWN keys)
        self.steering_limit = 1.0  # Maximum steering magnitude (can be adjusted with LEFT/RIGHT keys)
        
        # Key states
        self.key_w_pressed = False
        self.key_s_pressed = False
        self.key_a_pressed = False
        self.key_d_pressed = False
        
        # Connection tracking
        self.last_successful_send = 0
        self.connection_ok = False
        self.received_response = False
        
        # Video variables
        self.video_enabled = True
        self.frame_queue = queue.Queue(maxsize=10)  # Queue for frames between threads
        self.ui_frame = self.create_base_ui_frame()  # Create initial UI frame
        self.last_video_frame = None  # Store the last received video frame
        
        # Data collection
        self.data_collector = DataCollector()
        self.recording = False  # Flag to indicate if we're recording data
        
        # UDP client setup for control commands
        self.setup_udp_client()
        
        # Set up keyboard listener
        self.running = True
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        
        # Start command sender thread
        self.sender_thread = threading.Thread(target=self.command_sender)
        self.sender_thread.daemon = True
        
        # Start response receiver thread
        self.receiver_thread = threading.Thread(target=self.response_receiver)
        self.receiver_thread.daemon = True
        
        # Start video receiver thread (just receives data, doesn't display)
        self.video_thread = threading.Thread(target=self.video_receiver)
        self.video_thread.daemon = True
        
    def create_base_ui_frame(self):
        """Create the base UI frame with fixed elements."""
        # Create a blank frame with the desired size
        frame = np.zeros((UI_HEIGHT, UI_WIDTH, 3), dtype=np.uint8)
        frame[:] = COLOR_BG  # Set background color
        
        # Draw separator line between video and telemetry areas
        cv2.line(frame, (0, UI_HEIGHT - TELEMETRY_HEIGHT), 
                (UI_WIDTH, UI_HEIGHT - TELEMETRY_HEIGHT), COLOR_GRID, 2)
        
        # Draw title in the telemetry area instead of the video area
        title_y = UI_HEIGHT - TELEMETRY_HEIGHT + 25  # Position at the top of telemetry area
        cv2.putText(frame, "DeepRacer Controller", (20, title_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT, 2)
        
        return frame
    
    def draw_telemetry(self, frame, video_frame=None):
        """Draw telemetry information on the UI frame."""
        # Create a copy of the base frame
        if frame is None:
            frame = self.create_base_ui_frame()
        
        # Add the video frame if available
        if video_frame is not None:
            # Save this frame as our last valid frame
            self.last_video_frame = video_frame
        
        # Use the last valid frame if we have one and video is enabled
        if self.last_video_frame is not None and self.video_enabled:
            # Resize the video frame to fit the UI
            video_height = UI_HEIGHT - TELEMETRY_HEIGHT
            video_width = int(self.last_video_frame.shape[1] * video_height / self.last_video_frame.shape[0])
            
            # Center the video if it's smaller than the UI width
            if video_width < UI_WIDTH:
                x_offset = (UI_WIDTH - video_width) // 2
                resized_video = cv2.resize(self.last_video_frame, (video_width, video_height))
                frame[0:video_height, x_offset:x_offset+video_width] = resized_video
            else:
                # If video is wider, resize to fit width and crop height
                video_width = UI_WIDTH
                video_height_scaled = int(self.last_video_frame.shape[0] * video_width / self.last_video_frame.shape[1])
                resized_video = cv2.resize(self.last_video_frame, (video_width, video_height_scaled))
                # Crop to fit
                crop_y = (video_height_scaled - video_height) // 2 if video_height_scaled > video_height else 0
                frame[0:video_height, 0:video_width] = resized_video[crop_y:crop_y+video_height, 0:video_width]
        elif not self.video_enabled:
            # Draw a message if video is disabled
            video_area_center = (UI_WIDTH // 2, (UI_HEIGHT - TELEMETRY_HEIGHT) // 2)
            cv2.putText(frame, "Video Display Disabled", 
                       (video_area_center[0] - 150, video_area_center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2)
            cv2.putText(frame, "Press 'V' to Enable", 
                       (video_area_center[0] - 120, video_area_center[1] + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 1)
        
        # Redesigned layout with THREE DISTINCT COLUMNS
        # Start below the title
        panel_start_y = UI_HEIGHT - TELEMETRY_HEIGHT + 60  # Increased to leave room for title
        
        # SECTION DIVIDERS - add vertical lines to separate sections
        # Draw vertical separator lines
        # v_line1_x = UI_WIDTH // 3
        # v_line2_x = UI_WIDTH * 2 // 3
        # cv2.line(frame, (v_line1_x, UI_HEIGHT - TELEMETRY_HEIGHT), 
        #         (v_line1_x, UI_HEIGHT), COLOR_GRID, 1)
        # cv2.line(frame, (v_line2_x, UI_HEIGHT - TELEMETRY_HEIGHT), 
        #         (v_line2_x, UI_HEIGHT), COLOR_GRID, 1)
        
        # Add recording indicator if actively recording
        if self.recording:
            rec_x = UI_WIDTH - 120  # Position near the right edge
            rec_y = UI_HEIGHT - TELEMETRY_HEIGHT + 25  # Position at the top of telemetry area
            
            # Draw red recording circle
            cv2.circle(frame, (rec_x, rec_y), 10, COLOR_RECORDING, -1)
            
            # Draw REC text
            cv2.putText(frame, "REC", (rec_x + 20, rec_y + 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RECORDING, 2)
            
            # Get recording status
            status = self.data_collector.get_status()
            if status["frame_count"] > 0:
                cv2.putText(frame, f"Frames: {status['frame_count']}", 
                          (rec_x - 60, rec_y + 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
        
        # COLUMN 1 - LEFT: Connection and controls
        left_col_x = 50
        
        # Connection status
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
        cv2.putText(frame, f"Server: {self.server_ip}:{self.server_port}", 
                  (left_col_x, server_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 1)
        
        # Controls section (left column, bottom half)
        controls_y = server_y + 60
        cv2.putText(frame, "Controls:", (left_col_x, controls_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 1)
                   
        line_height = 40
        # Left side controls
        cv2.putText(frame, "W/S: Throttle", 
                   (left_col_x, controls_y + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 1)
        cv2.putText(frame, "A/D: Steering", 
                   (left_col_x, controls_y + line_height*2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 1)
        cv2.putText(frame, "V: Toggle video", 
                   (left_col_x, controls_y + line_height*3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 1)
        cv2.putText(frame, "R: Toggle recording", 
                   (left_col_x, controls_y + line_height*4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 1)
                   
        # Right side controls (shifted to the right in the same column)
        col2_x = left_col_x + 230
        cv2.putText(frame, "UP/DOWN: Throttle limit", 
                   (col2_x, controls_y + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 1)
        cv2.putText(frame, "LEFT/RIGHT: Steering limit", 
                   (col2_x, controls_y + line_height*2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 1)
        cv2.putText(frame, "ESC: Exit", 
                   (col2_x, controls_y + line_height*3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 1)
        
        # COLUMN 2 - CENTER: Throttle control
        center_col_x = UI_WIDTH // 3 + 50
        bar_width = 280
        
        # Throttle visualization
        throttle_y = panel_start_y
        cv2.putText(frame, "Throttle:", (center_col_x, throttle_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 1)
        
        # Draw throttle bar background
        cv2.rectangle(frame, (center_col_x, throttle_y + 15), 
                    (center_col_x + bar_width, throttle_y + 45), COLOR_GRID, -1)
        
        # Draw throttle value bar
        normalized_throttle = self.throttle  # Assuming throttle range is -1 to 1
        throttle_color = COLOR_THROTTLE_FORWARD if normalized_throttle >= 0 else COLOR_THROTTLE_REVERSE
        throttle_width = int(abs(normalized_throttle) * (bar_width / 2))
        
        if normalized_throttle >= 0:
            # Forward throttle (right half)
            bar_x = center_col_x + bar_width // 2
            cv2.rectangle(frame, (bar_x, throttle_y + 15), 
                        (bar_x + throttle_width, throttle_y + 45), throttle_color, -1)
        else:
            # Reverse throttle (left half)
            bar_x = center_col_x + bar_width // 2 - throttle_width
            cv2.rectangle(frame, (bar_x, throttle_y + 15), 
                        (bar_x + throttle_width, throttle_y + 45), throttle_color, -1)
        
        # Draw center marker
        cv2.line(frame, (center_col_x + bar_width // 2, throttle_y + 10), 
               (center_col_x + bar_width // 2, throttle_y + 50), (200, 200, 200), 2)
        
        # Add throttle value text
        cv2.putText(frame, f"{self.throttle:.2f}", (center_col_x + bar_width + 20, throttle_y + 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, throttle_color, 1)
        
        # Add throttle limit indicator
        limit_y = throttle_y + 70
        limit_text = f"Throttle Limit: {self.throttle_limit:.1f}"
        cv2.putText(frame, limit_text, (center_col_x, limit_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_THROTTLE_FORWARD, 1)
        
        # COLUMN 3 - RIGHT: Steering control 
        right_col_x = UI_WIDTH * 2 // 3 + 50
        
        # Steering visualization
        steering_y = panel_start_y  # Align with throttle
        cv2.putText(frame, "Steering:", (right_col_x, steering_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 1)
        
        # Draw steering bar background
        cv2.rectangle(frame, (right_col_x, steering_y + 15), 
                    (right_col_x + bar_width, steering_y + 45), COLOR_GRID, -1)
        
        # Draw steering value bar
        normalized_angle = self.angle  # Assuming angle range is -1 to 1
        steering_width = int(abs(normalized_angle) * (bar_width / 2))
        
        if normalized_angle <= 0:
            # Right steering (right half)
            bar_x = right_col_x + bar_width // 2
            cv2.rectangle(frame, (bar_x, steering_y + 15), 
                        (bar_x + steering_width, steering_y + 45), COLOR_STEERING, -1)
        else:
            # Left steering (left half)
            bar_x = right_col_x + bar_width // 2 - steering_width
            cv2.rectangle(frame, (bar_x, steering_y + 15), 
                        (bar_x + steering_width, steering_y + 45), COLOR_STEERING, -1)
        
        # Draw center marker
        cv2.line(frame, (right_col_x + bar_width // 2, steering_y + 10), 
               (right_col_x + bar_width // 2, steering_y + 50), (200, 200, 200), 2)
        
        # Add steering value text
        cv2.putText(frame, f"{self.angle:.2f}", (right_col_x + bar_width + 20, steering_y + 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_STEERING, 1)
        
        # Add steering limit indicator 
        limit_y = steering_y + 70
        limit_text = f"Steering Limit: {self.steering_limit:.1f}"
        cv2.putText(frame, limit_text, (right_col_x, limit_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_STEERING, 1)
        
        return frame
        
    def setup_udp_client(self):
        """Set up UDP client socket for control commands."""
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Set socket timeout for receive operations
        self.udp_socket.settimeout(1.0)
    
    def setup_video_socket(self):
        """Set up UDP socket for video reception."""
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.video_socket.bind(("0.0.0.0", VIDEO_RECEIVE_PORT))
        # Don't set a timeout - use blocking mode like the original code
        print(f"Video receiver listening on port {VIDEO_RECEIVE_PORT}")
        
    def start(self):
        """Start the controller and video receiver."""
        print(f"Connecting to DeepRacer controller at {self.server_ip}:{self.server_port}")
        print("WASD keyboard control enabled:")
        print("  W: Forward (throttle up to limit)")
        print("  S: Reverse (throttle up to limit)")
        print("  A: Left (steering up to limit)")
        print("  D: Right (steering up to limit)")
        print("  UP/DOWN: Adjust throttle limit")
        print("  LEFT/RIGHT: Adjust steering limit")
        print("  V: Toggle video display")
        print("  R: Toggle data recording")
        print("  ESC: Exit")
        
        # Send initial ping to test connection
        self.ping_server()
        
        # Set up video reception
        self.setup_video_socket()
        
        # Start all the threads
        self.listener.start()
        self.sender_thread.start()
        self.receiver_thread.start()
        self.video_thread.start()
        
        # Main thread loop - handles video display in the main thread
        try:
            last_status_check = time.time()
            cv2.namedWindow("DeepRacer Controller", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("DeepRacer Controller", UI_WIDTH, UI_HEIGHT)
            
            # Create and display initial UI
            ui_frame = self.draw_telemetry(None, None)
            cv2.imshow("DeepRacer Controller", ui_frame)
            
            # Track when we need to update the UI
            last_ui_update = time.time()
            ui_update_interval = 0.05  # Update UI at 20Hz even without new frames
            new_frame_received = False
            
            while self.running:
                current_time = time.time()
                
                # Process video frames in the main thread
                if self.video_enabled:
                    try:
                        # Non-blocking get from queue
                        video_frame = self.frame_queue.get(block=False)
                        new_frame_received = True
                    except queue.Empty:
                        # No frame available
                        new_frame_received = False
                
                # Update UI when we have a new frame or periodically for control updates
                if new_frame_received or (current_time - last_ui_update > ui_update_interval):
                    # Only pass the new frame to draw_telemetry if we actually received one
                    ui_frame = self.draw_telemetry(None, video_frame if new_frame_received else None)
                    cv2.imshow("DeepRacer Controller", ui_frame)
                    last_ui_update = current_time
                
                # Must call waitKey in the main thread for macOS compatibility
                # Setting a very short wait time to remain responsive but reduce CPU usage
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.video_enabled = False

                current_time = time.time()
                if current_time - last_status_check > CONNECTION_CHECK_INTERVAL:
                    if time.time() - self.last_successful_send > CONNECTION_CHECK_INTERVAL:
                        if self.connection_ok:
                            print("Warning: No recent successful communication with server")
                            self.connection_ok = False
                            # Try to ping the server again
                            self.ping_server()
                    last_status_check = current_time

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.stop()

    def ping_server(self):
        """Send a ping message to test server connection."""
        try:
            ping_msg = json.dumps({"ping": True}).encode('utf-8')
            self.udp_socket.sendto(ping_msg, (self.server_ip, self.server_port))
            print(f"Ping sent to {self.server_ip}:{self.server_port}")
        except Exception as e:
            print(f"Error pinging server: {str(e)}")

    def stop(self):
        """Stop the controller and clean up resources."""
        print("Stopping controller...")
        self.running = False

        # Stop recording if active
        if self.recording:
            self.data_collector.stop_recording()

        if hasattr(self, 'listener'):
            self.listener.stop()
        if hasattr(self, 'udp_socket'):
            self.udp_socket.close()
        if hasattr(self, 'video_socket'):
            self.video_socket.close()
        cv2.destroyAllWindows()

    def on_press(self, key):
        """Handle key press events."""
        # Check for special keys that don't have a char attribute first
        if key == keyboard.Key.up:
            # Increase throttle limit (max 1.0)
            self.throttle_limit = min(1.0, self.throttle_limit + 0.2)
            self.throttle_limit = round(self.throttle_limit, 1)
            print(f"Throttle limit increased to {self.throttle_limit:.1f}")
            return
        elif key == keyboard.Key.down:
            # Decrease throttle limit (min 0.0)
            self.throttle_limit = max(0.0, self.throttle_limit - 0.2)
            self.throttle_limit = round(self.throttle_limit, 1)
            print(f"Throttle limit decreased to {self.throttle_limit:.1f}")
            return
        elif key == keyboard.Key.left:
            # Decrease steering limit (min 0.0)
            self.steering_limit = max(0.0, self.steering_limit - 0.2)
            self.steering_limit = round(self.steering_limit, 1)
            print(f"Steering limit decreased to {self.steering_limit:.1f}")
            return
        elif key == keyboard.Key.right:
            # Increase steering limit (max 1.0)
            self.steering_limit = min(1.0, self.steering_limit + 0.2)
            self.steering_limit = round(self.steering_limit, 1)
            print(f"Steering limit increased to {self.steering_limit:.1f}")
            return
        elif key == keyboard.Key.esc:
            print("ESC pressed, stopping...")
            self.running = False
            return False

        try:
            if key.char == 'w':
                self.key_w_pressed = True
            elif key.char == 's':
                self.key_s_pressed = True
            elif key.char == 'a':
                self.key_a_pressed = True
            elif key.char == 'd':
                self.key_d_pressed = True
            elif key.char == 'c':
                print(f"Connection status: {'Connected' if self.connection_ok else 'Not connected'}")
                if not self.connection_ok:
                    print("Attempting to reconnect...")
                    self.ping_server()
            elif key.char == 'v':
                self.video_enabled = not self.video_enabled
                print(f"Video display {'enabled' if self.video_enabled else 'disabled'}")
                # Force UI update on toggle
                if not self.video_enabled:
                    self.last_video_frame = None
            elif key.char == 'r':
                # Toggle data recording
                self.recording = self.data_collector.toggle_recording()
                if self.recording:
                    print("Data recording started")
                else:
                    print("Data recording stopped")
        except AttributeError:
            pass

    def on_release(self, key):
        """Handle key release events."""
        try:
            if key.char == 'w':
                self.key_w_pressed = False
            elif key.char == 's':
                self.key_s_pressed = False
            elif key.char == 'a':
                self.key_a_pressed = False
            elif key.char == 'd':
                self.key_d_pressed = False
        except AttributeError:
            pass

    def update_control_values(self):
        """Update controls based on keys with gradual acceleration"""
        # Gradual throttle control (W/S keys)
        # Increase/decrease by 0.05 every 0.1s based on key states
        if self.key_w_pressed and not self.key_s_pressed:
            # Forward throttle - gradually increase
            self.throttle = min(self.throttle_limit, self.throttle + 0.2)
        elif self.key_s_pressed and not self.key_w_pressed:
            # Reverse throttle - gradually increase (negative)
            self.throttle = max(-self.throttle_limit, self.throttle - 0.2)
        else:
            # No throttle key pressed - gradually return to idle
            if self.throttle > 0:
                self.throttle = max(IDLE_THROTTLE, self.throttle - 0.05)
            elif self.throttle < 0:
                self.throttle = min(IDLE_THROTTLE, self.throttle + 0.05)
            else:
                self.throttle = IDLE_THROTTLE

        # Gradual steering control (A/D keys)
        # Increase/decrease by 0.15 every 0.1s based on key states
        if self.key_a_pressed and not self.key_d_pressed:
            # Left steering - gradually increase
            self.angle = min(self.steering_limit, self.angle + 0.25)
        elif self.key_d_pressed and not self.key_a_pressed:
            # Right steering - gradually increase (more negative)
            self.angle = max(-self.steering_limit, self.angle - 0.25)
        else:
            # No steering key pressed - gradually return to center
            if self.angle > 0:
                self.angle = max(0.0, self.angle - 0.25)
            elif self.angle < 0:
                self.angle = min(0.0, self.angle + 0.25)
            else:
                self.angle = 0.0

    def send_command(self):
        """Send control command to the server via UDP."""
        try:
            # Create command data
            command = {
                'angle': self.angle,
                'throttle': self.throttle
            }

            json_data = json.dumps(command).encode('utf-8')
            self.udp_socket.sendto(json_data,
                                   (self.server_ip, self.server_port))

            self.last_successful_send = time.time()
            return True
        except socket.error as se:
            if not self.connection_ok:
                print(f"Network error: {str(se)}")
            self.connection_ok = False
            return False
        except Exception as e:
            print(f"Error sending command: {str(e)}")
            self.connection_ok = False
            return False

    def response_receiver(self):
        """Thread that listens for server responses."""
        while self.running:
            try:
                data, addr = self.udp_socket.recvfrom(1024)
                response = json.loads(data.decode('utf-8'))

                if not self.connection_ok:
                    print(f"Connection established with server at {addr}")
                    self.connection_ok = True

                self.received_response = True
                # print(f"Server response: {response}")
            except socket.timeout:
                pass
            except Exception as e:
                if self.running:
                    print(f"Error receiving response: {str(e)}")
            time.sleep(0.01)

    def video_receiver(self):
        """Thread that receives video frames and puts them in a queue."""
        print("Starting video receiver...")  
        while self.running:
            try:
                data, addr = self.video_socket.recvfrom(65535)
                np_data = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

                if frame is not None:
                    if self.recording:
                        self.data_collector.add_frame(frame)

                    if self.video_enabled:
                        try:
                            self.frame_queue.put(frame, block=False)
                        except queue.Full:
                            pass

            except Exception as e:
                if self.running:
                    print(f"Error receiving video: {str(e)}")
                time.sleep(0.1)

    def command_sender(self):
        """Thread that continuously sends control commands."""
        last_report_time = 0
        send_interval = 0.1

        while self.running:
            current_time = time.time()

            self.update_control_values()
            success = self.send_command()

            if self.recording:
                self.data_collector.add_control_signal(self.throttle, self.angle)

            if success and (self.throttle != 0.0 or self.angle != 0.0) and (current_time - last_report_time > 0.5):
                print(f"Sending: angle={self.angle}, throttle={self.throttle}")
                last_report_time = current_time

            time.sleep(send_interval)


def main():
    if len(sys.argv) < 2:
        print("Usage: python controller_local.py <server_ip>")
        sys.exit(1)

    server_ip = sys.argv[1]
    controller = CombinedLocalController(server_ip)

    # Setup signal handler for clean shutdown on Ctrl+C
    def signal_handler(sig, frame):
        print("Ctrl+C pressed, shutting down...")
        controller.running = False
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    controller.start()


if __name__ == '__main__':
    main() 