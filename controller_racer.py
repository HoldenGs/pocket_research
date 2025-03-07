#!/usr/bin/env python3
import rclpy
import socket
import json
import threading
import time
import cv2
import numpy as np
from rclpy.node import Node
from deepracer_interfaces_pkg.msg import ServoCtrlMsg

# Constants
IDLE_THROTTLE = 0.0
VIDEO_PORT = 5005
VIDEO_QUALITY = 70
VIDEO_FPS = 15

class CombinedRacerController(Node):
    def __init__(self):
        super().__init__('combined_racer_controller')
        
        # Create servo control publisher
        self.pub = self.create_publisher(ServoCtrlMsg, '/ctrl_pkg/servo_msg', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Default control values
        self.angle = 0.0
        self.throttle = IDLE_THROTTLE
        
        # UDP controller server setup
        self.server_ip = '0.0.0.0'  # Listen on all network interfaces
        self.server_port = 9999
        self.client_addr = None
        self.setup_udp_server()
        
        # Video streaming setup
        self.video_client_addr = None
        self.video_enabled = False
        self.camera_device = 0  # Default camera device
        
        # Print server IP for user reference
        self.get_logger().info(f'UDP control server started on all interfaces (0.0.0.0):{self.server_port}')
        self.get_logger().info(f'Server IP addresses:')
        self.run_terminal_cmd('hostname -I')
        self.get_logger().info('Waiting for client connection...')

    def run_terminal_cmd(self, cmd):
        """Run a terminal command and log the output."""
        import subprocess
        try:
            output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
            for line in output.split('\n'):
                self.get_logger().info(f"  {line}")
            return output
        except Exception as e:
            self.get_logger().error(f"Command failed: {str(e)}")
            return None

    def setup_udp_server(self):
        """Set up UDP server socket and start listening thread."""
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Set socket option to allow reuse of address/port
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_socket.bind((self.server_ip, self.server_port))
        
        # Start UDP listening thread
        self.udp_thread = threading.Thread(target=self.udp_listener)
        self.udp_thread.daemon = True
        self.udp_thread.start()

    def setup_video_sender(self, client_addr):
        """Set up video streaming to the provided client address."""
        if client_addr is None:
            self.get_logger().warning("No client address provided for video streaming")
            return False
            
        try:
            # Set up video socket just like the original implementation
            self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.video_client_addr = (client_addr[0], VIDEO_PORT)
            
            # Try to open the camera
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.get_logger().error(f"Failed to open camera device {self.camera_device}")
                return False
                
            # Start video streaming thread
            self.video_enabled = True
            self.video_thread = threading.Thread(target=self.video_sender)
            self.video_thread.daemon = True
            self.video_thread.start()
            
            self.get_logger().info(f"Video streaming started to {self.video_client_addr}")
            return True
        except Exception as e:
            self.get_logger().error(f"Error setting up video streaming: {str(e)}")
            return False

    def udp_listener(self):
        """Listen for UDP control commands."""
        self.get_logger().info("UDP listener started and waiting for messages")
        while rclpy.ok():
            try:
                data, addr = self.udp_socket.recvfrom(1024)
                if not data:
                    break
                
                # Remember client address for potential responses
                if self.client_addr != addr:
                    self.client_addr = addr
                    self.get_logger().info(f'New client connected from {addr}')
                    
                    # Start video streaming to the new client
                    if not self.video_enabled:
                        self.setup_video_sender(addr)
                
                try:
                    # Parse JSON control commands
                    control_data = json.loads(data.decode('utf-8'))
                    
                    # Check for ping message
                    if 'ping' in control_data:
                        self.get_logger().info(f'Received ping from {addr}')
                        # Send acknowledgment back to client
                        ack = json.dumps({"status": "ok", "pong": True}).encode('utf-8')
                        self.udp_socket.sendto(ack, addr)
                        continue
                    
                    # Process control commands
                    if 'angle' in control_data:
                        self.angle = float(control_data['angle'])
                    if 'throttle' in control_data:
                        self.throttle = float(control_data['throttle'])
                    
                    # Send acknowledgment back to client
                    ack = json.dumps({"status": "ok", "received": control_data}).encode('utf-8')
                    self.udp_socket.sendto(ack, addr)
                    
                    self.get_logger().debug(f'Updated controls: angle={self.angle}, throttle={self.throttle}')
                except json.JSONDecodeError:
                    self.get_logger().warning(f'Received invalid JSON data: {data.decode("utf-8")}')
                except Exception as e:
                    self.get_logger().error(f'Error processing data: {str(e)}')
            except Exception as e:
                self.get_logger().error(f'UDP receive error: {str(e)}')

    def video_sender(self):
        """Thread that captures and sends video frames to the client."""
        self.get_logger().info("Starting video streaming thread")
        
        # Simple implementation similar to the original video_sender.py
        while rclpy.ok() and self.video_enabled:
            try:
                # Capture frame from camera - just like the original code
                ret, frame = self.cap.read()
                if not ret:
                    self.get_logger().warning("Failed to capture frame from camera")
                    time.sleep(0.1)  # Wait a bit before trying again
                    continue
                
                # Compress frame to JPEG - match original code's simplicity
                ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), VIDEO_QUALITY])
                if not ret:
                    self.get_logger().warning("Failed to encode frame")
                    continue
                
                # Send frame to client - simple send like the original
                data = buf.tobytes()
                self.video_socket.sendto(data, self.video_client_addr)
                
                # Brief sleep to control frame rate
                time.sleep(1.0/VIDEO_FPS)
                
            except Exception as e:
                self.get_logger().error(f"Error in video streaming: {str(e)}")
                time.sleep(0.1)  # Wait a bit before trying again

    def timer_callback(self):
        """Publish control commands to ROS2 at regular intervals."""
        if rclpy.ok():
            msg = ServoCtrlMsg()
            msg.angle = self.angle
            msg.throttle = self.throttle
            
            self.pub.publish(msg)
            self.get_logger().debug(f'Publishing: angle={msg.angle}, throttle={msg.throttle}')

    def cleanup(self):
        """Clean up resources when node is destroyed."""
        if hasattr(self, 'video_enabled'):
            self.video_enabled = False
        
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            
        if hasattr(self, 'video_socket'):
            self.video_socket.close()
            
        if hasattr(self, 'udp_socket'):
            self.udp_socket.close()

def main(args=None):
    rclpy.init(args=args)
    node = CombinedRacerController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 