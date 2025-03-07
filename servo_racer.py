#!/usr/bin/env python3
import rclpy, time
import socket
import json
import threading
import cv2
from rclpy.node import Node
from deepracer_interfaces_pkg.msg import ServoCtrlMsg

# Constants
IDLE_THROTTLE = 0.0


class SimpleController(Node):
    def __init__(self):
        super().__init__('simple_controller')
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

    def udp_listener(self):
        """Listen for UDP control commands."""
        self.get_logger().info("UDP listener started and waiting for messages")
        while True:
            
            try:
                data, addr = self.udp_socket.recvfrom(1024)
                if not data:
                    break
                
                # Remember client address for potential responses
                if self.client_addr != addr:
                    self.client_addr = addr
                    self.get_logger().info(f'New client connected from {addr}')
                
                self.get_logger().info(f'Received data from {addr}: {data.decode("utf-8")}')
                
                try:
                    # Parse JSON control commands
                    control_data = json.loads(data.decode('utf-8'))
                    if 'angle' in control_data:
                        self.angle = float(control_data['angle'])
                    if 'throttle' in control_data:
                        self.throttle = float(control_data['throttle'])
                    
                    # Send acknowledgment back to client
                    if self.client_addr:
                        ack = json.dumps({"status": "ok", "received": control_data}).encode('utf-8')
                        self.udp_socket.sendto(ack, self.client_addr)
                    
                    self.get_logger().info(f'Updated controls: angle={self.angle}, throttle={self.throttle}')
                except json.JSONDecodeError:
                    self.get_logger().warning(f'Received invalid JSON data: {data.decode("utf-8")}')
                except Exception as e:
                    self.get_logger().error(f'Error processing data: {str(e)}')
            except Exception as e:
                self.get_logger().error(f'UDP receive error: {str(e)}')

    def timer_callback(self):
        """Publish control commands to ROS2 at regular intervals."""
        msg = ServoCtrlMsg()
        msg.angle = self.angle
        msg.throttle = self.throttle
        
        self.pub.publish(msg)
        self.get_logger().info(f'Publishing: angle={msg.angle}, throttle={msg.throttle}')

def main(args=None):
    rclpy.init(args=args)
    node = SimpleController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'udp_socket'):
            node.udp_socket.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
