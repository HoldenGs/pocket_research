#!/usr/bin/env python3
import socket
import json
import sys
import threading
import time
import signal
from pynput import keyboard

# Constants
IDLE_THROTTLE = 0.50

class RemoteController:
    def __init__(self, server_ip, server_port=9999):
        self.server_ip = server_ip
        self.server_port = server_port
        
        # Control values
        self.angle = 0.0
        self.throttle = IDLE_THROTTLE
        
        # Key states
        self.key_w_pressed = False
        self.key_s_pressed = False
        self.key_a_pressed = False
        self.key_d_pressed = False
        
        # Connection tracking
        self.last_successful_send = 0
        self.connection_ok = False
        self.received_response = False
        
        # UDP client setup
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
        
    def setup_udp_client(self):
        """Set up UDP client socket."""
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Set socket timeout for receive operations
        self.udp_socket.settimeout(1.0)
        
    def start(self):
        """Start the controller."""
        print(f"Connecting to DeepRacer controller at {self.server_ip}:{self.server_port}")
        print("WASD keyboard control enabled:")
        print("  W: Forward (throttle = 1.0)")
        print("  S: Reverse (throttle = -1.0)")
        print("  A: Left (angle = 1.0)")
        print("  D: Right (angle = -1.0)")
        print("  ESC: Exit")
        print("  C: Check connection")
        
        # Send initial ping to test connection
        self.ping_server()
        
        # Start the keyboard listener and command sender
        self.listener.start()
        self.sender_thread.start()
        self.receiver_thread.start()
        
        try:
            # Keep the main thread alive and show connection status
            while self.running:
                # Check connection status every 5 seconds
                if time.time() - self.last_successful_send > 5:
                    if self.connection_ok:
                        print("Warning: No recent successful communication with server")
                        self.connection_ok = False
                        # Try to ping the server again
                        self.ping_server()
                
                time.sleep(1)
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
        """Stop the controller."""
        print("Stopping controller...")
        self.running = False
        if hasattr(self, 'listener'):
            self.listener.stop()
        if hasattr(self, 'udp_socket'):
            self.udp_socket.close()
            
    def on_press(self, key):
        """Handle key press events."""
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
        except AttributeError:
            # Special key that doesn't have a char attribute
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
            # Special key that doesn't have a char attribute
            pass
            
        # Stop on ESC key
        if key == keyboard.Key.esc:
            print("ESC pressed, stopping...")
            self.running = False
            return False
            
    def update_control_values(self):
        """Update control values based on key states."""
        # Handle throttle (W/S keys)
        if self.key_w_pressed and not self.key_s_pressed:
            self.throttle = 1.5
        elif self.key_s_pressed and not self.key_w_pressed:
            self.throttle = -1.0
        else:
            self.throttle = IDLE_THROTTLE
            
        # Handle steering (A/D keys)
        if self.key_a_pressed and not self.key_d_pressed:
            self.angle = 1.0
        elif self.key_d_pressed and not self.key_a_pressed:
            self.angle = -1.0
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
            
            # Encode and send
            json_data = json.dumps(command).encode('utf-8')
            self.udp_socket.sendto(json_data, (self.server_ip, self.server_port))
            
            # Update tracking
            self.last_successful_send = time.time()
            return True
        except socket.error as se:
            if not self.connection_ok:  # Only print errors when connection status changes
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
                
                # Update connection status
                if not self.connection_ok:
                    print(f"Connection established with server at {addr}")
                    self.connection_ok = True
                
                self.received_response = True
                # Uncomment to print every response (could be noisy)
                # print(f"Server response: {response}")
            except socket.timeout:
                # This is expected due to the socket timeout
                pass
            except Exception as e:
                if self.running:  # Only print errors if we're still supposed to be running
                    print(f"Error receiving response: {str(e)}")
            time.sleep(0.01)
            
    def command_sender(self):
        """Thread that continuously sends control commands."""
        last_status_check = 0
        last_report_time = 0
        send_interval = 0.1  # Send commands at 10Hz
        
        while self.running:
            current_time = time.time()
            
            # Update and send control values
            self.update_control_values()
            success = self.send_command()
            
            # Print status at a reasonable rate to avoid flooding the console
            if success and (self.throttle != 0.0 or self.angle != 0.0) and (current_time - last_report_time > 0.5):
                print(f"Sending: angle={self.angle}, throttle={self.throttle}")
                last_report_time = current_time
            
            time.sleep(send_interval)

def main():
    if len(sys.argv) < 2:
        print("Usage: python controller_client.py <server_ip>")
        sys.exit(1)
        
    server_ip = sys.argv[1]
    controller = RemoteController(server_ip)
    
    # Setup signal handler for clean shutdown on Ctrl+C
    def signal_handler(sig, frame):
        print("Ctrl+C pressed, shutting down...")
        controller.running = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    controller.start()

if __name__ == '__main__':
    main()