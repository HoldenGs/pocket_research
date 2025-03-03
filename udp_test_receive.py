#!/usr/bin/env python3
import socket
import json
import sys
import time

# Configuration
SERVER_IP = '0.0.0.0'  # Listen on all network interfaces
SERVER_PORT = 9999     # Same port as used in servo_receiver.py

# Print network interfaces to help with debugging
import subprocess
try:
    print("Network interfaces:")
    interfaces = subprocess.check_output("ifconfig | grep 'inet ' | grep -v 127.0.0.1", shell=True).decode('utf-8')
    for line in interfaces.splitlines():
        print(f"  {line.strip()}")
except Exception as e:
    print(f"Could not get network interfaces: {e}")

# Create UDP socket with more debugging
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow reuse of address

try:
    sock.bind((SERVER_IP, SERVER_PORT))
    print(f"Successfully bound to {SERVER_IP}:{SERVER_PORT}")
except Exception as e:
    print(f"Error binding to socket: {e}")
    sys.exit(1)

print(f"UDP test listener started on {SERVER_IP}:{SERVER_PORT}")
print("Waiting for messages... (Press Ctrl+C to stop)")

# Set a small timeout so we can print a heartbeat message
sock.settimeout(5)

try:
    last_heartbeat = time.time()
    message_count = 0
    
    while True:
        try:
            # Receive data (buffer size 1024 bytes)
            data, addr = sock.recvfrom(1024)
            message_count += 1
            
            # Reset heartbeat timer
            last_heartbeat = time.time()
            
            # Try to decode as JSON
            try:
                message = json.loads(data.decode('utf-8'))
                print(f"Received from {addr}: {message}")
                
                # Send acknowledgment
                ack = json.dumps({
                    "status": "received", 
                    "data": message,
                    "timestamp": time.time(),
                    "message_count": message_count
                }).encode('utf-8')
                sock.sendto(ack, addr)
                print(f"Sent acknowledgment to {addr}")
                
            except json.JSONDecodeError:
                print(f"Received non-JSON data from {addr}: {data.decode('utf-8', errors='replace')}")
        
        except socket.timeout:
            # Print heartbeat message every 5 seconds if no messages received
            current_time = time.time()
            if current_time - last_heartbeat >= 5:
                print(f"Still listening... (received {message_count} messages so far)")
                last_heartbeat = current_time
        
        except Exception as e:
            print(f"Error receiving data: {e}")
            time.sleep(1)  # Prevent tight loop in case of repeated errors
        
except KeyboardInterrupt:
    print("Stopped by user")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    sock.close()
    print("Socket closed") 