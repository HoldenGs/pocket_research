#!/usr/bin/env python3
import socket
import json
import time

# Configuration
SERVER_IP = '192.168.1.53'  # Target IP
SERVER_PORT = 9999          # Target port
LOCAL_PORT = 8888           # Local port to bind to

# Create UDP socket with explicit binding to make it more predictable
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Get local IP
import subprocess
local_ip = subprocess.check_output("ifconfig | grep 'inet ' | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1", shell=True).decode('utf-8').strip()
print(f"Local IP: {local_ip}")

# Bind to specific local port (makes NAT traversal more predictable)
try:
    sock.bind((local_ip, LOCAL_PORT))
    print(f"Socket bound to {local_ip}:{LOCAL_PORT}")
except Exception as e:
    print(f"Could not bind to specific port: {e}")
    print("Continuing with automatic port assignment")

# Set socket timeout to receive possible responses
sock.settimeout(1)

print(f"Sending test messages to {SERVER_IP}:{SERVER_PORT}")
print("Press Ctrl+C to stop")

try:
    counter = 0
    while True:
        # Create test message
        message = {
            "angle": 0.5,
            "throttle": 0.3,
            "test_counter": counter,
            "source": "test_sender"
        }
        
        # Encode and send
        json_data = json.dumps(message).encode('utf-8')
        bytes_sent = sock.sendto(json_data, (SERVER_IP, SERVER_PORT))
        print(f"Sent message #{counter} ({bytes_sent} bytes): {message}")
        
        # Try to receive a response
        try:
            data, addr = sock.recvfrom(1024)
            print(f"Received response from {addr}: {data.decode('utf-8')}")
        except socket.timeout:
            print("No response received (timeout)")
        
        counter += 1
        time.sleep(1)
        
except KeyboardInterrupt:
    print("Stopped by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    sock.close()
    print("Socket closed") 