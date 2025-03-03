#!/usr/bin/env python3
import socket
import subprocess
import json
import platform
import os
import time

def run_cmd(cmd):
    """Run a command and return its output."""
    print(f"Running: {cmd}")
    try:
        output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        return output
    except subprocess.CalledProcessError as e:
        return f"Error (code {e.returncode}): {e.output.decode('utf-8').strip() if e.output else 'No output'}"

def print_section(title):
    """Print a section title."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)

# System information
print_section("SYSTEM INFORMATION")
print(f"Platform: {platform.platform()}")
print(f"Python version: {platform.python_version()}")
print(f"Hostname: {platform.node()}")
print(f"User: {os.getlogin()}")

# Network interfaces
print_section("NETWORK INTERFACES")
if platform.system() == "Linux" or platform.system() == "Darwin":
    interfaces = run_cmd("ifconfig")
else:
    interfaces = run_cmd("ipconfig /all")
print(interfaces)

# IP routing
print_section("IP ROUTING")
if platform.system() == "Linux" or platform.system() == "Darwin":
    routes = run_cmd("netstat -rn")
else:
    routes = run_cmd("route print")
print(routes)

# Test UDP port binding
print_section("UDP PORT BINDING TEST")
test_ports = [9999, 8888, 7777]
for port in test_ports:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', port))
        print(f"Successfully bound to UDP port {port}")
        sock.close()
    except Exception as e:
        print(f"Failed to bind to UDP port {port}: {e}")

# Local UDP echo test
print_section("LOCAL UDP ECHO TEST")
try:
    # Create server socket
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('127.0.0.1', 7777))
    server_sock.settimeout(3)
    
    # Create client socket
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_sock.settimeout(3)
    
    # Send test message
    test_msg = json.dumps({"test": "message", "timestamp": time.time()}).encode('utf-8')
    client_sock.sendto(test_msg, ('127.0.0.1', 7777))
    print("Sent test message to local UDP port 7777")
    
    # Try to receive
    data, addr = server_sock.recvfrom(1024)
    print(f"Received: {data.decode('utf-8')} from {addr}")
    
    # Send response
    response = json.dumps({"echo": json.loads(data.decode('utf-8'))}).encode('utf-8')
    server_sock.sendto(response, addr)
    
    # Receive response
    data, addr = client_sock.recvfrom(1024)
    print(f"Received echo response: {data.decode('utf-8')} from {addr}")
    
    print("Local UDP echo test successful!")
except Exception as e:
    print(f"Local UDP echo test failed: {e}")
finally:
    server_sock.close()
    client_sock.close()

# Firewall check
print_section("FIREWALL STATUS")
if platform.system() == "Linux":
    print(run_cmd("sudo iptables -L | grep -i udp || echo 'No UDP rules found'"))
elif platform.system() == "Darwin":
    print(run_cmd("sudo pfctl -s rules 2>/dev/null | grep -i udp || echo 'No UDP rules found'"))
else:
    print(run_cmd("netsh advfirewall show currentprofile"))

print("\nDiagnostics complete. Use this information to troubleshoot UDP connectivity issues.") 