#!/bin/bash

# DeepRacer Controller Startup Script
# This script sources the required ROS2 environment and starts the controller

set -e  # Exit on any error

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTROLLER_SCRIPT="$SCRIPT_DIR/controller_racer.py"

# Source ROS2 environment
echo "Sourcing ROS2 environment..."
source /opt/ros/foxy/setup.bash

# Source DeepRacer interfaces
echo "Sourcing DeepRacer interfaces..."
source ~/deepracer_ws/aws-deepracer-interfaces-pkg/install/setup.bash

# Navigate to script directory
cd "$SCRIPT_DIR"

# Kill any existing process using the camera
echo "Checking for processes using camera..."
CAMERA_PROCESSES=$(lsof /dev/video0 2>/dev/null | grep -v COMMAND | awk '{print $2}' | head -n 1 || true)
if [ ! -z "$CAMERA_PROCESSES" ]; then
    echo "Killing existing camera process: $CAMERA_PROCESSES"
    kill -9 $CAMERA_PROCESSES 2>/dev/null || true
    sleep 2
fi

# Also check video1 just in case
CAMERA_PROCESSES=$(lsof /dev/video1 2>/dev/null | grep -v COMMAND | awk '{print $2}' | head -n 1 || true)
if [ ! -z "$CAMERA_PROCESSES" ]; then
    echo "Killing existing camera process on video1: $CAMERA_PROCESSES"
    kill -9 $CAMERA_PROCESSES 2>/dev/null || true
    sleep 2
fi

echo "Starting DeepRacer controller..."
exec python3 "$CONTROLLER_SCRIPT" 