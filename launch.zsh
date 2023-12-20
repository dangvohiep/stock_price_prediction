#!/bin/zsh

# Function to get the current IP address
function get_ip_address() {
    local ip=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | grep -v "169.254" | awk '{print $2}' | head -n 1)
    echo $ip
}

# Get the current IP address
IP_ADDRESS=$(get_ip_address)

# Activate the Python virtual environment
source .venv/bin/activate

# Check if activation was successful
if [ $? -eq 0 ]; then
    echo "Virtual environment activated successfully."
else
    echo "Failed to activate virtual environment."
    exit 1
fi

# Change to the 'stock' directory
cd stock

# Check if directory change was successful
if [ $? -eq 0 ]; then
    echo "Changed to 'stock' directory."
else
    echo "Failed to change to 'stock' directory."
    exit 1
fi

# Start the Django server with nohup and redirect output to log files
nohup python3 manage.py runserver $IP_ADDRESS:8000 1>> stdout.log 2>> stderr.log &

# Check if Django server started successfully
if [ $? -eq 0 ]; then
    echo "Django server started successfully."
    echo "Try this: $IP_ADDRESS:8000/stock/?symbol=IBM&backward=100&forward=10"
else
    echo "Failed to start Django server."
    exit 1
fi
