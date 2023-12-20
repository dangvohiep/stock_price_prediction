#!/bin/zsh

# Find the PIDs of the manage.py process
PIDs=$(ps aux | grep '[m]anage.py runserver' | awk '{print $2}')

# Check if PIDs were found
if [[ -z $PIDs ]]; then
    echo "Django process manage.py not found."
else
    # Kill each process
    for PID in ${(f)PIDs}; do
        kill $PID
        echo "Django process manage.py (PID: $PID) has been terminated."
    done
fi
