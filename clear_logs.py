import os
import json
import time
import datetime
from threading import Lock

# Path to the logs directory and access logs file
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)  # Ensure logs directory exists
ACCESS_LOGS_FILE = os.path.join(LOGS_DIR, 'access_logs.json')
FALLBACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'access_logs_fallback.json')
MAIN_LOGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'access_logs.json')

def clear_logs_directly():
    """
    Clear logs directly by writing an empty array to all possible log files.
    This is a brute force approach that should work regardless of the current state.
    """
    print("Starting direct log clearing process...")
    
    # List of all possible log file locations
    log_files = [
        ACCESS_LOGS_FILE,
        FALLBACK_FILE,
        MAIN_LOGS_FILE
    ]
    
    success = False
    
    # Try to clear each file
    for log_file in log_files:
        try:
            print(f"Attempting to clear log file: {log_file}")
            
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                print(f"Created directory: {log_dir}")
            
            # Write empty array to file
            with open(log_file, 'w') as f:
                json.dump([], f)
            
            print(f"Successfully cleared log file: {log_file}")
            success = True
        except Exception as e:
            print(f"Error clearing log file {log_file}: {e}")
    
    if success:
        print("Successfully cleared at least one log file.")
        return True
    else:
        print("Failed to clear any log files.")
        return False

if __name__ == "__main__":
    # This allows the script to be run directly
    result = clear_logs_directly()
    print(f"Log clearing result: {result}")
