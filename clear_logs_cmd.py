import os
import json
import sys

def clear_logs_cmd():
    """
    Command-line script to clear all logs.
    This is a very simple script that just writes an empty array to the log file.
    """
    try:
        # Find all possible log files
        log_files = []
        
        # Main directory
        main_dir = os.path.dirname(os.path.abspath(__file__))
        log_files.append(os.path.join(main_dir, 'access_logs.json'))
        
        # Logs directory
        logs_dir = os.path.join(main_dir, 'logs')
        if os.path.exists(logs_dir):
            log_files.append(os.path.join(logs_dir, 'access_logs.json'))
        
        # Clear each file
        cleared = False
        for log_file in log_files:
            try:
                print(f"Attempting to clear {log_file}...")
                with open(log_file, 'w') as f:
                    json.dump([], f)
                print(f"Successfully cleared {log_file}")
                cleared = True
            except Exception as e:
                print(f"Error clearing {log_file}: {e}")
        
        if cleared:
            print("Successfully cleared logs.")
            return 0
        else:
            print("Failed to clear any log files.")
            return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(clear_logs_cmd())
