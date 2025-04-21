import os
import json
import time
import datetime
from threading import Lock

# Path to the logs directory and access logs file
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)  # Ensure logs directory exists
ACCESS_LOGS_FILE = os.path.join(LOGS_DIR, 'access_logs.json')
# Lock for thread-safe file operations
log_lock = Lock()

def initialize_logs():
    """Initialize the access logs file if it doesn't exist."""
    try:
        # Ensure logs directory exists
        os.makedirs(LOGS_DIR, exist_ok=True)

        if not os.path.exists(ACCESS_LOGS_FILE):
            with open(ACCESS_LOGS_FILE, 'w') as f:
                json.dump([], f)
            print(f"Created new log file at {ACCESS_LOGS_FILE}")
            return []

        try:
            with open(ACCESS_LOGS_FILE, 'r') as f:
                logs = json.load(f)
                print(f"Loaded {len(logs)} log entries from {ACCESS_LOGS_FILE}")
                return logs
        except json.JSONDecodeError:
            print(f"Log file {ACCESS_LOGS_FILE} is corrupted, creating new one")
            # If file is corrupted, create a new one
            with open(ACCESS_LOGS_FILE, 'w') as f:
                json.dump([], f)
            return []
    except Exception as e:
        print(f"Error initializing logs: {e}")
        # Create a fallback file in the root directory if logs directory is inaccessible
        fallback_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'access_logs_fallback.json')
        print(f"Using fallback log file: {fallback_file}")
        with open(fallback_file, 'w') as f:
            json.dump([], f)
        return []

def save_logs(logs):
    """Save logs to the file."""
    try:
        # Ensure logs directory exists
        os.makedirs(LOGS_DIR, exist_ok=True)

        # Create a temporary file first
        temp_file = ACCESS_LOGS_FILE + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(logs, f, indent=2)

        # Then rename it to the actual file (atomic operation)
        if os.path.exists(ACCESS_LOGS_FILE):
            # On Windows, we need to remove the destination file first
            try:
                os.remove(ACCESS_LOGS_FILE)
            except Exception as e:
                print(f"Warning: Could not remove existing log file: {e}")

        os.rename(temp_file, ACCESS_LOGS_FILE)
        print(f"Successfully saved {len(logs)} log entries to {ACCESS_LOGS_FILE}")
        return True
    except Exception as e:
        print(f"Error saving logs: {e}")
        # Try fallback location
        try:
            fallback_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'access_logs_fallback.json')
            with open(fallback_file, 'w') as f:
                json.dump(logs, f, indent=2)
            print(f"Saved logs to fallback location: {fallback_file}")
            return True
        except Exception as fallback_error:
            print(f"Error saving to fallback location: {fallback_error}")
            raise

def log_access(user_name, access_granted, method="Face Recognition"):
    """
    Log an access attempt.

    Args:
        user_name (str): Name of the user attempting access
        access_granted (bool): Whether access was granted
        method (str): Method of access (Face Recognition, Manual, etc.)
    """
    with log_lock:
        logs = initialize_logs()

        # Create a new log entry
        log_entry = {
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_name,
            "access_granted": access_granted,
            "method": method
        }

        # Add to logs and save
        logs.append(log_entry)
        save_logs(logs)

        return log_entry

def get_logs(limit=100, offset=0, user_filter=None, access_filter=None):
    """
    Get access logs with optional filtering.

    Args:
        limit (int): Maximum number of logs to return
        offset (int): Offset for pagination
        user_filter (str, optional): Filter logs by user name
        access_filter (bool, optional): Filter logs by access granted/denied

    Returns:
        list: Filtered logs
    """
    with log_lock:
        logs = initialize_logs()

        # Apply filters
        filtered_logs = logs

        if user_filter:
            filtered_logs = [log for log in filtered_logs if user_filter.lower() in log["user"].lower()]

        if access_filter is not None:
            filtered_logs = [log for log in filtered_logs if log["access_granted"] == access_filter]

        # Sort by timestamp (newest first)
        filtered_logs.sort(key=lambda x: x["timestamp"], reverse=True)

        # Apply pagination
        paginated_logs = filtered_logs[offset:offset+limit]

        return paginated_logs, len(filtered_logs)

def clear_logs():
    """Clear all logs (for maintenance purposes)."""
    try:
        with log_lock:
            print(f"Attempting to clear logs from {ACCESS_LOGS_FILE}")

            # Check if the file exists
            if not os.path.exists(ACCESS_LOGS_FILE):
                print(f"Log file {ACCESS_LOGS_FILE} does not exist, creating empty one")
                save_logs([])
                return True

            # Check if we can read the file
            try:
                with open(ACCESS_LOGS_FILE, 'r') as _:
                    print(f"Successfully opened log file for reading")
            except Exception as e:
                print(f"Error opening log file for reading: {e}")
                # Try to delete the file directly
                try:
                    os.remove(ACCESS_LOGS_FILE)
                    print(f"Removed problematic log file")
                    save_logs([])
                    return True
                except Exception as del_error:
                    print(f"Could not remove log file: {del_error}")
                    raise

            # Then try to save empty logs
            print("Saving empty log file")
            save_logs([])
            print("Logs cleared successfully")
            return True
    except Exception as e:
        print(f"Error clearing logs: {e}")
        # Try to create a new empty file as a last resort
        try:
            fallback_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'access_logs_fallback.json')
            with open(fallback_file, 'w') as f:
                json.dump([], f)
            print(f"Created empty fallback log file: {fallback_file}")
            # We'll use the fallback file but won't modify the global variable
            # as that's causing a syntax error
            return True
        except Exception as fallback_error:
            print(f"Failed to create fallback log file: {fallback_error}")
            raise

def clear_logs_directly():
    """Clear logs using direct file write - most reliable method."""
    try:
        # Try to clear both the main log file and the fallback
        main_log_file = ACCESS_LOGS_FILE
        fallback_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'access_logs_fallback.json')
        root_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'access_logs.json')

        # List of all possible log files
        log_files = [main_log_file, fallback_file, root_log_file]

        success = False
        for log_file in log_files:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(log_file), exist_ok=True)

                # Write empty array to file
                with open(log_file, 'w') as f:
                    json.dump([], f)
                print(f"Successfully cleared log file: {log_file}")
                success = True
            except Exception as e:
                print(f"Error clearing log file {log_file}: {e}")

        return success
    except Exception as e:
        print(f"Error in clear_logs_directly: {e}")
        return False
