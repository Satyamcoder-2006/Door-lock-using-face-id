import os
import json
from werkzeug.security import generate_password_hash, check_password_hash

# Get the absolute path to the config file
ADMIN_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'admin_config.json')
DEFAULT_PASSWORD = 'admin'

def load_admin_config():
    if not os.path.exists(ADMIN_CONFIG_FILE):
        # Create default config with hashed password
        config = {
            'password_hash': generate_password_hash(DEFAULT_PASSWORD)
        }
        with open(ADMIN_CONFIG_FILE, 'w') as f:
            json.dump(config, f)
        return config
    
    with open(ADMIN_CONFIG_FILE, 'r') as f:
        return json.load(f)

def save_admin_config(config):
    with open(ADMIN_CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def verify_password(password):
    config = load_admin_config()
    return check_password_hash(config['password_hash'], password)

def change_password(current_password, new_password):
    config = load_admin_config()
    if not check_password_hash(config['password_hash'], current_password):
        return False
    
    config['password_hash'] = generate_password_hash(new_password)
    save_admin_config(config)
    return True 