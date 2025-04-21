"""
Telegram notification module for FaceLock system.
This module handles sending notifications to Telegram when door events occur.
"""
import requests
import logging
import os
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration file path
CONFIG_FILE = 'telegram_config.json'

class TelegramNotifier:
    """Class to handle Telegram notifications for door events."""
    
    def __init__(self):
        """Initialize the Telegram notifier with configuration."""
        self.config = self._load_config()
        self.enabled = self.config.get('enabled', False)
        self.bot_token = self.config.get('bot_token', '')
        self.chat_ids = self.config.get('chat_ids', [])
        
        # Validate configuration
        if self.enabled and (not self.bot_token or not self.chat_ids):
            logger.warning("Telegram notifications enabled but missing bot_token or chat_ids")
            self.enabled = False
    
    def _load_config(self):
        """Load configuration from file or create default if not exists."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading Telegram config: {e}")
                return self._create_default_config()
        else:
            return self._create_default_config()
    
    def _create_default_config(self):
        """Create and save default configuration."""
        default_config = {
            'enabled': False,
            'bot_token': '',
            'chat_ids': [],
            'notification_types': {
                'door_unlock': True,
                'door_lock': True,
                'failed_access': True
            }
        }
        
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(default_config, f, indent=4)
            logger.info(f"Created default Telegram configuration at {CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Error creating default Telegram config: {e}")
        
        return default_config
    
    def save_config(self, config):
        """Save updated configuration to file."""
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            self.config = config
            self.enabled = config.get('enabled', False)
            self.bot_token = config.get('bot_token', '')
            self.chat_ids = config.get('chat_ids', [])
            logger.info("Telegram configuration updated successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving Telegram config: {e}")
            return False
    
    def send_door_unlock_notification(self, user_name, method="Face Recognition"):
        """Send notification when door is unlocked."""
        if not self.enabled or not self.config.get('notification_types', {}).get('door_unlock', True):
            return False
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"🔓 Door Unlocked\n\n👤 User: {user_name}\n🔑 Method: {method}\n⏰ Time: {timestamp}"
        return self._send_message(message)
    
    def send_door_lock_notification(self, user_name=None):
        """Send notification when door is locked."""
        if not self.enabled or not self.config.get('notification_types', {}).get('door_lock', True):
            return False
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if user_name:
            message = f"🔒 Door Locked\n\n👤 User: {user_name}\n⏰ Time: {timestamp}"
        else:
            message = f"🔒 Door Locked\n\n⏰ Time: {timestamp}\n📝 Note: Auto-locked due to timeout"
        
        return self._send_message(message)
    
    def send_failed_access_notification(self, user_name, method="Face Recognition"):
        """Send notification when access is denied."""
        if not self.enabled or not self.config.get('notification_types', {}).get('failed_access', True):
            return False
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"⛔ Access Denied\n\n👤 User: {user_name}\n🔑 Method: {method}\n⏰ Time: {timestamp}"
        return self._send_message(message)
    
    def _send_message(self, message):
        """Send message to all configured chat IDs."""
        if not self.enabled or not self.bot_token or not self.chat_ids:
            return False
        
        success = True
        for chat_id in self.chat_ids:
            try:
                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                payload = {
                    'chat_id': chat_id,
                    'text': message,
                    'parse_mode': 'HTML'
                }
                response = requests.post(url, data=payload, timeout=5)
                
                if response.status_code != 200:
                    logger.error(f"Failed to send Telegram message to {chat_id}: {response.text}")
                    success = False
                
            except Exception as e:
                logger.error(f"Error sending Telegram message to {chat_id}: {e}")
                success = False
        
        return success
    
    def test_notification(self):
        """Send a test notification to verify configuration."""
        if not self.enabled or not self.bot_token or not self.chat_ids:
            return False
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"🔔 Test Notification\n\n✅ Your FaceLock notification system is working correctly!\n⏰ Time: {timestamp}"
        return self._send_message(message)


# Create a singleton instance
notifier = TelegramNotifier()

# Function to get the notifier instance
def get_notifier():
    """Get the Telegram notifier instance."""
    return notifier
