import os
import threading
import time
import sys
from gtts import gTTS

# Try to import pygame, but provide fallback if it fails
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    print("Warning: pygame not available. Voice functionality will be limited.")
    PYGAME_AVAILABLE = False

class VoiceModule:
    def __init__(self):
        """Initialize the voice module."""
        self.audio_dir = "audio"
        self.temp_file = os.path.join(self.audio_dir, "temp_audio.mp3")

        # Create audio directory if it doesn't exist
        os.makedirs(self.audio_dir, exist_ok=True)

        # Initialize pygame mixer for audio playback if available
        self.pygame_initialized = False
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
                self.pygame_initialized = True
                print("🔊 Voice module initialized with pygame")
            except Exception as e:
                print(f"Warning: Failed to initialize pygame mixer: {e}")

        # Cache for common messages
        self._initialize_cache()

        if not self.pygame_initialized:
            print("🔊 Voice module initialized (text-to-speech only, no audio playback)")

    def _initialize_cache(self):
        """Initialize cache with common voice messages."""
        common_messages = {
            "welcome": "Welcome, access granted.",
            "door_unlocked": "Door unlocked.",
            "door_locked": "Door locked.",
            "access_denied": "Access denied. Unauthorized user."
        }

        print("🔊 Initializing voice message cache...")
        for key, message in common_messages.items():
            file_path = os.path.join(self.audio_dir, f"{key}.mp3")
            if not os.path.exists(file_path):
                try:
                    tts = gTTS(text=message, lang='en', slow=False)
                    tts.save(file_path)
                    print(f"🔊 Cached voice message: {key}")
                except Exception as e:
                    print(f"❌ Error caching voice message {key}: {e}")
            else:
                print(f"✅ Found cached voice message: {key}")

    def speak(self, text, async_mode=True):
        """Speak the given text.

        Args:
            text (str): The text to speak
            async_mode (bool): If True, speak in a separate thread
        """
        if async_mode:
            threading.Thread(target=self._speak_thread, args=(text,)).start()
        else:
            self._speak_thread(text)

    def _speak_thread(self, text):
        """Thread function to speak text."""
        try:
            # Generate a filename based on the text (for caching)
            filename = "".join(c for c in text.lower() if c.isalnum())[:20]
            file_path = os.path.join(self.audio_dir, f"{filename}.mp3")

            # Check if we already have this audio file
            if not os.path.exists(file_path):
                # Generate and save the audio file
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(file_path)
                print(f"🔊 Generated speech file: {filename}.mp3")

            # Play the audio file if pygame is available
            if self.pygame_initialized:
                try:
                    pygame.mixer.music.load(file_path)
                    pygame.mixer.music.play()

                    # Wait for the audio to finish playing
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                except Exception as e:
                    print(f"⚠️ Error playing audio: {e}")
            else:
                # Just print what would have been spoken if pygame is not available
                print(f"🔊 Would speak: '{text}' (pygame not available)")

        except Exception as e:
            print(f"❌ Error in voice module: {e}")

    def welcome_user(self, name):
        """Speak a welcome message for a recognized user."""
        self.speak(f"Welcome {name}, access granted.")

    def door_unlocked(self):
        """Announce that the door has been unlocked."""
        self.speak("Door unlocked.")

    def door_locked(self):
        """Announce that the door has been locked."""
        self.speak("Door locked.")

    def access_denied(self):
        """Announce that access has been denied."""
        self.speak("Access denied. Unauthorized user.")

# Singleton instance
_voice_module = None

def get_voice_module():
    """Get the singleton voice module instance."""
    global _voice_module
    if _voice_module is None:
        _voice_module = VoiceModule()
    return _voice_module
