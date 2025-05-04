import serial
import time
import logging
from typing import Optional

class ArduinoManager:
    def __init__(self, port: str, baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self.connection: Optional[serial.Serial] = None
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> bool:
        try:
            self.connection = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            self.logger.info(f"Successfully connected to Arduino on {self.port}")
            return True
        except serial.SerialException as e:
            self.logger.error(f"Failed to connect to Arduino: {str(e)}")
            return False
            
    def send_command(self, command: str) -> bool:
        if not self.connection:
            self.logger.error("No Arduino connection available")
            return False
            
        try:
            self.connection.write(command.encode())
            return True
        except serial.SerialException as e:
            self.logger.error(f"Failed to send command: {str(e)}")
            return False
            
    def close(self):
        if self.connection:
            self.connection.close()
            self.logger.info("Arduino connection closed")