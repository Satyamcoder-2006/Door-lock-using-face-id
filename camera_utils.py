import cv2
import threading
import time
import numpy as np

class ThreadedCamera:
    """
    Camera class that runs in a separate thread to decouple frame capture from processing.
    This significantly reduces lag by ensuring frame capture happens continuously.
    """
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.width = width
        self.height = height
        self.capture = cv2.VideoCapture(src)
        
        # Set optimal camera parameters
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Initialize variables
        self.ret, self.frame = self.capture.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.thread = None
    
    def start(self):
        """Start the camera thread"""
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self
    
    def _update(self):
        """Update thread that continuously captures frames"""
        while self.started:
            ret, frame = self.capture.read()
            if not ret:
                # If frame capture fails, wait a bit and try again
                time.sleep(0.1)
                continue
                
            with self.read_lock:
                self.ret = ret
                self.frame = frame
    
    def read(self):
        """Thread-safe frame reading"""
        with self.read_lock:
            if not self.ret:
                return False, None
            return self.ret, self.frame.copy()
    
    def isOpened(self):
        """Check if camera is opened"""
        return self.capture.isOpened() if self.capture else False
    
    def release(self):
        """Release camera resources"""
        self.stop()
        if self.capture:
            self.capture.release()
    
    def stop(self):
        """Stop the camera thread"""
        self.started = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        
    def __enter__(self):
        self.start()
        return self