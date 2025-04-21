import cv2
import numpy as np
import time

class MockCamera:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.is_open = True
        self.frame_count = 0
        
    def read(self):
        # Create a black frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add some dynamic content
        self.frame_count += 1
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, self.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add a moving circle
        radius = 30
        x = int(self.width/2 + (self.width/3) * np.sin(self.frame_count / 30))
        y = int(self.height/2 + (self.height/3) * np.cos(self.frame_count / 20))
        cv2.circle(frame, (x, y), radius, (0, 0, 255), -1)
        
        # Add text
        cv2.putText(frame, "Mock Camera Active", (self.width//2 - 100, self.height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return True, frame
    
    def isOpened(self):
        return self.is_open
    
    def release(self):
        self.is_open = False
    
    def set(self, propId, value):
        # Mock implementation - just return True
        return True

# Test the mock camera
if __name__ == "__main__":
    mock_cam = MockCamera()
    
    # Display the mock camera feed
    while True:
        ret, frame = mock_cam.read()
        cv2.imshow("Mock Camera", frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    mock_cam.release()
    cv2.destroyAllWindows()
