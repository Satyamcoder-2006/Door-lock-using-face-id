import cv2
import time

def test_camera(index):
    print(f"Testing camera at index {index}...")
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"Failed to open camera at index {index}")
        return False
    
    print(f"Camera at index {index} opened successfully!")
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame from camera at index {index}")
        cap.release()
        return False
    
    print(f"Successfully read frame from camera at index {index}")
    
    # Save the frame to a file
    cv2.imwrite(f"camera_{index}_test.jpg", frame)
    print(f"Saved test image to camera_{index}_test.jpg")
    
    # Release the camera
    cap.release()
    return True

# Try different camera indices
for i in range(5):
    print("\n" + "="*50)
    success = test_camera(i)
    if success:
        print(f"Camera at index {i} is working!")
    else:
        print(f"Camera at index {i} is not working.")
    print("="*50 + "\n")
    time.sleep(1)  # Wait a bit between tests

print("\nCamera test completed.")
