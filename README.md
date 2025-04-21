# Face Recognition Door Lock System

A smart door lock system that uses facial recognition for access control, built with Python, OpenCV, and Arduino.

## Features

- 🎥 Real-time face detection and recognition
- 🚪 Arduino-controlled door lock
- 🌐 Web interface for management
- 👥 Multiple user support
- 📊 Training interface for new users
- 🔒 Secure access control

## Prerequisites

- Python 3.8 or higher
- Arduino board with appropriate door lock mechanism
- Webcam
- Arduino IDE (for uploading the door lock code)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Connect your Arduino board and upload the door lock code (see Arduino Setup below)

4. Create the required directories:
   ```bash
   mkdir -p userimage
   ```

## Configuration

1. Update the Arduino port in `app.py` if needed:
   ```python
   arduino_port = 'COM7'  # Change this to match your Arduino port
   ```

2. Adjust the camera index in `app.py` if needed:
   ```python
   camera_index = 0  # Try 1 if the default camera isn't working
   ```

## Usage

1. Start the web application:
   ```bash
   python app.py
   ```

2. Open a web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. To add a new user:
   - Click "Add User"
   - Enter the user's name
   - Follow the prompts to capture face images
   - Wait for training to complete

4. To verify access:
   - Click "Verify Face"
   - Look at the camera
   - The system will automatically unlock the door if recognized

## Arduino Setup

1. Connect your Arduino board
2. Upload the appropriate door lock code
3. Connect the door lock mechanism according to your hardware specifications
4. Make sure the Arduino is connected to the specified COM port

## Troubleshooting

### Camera Issues
- If the camera doesn't work with index 0, try changing to index 1
- Make sure no other application is using the camera
- Check if the camera is properly connected

### Arduino Issues
- Verify the correct COM port is set
- Check Arduino connections
- Make sure the door lock mechanism is properly wired

### Face Recognition Issues
- Ensure good lighting conditions
- Try retraining with more face images
- Adjust the confidence threshold if needed

## Security Notes

- Keep the `face_recognizer_model.xml` and `label_map.pkl` files secure
- Regularly update the face recognition model
- Monitor access logs
- Consider implementing additional security measures for critical installations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 