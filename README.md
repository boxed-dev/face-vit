# Face Detection Application 

The **Face Detection Application** is a Python program that uses the OpenCV and face_recognition libraries to perform real-time face detection on a live video feed from your webcam. It can also recognize and label faces in the video by comparing them to a set of known faces. The application is built using Streamlit, making it easy to deploy and use via a web interface.

## Prerequisites

Before using this application, ensure you have the following prerequisites:

- Python
- OpenCV
- face_recognition
- Streamlit

You can install the required libraries using `pip`:

```bash
pip install opencv-python-headless
pip install face-recognition
pip install streamlit
```

## Usage

1. Clone the repository or download the application to your local machine.

2. Open a terminal and navigate to the application's directory.

3. Run the application using the following command:

   ```bash
   streamlit run your_app_name.py
   ```

   Replace `your_app_name.py` with the actual name of your Python script.

4. The application will open in your default web browser, displaying the live video feed from your webcam with real-time face detection and recognition.

## Features

### Real-Time Face Detection

The application uses OpenCV to capture video from your webcam and detect faces in real-time. Detected faces are highlighted with green rectangles.

### Face Recognition

The application can recognize and label faces by comparing them to a set of known faces. Known faces and their corresponding labels should be stored in an "Images" directory in the application's directory. The labels are displayed above the recognized faces.

### Streamlit Web Interface

The application is built using Streamlit, providing an easy-to-use web interface for interacting with the face detection and recognition features.

## Customize and Enhance

You can customize and enhance the application in several ways:

- Add more known faces to the "Images" directory for better recognition.
- Modify the label display style or position.
- Implement additional features such as face tracking, emotion detection, or face mask recognition.
- Improve the application's performance for real-world use by optimizing the code.

## Disclaimer

This application is a basic example and may require further enhancements for production use. Ensure that you have the necessary permissions to capture video from your webcam.

Please use this application responsibly and respect privacy and legal considerations when using it.

Feel free to contribute to this project and make it even better!

---

📷 For questions and support, please contact me: [Your Contact Information]

👤 Feel free to contribute to this project and make it even better!
