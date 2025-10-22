import cv2 as cv #This actually imports the OpenCV library which is how we will be able to access a webcam
import time # Importing time module to add delays if necessary and will also do something else

CAM_INDEX = 0  # Default camera index (usually 0 for built-in webcam, change it if needed)
BACKEND = cv.CAP_AVFOUNDATION  # Use AVFoundation backend for macOS (you can change it based on your OS so for you Abhiram you have ot switch it to cv.CAP_DSHOW for Windows)

cap = cv.VideoCapture(CAM_INDEX, BACKEND)  # Create a VideoCapture object to access the webcam (Because Python is object oriented we create an object of the VideoCapture class)
if not cap.isOpened():  # Check if the webcam is opened correctly
    print("Error: Could not open webcam, try a different index or check your camera connection.")
    exit()
