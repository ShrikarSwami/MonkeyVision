import cv2 as cv #This actually imports the OpenCV library which is how we will be able to access a webcam
import time # Importing time module to add delays if necessary and will also do something else
import mediapipe as mp # Importing MediaPipe library for advanced computer vision tasks 
mp_hands = mp.solutions.hands # This is specifically importing the hands solution from MediaPipe
mp_draw   = mp.solutions.drawing_utils # This is for drawing the landmarks on the hands
mp_styles = mp.solutions.drawing_styles # This is for styling the landmarks and connections
import sys # Importing sys module to handle system-specific parameters and functions (will help for windows vs macOS compatibility)
import argparse # Importing argparse module to handle command-line arguments 
import numpy as np #This imports the numpy library which is useful for numerical operations and handling arrays 
mp_face = mp.solutions.face_mesh # Importing the face mesh solution from MediaPipe this is how we will be able to detect faces
import math # Importing math module for mathematical operations 
from collections import deque # Importing deque from collections module for efficient appending and popping of elements from both ends 



history = deque(maxlen=10) # this will store the last 5 positions of the mouth for smoothing

def lmk_xy(lms, idx, w, h):
    #normalize landmarks or face mesh points to pixel coordinates
    return int(lms[idx].x * w), int(lms[idx].y * h) #This looks complicated but its just converting the normalized landmark coordinates to pixel coordinates based on the frame width and height which means multiplying by w and h

def dist(a, b):
    #Calculate Euclidean distance between two points (Euclidean distance is the straight line distance between two points in 2D space)
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) #This is just the distance formula on some pythagorean theorem type stuff

#Monkey pictures:
img_neutral = cv.imread("monkey_neutral.webp")
img_middle  = cv.imread("monkey_say.jpg")
img_shush   = cv.imread("monkey_think.jpg")

REACTION = "Monkey"
cv.namedWindow(REACTION, cv.WINDOW_NORMAL)  # Create a named window for displaying the monkey reaction

#This is a sanity check to make sure your webcam works and the images load properly
for name, im in [("neutral", img_neutral), ("middle", img_middle), ("shush", img_shush)]:
    if im is None:
        print(f"Error: Could not load image '{name}'. Please ensure the file exists and the path is correct.")

CAM_INDEX = 0  # Default camera index (usually 0 for built-in webcam, change it if needed)
BACKEND = cv.CAP_AVFOUNDATION  # Use AVFoundation backend for macOS (you can change it based on your OS so for you Abhiram you have ot switch it to cv.CAP_DSHOW for Windows)

cap = cv.VideoCapture(CAM_INDEX, BACKEND)  # Create a VideoCapture object to access the webcam (Because Python is object oriented we create an object of the VideoCapture class)
if not cap.isOpened():  # Check if the webcam is opened correctly
    print("Error: Could not open webcam, try a different index or check your camera connection.") # Print an error message if your webcam doesn't work
    exit()


#(This is setting the resolution of the webcam to 1080 progressive scan (What the p stands for))
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)  # Set the width of the frames to 1920 pixels
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)  # Set the height of the frames to 1080 pixels

WIN = "Monkey" #This is what the window will be called
t_prev = time.perf_counter() # Previous time for FPS calculation


#Hand detection setup
hands = mp_hands.Hands(  # Initialize the MediaPipe Hands solution this is needed to detect hands
    static_image_mode=False,  # Set to False for video input (not static images because thats stupid)
    max_num_hands=2,  # Maximum number of hands to detect (2 for both hands because that's how many we have but one also works because the monkey has one hand that changes)
    min_detection_confidence=0.5,  # Minimum confidence for detection (50% confidence to consider a detection valid)
    min_tracking_confidence=0.5  # Minimum confidence for tracking (50% confidence to consider a tracking valid)
)

#Face mesh detection setup
face = mp_face.FaceMesh( # Initialize the MediaPipe Face Mesh solution this is used to detect face mesh
    static_image_mode=False,  # Set to False for video input (because again thats stupid)
    max_num_faces=1,  # Maximum number of faces to detect (1 for single face one single monkey)
    min_detection_confidence=0.6,  # Minimum confidence for detection
    min_tracking_confidence=0.6  # Minimum confidence for tracking
)

# Main loop to continuously capture frames from the webcam and do all the processing
while True:  # Start an infinite loop to continuously capture frames from the webcam 
    ret, frame = cap.read()  # Read a frame from the webcam

    # Check if frame is read correctly
    if not ret:  # If the frame is not read correctly, break the loop (Fix your webcam)
        print("Error: Could not read frame.")
        break


    # Flip the frame horizontally for a mirror effect and so it looks normal to us
    frame = cv.flip(frame, 1)  # Flip the frame horizontally for a mirror effect and so it looks normal to us


    # Convert the BGR frame to RGB as MediaPipe requires RGB input and process it to detect hands
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Convert the frame from BGR to RGB color space (MediaPipe uses RGB format while OpenCV uses BGR)
    out = hands.process(rgb)  # Process the RGB frame to detect hands
    if out.multi_hand_landmarks:  # If hands are detected in the frame
        for hl in out.multi_hand_landmarks:  # Iterate through each detected hand
            mp_draw.draw_landmarks(  # Draw the hand landmarks on the original frame
                frame, hl, mp_hands.HAND_CONNECTIONS,  # Draw landmarks and connections
                mp_styles.get_default_hand_landmarks_style(),  # Use default style for landmarks
                mp_styles.get_default_hand_connections_style()  # Use default style for connections
            )

            

    #Face mesh detection and drawing
    face_out=face.process(rgb)  # Process the RGB frame to detect face mesh
    if face_out.multi_face_landmarks:  # If face mesh is detected in the frame
        h, w = frame.shape[:2]  # Get the height and width of the frame
        fl = face_out.multi_face_landmarks[0].landmark  # Get the first detected face landmarks

        x13, y13 = int(fl[13].x * w), int(fl[13].y * h)  # Get the coordinates of landmark 13 or lower lip
        x14, y14 = int(fl[14].x * w), int(fl[14].y * h)  # Get the coordinates of landmark 14 or upper lip

        mouth_x = (x13 + x14) // 2  # Calculate the x-coordinate of the mouth center
        mouth_y = (y13 + y14) // 2  # Calculate the y-coordinate of the mouth center

        #Draw a small circle at the mouth center
        cv.circle(frame, (mouth_x, mouth_y), 5, (0, 255, 255), -1) # Draw a small yellow circle at the mouth center
        #optional label for debugging (comment out later)
        cv.putText(frame, "mouth", (mouth_x + 6, mouth_y - 6), # Label the mouth center for debugging
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) # Label the mouth center for debugging
        mouth_pt = (mouth_x, mouth_y)  # Create a tuple for the mouth point which will be used for distance calculation
    else:
        mouth_pt = None  # If no face is detected, set mouth_pt to None

        label = "NONE"
h, w = frame.shape[:2]

if out.multi_hand_landmarks:
    # use the first detected hand for now
    hl = out.multi_hand_landmarks[0] # Get the first detected hand landmarks
    lm = hl.landmark # Get the list of landmarks for the hand

    # tip vs PIP test (True == finger up). order: [thumb, index, middle, ring, pinky]
    TIPS = [4, 8, 12, 16, 20] # Landmark indices for finger tips (Landmark means specific points on the hand that MediaPipe detects)
    PIPS = [2, 6, 10, 14, 18] # Landmark indices for finger PIP joints (Pip means Proximal Interphalangeal Joint which is the middle joint of the finger)
    fingers_up = [(lm[t].y < lm[p].y) for t, p in zip(TIPS, PIPS)] # Determine if each finger is up by comparing tip and PIP y-coordinates
    thumb_up, index_up, middle_up, ring_up, pinky_up = fingers_up # Unpack the finger states into individual variables

    # 1) only middle up (ignore thumb)
    if middle_up and not index_up and not ring_up and not pinky_up:
        label = "MIDDLE"

    # 2)


    # Calculate FPS
    t_curr = time.perf_counter()  # Current time
    fps = 1.0 / max(1e-6, (t_curr - t_prev))  # FPS is 1 divided by the time difference between the current and previous frame
    t_prev = t_curr  # Update the previous time

    # translucent HUD bar across the top
    overlay = frame.copy() # Create a copy of the frame to draw on
    cv.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1) #Draw a black rectangle at the top of the frame
    alpha = 0.6 # opacity factor
    frame = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0) # Blend the overlay with the original frame


    # Overlay FPS on the frame 
    cv.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # choose a reaction image based on the detected gesture
    if   stable == "MIDDLE" and img_middle  is not None: show = img_middle
    elif stable == "SHUSH"  and img_shush   is not None: show = img_shush
    else:                                                  show = img_neutral

    # display reaction window
    if show is not None:
        sh = int(frame.shape[0] * 0.9)
        scale = sh / show.shape[0]
        sw = int(show.shape[1] * scale)
        preview = cv.resize(show, (sw, sh), interpolation=cv.INTER_AREA)
        cv.imshow(REACTION, preview)

    # draw the current gesture on your HUD
    cv.putText(frame, f"Gesture: {stable}", (160, 30),
            cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    cv.imshow(WIN, frame)  # Display the frame in a window named "Monkey" 

    if cv.waitKey(1) & 0xFF == ord('q'):  # Wait for 1 ms and check if 'q' key is pressed to exit
        break # Exit the loop if 'q' is pressed

cap.release()  # Release the webcam resource
cv.destroyAllWindows()  # Close all OpenCV windows
hands.close()  # Close the MediaPipe Hands solution
face.close()  # Close the MediaPipe Face Mesh solution