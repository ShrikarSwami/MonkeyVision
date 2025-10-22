import cv2 as cv #This actually imports the OpenCV library which is how we will be able to access a webcam
import time # Importing time module to add delays if necessary and will also do something else
import mediapipe as mp # Importing MediaPipe library for advanced computer vision tasks 
mp_hands = mp.solutions.hands # This is specifically importing the hands solution from MediaPipe
mp_draw   = mp.solutions.drawing_utils # This is for drawing the landmarks on the hands
mp_styles = mp.solutions.drawing_styles # This is for styling the landmarks and connections
import sys # Importing sys module to handle system-specific parameters and functions (will help for windows vs macOS compatibility)
import argparse # Importing argparse module to handle command-line arguments (not used in this code but useful for future enhancements)




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

hands = mp_hands.Hands(  # Initialize the MediaPipe Hands solution
    static_image_mode=False,  # Set to False for video input (not static images because thats stupid)
    max_num_hands=2,  # Maximum number of hands to detect (2 for both hands)
    min_detection_confidence=0.5,  # Minimum confidence for detection (50% confidence to consider a detection valid)
    min_tracking_confidence=0.5  # Minimum confidence for tracking (50% confidence to consider a tracking valid)
)


while True:  # Start an infinite loop to continuously capture frames from the webcam 
    ret, frame = cap.read()  # Read a frame from the webcam

    if not ret:  # If the frame is not read correctly, break the loop (Fix your webcam)
        print("Error: Could not read frame.")
        break

    frame = cv.flip(frame, 1)  # Flip the frame horizontally for a mirror effect and so it looks normal to us

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Convert the frame from BGR to RGB color space (MediaPipe uses RGB format while OpenCV uses BGR)
    out = hands.process(rgb)  # Process the RGB frame to detect hands
    if out.multi_hand_landmarks:  # If hands are detected in the frame
        for hl in out.multi_hand_landmarks:  # Iterate through each detected hand
            mp_draw.draw_landmarks(  # Draw the hand landmarks on the original frame
                frame, hl, mp_hands.HAND_CONNECTIONS,  # Draw landmarks and connections
                mp_styles.get_default_hand_landmarks_style(),  # Use default style for landmarks
                mp_styles.get_default_hand_connections_style()  # Use default style for connections
            )


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

    cv.imshow(WIN, frame)  # Display the frame in a window named "Monkey" 

    if cv.waitKey(1) & 0xFF == ord('q'):  # Wait for 1 ms and check if 'q' key is pressed to exit
        break # Exit the loop if 'q' is pressed

cap.release()  # Release the webcam resource
cv.destroyAllWindows()  # Close all OpenCV windows
hands.close()  # Close the MediaPipe Hands solution
