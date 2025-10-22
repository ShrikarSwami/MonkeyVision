import cv2 as cv #This actually imports the OpenCV library which is how we will be able to access a webcam
import time # Importing time module to add delays if necessary and will also do something else

CAM_INDEX = 0  # Default camera index (usually 0 for built-in webcam, change it if needed)
BACKEND = cv.CAP_AVFOUNDATION  # Use AVFoundation backend for macOS (you can change it based on your OS so for you Abhiram you have ot switch it to cv.CAP_DSHOW for Windows)

cap = cv.VideoCapture(CAM_INDEX, BACKEND)  # Create a VideoCapture object to access the webcam (Because Python is object oriented we create an object of the VideoCapture class)
if not cap.isOpened():  # Check if the webcam is opened correctly
    print("Error: Could not open webcam, try a different index or check your camera connection.")
    exit()


#(This is setting the resolution of the webcam to 1080 progressive scan (What the p stands for))
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)  # Set the width of the frames to 1920 pixels
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)  # Set the height of the frames to 1080 pixels

WIN = "Monkey" #This is what the window will be called
t_prev = time.perf_counter() # Previous time for FPS calculation

while True:  # Start an infinite loop to continuously capture frames from the webcam 
    ret, frame = cap.read()  # Read a frame from the webcam
    frame = cv.flip(frame, 1)  # Flip the frame horizontally for a mirror effect and so it looks normal to us
    if not ret:  # If the frame is not read correctly, break the loop (Fix your webcam)
        print("Error: Could not read frame.")
        break

    # Calculate FPS
    t_curr = time.perf_counter()  # Current time
    fps = 1 / (t_curr - t_prev)  # FPS is 1 divided by the time difference between the current and previous frame
    t_prev = t_curr  # Update the previous time

    # Overlay FPS on the frame 
    cv.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow(WIN, frame)  # Display the frame in a window named "Monkey" 

    if cv.waitKey(1) & 0xFF == ord('q'):  # Wait for 1 ms and check if 'q' key is pressed to exit
        break # Exit the loop if 'q' is pressed

cap.release()  # Release the webcam resource
cv.destroyAllWindows()  # Close all OpenCV windows