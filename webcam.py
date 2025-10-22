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
import pygame #This will be used for music

#Windows heads like Abhiram should change this to cv.CAP_DSHOW
import platform
from pathlib import Path
import argparse


# Alpha-blending function for overlaying images with transparency basically what this means is putting one image on top of another with some see-through effect
def overlay_bgra(dst, src, x, y, target_w=None, target_h=None):  # Function to overlay images with alpha blending
    """
    Alpha-blends src (BGR or BGRA) onto dst at (x,y). If target_w/h are given, resizes src first.
    Handles PNG with alpha or JPG without alpha (opaque).
    All this means is that it puts one image on top of another with some see-through effect
    """
    # resize if requested 
    if target_w is not None and target_h is not None: # If target width and height are provided
        src = cv.resize(src, (target_w, target_h), interpolation=cv.INTER_AREA) # Resize the source image to the target dimensions

    # ensure 4 channels for blending
    if src.shape[2] == 3: # If the source image has no alpha channel
        # no alpha -> make fully opaque
        bgr = src # Get the BGR channels of the source image
         # Create a full alpha channel (255 means fully opaque)
        alpha = np.full((src.shape[0], src.shape[1]), 255, dtype=np.uint8) # Create an alpha channel with full opacity
    else: # If the source image has an alpha channel
        bgr = src[:, :, :3] # Get the BGR channels of the source image
        alpha = src[:, :, 3] # Get the alpha channel of the source image

    h, w = bgr.shape[:2] # Get the height and width of the source image

    # clip ROI to frame ROI means region of interest which is basically the part of the image we care about
    x0, y0 = max(0, x), max(0, y) # Calculate the top-left corner of the region of interest in the destination image
    x1, y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h) # Calculate the bottom-right corner of the region of interest in the destination image
    if x0 >= x1 or y0 >= y1: # If the region of interest is invalid (no overlap)
        return  # nothing visible

    # compute corresponding region in src
    sx0, sy0 = x0 - x, y0 - y # Calculate the top-left corner of the corresponding region in the source image
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0) # Calculate the bottom-right corner of the corresponding region in the source image

    roi = dst[y0:y1, x0:x1] # Get the region of interest in the destination image
    src_crop = bgr[sy0:sy1, sx0:sx1] # Get the corresponding region in the source image
    a = alpha[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0 # Normalize the alpha channel to [0, 1] which basically means converting the alpha values from 0-255 to 0-1
    a = a[..., None]  # (H,W,1) # Expand dimensions for broadcasting

    # alpha blend
    roi[:] = (a * src_crop.astype(np.float32) + (1 - a) * roi.astype(np.float32)).astype(np.uint8) # Perform alpha blending and update the region of interest in the destination image


history = deque(maxlen=10) # this will store the last 5 positions of the mouth for smoothing

def lmk_xy(lms, idx, w, h): #Function to get the pixel coordinates of a specific landmark a landmark is a point on the hand or face mesh
    #normalize landmarks or face mesh points to pixel coordinates
    return int(lms[idx].x * w), int(lms[idx].y * h) #This looks complicated but its just converting the normalized landmark coordinates to pixel coordinates based on the frame width and height which means multiplying by w and h

def dist(a, b):
    #Calculate Euclidean distance between two points (Euclidean distance is the straight line distance between two points in 2D space)
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) #This is just the distance formula on some pythagorean theorem type stuff

#Monkey pictures:
ASSETS = Path(__file__).parent

img_neutral = cv.imread(str(ASSETS / "monkey_neutral.png"), cv.IMREAD_UNCHANGED)
img_middle  = cv.imread(str(ASSETS / "monkey_say.png"),     cv.IMREAD_UNCHANGED)
img_think   = cv.imread(str(ASSETS / "monkey_think.png"),   cv.IMREAD_UNCHANGED)


REACTION = "Monkey"
cv.namedWindow(REACTION, cv.WINDOW_NORMAL)  # Create a named window for displaying the monkey reaction

#This is a sanity check to make sure your webcam works and the images load properly
for name, im in [("neutral", img_neutral), ("middle", img_middle), ("think", img_think)]:
    if im is None:
        print(f"Error: Could not load image '{name}'. Please ensure the file exists and the path is correct.")

# --- music setup ---
MUSIC_PATH = "music.mp3" # Path to the music file
MUSIC_READY = False # Flag to indicate if music is ready to play
try: # Try to initialize the music player
    pygame.mixer.init()             # init audio device
    pygame.mixer.music.load(MUSIC_PATH) # load music file
    pygame.mixer.music.set_volume(0.8)  # 0.0 - 1.0 set volume
     # If everything is successful, set MUSIC_READY to True
    MUSIC_READY = True # Music is ready to play
except Exception as e: # If there is an error during initialization or loading
    print(f"[audio] init/load failed: {e}") # Print the error message

# --- state variables ---
played_song = False     # have we already played the song once?
prev_stable = "NEUTRAL" # last frame's smoothed label


CAM_INDEX = 0  # Default camera index (usually 0 for built-in webcam, change it if needed)
BACKEND = cv.CAP_AVFOUNDATION  # Use AVFoundation backend for macOS (you can change it based on your OS so for you Abhiram you have ot switch it to cv.CAP_DSHOW for Windows)

def pick_backend(name: str | None) -> int | None:
    """
    Map a readable backend name to OpenCV's CAP_* constant.
    Returns None to use OpenCV default if unknown.
    """
    if not name:
        return None
    name = name.lower()
    m = {
        "avfoundation": cv.CAP_AVFOUNDATION,  # macOS
        "dshow":        cv.CAP_DSHOW,         # Windows
        "msmf":         cv.CAP_MSMF,          # Windows (alt)
        "v4l2":         cv.CAP_V4L2,          # Linux
        "gstreamer":    cv.CAP_GSTREAMER,
        "any":          None,                 # let OpenCV decide
        "auto":         -1,                   # we'll compute below
    }
    return m.get(name, None)

def auto_backend_for_os() -> int | None:
    sysname = platform.system()
    if sysname == "Darwin":   # macOS
        return cv.CAP_AVFOUNDATION
    elif sysname == "Windows":
        # On many Windows systems, DSHOW is the safest default
        return cv.CAP_DSHOW
    else:
        # Linux
        return cv.CAP_V4L2

# ---- CLI flags (optional, but handy) ----
parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
parser.add_argument("--backend", type=str, default="auto",
                    help="Camera backend: auto|avfoundation|dshow|msmf|v4l2|gstreamer|any")
parser.add_argument("--width",  type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--music",  type=str, default=str(ASSETS / "music.mp3"),
                    help="Path to mp3 for POINT celebration")
args, _ = parser.parse_known_args()

# Decide backend
backend = pick_backend(args.backend)
if backend == -1:  # "auto"
    backend = auto_backend_for_os()

# Open camera (with or without explicit backend)
if backend is None:
    cap = cv.VideoCapture(args.camera)
else:
    cap = cv.VideoCapture(args.camera, backend)

if not cap.isOpened():
    print("Error: Could not open webcam. Try a different index/backend.")
    raise SystemExit(1)

cap.set(cv.CAP_PROP_FRAME_WIDTH,  args.width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)


#(This is setting the resolution of the webcam to 1080 progressive scan (What the p stands for))
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)  # Set the width of the frames to 1920 pixels
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)  # Set the height of the frames to 1080 pixels

WIN = "Monkey" #This is what the window will be called
t_prev = time.perf_counter() # Previous time for FPS calculation


"""
Hand detection setup
"""
hands = mp_hands.Hands(  # Initialize the MediaPipe Hands solution this is needed to detect hands
    static_image_mode=False,  # Set to False for video input (not static images because thats stupid)
    max_num_hands=1,  # Maximum number of hands to detect (1 for single hand because we only need one hand to do gestures)
    min_detection_confidence=0.5,  # Minimum confidence for detection (50% confidence to consider a detection valid)
    min_tracking_confidence=0.5  # Minimum confidence for tracking (50% confidence to consider a tracking valid)
)

"""
Face mesh detection setup
"""
face = mp_face.FaceMesh( # Initialize the MediaPipe Face Mesh solution this is used to detect face mesh
    static_image_mode=False,  # Set to False for video input (because again thats stupid)
    max_num_faces=1,  # Maximum number of faces to detect (1 for single face one single monkey)
    min_detection_confidence=0.6,  # Minimum confidence for detection
    min_tracking_confidence=0.6  # Minimum confidence for tracking
)

"""
Main loop to continuously capture frames from the webcam and do all the processing
"""
while True:  # Start an infinite loop to continuously capture frames from the webcam
    ret, frame = cap.read() # Read a frame from the webcam (this returns a boolean ret which is True if the frame was read successfully and the actual frame itself)
    if not ret: # If the frame was not read successfully
        print("Error: Could not read frame.") # Print an error message and fix your webcam
        break # Break the loop

    frame = cv.flip(frame, 1) # Flip the frame horizontally for a mirror-like effect so it looks normal to us

    # --- MediaPipe input ---
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # Convert the frame from BGR to RGB color space because MediaPipe uses RGB format while OpenCV uses BGR by default
    out = hands.process(rgb) # Process the RGB frame to detect hands (using the hands object we created earlier)

    # --- Draw first hand (one hand only) ---
    if out.multi_hand_landmarks: # If any hand landmarks were detected a hand landmark is a point on the hand like fingertip or knuckle
        hl = out.multi_hand_landmarks[0] # Get the first detected hand landmarks (we only care about one hand)
        mp_draw.draw_landmarks( # Draw the hand landmarks on the frame
            frame, hl, mp_hands.HAND_CONNECTIONS, # Draw the connections between the landmarks
            mp_styles.get_default_hand_landmarks_style(), # Use default style for landmarks
            mp_styles.get_default_hand_connections_style() # Use default style for connections
        )

    # --- Face mesh + mouth center + head box ---
    face_out = face.process(rgb) # Process the RGB frame to detect face mesh (using the face object we created earlier)
     # If any face landmarks were detected a landmark is basically just a point on the face mesh
    if face_out.multi_face_landmarks: # If any face landmarks were detected
        h, w = frame.shape[:2] # Get the height and width of the frame which is done so we can convert normalized coordinates to pixel coordinates
        lms = face_out.multi_face_landmarks[0].landmark # Get the first detected face landmarks

        # mouth center from landmarks 13 & 14 (kept for THINK gesture)
        x13, y13 = int(lms[13].x * w), int(lms[13].y * h) # Get the pixel coordinates of landmark 13 (left corner of the mouth)
        x14, y14 = int(lms[14].x * w), int(lms[14].y * h) # Get the pixel coordinates of landmark 14 (right corner of the mouth)
        mouth_x = (x13 + x14) // 2 # Calculate the x-coordinate of the mouth center
        mouth_y = (y13 + y14) // 2 # Calculate the y-coordinate of the mouth center
        mouth_pt = (mouth_x, mouth_y) # Create a tuple for the mouth center point

        # compute a tight face bounding box from all mesh points
        xs = [int(p.x * w) for p in lms] # Get the x-coordinates of all landmarks
        ys = [int(p.y * h) for p in lms] # Get the y-coordinates of all landmarks
        x0, x1 = max(0, min(xs)), min(w - 1, max(xs)) # Calculate the left and right x-coordinates of the bounding box
        y0, y1 = max(0, min(ys)), min(h - 1, max(ys)) # Calculate the top and bottom y-coordinates of the bounding box

        # add a small margin so the box isn’t too tight
        side = max(x1 - x0, y1 - y0) # Calculate the side length of the bounding box
        m = int(0.10 * side)  # 10% padding # Calculate the margin (10% of the side length)
        x0 = max(0, x0 - m); y0 = max(0, y0 - m) # Adjust the top-left corner of the bounding box with margin
        x1 = min(w - 1, x1 + m); y1 = min(h - 1, y1 + m) # Adjust the bottom-right corner of the bounding box with margin

        # draw the monkey box
        cv.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2) # Draw the bounding box on the frame
        # Label the box as "monkey"
        cv.putText(frame, "monkey", (x0, max(0, y0 - 6)), # Put the text "monkey" above the bounding box
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) # Font settings for the text
        
        face_box = (x0, y0, x1, y1) # Store the face bounding box as a tuple


        # (optional) draw the mouth dot for debugging; comment out later
        # cv.circle(frame, (mouth_x, mouth_y), 5, (0, 255, 255), -1)

    else:
        mouth_pt = None
        face_box = None



    # --- Decide gesture from hand + mouth (one hand) ---
    label = "NEUTRAL" # Default label is NEUTRAL
    h, w = frame.shape[:2] # Get the height and width of the frame

    if out.multi_hand_landmarks: # If any hand landmarks were detected
        lm = out.multi_hand_landmarks[0].landmark # Get the first detected hand landmarks

        # True == finger up (compare tip vs PIP)
        TIPS = [4, 8, 12, 16, 20] # Indices of fingertip landmarks
        PIPS = [2, 6, 10, 14, 18] # Indices of PIP (proximal interphalangeal) landmarks A PIP joint is the middle joint of the finger
         # Check if each finger is up by comparing the y-coordinates of the tip and PIP landmarks
        fingers_up = [(lm[t].y < lm[p].y) for t, p in zip(TIPS, PIPS)] # List comprehension to create a list of boolean values indicating if each finger is up this basically just means if the y coordinate of the tip is less than the y coordinate of the PIP joint then the finger is up because in image coordinates y increases downwards
         # Unpack the list of boolean values into individual variables for each finger
        thumb_up, index_up, middle_up, ring_up, pinky_up = fingers_up # Unpack the list into individual finger variables which means thumb_up is fingers_up[0], index_up is fingers_up[1], and so on

        # POINT: index up, others (except thumb) down
        if index_up and not middle_up and not ring_up and not pinky_up: # If only the index finger is up and all other fingers (except thumb) are down
            label = "POINT" # Set label to POINT

        # THINK: index fingertip near the mouth
        if mouth_pt is not None: # If mouth point is detected
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)  # index tip
            diag = math.hypot(w, h) # Calculate the diagonal length of the frame (used for normalization)
             # Check if the index fingertip is near the mouth (within 6% of the frame diagonal)
            if math.hypot(ix - mouth_pt[0], iy - mouth_pt[1]) < 0.06 * diag: # If the distance between the index fingertip and mouth center is less than 6% of the frame diagonal
                label = "THINK" # Set label to THINK

    # --- Smooth the label over recent frames ---
    history.append(label) # Append the current label to the history deque
    stable = max(set(history), key=history.count) if history else "NEUTRAL" # Get the most frequent label in the history for stability

    # --- play once when we FIRST enter POINT ---
    if MUSIC_READY and not played_song and prev_stable != "POINT" and stable == "POINT": # If music is ready, song has not been played yet, previous stable label was not POINT, and current stable label is POINT
        #  play the song
        try: # Try to play the music
            pygame.mixer.music.play()  # non-blocking # Play the music file (non-blocking means it won't stop the program while playing)
             # Set played_song to True to indicate that the song has been played
            played_song = True # Mark that the song has been played
        except Exception as e: # If there is an error while trying to play the music
             # Print the error message
            print(f"[audio] play failed: {e}") # Print the error message
    # Reset the played_song flag when leaving POINT
    prev_stable = stable # Update the previous stable label to the current stable label
 

    # --- HUD: FPS bar ---
    # This could also be commented out if you want
    t_curr = time.perf_counter() # Current time for FPS calculation
    fps = 1.0 / max(1e-6, (t_curr - t_prev)) # Calculate FPS (frames per second) using the time difference between current and previous frame
    t_prev = t_curr # Update previous time to current time

    overlay = frame.copy() # Create a copy of the frame for overlay
    cv.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1) # Draw a semi-transparent rectangle at the top for HUD background
    frame = cv.addWeighted(overlay, 0.6, frame, 0.4, 0) # Blend the overlay with the original frame for transparency effect

    cv.putText(frame, f"FPS: {fps:.2f}", (10, 30), # Put the FPS text on the HUD
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # Font settings for the FPS text

    # --- Pick reaction image ---
    if   stable == "POINT" and img_middle is not None: # If the stable gesture is POINT and the middle image is loaded
        show = img_middle # Show the middle image
    elif stable == "THINK" and img_think  is not None: # If the stable gesture is THINK and the think image is loaded
        show = img_think # Show the think image
    else: # Default case 
        show = img_neutral # Show the neutral image

    # --- Overlay the monkey image on the main frame ---
    if show is not None: # If the show image is loaded 
        # Decide where to place it:
        if face_box is not None: # If face bounding box is detected
            x0, y0, x1, y1 = face_box # Unpack the face bounding box
            box_w = x1 - x0 # Calculate the width of the face bounding box
            box_h = y1 - y0 # Calculate the height of the face bounding box

            # size monkey to ~ the width of the head box (tweak 0.9 if you like)
            target_w = int(box_w * 0.9) # Set the target width of the monkey image to 90% of the face box width
            target_h = int(target_w * show.shape[0] / show.shape[1])  # keep monkey aspect ratio

            # place monkey just above the head box with a small margin
            margin = int(0.05 * box_h) # Calculate a small margin (5% of the face box height)
            ox = x0 + (box_w - target_w) // 2 # Center the monkey image horizontally above the face box
            oy = y0 - margin - target_h # Place the monkey image just above the face box with margin
            # if above goes offscreen, pin to top
            oy = max(0, oy) # Ensure the y-coordinate is not offscreen (not less than 0)
        else:
            # Fallback: bottom-right corner when no face detected
            target_w = 300 # Set a default target width for the monkey image
            target_h = int(target_w * show.shape[0] / show.shape[1]) #  keep monkey aspect ratio
            ox = frame.shape[1] - target_w - 20 # Set the x-coordinate to place the monkey image in the bottom-right corner with a margin
            oy = frame.shape[0] - target_h - 20 # Set the y-coordinate to place the monkey image in the bottom-right corner with a margin

        # If your monkey files are PNG with alpha, blending will honor transparency.
        # JPG/WebP will render opaque (still fine).
        overlay_bgra(frame, show, ox, oy, target_w, target_h) # Overlay the monkey image on the main frame at the calculated position with the target size
        #basically what this does is it puts the monkey image on top of the main frame at the calculated position with the target size



    # Gesture label on HUD
    cv.putText(frame, f"Gesture: {stable}", (160, 30), # Put the gesture label on the HUD
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # Font settings for the gesture label

    cv.imshow(WIN, frame) # Show the processed frame in the main window

    if (k := cv.waitKey(1) & 0xFF) in (ord('q'), 27): # Wait for 'q' or ESC key to exit
        break # Exit the loop





"""
The lines below this point are for cleanup and releasing resources after the main loop ends.
"""
if MUSIC_READY: # If music was initialized successfully
    pygame.mixer.music.stop() # Stop the music playback
    pygame.mixer.quit() # Quit the pygame mixer to release audio resources
cap.release()  # Release the webcam resource
cv.destroyAllWindows()  # Close all OpenCV windows
hands.close()  # Close the MediaPipe Hands solution
face.close()  # Close the MediaPipe Face Mesh solution