import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# --- Configuration ---
# Set the desired camera width and height
W_CAM, H_CAM = 640, 480

# Define the area on the screen the hand movement will control
FRAME_REDUCTION = 100
SMOOTHING_FACTOR = 3 # Increased responsiveness by lowering smoothing
CLICK_COOLDOWN = 0.2 # Very short cooldown, as the state machine handles repetition

# --- Click Detection Thresholds (Highly Precise) ---
# 1. Pinch Threshold (Tip-to-Thumb distance): Reduced for a very deliberate pinch
PINCH_THRESHOLD = 20 
# 2. Curl Confirmation Threshold (Tip-to-DIP distance): Must be small to confirm the finger is bent
DIP_THRESHOLD = 30 
# 3. Verticality check offset (normalized y-difference)
Y_DIFF_THRESHOLD = 0.02 

# --- Click State Machine ---
# States: 0 = READY, 1 = PINCHED (Click performed, waiting for release)
LEFT_CLICK_STATE = 0 
RIGHT_CLICK_STATE = 0 

# --- PyAutoGUI Setup ---
# Get screen resolution
W_SCREEN, H_SCREEN = pyautogui.size()

# Set up the hand tracking model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
    model_complexity=0 
)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W_CAM)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H_CAM)

# Variables for smoothing and click delay
previous_x, previous_y = 0, 0
current_x, current_y = 0, 0
last_click_time = time.time()


print("Virtual Mouse Controller Started. Press 'q' to exit.")

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two MediaPipe landmarks in pixels."""
    x1, y1 = p1.x * W_CAM, p1.y * H_CAM
    x2, y2 = p2.x * W_CAM, p2.y * H_CAM
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def handle_click_state(is_pinched, click_type):
    """Manages the click state machine (READY -> PINCHED) for single click execution."""
    global LEFT_CLICK_STATE, RIGHT_CLICK_STATE, last_click_time

    if click_type == 'LEFT':
        state = LEFT_CLICK_STATE
    else:
        state = RIGHT_CLICK_STATE

    # Check for click transition (READY to PINCHED)
    if is_pinched and state == 0 and (time.time() - last_click_time > CLICK_COOLDOWN):
        if click_type == 'LEFT':
            pyautogui.click()
            LEFT_CLICK_STATE = 1
            print(f"Action: {click_type} Click")
        else: # RIGHT CLICK
            pyautogui.rightClick()
            RIGHT_CLICK_STATE = 1
            print(f"Action: {click_type} Click")
        last_click_time = time.time()
        
    # Check for release transition (PINCHED to READY)
    elif not is_pinched and state == 1:
        if click_type == 'LEFT':
            LEFT_CLICK_STATE = 0
        else:
            RIGHT_CLICK_STATE = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from webcam.")
        break

    img = cv2.flip(img, 1)

    # Use explicit BGR2RGB for efficiency and set image to be non-writable for speed
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False
    results = hands.process(img_rgb)
    img_rgb.flags.writeable = True

    # Initialize click states for visualization (Assume not clicked unless proven otherwise)
    is_left_clicked = False
    is_right_clicked = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get necessary landmarks
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            # Index Finger landmarks
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP] 

            # Middle Finger landmarks
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP] 

            # Convert normalized coordinates of Index Tip to pixel coordinates
            x1 = int(index_tip.x * W_CAM)
            y1 = int(index_tip.y * H_CAM)

            # --- 1. Movement: Map hand position to screen coordinates ---
            cv2.rectangle(img, (FRAME_REDUCTION, FRAME_REDUCTION),
                          (W_CAM - FRAME_REDUCTION, H_CAM - FRAME_REDUCTION),
                          (255, 0, 255), 2)

            map_x = np.interp(x1, (FRAME_REDUCTION, W_CAM - FRAME_REDUCTION), (0, W_SCREEN))
            map_y = np.interp(y1, (FRAME_REDUCTION, H_CAM - FRAME_REDUCTION), (0, H_SCREEN))

            # Smooth the movement
            current_x = previous_x + (map_x - previous_x) / SMOOTHING_FACTOR
            current_y = previous_y + (map_y - previous_y) / SMOOTHING_FACTOR

            pyautogui.moveTo(current_x, current_y, _pause=False)
            previous_x, previous_y = current_x, current_y

            # Draw a circle on the index finger to show the cursor point (default color)
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)


            # --- 2. Left Click Gesture Check ---
            left_pinch_dist = calculate_distance(index_tip, thumb_tip)
            index_curl_dist = calculate_distance(index_tip, index_dip)
            # Index Tip (y) must be *above* the Index DIP (knuckle) (or very close)
            index_is_curled_vertically = index_tip.y - index_dip.y < Y_DIFF_THRESHOLD
            
            is_left_pinched = (left_pinch_dist < PINCH_THRESHOLD and 
                               index_curl_dist < DIP_THRESHOLD and
                               index_is_curled_vertically)
            
            handle_click_state(is_left_pinched, 'LEFT')
            
            # --- 3. Right Click Gesture Check ---
            right_pinch_dist = calculate_distance(middle_tip, thumb_tip)
            middle_curl_dist = calculate_distance(middle_tip, middle_dip)
            # Middle Tip (y) must be *above* the Middle DIP (knuckle)
            middle_is_curled_vertically = middle_tip.y - middle_dip.y < Y_DIFF_THRESHOLD
            
            is_right_pinched = (right_pinch_dist < PINCH_THRESHOLD and 
                                middle_curl_dist < DIP_THRESHOLD and
                                middle_is_curled_vertically)
                                
            handle_click_state(is_right_pinched, 'RIGHT')
            
            # --- 4. Visualization Update ---
            if LEFT_CLICK_STATE == 1:
                cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED) # Green for Left Click
            
            # If the index finger is not controlling the cursor (e.g., performing a right click gesture), 
            # we need to draw the middle finger tip to confirm the gesture point.
            if RIGHT_CLICK_STATE == 1:
                 mid_x = int(middle_tip.x * W_CAM)
                 mid_y = int(middle_tip.y * H_CAM)
                 cv2.circle(img, (mid_x, mid_y), 10, (0, 0, 255), cv2.FILLED) # Red for Right Click

    
    # Display the resulting frame
    cv2.imshow("Virtual Mouse Tracker", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
