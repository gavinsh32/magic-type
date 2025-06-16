# label.py
# Gavin Haynes
# June 14, 2024
# Create a labeled dataset for hand typing. Whenever a key is pressed, the current hand positions are saved.

import cv2 as cv
import mediapipe as mp
import json
import time
import numpy as np

DATA_COLLECTION_DELAY_MS = 1000 // 30   # 30 FPS

ALLOWED_LEFT_HAND_CHARS = set(['q', 'w', 'e', 'r', 't', 'a', 's', 'd', 'f', 'g', 'z', 'x', 'c', 'v', 'b'])
ALLOWED_RIGHT_HAND_CHARS = set(['y', 'u', 'i', 'o', 'p', 'h', 'j', 'k', 'l', 'm', 'n'])

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv.VideoCapture(0)  # Open the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

hands = mp_hands.Hands(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

last_key_pressed_display = None # For displaying on screen
collected_data = []
data_id_counter = 0

while cap.isOpened():
    current_key_for_data = None # Key to be stored for this frame's data point
    
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Convert the BGR image to RGB
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)
    
    current_frame_left_points_np = None
    current_frame_right_points_np = None

    # Draw hand landmarks and connections
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Convert landmarks to NumPy array [21, 3]
            points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            
            hand_label = handedness_info.classification[0].label # 'Left' or 'Right'
            if hand_label == "Left":
                current_frame_left_points_np = points
            elif hand_label == "Right":
                current_frame_right_points_np = points

            # Draw the landmarks on the frame for visualization
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    # Display the last key pressed on the frame
    if last_key_pressed_display:
        cv.putText(frame, f"Last Key: {last_key_pressed_display}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    # Display the frame
    cv.imshow('Hand Tracking', frame)

    # Read keyboard input using cv.waitKey
    key_code = cv.waitKey(DATA_COLLECTION_DELAY_MS) & 0xFF
    
    # current_key_for_data is already None. It will be updated if an allowed key is pressed.
    if key_code != 0xFF and key_code != 27:  # A key was pressed (and not ESC)
        char_candidate = chr(key_code)
        # Check if the pressed key is in either of the allowed lists
        if char_candidate in ALLOWED_LEFT_HAND_CHARS or char_candidate in ALLOWED_RIGHT_HAND_CHARS:
            current_key_for_data = char_candidate
            last_key_pressed_display = char_candidate # Update display only for allowed keys
    # If no key was pressed (key_code == 0xFF) or an unallowed key was pressed,
    # current_key_for_data remains None.
    # last_key_pressed_display persists the last *allowed* key.

    # Prepare data point for this frame
    # Convert numpy arrays to lists for JSON serialization
    left_points_for_json = current_frame_left_points_np.tolist() if current_frame_left_points_np is not None else None
    right_points_for_json = current_frame_right_points_np.tolist() if current_frame_right_points_np is not None else None

    data_point = {
        "id": data_id_counter,
        "key": current_key_for_data,
        "left_points": left_points_for_json,
        "right_points": right_points_for_json
    }

    if current_key_for_data is not None:
        print(data_id_counter, current_key_for_data)

    collected_data.append(data_point)
    data_id_counter += 1
    
    if key_code == 27:  # Exit on 'ESC' key
        break

hands.close()
cap.release()
cv.destroyAllWindows()

# Save the collected data to a JSON file
output_filename = "hand_gesture_data.json"
with open(output_filename, 'w') as f:
    json.dump(collected_data, f, indent=4)
print(f"Data collection complete. {len(collected_data)} frames saved to {output_filename}")