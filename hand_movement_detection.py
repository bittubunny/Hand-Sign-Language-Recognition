import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize hand tracking
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize the video capture
cap = cv2.VideoCapture(0)

data = []
labels = []

# Initialize label
current_label = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image color to RGB (MediaPipe requires RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(rgb_frame)

    # Draw hand landmarks and save the data
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark positions
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            # Convert to numpy array and flatten
            landmark_array = np.array(landmarks).flatten()

            if landmark_array.shape[0] == 63:
                data.append(landmark_array)
                labels.append(current_label)

    # Display the resulting frame
    cv2.imshow('Hand Tracking', frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('z'):  # Press 'z' to move to the next label
        current_label += 1
        print(f"Current label: {current_label}")
    elif key == ord('a'):  # Press 'a' to print the current label
        print(f"Current label: {current_label}")

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save the data to a CSV file
data_df = pd.DataFrame(data)
data_df['label'] = labels
data_df.to_csv('hand_landmarks_data.csv', index=False)
