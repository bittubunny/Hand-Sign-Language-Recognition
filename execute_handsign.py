import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import pandas as pd

# Load the model
model = load_model('hand_sign_model.h5')

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize hand tracking
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Load label binarizer
data = pd.read_csv('hand_landmarks_data.csv')
lb = LabelBinarizer()
lb.fit(data['label'].values)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image color to RGB (MediaPipe requires RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(rgb_frame)

    # Draw hand landmarks and predict hand sign
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark positions
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            # Convert to numpy array and reshape
            landmark_array = np.array(landmarks).flatten()
            
            # Assuming landmarks are normalized, adjust if needed
            if landmark_array.shape[0] == 63:
                # Make prediction
                prediction = model.predict(landmark_array.reshape(1, -1))
                predicted_label = lb.inverse_transform(prediction)[0]
                cv2.putText(frame, f'Sign: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Hand Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
