import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
from collections import deque
from tensorflow.keras.models import load_model

# Load trained LSTM model
MODEL_PATH = "best_lstm_model.keras"
lstm_model = load_model(MODEL_PATH)

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Gesture label mapping
gesture_labels = {
    0: "Call", 1: "Dislike", 2: "Fist", 3: "Four", 4: "Like",
    5: "Mute", 6: "OK", 7: "One", 8: "Palm", 9: "Peace",
    10: "Peace Inverted", 11: "Rock", 12: "Stop", 13: "Stop Inverted",
    14: "Three", 15: "Two Up", 16: "Two Up Inverted"
}

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)  # Adjust speech speed
tts_engine.setProperty("volume", 1.0)  # Set volume

# Set up real-time gesture sequence tracking
SEQUENCE_LENGTH = 30
frame_sequence = deque(maxlen=SEQUENCE_LENGTH)

cap = cv2.VideoCapture(0)
previous_label = None  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe processing
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    # Extract landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract 21 landmark points (x, y, z)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            
            if len(landmarks) == 63:  # Ensure correct shape
                frame_sequence.append(landmarks)  # Append to sequence

            # If we have collected enough frames, make a prediction
            if len(frame_sequence) == SEQUENCE_LENGTH:
                lstm_input = np.expand_dims(np.array(frame_sequence), axis=0)  # Add batch dimension
                lstm_prediction = lstm_model.predict(lstm_input)
                sequence_label_index = np.argmax(lstm_prediction)

                # Get recognized gesture label
                predicted_text = gesture_labels.get(sequence_label_index, "Unknown")
                
                # Display gesture on video
                cv2.putText(frame, f"Gesture: {predicted_text}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Speak out the detected gesture (if different from previous)
                if predicted_text != previous_label:
                    tts_engine.say(predicted_text)
                    tts_engine.runAndWait()
                    previous_label = predicted_text  

    # Show webcam output with subtitles
    cv2.imshow("Real-Time Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
