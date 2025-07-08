import os
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Input and output paths
IMAGE_DATASET_PATH = "D:/gesture_recognition_project/dataset"  # Your image dataset
LANDMARK_DATASET_PATH = "D:/gesture_recognition_project/dataset_landmarks"  # New folder for landmarks

# Create the landmark dataset directory
os.makedirs(LANDMARK_DATASET_PATH, exist_ok=True)

# Process each class (gesture type)
for class_name in os.listdir(IMAGE_DATASET_PATH):
    class_path = os.path.join(IMAGE_DATASET_PATH, class_name)
    landmark_class_path = os.path.join(LANDMARK_DATASET_PATH, class_name)
    os.makedirs(landmark_class_path, exist_ok=True)

    # Process each image
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Extract (x, y, z) for all 21 landmarks
                landmark_vector = []
                for lm in hand_landmarks.landmark:
                    landmark_vector.extend([lm.x, lm.y, lm.z])

                # Save as .npy file
                landmark_file = os.path.join(landmark_class_path, image_name.replace(".jpg", ".npy"))  
                np.save(landmark_file, np.array(landmark_vector, dtype=np.float32))

print("Landmark dataset created successfully!")
SEQUENCE_LENGTH = 30
# Step 1: Convert Landmark Frames to Sequences
def create_landmark_sequences():
    """Convert individual landmark frames into sequences of 30 frames each."""
    for class_name in sorted(os.listdir(LANDMARK_DATASET_PATH)):
        class_path = os.path.join(LANDMARK_DATASET_PATH, class_name)
        sequence_class_path = os.path.join(SEQUENCE_DATASET_PATH, class_name)
        os.makedirs(sequence_class_path, exist_ok=True)

        landmark_files = sorted(os.listdir(class_path))
        sequence = []

        for landmark_file in landmark_files:
            landmark_path = os.path.join(class_path, landmark_file)
            landmarks = np.load(landmark_path)  # Shape: (63,)
            sequence.append(landmarks)

            if len(sequence) == SEQUENCE_LENGTH:
                seq_filename = os.path.join(sequence_class_path, f"seq_{len(os.listdir(sequence_class_path)) + 1}.npy")
                np.save(seq_filename, np.array(sequence, dtype=np.float32))  
                sequence = []  # Reset sequence

create_landmark_sequences()
print("Landmark sequences created successfully!")

