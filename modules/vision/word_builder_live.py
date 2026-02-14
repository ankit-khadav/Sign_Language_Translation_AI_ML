import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from modules.speech.speak_word import speak

# Load landmark model
model = joblib.load("models/landmark_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Smoothing
pred_queue = deque(maxlen=15)

# Word builder variables
current_word = ""
stable_label = ""
stable_count = 0
STABLE_THRESHOLD = 25   # ~3 sec delay
letter_added = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)
    label = "Detecting..."

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks).reshape(1, -1)

            prediction = model.predict(landmarks)[0]
            pred_queue.append(prediction)

            if len(pred_queue) == 15:
                label = Counter(pred_queue).most_common(1)[0][0]

    # Stability logic
    if label == stable_label:
        stable_count += 1
    else:
        stable_label = label
        stable_count = 0
        letter_added = False

    # Add letter after stable hold
    if stable_count == STABLE_THRESHOLD and not letter_added:

        if stable_label == "space":
            current_word += " "

        elif stable_label == "del":
            current_word = current_word[:-1]

        elif stable_label not in ["nothing", "Detecting..."]:
            current_word += stable_label

        letter_added = True

    # Display prediction
    cv2.putText(frame, f"Sign: {label}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    # Display word
    cv2.putText(frame, f"Word: {current_word}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2)

    cv2.putText(frame, "Hold sign 3 sec | Press S to Speak",
                (50, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,255,255), 2)

    cv2.imshow("Landmark Word Builder", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if current_word.strip() != "":
            speak(current_word)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
