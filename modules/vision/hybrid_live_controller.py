import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import joblib

# Load models
cnn_model = tf.keras.models.load_model("models/asl_cnn_model.h5")
landmark_model = joblib.load("models/landmark_model.pkl")

classes = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z','del','nothing','space'
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

IMG_SIZE = 64
mode = 2

print("Press 1: CNN Mode")
print("Press 2: Landmark Mode")
print("Press 3: Hybrid Mode")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    # ROI box
    x1, y1, x2, y2 = 100, 50, 500, 400
    roi = frame[y1:y2, x1:x2]

    # Draw box
    cv2.rectangle(display_frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # ---------- CNN ----------
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi_resized = roi_resized / 255.0
    roi_resized = np.reshape(roi_resized, (1, IMG_SIZE, IMG_SIZE, 3))

    cnn_pred = cnn_model.predict(roi_resized, verbose=0)
    cnn_conf = float(np.max(cnn_pred))
    cnn_label = classes[np.argmax(cnn_pred)]

    # ---------- LANDMARK (ONLY FROM ROI) ----------
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_roi)

    landmark_label = ""
    landmark_conf = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Draw only in landmark/hybrid mode
            if mode in [2, 3]:
                mp_draw.draw_landmarks(
                    display_frame[y1:y2, x1:x2],
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks).reshape(1, -1)

            probs = landmark_model.predict_proba(landmarks)[0]
            landmark_conf = float(np.max(probs))
            landmark_label = landmark_model.classes_[np.argmax(probs)]

    # ---------- MODE LOGIC ----------
    if mode == 1:
        final_label = cnn_label
        mode_text = "CNN Mode"

    elif mode == 2:
        final_label = landmark_label if landmark_label else "No Hand"
        mode_text = "Landmark Mode"


    elif mode == 3:

        mode_text = "Hybrid Mode"

        if landmark_label == "":

            final_label = cnn_label

        elif cnn_label == landmark_label:

            final_label = cnn_label

        else:

            final_label = landmark_label

    # ---------- DISPLAY ----------
    cv2.putText(display_frame, f"Mode: {mode_text}",
                (50,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.putText(display_frame, f"Prediction: {final_label}",
                (50,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Show confidence ONLY in Hybrid mode
    if mode == 3:
        cv2.putText(display_frame, f"CNN: {cnn_label} ({cnn_conf:.2f})",
                    (50,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(display_frame, f"Landmark: {landmark_label} ({landmark_conf:.2f})",
                    (50,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(display_frame, "Press 1:CNN  2:Landmark  3:Hybrid  Q:Quit",
                (50,450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Hybrid Sign Recognition", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        mode = 1
    elif key == ord('2'):
        mode = 2
    elif key == ord('3'):
        mode = 3
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
