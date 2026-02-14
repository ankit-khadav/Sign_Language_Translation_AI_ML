import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter

# Load trained model
model = tf.keras.models.load_model("models/asl_cnn_model.h5")

# Class labels
classes = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'del','nothing','space'
]

# Start camera
cap = cv2.VideoCapture(0)

IMG_SIZE = 64

# Stronger smoothing queue
pred_queue = deque(maxlen=25)

CONFIDENCE_THRESHOLD = 0.80

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Bigger ROI box
    x1, y1, x2, y2 = 100, 50, 500, 400
    roi = frame[y1:y2, x1:x2]

    # Slight blur to reduce noise
    roi = cv2.GaussianBlur(roi, (5,5), 0)

    # Resize for model
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    # Prediction
    pred = model.predict(img, verbose=0)
    class_index = np.argmax(pred)
    confidence = np.max(pred)
    raw_label = classes[class_index]

    # Confidence filtering
    if confidence > CONFIDENCE_THRESHOLD:
        pred_queue.append(raw_label)

    # Get stable prediction
    if len(pred_queue) == 25:
        label = Counter(pred_queue).most_common(1)[0][0]
    else:
        label = "Detecting..."

    # Draw box
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # Show prediction
    cv2.putText(frame, f"Sign: {label}",
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,0,255), 2)

    # Show confidence
    cv2.putText(frame, f"Confidence: {confidence:.2f}",
                (50,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255,0,0), 2)

    # User guide
    cv2.putText(frame, "Keep hand steady inside box",
                (50,450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,255,255), 2)

    cv2.imshow("ASL Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
