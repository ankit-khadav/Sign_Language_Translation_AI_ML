import os
import cv2
import mediapipe as mp
import csv

# Dataset path
DATASET_PATH = "data/dataset/asl_alphabet_train"
OUTPUT_CSV = "data/dataset/landmarks_dataset.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

data = []

print("Starting landmark extraction...")

for label in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder_path):
        continue

    print(f"Processing: {label}")

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            row = [label]

            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            data.append(row)

print("Saving CSV...")

with open(OUTPUT_CSV, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f"Done! Landmarks saved to {OUTPUT_CSV}")
