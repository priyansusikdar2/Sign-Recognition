import cv2
import numpy as np
import mediapipe as mp
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Finger tip IDs
tip_ids = [4, 8, 12, 16, 20]

# Get angle between three points
def angle_between(p1, p2, p3):
    a = np.array([p1.x - p2.x, p1.y - p2.y])
    b = np.array([p3.x - p2.x, p3.y - p2.y])
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

# Get finger states
def get_finger_states(hand_landmarks):
    fingers = []

    # Thumb
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for i in range(1, 5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# Classify gestures using rules
def classify_gesture(fingers, landmarks):
    if fingers == [0, 1, 1, 0, 0]:
        return "V (Peace)"
    elif fingers == [0, 1, 0, 0, 0]:
        return "1"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Hand"
    elif fingers == [0, 0, 0, 0, 0]:
        return "Fist (A)"
    elif fingers == [1, 0, 0, 0, 1]:
        return "Y (Shaka)"
    elif fingers == [0, 1, 1, 1, 1]:
        angle = angle_between(landmarks.landmark[5], landmarks.landmark[9], landmarks.landmark[13])
        if 40 < angle < 60:
            return "B"
    elif fingers == [1, 1, 0, 0, 0]:
        return "L"
    elif fingers == [0, 1, 1, 0, 1]:
        return "W"
    return "Unknown"

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = get_finger_states(hand_landmarks)
            gesture = classify_gesture(fingers, hand_landmarks)

            # Draw Gesture
            cv2.putText(frame, f'Gesture: {gesture}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Draw Finger Status
            for i, f in enumerate(['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']):
                cv2.putText(frame, f'{f}: {"Up" if fingers[i] else "Down"}', (10, 70 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
