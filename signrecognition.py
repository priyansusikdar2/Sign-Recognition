import cv2
import mediapipe as mp
import pyttsx3
import time

# ========== Text-to-Speech ==========
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ========== MediaPipe Initialization ==========
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ========== Finger State Extraction ==========
def get_finger_states(hand_landmarks):
    finger_states = []
    tip_ids = [4, 8, 12, 16, 20]

    # Thumb: x-coordinates
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        finger_states.append(1)
    else:
        finger_states.append(0)

    # Other fingers: y-coordinates
    for i in range(1, 5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
            finger_states.append(1)
        else:
            finger_states.append(0)

    return finger_states

# ========== Gesture to Letter Mapping ==========
gesture_map = {
    (0, 1, 0, 0, 0): 'D',
    (0, 1, 1, 0, 0): 'U',
    (0, 1, 1, 1, 1): 'B',
    (0, 0, 0, 0, 0): 'A',
    (1, 1, 1, 1, 1): '5',
    (0, 1, 0, 0, 1): 'L',
    (1, 0, 0, 0, 0): 'Thumbs Up',
     (0, 1, 0, 0, 0): 'D',
    (0, 1, 1, 0, 0): 'U',
    (0, 1, 1, 1, 1): 'B',
    (1, 1, 0, 0, 0): 'R',
    (1, 1, 1, 0, 0): 'W',
    (0, 1, 0, 0, 1): 'L',

}

def classify_gesture(states):
    return gesture_map.get(tuple(states))

# ========== Main Loop ==========
cap = cv2.VideoCapture(0)
prev_letter = ""
prev_time = 0
phrase = ""
last_detected_time = 0
stability_threshold = 1.2  # seconds
stable_letter = None

log_file = open("sign_output_log.txt", "w")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_states = get_finger_states(hand_landmarks)
            current_letter = classify_gesture(finger_states)

            if current_letter:
                current_time = time.time()
                if current_letter == stable_letter:
                    if current_time - last_detected_time > stability_threshold:
                        if current_letter != prev_letter:
                            phrase += current_letter + " "
                            speak(current_letter)
                            log_file.write(f"{time.strftime('%H:%M:%S')} - {current_letter}\n")
                            prev_letter = current_letter
                        last_detected_time = current_time
                else:
                    stable_letter = current_letter
                    last_detected_time = current_time

                # Show detected gesture
                cv2.putText(frame, f"Detected: {current_letter}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    # Show phrase buffer
    cv2.putText(frame, f"Phrase: {phrase.strip()}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Sign Language Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to quit
        break
    elif key == ord('c'):
        phrase = ""
        speak("Cleared")
        prev_letter = ""
        stable_letter = None

cap.release()
cv2.destroyAllWindows()
log_file.close()
