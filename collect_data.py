import cv2
import mediapipe as mp
import pyttsx3
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Speed of speech

# Gesture Mapping (Finger Patterns & Sentences)
GESTURE_LABELS = {
    "HELLO": ([1, 1, 1, 1, 1], "Hello! How are you?"),
    "BABY": ([1, 0, 0, 0, 0], "This is a baby."),
    "LOVE": ([1, 0, 0, 0, 1], "I love you."),
    "BATHROOM": ([0, 0, 0, 0, 1], "Where is the bathroom?"),
    "FRIEND": ([1, 1, 0, 0, 0], "You are my friend."),
    "FATHER": ([1, 1, 1, 0, 0], "This is my father."),
    "MOTHER": ([0, 1, 1, 0, 0], "This is my mother."),
    "SCHOOL": ([0, 0, 1, 1, 1], "I am going to school."),
    "YES": ([0, 1, 0, 0, 0], "Yes, that's correct."),
    "NO": ([1, 0, 0, 0, 0], "No, that’s wrong."),
    "PLEASE": ([1, 1, 1, 1, 1], "Please help me."),
    "THANK YOU": ([1, 1, 1, 1, 1], "Thank you so much!")
}

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

def classify_gesture(fingers):
    """ Match detected finger positions to predefined gestures """
    for gesture, (pattern, sentence) in GESTURE_LABELS.items():
        if fingers == pattern:
            return gesture, sentence
    return "UNKNOWN", ""

last_detected = ""
last_spoken_time = 0  # Track time for speech delay
frame_skip = 2  # Skip every alternate frame to reduce load
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frame processing to improve speed

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark points
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

            # Finger state detection (1 = Up, 0 = Down)
            fingers = []
            for tip, base in [(8, 6), (12, 10), (16, 14), (20, 18)]:  # Index, Middle, Ring, Pinky
                fingers.append(1 if landmarks[tip][1] < landmarks[base][1] else 0)
            
            thumb_up = landmarks[4][0] > landmarks[3][0]  # Thumb detection
            fingers.insert(0, int(thumb_up))

            # Detect gesture
            detected_gesture, detected_sentence = classify_gesture(fingers)

            # Speak detected sentence only if it's new and enough time has passed
            if detected_gesture != "UNKNOWN" and detected_gesture != last_detected and time.time() - last_spoken_time > 2:
                print(f"Detected: {detected_gesture} - {detected_sentence}")
                engine.say(detected_sentence)
                engine.runAndWait()
                last_detected = detected_gesture
                last_spoken_time = time.time()  # Update last spoken time

            # Display result
            cv2.putText(frame, detected_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-Time Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
