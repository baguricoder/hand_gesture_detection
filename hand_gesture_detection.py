import cv2
import mediapipe as mp

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for natural movement
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Finger tip landmarks (index, middle, ring, pinky, thumb)
            finger_tips = [8, 12, 16, 20]
            thumb_tip = 4
            count = 0

            # Get landmark positions
            landmarks = hand_landmarks.landmark

            # Check if fingers are extended
            for tip in finger_tips:
                if landmarks[tip].y < landmarks[tip - 2].y:
                    count += 1

            # Thumb detection
            if landmarks[thumb_tip].x < landmarks[thumb_tip - 1].x:
                count += 1

            # Determine Even or Odd
            result_text = f"Number: {count} - {'Even' if count % 2 == 0 else 'Odd'}"

            # Display text on screen
            cv2.putText(frame, result_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the video feed
    cv2.imshow("Hand Gesture Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release
cv2.destroyAllWindows()