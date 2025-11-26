import cv2
import mediapipe as mp
import math
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Hands configuration
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# For storing wrist positions to smooth movement
positions = deque(maxlen=5)  # store last 5 positions (x, y)

def get_direction(prev_point, current_point, move_threshold=20):
    """
    Decide movement direction based on change in x, y.
    move_threshold: minimum pixel change to count as movement
    """
    if prev_point is None or current_point is None:
        return "NO HAND"

    px, py = prev_point
    cx, cy = current_point

    dx = cx - px
    dy = cy - py

    # Calculate distance moved
    distance = math.sqrt(dx**2 + dy**2)
    if distance < move_threshold:
        return "STILL"

    # Horizontal movement
    if abs(dx) > abs(dy):
        if dx > 0:
            return "RIGHT"
        else:
            return "LEFT"
    else:
        if dy > 0:
            return "DOWN"
        else:
            return "UP"


def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_center = None
    direction_text = "NO HAND"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip the frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find hands
        result = hands.process(rgb_frame)

        current_center = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks on the hand
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Get WRIST landmark
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                cx = int(wrist.x * w)
                cy = int(wrist.y * h)
                current_center = (cx, cy)

                # Draw a circle at wrist
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

                # Save position for path effect
                positions.append(current_center)

                # Draw path (trail) of wrist
                for i in range(1, len(positions)):
                    if positions[i - 1] is None or positions[i] is None:
                        continue
                    cv2.line(frame, positions[i - 1], positions[i], (255, 0, 0), 2)

                break  # Only first hand considered

        # Decide direction based on previous and current center
        direction_text = get_direction(prev_center, current_center)
        prev_center = current_center

        # Show direction text on screen
        cv2.putText(frame, f"Direction: {direction_text}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, "Press 'q' to Quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # Show result
        cv2.imshow("Hand Movement Detection", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
