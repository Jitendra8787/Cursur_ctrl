import cv2
import mediapipe as mp
import pyautogui
import math

# Get screen size for mapping
screen_w, screen_h = pyautogui.size()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Click state to avoid repeated clicking
click_down = False

# Distance threshold for pinch (thumb + index) to click
PINCH_THRESHOLD = 40  # pixels (you can adjust)

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def main():
    global click_down

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # width
    cap.set(4, 480)  # height

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame.")
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert to RGB for mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        index_tip_pos = None
        thumb_tip_pos = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Index finger tip
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                index_tip_pos = (ix, iy)

                # Thumb tip
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                thumb_tip_pos = (tx, ty)

                # Draw circles
                cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)
                cv2.circle(frame, (tx, ty), 10, (0, 255, 255), -1)
                cv2.line(frame, (ix, iy), (tx, ty), (255, 0, 0), 2)

                # Map index finger position to screen coordinates
                # NOTE: we use frame width/height -> screen width/height
                mouse_x = int(ix / w * screen_w)
                mouse_y = int(iy / h * screen_h)

                # Move the mouse
                pyautogui.moveTo(mouse_x, mouse_y)

                # Detect pinch for click
                if index_tip_pos and thumb_tip_pos:
                    dist = euclidean_distance(index_tip_pos, thumb_tip_pos)

                    cv2.putText(
                        frame,
                        f"Dist: {int(dist)}",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # If fingers are close -> mouse down (click & hold)
                    if dist < PINCH_THRESHOLD and not click_down:
                        pyautogui.mouseDown()
                        click_down = True
                    # If fingers are apart -> mouse up (release)
                    elif dist >= PINCH_THRESHOLD and click_down:
                        pyautogui.mouseUp()
                        click_down = False

                break  # only first hand

        else:
            # No hand -> ensure mouse is not stuck in click
            if click_down:
                pyautogui.mouseUp()
                click_down = False

        # Info Text
        cv2.putText(
            frame,
            "Move index finger = move mouse",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Pinch index + thumb = click (hold)",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Hand Mouse Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
