import cv2
import mediapipe as mp
import pyautogui
import math

hand_detector = mp.solutions.hands.Hands()
hand_drawer = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

index_x = index_y = thumb_x = thumb_y = 0

while True:
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_result = hand_detector.process(rgb_frame)
    detected_hands = detection_result.multi_hand_landmarks

    if detected_hands:
        for single_hand in detected_hands:
            hand_drawer.draw_landmarks(frame, single_hand)
            all_landmarks = single_hand.landmark
            for landmark_id, landmark_point in enumerate(all_landmarks):
                pixel_x = int(landmark_point.x * frame_width)
                pixel_y = int(landmark_point.y * frame_height)

                if landmark_id == 8:  # Index fingertip
                    cv2.circle(img=frame, center=(pixel_x, pixel_y), radius=8, color=(255, 0, 0), thickness=-1)
                    index_x = pixel_x
                    index_y = pixel_y

                if landmark_id == 4:  # Thumb tip
                    cv2.circle(img=frame, center=(pixel_x, pixel_y), radius=8, color=(0, 255, 0), thickness=-1)
                    thumb_x = pixel_x
                    thumb_y = pixel_y

            if index_x and thumb_x and index_y and thumb_y:
                distance = math.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)
                if distance > 60:
                    pyautogui.press("volumeup")
                else:
                    pyautogui.press("volumedown")
                cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), (0, 0, 255), 4)

    cv2.imshow("Hand Volume Control", frame)
    key_pressed = cv2.waitKey(10)
    if key_pressed == 27:  # ESC key
        break

camera.release()
cv2.destroyAllWindows()
