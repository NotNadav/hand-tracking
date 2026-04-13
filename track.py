import cv2
import mediapipe as mp
import time

# Initialize video capture with lower resolution
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

pTime = time.time()  # Fix bug 1: init to now, not 0, avoids misleading FPS on first frame

# Process only every nth frame for better performance
frame_skip = 2
frame_count = 0
last_results = None  # Fix bug 2: cache results to redraw on skipped frames
fps = 0             # Fix bug 3: track FPS separately, only update on processed frames

while True:
    success, img = cap.read()
    frame_count += 1
    if not success:
        break

    # Only process every nth frame
    if frame_count % frame_skip == 0:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        last_results = hands.process(imgRGB)

        # Fix bug 3: update FPS only when actually processing frames
        cTime = time.time()
        elapsed = cTime - pTime
        if elapsed > 0:  # Fix bug 1: guard against zero division
            fps = 1 / elapsed
        pTime = cTime

    # Fix bug 2: always draw from cached results so landmarks don't flicker on skipped frames
    if last_results and last_results.multi_hand_landmarks:
        for handLms in last_results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Display the video feed
    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
