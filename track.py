import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

pTime = time.time()
frame_skip = 2
frame_count = 0
last_results = None  # cached for redraw on skipped frames
fps = 0

while True:
    success, img = cap.read()
    frame_count += 1
    if not success:
        break

    if frame_count % frame_skip == 0:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        last_results = hands.process(imgRGB)

        cTime = time.time()
        elapsed = cTime - pTime
        if elapsed > 0:  # avoid division by zero
            fps = 1 / elapsed
        pTime = cTime

    if last_results and last_results.multi_hand_landmarks:
        for handLms in last_results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
