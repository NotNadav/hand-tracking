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

pTime = 0

# Process only every nth frame for better performance
frame_skip = 2
frame_count = 0

while True:
    success, img = cap.read()
    frame_count += 1
    if not success:
        break

    # Only process every nth frame
    if frame_count % frame_skip == 0:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        # Draw landmarks if hand is detected
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    # Calculate FPS (only every nth frame to save processing)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Display the video feed
    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
