import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import urllib.request
import os

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

# Hand skeleton connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

def draw_landmarks(img, landmarks):
    h, w = img.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for x, y in points:
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
    for start, end in HAND_CONNECTIONS:
        cv2.line(img, points[start], points[end], (255, 255, 255), 2)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        last_results = detector.detect(mp_image)

        cTime = time.time()
        elapsed = cTime - pTime
        if elapsed > 0:  # avoid division by zero
            fps = 1 / elapsed
        pTime = cTime

    if last_results and last_results.hand_landmarks:
        for hand_landmarks in last_results.hand_landmarks:
            draw_landmarks(img, hand_landmarks)

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Hand Tracking", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
