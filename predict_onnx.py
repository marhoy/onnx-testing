import cv2
from PIL import Image
from onnx_testing.predict_onnx import predict
import time

cap = cv2.VideoCapture(0)

# Set resolution equal to model input
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

# Set FPS to slightly higher than what we expect to get
cap.set(cv2.CAP_PROP_FPS, 1)

last_logged = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, -1)  # Flip camera vertically

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = predict(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("frame", frame)

    # log model performance
    frame_count += 1
    now = time.time()
    if now - last_logged > 1:
        print(f"{frame_count / (now-last_logged)} fps")
        last_logged = now
        frame_count = 0

    k = cv2.waitKey(30) & 0xFF
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
