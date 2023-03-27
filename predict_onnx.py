import cv2
from PIL import Image
from onnx_testing.predict_onnx import predict

cap = cv2.VideoCapture(0)

# Set resolution equal to model input
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

# Set FPS to slightly higher than what we expect to get
cap.set(cv2.CAP_PROP_FPS, 36)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, -1)  # Flip camera vertically

    cv2.imshow("frame", frame)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:  # press 'ESC' to quit
        break

# image = predict(frame)


cap.release()
cv2.destroyAllWindows()
