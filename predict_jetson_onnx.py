import cv2
from onnx_testing.predict_onnx import predict
import time

def open_cam_onboard(width, height):
    # On versions of L4T previous to L4T 28.1, flip-method=2
    # Use Jetson onboard camera
    gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM)," \
              "width=(int)1920, height=(int)1080, format=(string)NV12, " \
              "framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, " \
              "width=(int){}, height=(int){}, format=(string)BGRx ! " \
              "videoconvert ! video/x-raw, format=(string)BGR !" \
              "appsink").format(width, height)

    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


# Create a view writer / viewer
gst = "appsrc ! queue ! videoconvert ! video/x-raw,format=RGBA ! nvvidconv ! nvegltransform ! nveglglessink "
vw = cv2.VideoWriter(gst, cv2.CAP_GSTREAMER, 0, 30, (1280, 720))

# cap = cv2.VideoCapture(0)

# Set resolution equal to model input
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cap = open_cam_onboard(1280, 720)

# Set FPS to slightly higher than what we expect to get
#cap.set(cv2.CAP_PROP_FPS, 1)

last_logged = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    frame = predict(frame)

    #cv2.imshow("frame", frame)
    vw.write(frame)

    # log model performance
    frame_count += 1
    now = time.time()
    if now - last_logged > 1:
        print(f"{frame_count / (now-last_logged)} fps")
        last_logged = now
        frame_count = 0

    #k = cv2.waitKey(30) & 0xFF
    #if k == 27:  # press 'ESC' to quit
    #    break

cap.release()
cv2.destroyAllWindows()
