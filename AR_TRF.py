import cv2
from omnicv import fisheyeImgConv


WINDOW_NAME = "360 Viewer"
Frame_Size = 400

mapper = fisheyeImgConv()

cv2.namedWindow(WINDOW_NAME)

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    FOV = 90
    theta = 0
    phi = 0

    outputFlat = mapper.eqruirect2persp(img, FOV, theta, phi, Frame_Size, Frame_Size)
    outputCurve = outputFlat

    cv2.imshow(WINDOW_NAME, outputFlat)
    cv2.imshow(WINDOW_NAME, outputCurve)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
