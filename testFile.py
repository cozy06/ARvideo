import cv2
from omnicv import fisheyeImgConv
import numpy as np



WINDOW_NAME = "360 Viewer"
Frame_Size = 400

mapper = fisheyeImgConv()

cv2.namedWindow(WINDOW_NAME)

cap = cv2.VideoCapture(0)

exp = 1.5       # 볼록, 오목 지수 (오목 : 0.1 ~ 1, 볼록 : 1.1~)

mapy, mapx = np.indices((Frame_Size, Frame_Size),dtype=np.float32)

mapx = 2*mapx/(Frame_Size-1)-1
mapy = 2*mapy/(Frame_Size-1)-1

r, theta = cv2.cartToPolar(mapx, mapy)
r = r **exp

mapx, mapy = cv2.polarToCart(r, theta)

mapx = ((mapx + 1)*Frame_Size-1)/2
mapy = ((mapy + 1)*Frame_Size-1)/2


while True:
    _, img = cap.read()

    FOV = 90
    theta = 0
    phi = 0

    outputFlat = mapper.eqruirect2persp(img, FOV, theta, phi, Frame_Size, Frame_Size)
    outputCurve = cv2.remap(outputFlat,mapx,mapy,cv2.INTER_LINEAR)

    # cv2.imshow(WINDOW_NAME + " Flatten", outputFlat)
    cv2.imshow(WINDOW_NAME + " Curved", outputCurve)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
