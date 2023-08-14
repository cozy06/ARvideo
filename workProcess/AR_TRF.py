import threading

import cv2
from omnicv import fisheyeImgConv
import numpy as np

Thread = True

videoSource = "/Users/cozy06/Desktop/cozy06-Dev/python/ARvideo/data/testVideo.mp4"  # 실시간은 0, 웹캠 사용시 1

WINDOW_NAME = "AR Viewer"
Frame_Size = 400

FOV = 90
theta = 0
phi = 0

def mapping(exp, Frame):
    mapy, mapx = np.indices((Frame, Frame), dtype=np.float32)

    mapx = 2 * mapx / (Frame - 1) - 1
    mapy = 2 * mapy / (Frame - 1) - 1

    r, theta = cv2.cartToPolar(mapx, mapy)
    r = r ** exp

    mapx, mapy = cv2.polarToCart(r, theta)

    mapx = ((mapx + 1) * Frame - 1) / 2
    mapy = ((mapy + 1) * Frame - 1) / 2

    return mapx, mapy

def movement():
    global FOV, theta, phi, Thread
    while Thread:
        ip = input("[FOV theta phi]: ").split(" ")
        if ip[0] == "stop" or ip[0] == "q":
            Thread = False
        else:
            FOV = int(ip[0])
            theta = int(ip[1])
            phi = int(ip[2])


inputThread = threading.Thread(target=movement)
inputThread.start()


mapper = fisheyeImgConv()

cap = cv2.VideoCapture(videoSource)

while Thread:
    _, img = cap.read()

    outputFlat = mapper.eqruirect2persp(img, FOV, theta, phi, Frame_Size, Frame_Size)

    exp = 1.5  # 볼록, 오목 지수 (오목 : 0.1 ~ 1, 볼록 : 1.1~)
    mapx, mapy = mapping(exp, Frame_Size)
    outputCurve = cv2.remap(outputFlat, mapx, mapy, cv2.INTER_LINEAR)

    cv2.imshow("input image", img)
    cv2.imshow(WINDOW_NAME + " Flatten", outputFlat)
    cv2.imshow(WINDOW_NAME + " Curved", outputCurve)
    cv2.waitKey(1)
