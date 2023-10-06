import threading
import cv2
from omnicv import fisheyeImgConv
import numpy as np

Thread = True

# videoSource = "/Users/cozy06/Desktop/cozy06-Dev/python/ARvideo/data/testVideo.mp4"  # 실시간은 0, 웹캠 사용시 1
videoSource = 0
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
# YOLO 가중치 파일과 CFG 파일 로드
YOLO_net = cv2.dnn.readNet("/Users/cozy06/Desktop/cozy06-Dev/python/ARvideoYolo/workProcess/yolov2/yolov2-tiny.weights","/Users/cozy06/Desktop/cozy06-Dev/python/ARvideoYolo/workProcess/yolov2/yolov2-tiny.cfg")

# YOLO NETWORK 재구성
classes = []
with open("/Users/cozy06/Desktop/cozy06-Dev/python/ARvideoYolo/workProcess/yolov2/yolo.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = YOLO_net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in YOLO_net.getUnconnectedOutLayers()]

while Thread:
    _, img = cap.read()
    outputFlat = mapper.eqruirect2persp(img, FOV, theta, phi, Frame_Size, Frame_Size)
    exp = 1.5  # 볼록, 오목 지수 (오목 : 0.1 ~ 1, 볼록 : 1.1~)
    mapx, mapy = mapping(exp, Frame_Size)

    frame = outputFlat
    h, w, c = frame.shape

    # YOLO 입력
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0),
                                 True, crop=False)
    YOLO_net.setInput(blob)
    outs = YOLO_net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:

        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                dw = int(detection[2] * w)
                dh = int(detection[3] * h)
                # Rectangle coordinate
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            score = confidences[i]

            # 경계상자와 클래스 정보 이미지에 입력
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
            cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5,
                        (255, 255, 255), 1)

    outputCurve = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    cv2.imshow("input image", img)
    cv2.imshow(WINDOW_NAME + " Flatten", outputFlat)
    cv2.imshow(WINDOW_NAME + " Curved", outputCurve)
    cv2.waitKey(1)
