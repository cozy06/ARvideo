import threading
import cv2
from omnicv import fisheyeImgConv
import numpy as np
import time
Thread = True

# videoSource = "/Users/cozy06/Desktop/cozy06-Dev/python/ARvideo/data/testVideo.mp4"  # 실시간은 0, 웹캠 사용시 1
videoSource = 0
WINDOW_NAME = "AR Viewer"
Frame_Size = 400
FOV = 90
theta = 0
phi = 0

# Load Yolo
net = cv2.dnn.readNet("/Users/cozy06/Desktop/cozy06-Dev/python/ARvideo/workProcess/re/yolov2-tiny.weights", "/Users/cozy06/Desktop/cozy06-Dev/python/ARvideo/workProcess/re/yolov2-tiny.cfg")
classes = []
with open("/Users/cozy06/Desktop/cozy06-Dev/python/ARvideo/workProcess/re/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
starting_time = time.time()
img_id=0
font = cv2.FONT_HERSHEY_PLAIN

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
    ret, img = cap.read()  # Add 'ret' to check if frame reading was successful
    img = mapper.eqruirect2persp(img, FOV, theta, phi, Frame_Size, Frame_Size)
    img_id += 1
    height, width,channels = img.shape
    if not ret:
        # If frame reading failed, break out of the loop
        break
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(img, (x, y), (x + w, y + 30), color, -1)
            cv2.putText(img, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)
        elapsed_time = time.time() - starting_time
        fps = img_id / elapsed_time
        cv2.putText(img, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)
        # key = cv2.waitKey(1)
    exp = 1.5  # 볼록, 오목 지수 (오목 : 0.1 ~ 1, 볼록 : 1.1~)
    mapx, mapy = mapping(exp, Frame_Size)

    cv2.imshow(WINDOW_NAME + " Flattenwr", img)
    outputCurve = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    cv2.imshow(WINDOW_NAME + " Flatten", img)
    cv2.imshow(WINDOW_NAME + " Curved", outputCurve)
    cv2.waitKey(1)
        #if key == 27:
            #break
cap.release()
cv2.destroyAllWindows()




    #outputFlat = mapper.eqruirect2persp(img, FOV, theta, phi, Frame_Size, Frame_Size)
    #exp = 1.5  # 볼록, 오목 지수 (오목 : 0.1 ~ 1, 볼록 : 1.1~)
    #mapx, mapy = mapping(exp, Frame_Size)

    #outputCurve = cv2.remap(outputFlat, mapx, mapy, cv2.INTER_LINEAR)
    #cv2.imshow("input image", img)
    #cv2.imshow(WINDOW_NAME + " Flatten", outputFlat)
    #cv2.imshow(WINDOW_NAME + " Curved", outputCurve)
    #cv2.waitKey(1)
