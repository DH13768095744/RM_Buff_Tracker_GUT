import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.buffTracker import F_BuffTracker, BBox
from utils.parameterUtils import Parameter
from utils.angleProcessor import bigPredictor, smallPredictor, angleObserver, trans, mode

FPS_ = []
angles = []
isImshow = True

paramterPath = r"./examples/2_12mm_red_bright/parameter.yaml"
param = Parameter(paramterPath)

cap = cv2.VideoCapture(param.videoRelPath)
ret, frame = cap.read()
if ret is False:
    exit(-1)
# height, width, channel = frame.shape
# fourcc = cv2.VideoWriter_fourcc(*'PIM1')
# out = cv2.VideoWriter(r'C:\Users\24382\Desktop\2_12mm_red_bright.avi', fourcc, 40, (width * 2, height * 2))

# 建议example 5旧符放大两倍
xy = []
frameCount = 0
while ret:
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.resize(frame, (width * 2, height * 2))
    frameCount += 1
    print(frameCount)
    if frameCount >= param.start:
        if frameCount == param.start:
            rect = cv2.selectROI('roi', frame)
            rect2 = cv2.selectROI("roi2", frame)
            R_Box = BBox(rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])
            fanBladeBox = BBox(rect2[0], rect2[1], rect2[0] + rect2[2], rect2[1] + rect2[3])
            tracker = F_BuffTracker(fanBladeBox, R_Box, param, isImshow=isImshow)
            observer = angleObserver()
            predictor = bigPredictor(freq=50, deltaT=0.2)
            interval = int(50 * 0.2)


        flag = tracker.update(frame, True)
        if flag is False:
            print(False)
        x, y = tracker.fanBladeBox.center_2f - tracker.R_Box.center_2f
        angle = observer.update(x, y, tracker.radius, clockModel=mode.anticlockwise)
        angles.append(angle)
        flag, deltaAngle = predictor.update(angle)
        if flag is True:
            angle = trans(x, y) + deltaAngle
            x = np.cos(angle) * tracker.radius
            y = np.sin(angle) * tracker.radius
            x, y = np.array([x, y]) + tracker.R_Box.center_2f
            xy.append([x, y])
            cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), -1)
            cv2.putText(frame, "now predict", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            if len(xy) >= interval:
                x, y = xy[len(xy) - interval]
                cv2.putText(frame, "before predict", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                pass

        cv2.rectangle(frame, tracker.fanBladeBox.p1, tracker.fanBladeBox.p2, (0, 255, 0), 3)

        cv2.imshow("frame", frame)
        c = cv2.waitKey(1)
        if c == ord('q') or c == ord('Q'):
            break

plt.plot(range(len(angles)), angles)
plt.show()
# with open("6_dark_blue_big.txt", 'w') as f:
#     for angle in angles:
#         f.write("{}\n".format(angle))

print("DONE")
