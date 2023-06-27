import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from buffTracker import F_BuffTracker, BBox
from parameterUtils import Parameter

FPS_ = []
isImshow = False

paramterPath = "./examples/5_old_buff_red_dark/parameter.yaml"
param = Parameter(paramterPath)

cap = cv2.VideoCapture(param.videoRelPath)
ret, frame = cap.read()
if ret is False:
    exit(-1)
height, width, channel = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'PIM1')
out = cv2.VideoWriter(r'C:\Users\24382\Desktop\5_old_buff_red_dark.avi', fourcc, 40, (width * 2, height * 2))
# exit(0)
# 建议example 5旧符放大两倍

frameCount = 0
while ret:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (width * 2, height * 2))
    frameCount += 1
    print(frameCount)
    if frameCount >= param.start:
        if frameCount == param.start:
            rect = cv2.selectROI('roi', frame)
            rect2 = cv2.selectROI("roi2", frame)
            R_Box = BBox(rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])
            fanBladeBox = BBox(rect2[0], rect2[1], rect2[0] + rect2[2], rect2[1] + rect2[3])
            observer = F_BuffTracker(fanBladeBox, R_Box, param)

        start = time.perf_counter()
        flag = observer.update(frame, True)
        end = time.perf_counter()
        FPS = 1 / (end - start)
        print("Is it successful? : {}\nFPS={}".format(flag, FPS))
        FPS_.append(FPS)

        cv2.rectangle(frame, observer.R_Box.p1, observer.R_Box.p2, (0, 0, 255), thickness=3)
        cv2.circle(frame, center=observer.R_Box.center_2i, radius=int(observer.radius), color=(0, 0, 255),
                   thickness=1)
        cv2.putText(frame, "R", observer.R_Box.p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(frame, "press Q to quit", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.imshow("frame", frame)
        out.write(frame)
        c = cv2.waitKey(1)
        if c == -1:
            continue
        elif chr(c) == 'Q' or chr(c) == 'q':
            break

out.release()
# average = np.average(FPS_)
# plt.plot(range(len(FPS_)), FPS_, label="image shape={}\nisImshow={}".format(frame.shape, isImshow))
# plt.plot(range(len(FPS_)), [average] * len(FPS_), color="red", label="average FPS = {}".format(average))
# plt.title("Frame per Second")
# plt.legend()
# f = plt.gcf()
# f.savefig("FPS.png")
# plt.show()
# f.clear()
print("DONE")
