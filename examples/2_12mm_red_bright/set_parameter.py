import cv2
import numpy as np
from buff_tracker.buffTracker import BBox
from buff_tracker.parameterUtils import parameterLoad, parameterWrite

def nothing(x):
    pass


def select_parameter(imgPath="1.jpg", jsonPath="parameter.yaml", frame=None):
    # use track bar to perfectly define (1/2)
    # the lower and upper values for HSV color space(2/2)
    cv2.namedWindow("Tracking")
    cv2.resizeWindow("Tracking", 640, 480)
    # 参数：1 Lower/Upper HSV 3 startValue 4 endValue
    cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
    cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
    cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)
    cv2.createTrackbar("kernel", "Tracking", 0, 10, nothing)
    cv2.createTrackbar("outside rate", "Tracking", 100, 200, nothing)
    cv2.createTrackbar("inside rate", "Tracking", 0, 100, nothing)
    flag = True
    if frame is None:
        img = cv2.imread(imgPath)
        frame = img.copy()
    while True:
        if frame is None:
            break
        if flag:
            rect = cv2.selectROI('roi', frame)
            rect2 = cv2.selectROI("roi2", frame)
            R_Box = BBox(rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])
            fanBladeBox = BBox(rect2[0], rect2[1], rect2[0] + rect2[2], rect2[1] + rect2[3])
            flag = False
        R = R_Box.center_distance(fanBladeBox)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("LH", "Tracking")
        l_s = cv2.getTrackbarPos("LS", "Tracking")
        l_v = cv2.getTrackbarPos("LV", "Tracking")

        u_h = cv2.getTrackbarPos("UH", "Tracking")
        u_s = cv2.getTrackbarPos("US", "Tracking")
        u_v = cv2.getTrackbarPos("UV", "Tracking")

        kernel = cv2.getTrackbarPos("kernel", "Tracking")

        outsideRate = cv2.getTrackbarPos("outside rate", "Tracking")
        insideRate = cv2.getTrackbarPos("inside rate", "Tracking")

        lowerLimit = np.array([l_h, l_s, l_v])  # lower limit
        upperLimit = np.array([u_h, u_s, u_v])  # upper limit

        mask = cv2.inRange(hsv, lowerLimit, upperLimit)
        mask = cv2.dilate(mask, np.ones((kernel, kernel), np.uint8), iterations=1)

        res = cv2.bitwise_and(frame, frame, mask=mask)  # src1,src2

        cv2.circle(frame, center=R_Box.center_2i, radius=int(R * insideRate / 100), color=(255, 0, 0), thickness=-1)
        cv2.circle(frame, center=R_Box.center_2i, radius=int(R * outsideRate / 100), color=(255, 0, 0), thickness=3)
        cv2.circle(mask, center=R_Box.center_2i, radius=int(R * insideRate / 100), color=(0, 0, 0), thickness=-1)
        cv2.circle(mask, center=R_Box.center_2i, radius=int(R * outsideRate / 100), color=(255, 255, 255), thickness=3)

        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        cv2.imshow("res", res)
        frame = img.copy()
        key = cv2.waitKey(1)
        if key < 0:
            continue
        elif chr(key) == "Q" or chr(key) == "q":  # Esc
            data = parameterLoad(jsonPath)
            data["HSV"]["lowerLimit"] = lowerLimit.tolist()
            data["HSV"]["upperLimit"] = upperLimit.tolist()
            data["outsideRate"] = outsideRate / 100
            data["insideRate"] = insideRate / 100
            data["kernel"] = kernel
            parameterWrite(jsonPath, data)
            break



select_parameter()
