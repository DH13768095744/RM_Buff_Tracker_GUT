import cv2
import numpy as np
from utils.parameterUtils import Parameter


def __GetMaskByHSVThreshold(param, frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, param.HSV.lowerLimit, param.HSV.upperLimit)
    kernel = np.ones((param.kernel, param.kernel), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

paramterPath = "./examples/5_old_buff_red_dark/parameter.yaml"
param = Parameter(paramterPath)

cap = cv2.VideoCapture(param.videoRelPath)
ret, frame = cap.read()
height, width, channel = frame.shape

while ret:
    frame = cv2.resize(frame, (width * 2, height * 2))
    mask = __GetMaskByHSVThreshold(param, frame)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.waitKey()
    ret, frame = cap.read()
