# 用于获得调参的图片
import cv2

cap = cv2.VideoCapture("12mm_red_bright.mp4")
ret, frame = cap.read()
i = 1
while ret:
    cv2.imshow("frame", frame)
    c = cv2.waitKey()
    if chr(c) == 'q' or chr(c) == 'Q':
        cv2.imwrite("{}.jpg".format(i), frame)
        break
    ret, frame = cap.read()
    i += 1
