import cv2
from buff_tracker.buffTracker import  F_BuffTracker, BBox

cap = cv2.VideoCapture(r"old_buff_red_dark.avi")
ret, frame = cap.read()
if ret is False:
    exit(-1)
height, width, channel = frame.shape

imgCenter = [width, height]
frameCount = 0
while ret:
    ret, frame = cap.read()
    frameCount += 1
    print(frameCount)
    if frameCount >= 30:
        frame = cv2.resize(frame, dsize=(width * 2, height * 2))
        if frameCount == 30:
            rect = cv2.selectROI('roi', frame)
            rect2 = cv2.selectROI("roi2", frame)
            R_Box = BBox(rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])
            fanBladeBox = BBox(rect2[0], rect2[1], rect2[0] + rect2[2], rect2[1] + rect2[3])
            observer = F_BuffTracker(fanBladeBox, R_Box, "parameter.yaml")

        print("Is it successful? : {}".format(observer.update(frame, True)))  # 建议8mm为false， 12mm为True

        cv2.rectangle(frame, observer.R_Box.p1, observer.R_Box.p2, (0, 0, 255), thickness=3)
        cv2.circle(frame, center=observer.R_Box.center_2i, radius=int(observer.radius), color=(0, 0, 255), thickness=1)
        cv2.putText(frame, "R", observer.R_Box.p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        # cv2.imshow("mask", mask)
        cv2.imshow("frame", frame)
        cv2.waitKey()

# out.release()
print("DONE")