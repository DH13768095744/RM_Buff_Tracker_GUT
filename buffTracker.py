import cv2
import math
import numpy as np
from enum import Enum
from typing import List
from parameterUtils import Parameter


class IoU_Type(Enum):
    IoU = "IoU"
    GIoU = "GIoU"
    DIoU = "DIoU"
    CIoU = "CIoU"


class RotationRectangle:
    def __init__(self, points, R_Box_center):
        # p1，p2为距离中心R最远的两个点，p3为距离p1最远的，剩下的就是p4
        p = [[points[0], 0],
             [points[1], 0],
             [points[2], 0],
             [points[3], 0]]
        for i in range(len(p)):
            p[i][1] = euclidean_distance(p[i][0], R_Box_center)
        p.sort(key=lambda p_: p_[1], reverse=True)
        self.p1 = p[0][0]
        self.p2 = p[1][0]
        p[2][1] = euclidean_distance(p[0][0], p[2][0])
        p[3][1] = euclidean_distance(p[0][0], p[3][0])
        self.p3, self.p4 = [p[2][0], p[3][0]] if p[2][1] > p[3][1] else [p[3][0], p[2][0]]

        self.points = [self.p1, self.p2, self.p3, self.p4]

        # 两个对角线,p1-p3, p2-p4，用对角线的交点作为旋转矩形的中心点
        self.__k13, self.__b13 = self.__getLine(self.p1, self.p3)
        self.__k24, self.__b24 = self.__getLine(self.p2, self.p4)
        # p1-p2的中心点，p3-p4的中心点
        self.__lineCenter_p1 = self.__getLineCenter(self.p1, self.p2)
        self.__lineCenter_p2 = self.__getLineCenter(self.p3, self.p4)
        self.top, self.disTop, self.btm, self.disBtm = self.__lineCenter_p1, euclidean_distance(self.__lineCenter_p1,
                                                                                                R_Box_center), \
            self.__lineCenter_p2, euclidean_distance(self.__lineCenter_p2, R_Box_center)

    @staticmethod
    def __getLineCenter(p1, p2):  # 求p1， p2的线段的中心点
        x1, y1 = p1
        x2, y2 = p2
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        x = xmin + (xmax - xmin) / 2
        y = ymin + (ymax - ymin) / 2
        return np.array([x, y])

    @staticmethod
    def __getLine(p1, p2):  # 求p1， p2组成的连线
        x1, y1 = p1
        x2, y2 = p2
        k = (x2 - x1) / (y2 - y1)
        b = y1 - k * x1
        return k, b

    @property
    def center_2f(self):
        k = np.array([
            [1, -1 * self.__k13],
            [1, -1 * self.__k24]
        ])
        b = np.array([
            [self.__b13],
            [self.__b24]
        ])
        x, y = np.dot(np.linalg.inv(k), b)
        return np.array([x[0], y[0]])

    @property
    def center_2i(self):
        x, y = self.center_2f
        return np.array([int(x), int(y)])

    @property
    def width(self):
        return euclidean_distance(self.p1, self.p2)

    @property
    def height(self):
        return euclidean_distance(self.p1, self.p4)

    @property
    def area(self):
        return self.width * self.height


# 从网上改的，便于计算IOU变体
class BBox:
    def __init__(self, xmin, ymin, xmax, ymax, BBox_ID=-1):
        '''
        定义框，左上角及右下角坐标
        '''
        self.id = BBox_ID
        self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
        self.p1, self.p2 = np.array([xmin, ymin]), np.array([xmin + self.width, ymin + self.height])

    def __xor__(self, other):
        '''
        计算box和other的IoU
        '''
        cross = self & other
        union = self | other
        return cross / (union + 1e-6)

    def __or__(self, other):
        '''
        计算box和other的并集
        '''
        cross = self & other
        union = self.area + other.area - cross
        return union

    def __and__(self, other):
        '''
        计算box和other的交集
        '''
        xmax = min(self.xmax, other.xmax)
        ymax = min(self.ymax, other.ymax)
        xmin = max(self.xmin, other.xmin)
        ymin = max(self.ymin, other.ymin)
        cross_box = BBox(xmin, ymin, xmax, ymax)
        if cross_box.width <= 0 or cross_box.height <= 0:
            return 0
        return cross_box.area

    def boundof(self, other):
        '''
        计算box和other的边缘外包框，使得2个box都在框内的最小矩形
        '''
        xmin = min(self.xmin, other.xmin)
        ymin = min(self.ymin, other.ymin)
        xmax = max(self.xmax, other.xmax)
        ymax = max(self.ymax, other.ymax)
        return BBox(xmin, ymin, xmax, ymax)

    def center_distance(self, other):
        '''
        计算两个box的中心点距离
        '''
        return euclidean_distance(self.center_2f, other.center_2f)

    def bound_diagonal_distance(self, other):
        '''
        计算两个box的bound的对角线距离
        '''
        bound = self.boundof(other)
        return euclidean_distance((bound.xmin, bound.ymin), (bound.xmax, bound.ymax))

    @property
    def center_2f(self):
        return np.array([(self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2])

    @property
    def center_2i(self):
        return np.array([int((self.xmin + self.xmax) / 2), int((self.ymin + self.ymax) / 2)])

    @property
    def area(self):
        return self.width * self.height

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin  # + 1

    def set_id(self, BBox_ID):
        self.id = BBox_ID

    def create_new_bbox_by_center(self, center):
        xmin = int(center[0] - self.width / 2)
        ymin = int(center[1] - self.height / 2)
        xmax = int(center[0] + self.width / 2)
        ymax = int(center[1] + self.height / 2)
        return BBox(xmin, ymin, xmax, ymax, self.id)


def IoU(a, b):
    return a ^ b


def GIoU(a, b):
    bound_area = a.boundof(b).area
    union_area = a | b
    return IoU(a, b) - (bound_area - union_area) / bound_area


def DIoU(a, b):
    d = a.center_distance(b)
    c = a.bound_diagonal_distance(b)
    return IoU(a, b) - (d ** 2) / (c ** 2)


def CIoU(a, b):
    v = 4 / (math.pi ** 2) * (math.atan(a.width / a.height) - math.atan(b.width / b.height)) ** 2
    iou = IoU(a, b)
    alpha = v / (1 - iou + v)
    return DIoU(a, b) - alpha * v


def euclidean_distance(p1, p2):
    '''
    计算两个点的欧式距离
    '''
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# 给R box用的，iou变体排序用
class TargetStruct:
    def __init__(self, box: BBox, iou):
        self.box = box
        self.iou = iou


# 各类IOU的计算
def compareByIoU(lBox, boxs, type, pointsList=None) -> List[TargetStruct]:
    '''
    :param lBox: 上一个box
    :param boxs: 备选box
    :param type: 选择什么iou变种
    :param pointsList: 四点
    :return: iou最大的框
    '''
    ious = []
    if len(boxs) == 0:
        return ious
    for i in range(len(boxs)):
        if type == IoU_Type.IoU:
            ious.append(TargetStruct(boxs[i], IoU(lBox, boxs[i])))
        elif type == IoU_Type.CIoU:
            ious.append(TargetStruct(boxs[i], CIoU(lBox, boxs[i])))
        elif type == IoU_Type.GIoU:
            ious.append(TargetStruct(boxs[i], GIoU(lBox, boxs[i])))
        elif type == IoU_Type.DIoU:
            ious.append(TargetStruct(boxs[i], DIoU(lBox, boxs[i])))
    ious.sort(key=lambda t: t.iou, reverse=True)
    return ious


# 求两个向量的夹角，未被使用，后续更新
def IncludedAngle(a: np.ndarray, b: np.ndarray):
    '''
    :param a: 1 x 2
    :param b: 1 x 2
    :return: 夹角
    '''
    a_ = np.dot(a, b.T)
    b_ = np.linalg.norm(a) * np.linalg.norm(b)
    return np.arccos(a_ / b_)


def Rotation(theta, vector: np.ndarray):
    '''
    :param theta: radian
    :param vector: 1 x 2, vector是由点的xy组成的向量，原点需要在能量机关中心R上
    :return: vecotr
    '''
    rotationMatrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    vector_ = np.dot(rotationMatrix, vector)
    return vector_


# AnticlockwiseAngleTransformer在使用的全局变量
deltaAngle = [0]
lastX = [None]
lastY = [None]
lastAngle = [0]
fanSkipAngle = [0]
coordinateSkip = np.pi
minFanSkip = 2 * np.pi / 5
flag1, flag2 = [False], [False]


# 用于保障观测角度的连续性, 未使用函数，后续更新
def AnticlockwiseAngleTransformer(x, y, R):
    x, y = Rotation(theta=fanSkipAngle[0], vector=np.array([x, y]))
    tempRadian = np.arctan(y / x)

    # 解决扇叶跳转问题
    if (lastY[0] is not None and lastX[0] is not None) and euclidean_distance((x, y), (lastX[0], lastY[0])) > 0.5 * R:
        includedAngle = IncludedAngle(a=np.array([x, y]), b=np.array([lastX[0], lastY[0]]))
        if includedAngle > 0.8 * minFanSkip:
            tempSkipAngle = np.round(includedAngle / minFanSkip) * minFanSkip
            tempx, tempy = Rotation(theta=tempSkipAngle, vector=np.array([x, y]))
            if euclidean_distance((lastX[0], lastY[0]), (tempx, tempy)) > 0.7 * R:
                tempx, tempy = Rotation(theta=-1 * tempSkipAngle, vector=np.array([x, y]))
                tempSkipAngle *= -1
            fanSkipAngle[0] += tempSkipAngle
            x, y = tempx, tempy
            tempRadian = np.arctan(y / x)

    # 解决坐标在不同象限计算角度跳动的问题
    if np.fabs(tempRadian - lastAngle[0]) >= 0.8 * minFanSkip:
        skip = int(np.fabs(tempRadian - lastAngle[0]) / np.pi * 10) / 10
        if skip >= 0.8:
            deltaAngle[0] -= coordinateSkip

    lastAngle[0] = tempRadian
    lastX[0] = x
    lastY[0] = y
    return tempRadian + deltaAngle[0]


# 扇叶类，单个扇叶的矩形和旋转矩形
class FanBlade:
    def __init__(self, rect: BBox, rtn_rect: RotationRectangle):
        self.bbox = rect
        self.rtn_rect = rtn_rect
        self.state = None


# 能量机关的跟踪器
class F_BuffTracker:
    def __init__(self, fanBladeBox: BBox, R_Box: BBox, param: Parameter, isImshow: bool = True):
        self.param = param
        self.fanBladeBox = fanBladeBox
        self.R_Box = R_Box
        self.FanBladeList = [FanBlade(BBox(0, 0, 0, 0), RotationRectangle(np.random.uniform(size=(4, 2)),
                                                                          R_Box.center_2f)) for i in range(5)]
        self.radius = R_Box.center_distance(fanBladeBox)
        self.states = ["target"] + ["unlighted"] * 4
        self.fanNum = 0
        self.center = R_Box.center_2f
        self.isImshow = isImshow
        self.frame = None

    @staticmethod
    def __Points2BBox(points) -> BBox:
        xs = points[:, 0:1]
        ys = points[:, 1:2]
        xmin, ymin, xmax, ymax = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
        return BBox(xmin, ymin, xmax, ymax)

    def __GetMaskByHSVThreshold(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.param.HSV.lowerLimit, self.param.HSV.upperLimit)
        kernel = np.ones((self.param.kernel, self.param.kernel), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def __getFanBlade(self, mask) -> List[FanBlade]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fanBladeList = []  # 备选扇叶框
        realFanBladeList = []  # 有效扇叶框
        for cont in contours:
            rect_pts = cv2.boxPoints(cv2.minAreaRect(cont)).astype(np.int32)
            rtn_rect = RotationRectangle(rect_pts, self.R_Box.center_2f)
            if 0.4 * self.radius < rtn_rect.disBtm < rtn_rect.disTop < 1.5 * self.radius and rtn_rect.area > 2 * self.R_Box.area:
                #  在0.4半径和1.5半径之内， 且面积起码得是中心R的box面积两倍以上的框才可能是备选框
                x, y, w, h = cv2.boundingRect(cont)
                if self.isImshow:
                    cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), thickness=3)
                fanBladeList.append(FanBlade(BBox(x, y, x + w, y + h), rtn_rect))

        if self.isImshow:
            cv2.imshow("mask__", mask)
        if self.FanBladeList[0].bbox.area == 0 and len(fanBladeList) == 1:  # 刚初始化后的特殊操作，仅第一帧会进入
            fanBladeList[0].bbox.id = 0
            return fanBladeList
        elif self.FanBladeList[0].bbox.area == 0 and len(fanBladeList) != 1:
            return None

        # 解决装甲板被击打后，装甲板中间短暂镂空问题（待击打扇叶被击打的下一帧中心会镂空），或是非一环的情况
        tempList = dict()
        """
        由于所有的扇叶都是围绕着中心点产生约束关系的上一帧的bbox和rtn_box信息是在上一帧的R_Box约束之下的，
        需要将上一帧的扇叶bbox和rtn_box信息转到当前R_Box之下,才能在云台剧烈运动使得能量机关在图像坐标系下也剧烈运动的情况下
        让IOU得到一个好的结果，也可以理解为对IOU的计算引入了以R_Box为圆心的极坐标系下的角度差为
        """
        correctFanBlade = []
        for i in range(len(self.FanBladeList)):
            tempXY = self.FanBladeList[i].bbox.center_2f - self.center + self.R_Box.center_2f
            tempBox = self.FanBladeList[i].bbox.create_new_bbox_by_center(tempXY)
            tempPoints = []
            for p in self.FanBladeList[i].rtn_rect.points:
                tempPoints.append(p - self.center + self.R_Box.center_2f)
            correctFanBlade.append(FanBlade(tempBox, RotationRectangle(tempPoints, self.R_Box.center_2f)))
            if self.isImshow:
                cv2.rectangle(self.frame, tempBox.p1, tempBox.p2, (255, 0, 0), 3)
                cv2.putText(self.frame, "id = {} | lastFrame".format(tempBox.id), tempBox.p1 - 30, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        if self.isImshow:
            cv2.imshow("temp_", self.frame)

        for lastFan in correctFanBlade:
            box = lastFan.bbox
            # cv2.rectangle(mask, box.p1, box.p2, (255, 255, 255), 3)
            tempList[box.id] = []
            for fan in fanBladeList:
                if IoU(box, fan.bbox) > 0:  # 如果现在的备选box和上一帧的扇叶框IOU > 0则大概率是扇叶或其部分
                    tempList[box.id].append(fan)
                    # cv2.rectangle(mask, fan.bbox.p1, fan.bbox.p2, (255, 255, 255), 3)
            if len(tempList[box.id]) == 1:  # 如果扇叶连通域只有一个（一环或待击打扇叶）
                tempList[box.id][0].bbox.id = box.id
                realFanBladeList.append(tempList[box.id][0])
            elif len(tempList[box.id]) >= 2:
                # 如果是镂空，或者非一环的情况时，此扇叶框的连通域不连接，框不止一个需要合并
                tempP = {1: [], 2: [], 3: [], 4: []}
                for rtn in tempList[box.id]:
                    # 组成tempP[i]的组成为pi和pi到中心R的距离
                    tempP[1] += [[rtn.rtn_rect.p1, euclidean_distance(rtn.rtn_rect.p1, self.R_Box.center_2f)]]
                    tempP[2] += [[rtn.rtn_rect.p2, euclidean_distance(rtn.rtn_rect.p2, self.R_Box.center_2f)]]
                    tempP[3] += [[rtn.rtn_rect.p3, euclidean_distance(rtn.rtn_rect.p3, self.R_Box.center_2f)]]
                    tempP[4] += [[rtn.rtn_rect.p4, euclidean_distance(rtn.rtn_rect.p4, self.R_Box.center_2f)]]
                # 距离中心R 最院的P1和P2
                p1 = max(tempP[1], key=lambda x: x[1])[0]
                p2 = max(tempP[2], key=lambda x: x[1])[0]
                # 距离中心R 最近的P3和P4
                p3 = min(tempP[3], key=lambda x: x[1])[0]
                p4 = min(tempP[4], key=lambda x: x[1])[0]

                bbox = self.__Points2BBox(np.array([p1, p2, p3, p4]))  # 将四点转化为bbox，用最大和最小的xy
                bbox.id = box.id
                realFanBladeList.append(FanBlade(bbox,
                                                 RotationRectangle([p1, p2, p3, p4], self.R_Box.center_2f)))

        # 更新扇叶状态（待击打目标，未亮起，已击打）
        for i in range(len(realFanBladeList)):
            if len(realFanBladeList) == 1:  # 如果只有一个扇叶亮起，则这一定是待击打目标，其他必然为未亮起
                realFanBladeList[0].bbox.id = 0
                self.states[0] = "target"
                for i_ in range(1, 5):
                    self.states[i_] = "unlighted"
            elif len(realFanBladeList) > self.fanNum:
                # 如果现在的亮起个数大于上一帧亮起个数，则之前为未亮起的现在一定是待击打目标
                # 上一帧是待击打目标的现在一定是已击打目标，其他不变
                id_x = realFanBladeList[i].bbox.id
                if self.states[id_x] == "target":
                    self.states[id_x] = "shot"
                elif self.states[id_x] == "unlighted":
                    self.states[id_x] = "target"
        return realFanBladeList

    def __MayBeTarget(self, w, h, flag: bool):
        if flag is True:
            return min(w * h, self.R_Box.area) / max(w * h, self.R_Box.area) > self.param.MayBeTarget.area and \
                min(w, self.R_Box.width) / max(w, self.R_Box.width) > self.param.MayBeTarget.width and \
                min(h, self.R_Box.height) / max(h, self.R_Box.height) > self.param.MayBeTarget.height
        else:
            return True

    def __GetAlternateBoxs(self, mask_, flag):  # , lastArea, lastWidth, lastHeight):
        contours, _ = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxs = []
        for cont in contours:
            rect = cv2.boundingRect(cont)
            x, y, w, h = rect
            dis = euclidean_distance((x + w / 2, y + h / 2), self.R_Box.center_2f)  # 框和中心R框的距离
            if self.__MayBeTarget(w, h, flag) and dis < self.radius * 2:
                # 如果flag == True，则将面积，宽高放入筛选，如果flag == False
                #  无论flag为什么，都只能考虑半径内的框
                xmin, ymin = x, y
                xmax, ymax = x + w, y + h
                temp = BBox(xmin, ymin, xmax, ymax)
                boxs.append(temp)
        return boxs

    def update(self, frame, isOpenMaybeTarget: bool) -> bool:
        """
        :param frame: 输入的RGB图像
        :param isOpenMaybeTarget: 是否打开MaybeTarget函数
        :return: 返回bool结构，update是否成功
        """
        if self.isImshow:
            self.frame = frame
        lightedFanBlade_IDList = [0]  # 亮起扇叶的ID的List，默认扇叶0肯定是亮起的
        mask = self.__GetMaskByHSVThreshold(frame)  # HSV二值化
        if self.isImshow:
            cv2.imshow("maskI", mask)
        boxs = self.__GetAlternateBoxs(mask, isOpenMaybeTarget)  # 获取中心R box的备选框
        box_and_iou = compareByIoU(self.R_Box, boxs, IoU_Type.CIoU)  # 通过ciou计算哪个框最有可能是这一帧的中心R box
        if len(box_and_iou) != 0 and box_and_iou[0].iou > -1:  # 如果没有找到中心R 则返回False
            self.R_Box = box_and_iou[0].box
        else:
            return False
        cv2.circle(mask, center=self.R_Box.center_2i, radius=int(self.radius * self.param.insideRate), color=(0, 0, 0),
                   thickness=-1)  # 把中心R 和流水灯都遮挡住，这留下扇叶顶准心
        cv2.circle(mask, center=self.R_Box.center_2i, radius=int(self.radius * self.param.outsideRate), color=(0, 0, 0),
                   thickness=3)  # 将外界和扇叶顶准心隔开，避免链接在一起影响外接矩形和最小外接矩形的计算
        fanBladeList = self.__getFanBlade(mask)  # 获取扇叶顶的准心
        self.center = self.R_Box.center_2f
        if fanBladeList is None:
            return False
        self.fanNum = len(fanBladeList)  # 当前帧的亮起扇叶个数
        for fan in fanBladeList:
            box = fan.bbox
            if box.id != 0:
                lightedFanBlade_IDList.append(box.id)
            self.FanBladeList[box.id].bbox = box
            print(box.center_2f)
            if self.isImshow:
                cv2.rectangle(frame, box.p1, box.p2, (0, 0, 255), 2)
                cv2.putText(frame, "id = {} | ".format(box.id) + self.states[box.id], box.p1, cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 255), 2)
        # 更新未亮起的扇叶顶的信息
        for i in range(5):
            if i in lightedFanBlade_IDList:  # 跳过亮起的扇叶ID
                continue
            self.FanBladeList[i].bbox = self.fanBladeBox.create_new_bbox_by_center(
                Rotation(np.pi * 2 / 5 * i,
                         self.FanBladeList[0].bbox.center_2f - self.R_Box.center_2f) + self.R_Box.center_2f
            )
            self.FanBladeList[i].bbox.id = i
        return True
