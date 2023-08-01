import math
import matplotlib.pyplot as plt
from enum import Enum
from utils.dataProcessor import *
import scipy.optimize as opt


def draw(xs, ys):
    plt.figure(figsize=(7, 7))
    plt.ion()
    for x, y in zip(xs, ys):
        plt.cla()

        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
        plt.xlim(-12, 12)
        plt.ylim(-12, 12)
        plt.plot([0, x], [0, y])
        plt.pause(0.001)
    plt.show()


class bigPredictor:
    def __init__(self, deltaT, freq):
        """
        :param deltaT: 你需要向后预测的时间
        :param freq:  相机采样频率(n张图/1s)
        """
        self.frame_interval = int(np.ceil(freq * deltaT))  # 向上取整
        self.__startFit = FitStartDetect()
        self.__smooth = MovAvg(window_size=20)
        # self.smooth = ExpMovAvg(decay=0.9)
        self.__slidWin = CircularQueue(self.frame_interval)  # 用一个固定大小的线性数组循环队列实现滑动窗口
        self.para = None
        self.__isStart = False
        self.y, self.diffY = [], []
        self.x = 0

    @staticmethod
    def __target_func(x, a0, a1, a2, a3):
        return a0 * np.sin(a1 * x + a2) + a3

    def __sinFit(self, x, y):
        freqs = np.fft.fftfreq(len(x), x[1] - x[0])
        Y = abs(np.fft.fft(y))
        freq = abs(freqs[np.argmax(Y[1:]) + 1])
        a0 = max(y) - min(y)
        a1 = 2 * np.pi * freq
        a2 = 0
        a3 = np.mean(y)
        p0 = [a0, a1, a2, a3]
        para, _ = opt.curve_fit(self.__target_func, x, y, p0=p0, maxfev=12000)
        return para

    def update(self, data):
        self.y.append(data)
        self.x += 1
        self.__slidWin.push(data)
        if self.__slidWin.isFull is True:
            diff = self.__slidWin.front - self.__slidWin.rear
            diff = self.__smooth.update(diff)
            self.diffY.append(diff)
            self.__slidWin.pop()
            if self.__isStart is False:
                flag, rear_ = self.__startFit.update(diff)
                self.__isStart = flag
            if self.__isStart is True:
                x = range(len(self.diffY))
                if self.para is None:
                    self.para = self.__sinFit(x, self.diffY)
                deltaY = self.__target_func(self.x + self.frame_interval, *self.para)
                return [True, deltaY]
        return [False, None]


class smallPredictor:
    def __init__(self, deltaT, freq):
        """
        :param deltaT:  向后预测的时间间隔
        :param freq:    相机的采样频率
        example：
        相机帧率50， 需要向后预测0.2s
        win_size = 50 * 0.2 = 10
        win_size可以理解为，你需要从当下预测未来第10张画面的情况
        """
        self.win_size = int(np.ceil(freq * deltaT))
        self.pred = 0
        self.y = []

    def update(self, data):
        self.y.append(data)
        if len(self.y) == self.win_size:
            '''
            这个if len == win_size: 代表着，你需要从0到deltaT，这段时间里采样后才能进行下面的部分

            diff = self.y[1: self.win_size] - self.y[0: self.win_size - 1]
            表示，你用【1个轮次】的间距做一阶差分，如果你采取【n个轮次】做差分，则会需要多采样n个轮次
            由于一元一次方程的特殊性，你要预测未来第n个轮次后的值，只需要（n * 【一个轮次差分值的均值】）即可
            即：
                return [True, self.pred * self.win_size]
            '''
            self.y = np.array(self.y)
            diff = self.y[1: self.win_size] - self.y[0: self.win_size - 1]
            self.y = self.y.tolist()
            self.pred = np.average(diff)
            return [True, self.pred * self.win_size]

        elif len(self.y) > self.win_size:
            return [True, self.pred * self.win_size]
        else:
            return [False, None]


class FitStartDetect:
    """
    三角函数的拟合最好包含一个周期在内精度会比较高，这个类就是用来检测出现两次【极值】的
    思路是，间隔一定距离，计算Δy和Δx的比作为导数，
    """

    def __init__(self, queue_capacity=15):
        self.__queue_capacity = queue_capacity  # 计算Δy和Δx的间隔
        self.__queue = CircularQueue(queue_capacity)
        self.derivative = []  # 导数
        self.__idx = 0
        self.__FlipCount = 0  # 增长下降趋势反转次数，如果反转次数等于2，就证明
        self.__lim = 20  # 在确定出现第二个【极值】后，为了更好拟合，多收集20个轮次的数据【可自行调节】
        self.__count = 0  # 为__lim计数

    def isFlip(self):
        return self.__queue.rear_index > 1 and \
            self.derivative[self.__idx - 2] * self.derivative[self.__idx - 1] < 0

    def update(self, data):
        # 输入的数据应当是经过滤波的【较为平滑的数据】
        # 如果返回False，则还需要收集数据，如果返回True则可以开始进行拟合
        self.__queue.push(data)
        if self.__queue.isFull is True:
            self.__queue.pop()
            self.derivative.append((self.__queue.front - self.__queue.rear) / self.__queue_capacity)
            self.__idx += 1
            if self.isFlip():
                self.__FlipCount += 1
            if self.__FlipCount == 2:
                if self.__count < self.__lim:
                    self.__count += 1  # 20个轮次以检验是不是由于滤波不够导致突发的单调性变化
                else:
                    return True, self.__idx
            if self.__FlipCount > 2:
                raise ValueError(
                    "不可能在20个轮次内单调性突然变化，你需要修改滤波算法及其参数，或queue_size参数[除非你帧率很低]")
        return False, -1


def euclidean_distance(p1, p2):
    """
    计算两个点的欧式距离
    """
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


class mode(Enum):
    big = "big"
    small = "small"


class clock(Enum):
    anticlockwise = "anticlockwise"
    clockwise = "clockwise"


def trans(x, y):
    angle = np.arctan(y / x)
    if x > 0 and y > 0:
        return angle
    elif x < 0:
        angle += np.pi
    else:
        angle += np.pi * 2
    return angle


class angleObserver:

    def __init__(self, clockMode):
        self.__lastY, self.__lastX, self.__lastAngle = -1, -1, []
        self.__delta = [0]
        self.__windowLength = 0.0
        self.__dataList = []
        self.__clockMode = clockMode

    @staticmethod
    def Rotation(theta, vector: np.ndarray):
        """
        :param theta: radian
        :param vector: 1 x 2, vector是由点的xy组成的向量，原点需要在能量机关中心R上
        :return: vecotr
        """
        rotationMatrix = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
        vector_ = np.dot(rotationMatrix, vector)
        return vector_

    def AngleTransformer(self, x, y):
        theta = np.arctan(y / x)
        if len(self.__lastAngle) == 0:
            self.__lastAngle.append(theta)
            return float(theta)
        delta = np.fabs(np.around((theta - self.__lastAngle[0]) / np.pi)) * np.pi
        if self.__clockMode == clock.anticlockwise:
            delta *= -1
        theta += delta
        self.__lastAngle[0] = theta
        return float(theta)

    def update(self, x, y, R):
        if self.__lastX == -1 and self.__lastY == -1:
            self.__lastX = x
            self.__lastY = y
        if self.__delta[0] != 0:
            x, y = self.Rotation(2 * np.pi / 5 * (5 - self.__delta[0]), np.array([x, y]))
        if euclidean_distance((self.__lastX, self.__lastY), (x, y)) > R * 0.5:
            points = [self.Rotation(2 * np.pi / 5 * time, np.array([self.__lastX, self.__lastY])) for time in range(5)]
            distances = [[euclidean_distance((x, y), (points[i][0], points[i][1])), i] for i in range(5)]
            tmp = min(distances, key=lambda dist: dist[0])[1]
            x, y = self.Rotation(2 * np.pi / 5 * (5 - tmp), np.array([x, y]))
            self.__delta[0] += tmp
        angle = self.AngleTransformer(x, y)
        self.__lastX, self.__lastY = x, y

        return angle
