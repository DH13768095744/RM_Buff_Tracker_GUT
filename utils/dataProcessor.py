import numpy as np


class CircularQueue:
    def __init__(self, queue_capacity):
        self.capacity = queue_capacity
        self.__circularQueue = [-1] * queue_capacity
        self.front_index, self.rear_index = 0, 0

    def push(self, data):
        if self.isFull:
            raise ValueError("队列满了，无法压入")
        self.__circularQueue[self.front_index % self.capacity] = data
        self.front_index += 1

    @property
    def size(self):
        return self.front_index - self.rear_index

    @property
    def isFull(self) -> bool:
        return self.size == self.capacity

    @property
    def isEmpty(self) -> bool:
        return self.front_index - self.rear_index == 0

    @property
    def front(self):
        if self.isEmpty:
            raise ValueError("队列空了，获取头部数据")
        return self.__circularQueue[(self.front_index - 1) % self.capacity]

    @property
    def rear(self):
        if self.isEmpty:
            raise ValueError("队列空了，获取尾部数据")
        return self.__circularQueue[self.rear_index % self.capacity]

    def pop(self):
        if self.isEmpty:
            raise ValueError("队列空了，无法弹出")
        self.rear_index += 1


class ExpMovAvg(object):
    def __init__(self, decay=0.9):
        self.shadow = 0
        self.decay = decay
        self.first_time = True

    def update(self, data):
        if self.first_time:
            self.shadow = data
            self.first_time = False
            return data
        else:
            self.shadow = self.decay * self.shadow + (1 - self.decay) * data
            return self.shadow


class MovAvg(object):
    # 用前缀和优化滑窗均值的计算速度
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.preSum = CircularQueue(window_size)  # 循环前缀和
        self.dataQueue = CircularQueue(window_size)  # 循环数组
        self.first_time = True

    def update(self, data):  # 单次update的复杂度 = O(1)
        if self.first_time:
            self.preSum.push(data)
            self.first_time = False
        else:
            if self.preSum.isFull is True:
                self.preSum.pop()
                self.preSum.push(self.preSum.front - self.dataQueue.rear + data)
                self.dataQueue.pop()
            else:
                self.preSum.push(self.preSum.front + data)

        self.dataQueue.push(data)
        return self.preSum.front / self.preSum.size


class SavGol(object):
    def __init__(self, window_size=11, rank=2):
        assert window_size % 2 == 1
        self.window_size = window_size
        self.rank = rank

        self.size = int((self.window_size - 1) / 2)
        self.mm = self.create_matrix(self.size)
        self.data_seq = []

    def create_matrix(self, size):
        line_seq = np.linspace(-size, size, 2 * size + 1)
        rank_seqs = [line_seq ** j for j in range(self.rank)]
        rank_seqs = np.mat(rank_seqs)
        kernel = (rank_seqs.T * (rank_seqs * rank_seqs.T).I) * rank_seqs
        mm = kernel[self.size].T
        return mm

    def update(self, data):  # 时间复杂度O(N)
        self.data_seq.append(data)
        if len(self.data_seq) > self.window_size:
            del self.data_seq[0]
        padded_data = self.data_seq.copy()
        if len(padded_data) < self.window_size:
            left = int((self.window_size - len(padded_data)) / 2)
            right = self.window_size - len(padded_data) - left
            for i in range(left):
                padded_data.insert(0, padded_data[0])
            for i in range(right):
                padded_data.insert(
                    len(padded_data), padded_data[len(padded_data) - 1])
        return (np.mat(padded_data) * self.mm).item()
