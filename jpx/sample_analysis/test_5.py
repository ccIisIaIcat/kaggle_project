import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus'] = False


class Jibbs:
    def __init__(self, m, n):
        self.x = [0, 0]
        self.samples = []
        self.mean = [0, 0]
        self.p = 0.5
        self.cov = [[1, self.p], [self.p, 1]]
        self.m = m
        self.n = n

    def sampling(self):
        for i in range(self.m):
            x = np.random.normal(self.p * self.x[1], 1 - self.p ** 2)
            self.x[0] = x
            x = np.random.normal(self.p * self.x[0], 1 - self.p ** 2)
            self.x[1] = x
        for i in range(self.n):
            x = np.random.normal(self.p * self.x[1], 1 - self.p ** 2)
            self.x[0] = x
            self.samples.append(self.x.copy())
            x = np.random.normal(self.p * self.x[0], 1 - self.p ** 2)
            self.x[1] = x
            self.samples.append(self.x.copy())
        self.samples = np.array(self.samples)

    def process(self):  # 展示抽样过程
        plt.pause(1)
        for i in range(self.samples.shape[0]):
            plt.plot(self.samples[:i, 0], self.samples[:i, 1], color='blue')
            plt.pause(0.01)
        plt.pause(3)

    def contrast(self):  # 与标准进行对比
        p1 = plt.subplot(1, 2, 1)
        p1.set_title('抽样结果')
        plt.axis([-5, 5, -5, 5])
        plt.scatter(self.samples[:, 0], self.samples[:, 1])
        p2 = plt.subplot(1, 2, 2)
        p2.set_title('标准')
        plt.axis([-5, 5, -5, 5])
        X = np.random.multivariate_normal(self.mean, self.cov, self.samples.shape[0])
        plt.scatter(X[:, 0], X[:, 1])
        plt.show()


J = Jibbs(100, 150)
J.sampling()
J.process()
