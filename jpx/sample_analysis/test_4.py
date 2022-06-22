import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus'] = False


class MCMC:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.x = 2.5
        self.samples = []

    def p(self, x):  # 假设f1为我们想要进行抽样的分布
        return 0.3/np.sqrt(np.pi*2)*np.exp(-(x+2)**2/2) + 0.7/np.sqrt(np.pi*2)*np.exp(-(x-4)**2/2)

    def q(self, m, x):
        return 1/np.sqrt(2*np.pi)*np.exp(-(x-m)**2/2)

    def plot(self):
        x = np.arange(-10, 10, 0.01)
        plt.plot(x, self.p(x), linewidth=3)

    def sampling(self):
        for i in range(self.m):
            x = np.random.normal(self.x, 1)
            t = min(1, self.p(x)*self.q(x, self.x)/(self.p(self.x)*self.q(self.x, x)))
            u = np.random.random()
            if u < t:
                self.x = x
        for i in range(self.n):
            x = np.random.normal(self.x, 1)
            t = min(1, self.p(x) * self.q(x, self.x) / (self.p(self.x) * self.q(self.x, x)))
            u = np.random.random()
            if u < t:
                self.x = x
            self.samples.append(self.x)

    def test(self):
        plt.hist(self.samples, bins=200, density=True,stacked=True)
        plt.show()


M = MCMC(10000, 500000)
M.sampling()
M.plot()
M.test()