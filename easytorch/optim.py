import numpy as np
import abc


class Optimizer(metaclass=abc.ABCMeta):

    def __init__(self, params, lr=3e-4):
        self.params = params
        self.lr = lr
        self.V = []
        self.m = []
        for param in self.params:
            self.V.append(np.zeros_like(param.data))
            self.m.append(np.zeros_like(param.data))

    def zero_grad(self):
        for param in self.params:
            param.grad = 0

    @abc.abstractmethod
    def step(self):
        pass


class SGD(Optimizer):

    def __init__(self, params, lr=3e-4):
        super(SGD, self).__init__(params, lr)

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad


class Adagrad(Optimizer):

    def __init__(self, params, lr=1e-2):
        super(Adagrad, self).__init__(params, lr)

    def step(self):
        for i in range(len(self.params)):
            self.V[i] += self.params[i].grad * self.params[i].grad
            self.params[i].data -= self.lr * self.params[i].grad / (self.V[i] + 1e-6)


class Moment(Optimizer):

    def __init__(self, params, lr=3e-4, beta=0.9):
        super(Moment, self).__init__(params, lr)
        self.beta = beta

    def step(self):
        for i in range(len(self.params)):
            self.m[i] = self.beta * self.m[i] + (1 - self.beta) * self.params[i].grad
            self.params[i].data -= self.lr * self.m[i]


class RMSprop(Optimizer):

    def __init__(self, params, lr=1e-2, alpha=0.99):
        super(RMSprop, self).__init__(params, lr)
        self.alpha = alpha

    def step(self):
        for i in range(len(self.params)):
            self.V[i] = self.alpha * self.V[i] + (1 - self.alpha) * (self.params[i].grad * self.params[i].grad)
            self.params[i].data -= self.lr * self.params[i].grad / (self.V[i] + 1e-6)


class Adam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999)):
        super(Adam, self).__init__(params, lr)
        self.betas = betas

    def step(self):
        for i in range(len(self.params)):
            self.m[i] = self.betas[0] * self.m[i] + (1-self.betas[0]) * self.params[i].grad
            self.V[i] = self.betas[1] * self.V[i] + (1-self.betas[1]) * (self.params[i].grad * self.params[i].grad)
            self.params[i].data -= self.lr * self.m[i] / (self.V[i] + 1e-6)
