import numpy as np
from collections import namedtuple


GRAD_NODE_FMT = namedtuple('grad_node', ['tensor', 'grad_fn'])


class Tensor:

    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data)
        self.requires_grad = requires_grad
        self.grad_node = []
        self.grad = None

    def __repr__(self):
        s = 'tensor({}'.format(self.data)
        if self.grad_node:
            s += ', grad_fn=<{}>)'.format(self.grad_node[0].grad_fn.__name__)
        elif self.requires_grad:
            s += ', requires_grad=True)'
        else:
            s += ')'
        return s

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    @property
    def shape(self):
        return self.data.shape

    def reshape(self, *shape):
        old_shape = self.data.shape
        t = Tensor(self.data.reshape(shape), self.requires_grad)
        if self.requires_grad:
            def ViewBackward(grad):
                grad = grad.reshape(old_shape)
                return grad
            t.grad_node.append(GRAD_NODE_FMT(self, ViewBackward))

        return t

    def backward(self, gradient=None):
        if not self.requires_grad:
            raise RuntimeError('tensor does not require grad')
        if self.grad is None:
            if self.data.shape == () or self.data.shape == (1, ):
                self.grad = np.ones(1)
            else:
                print(self.data.shape)
                raise RuntimeError('grad can be implicitly created only for scalar outputs')

        for node in self.grad_node:
            if node.tensor.grad is None:
                node.tensor.grad = node.grad_fn(self.grad)
            else:
                node.tensor.grad += node.grad_fn(self.grad)
            node.tensor.backward()

    def __add__(self, other):
        other = Tensor.astensor(other)
        data = self.data + other.data
        requires_grad = self.requires_grad or other.requires_grad
        t = Tensor(data, requires_grad)

        if self.requires_grad:
            def AddBackward(grad):
                grad = grad * np.ones_like(self.data)
                for _ in range(grad.ndim - self.data.ndim):
                    grad = grad.sum(axis=0)
                for i, d in enumerate(self.data.shape):
                    if d == 1:
                        grad = grad.sum(axis=i, keepdims=True)

                assert grad.shape == self.data.shape, 'AddBackward, grad.shape != data.shape'
                return grad
            t.grad_node.append(GRAD_NODE_FMT(self, AddBackward))

        if other.requires_grad:
            def AddBackward(grad):
                grad = grad * np.ones_like(other.data)
                for _ in range(grad.ndim - other.data.ndim):
                    grad = grad.sum(axis=0)

                for i, d in enumerate(other.data.shape):
                    if d == 1:
                        grad = grad.sum(axis=i, keepdims=True)

                assert grad.shape == other.data.shape, 'AddBackward, grad.shape != data.shape'
                return grad

            t.grad_node.append(GRAD_NODE_FMT(other, AddBackward))

        return t

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        return self + other

    def __sub__(self, other):
        # TODO: 重新写sub函数，目前的sub记录的grad_fn为AddBackward
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __isub__(self, other):
        return self - other

    def __neg__(self):
        data = - self.data
        requires_grad = self.requires_grad
        t = Tensor(data, requires_grad)

        if requires_grad:
            def NegBackward(grad):
                return -grad
            t.grad_node.append(GRAD_NODE_FMT(self, NegBackward))

        return t

    def __mul__(self, other):
        other = Tensor.astensor(other)
        data = self.data * other.data
        requires_grad = self.requires_grad or other.requires_grad
        t = Tensor(data, requires_grad)

        if requires_grad:
            def MulBackward(grad):
                grad = grad * other.data

                for _ in range(grad.ndim - self.data.ndim):
                    grad = grad.sum(0)
                for i, d in enumerate(self.data.shape):
                    if d == 1:
                        grad = grad.sum(axis=i, keepdims=True)

                assert grad.shape == self.data.shape, 'MulBackward, grad.shape != data.shape'
                return grad
            t.grad_node.append(GRAD_NODE_FMT(self, MulBackward))

        if other.requires_grad:
            def MulBackward(grad):
                grad = grad * self.data

                for _ in range(grad.ndim - other.data.ndim):
                    grad = grad.sum(0)
                for i, d in enumerate(self.data.shape):
                    if d == 1:
                        grad = grad.sum(axis=i, keepdims=True)

                assert grad.shape == other.data.shape, 'MulBackward, grad.shape != data.shape'
                return grad
            t.grad_node.append(GRAD_NODE_FMT(other, MulBackward))

        return t

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = Tensor.astensor(other)
        data = self.data / other.data
        requires_grad = self.requires_grad or other.requires_grad
        t = Tensor(data, requires_grad)

        if self.requires_grad:
            def DivBackward(grad):
                grad = grad / other.data

                for _ in range(grad.ndim - self.data.ndim):
                    grad = grad.sum(0)
                for i, d in enumerate(self.data.shape):
                    if d == 1:
                        grad = grad.sum(axis=i, keepdims=True)

                assert grad.shape == self.data.shape, 'DivBackward, grad.shape != data.shape'
                return grad
            t.grad_node.append(GRAD_NODE_FMT(self, DivBackward))

        if other.requires_grad:
            def DivBackward(grad):
                grad = - (self.data * grad) / (other.data**2)

                for _ in range(grad.ndim - other.data.ndim):
                    grad = grad.sum(0)
                for i, d in enumerate(other.shape):
                    if d == 1:
                        grad = grad.sum(axis=i, keepdims=True)

                assert grad.shape == other.data.shape, 'DivBackward, grad.shape != data.shape'
                return grad
            t.grad_node.append(GRAD_NODE_FMT(other, DivBackward))

        return t

    def __floordiv__(self, other):
        raise NotImplementedError('__floordiv__ not implemented')

    def sum(self, dim=None, keepdim=False):
        data = self.data.sum(axis=dim, keepdims=keepdim)
        requires_grad = self.requires_grad
        t = Tensor(data, requires_grad)

        if self.requires_grad:
            def SumBackward(grad):
                grad = grad * np.ones_like(self.data)
                return grad
            t.grad_node.append(GRAD_NODE_FMT(self, SumBackward))

        return t

    def __matmul__(self, other):
        other = Tensor.astensor(other)

        if self.data.ndim == 1 and other.data.ndim == 1:
            # TODO: fix this bug
            raise NotImplementedError('Special case for mul')

        data = self.data @ other.data
        requires_grad = self.requires_grad or other.requires_grad
        t = Tensor(data, requires_grad)

        if self.requires_grad:
            def DotBackward(grad):
                # d = other.data.reshape(-1, 1).T if other.data.ndim == 1 else other.data.T
                grad = grad @ other.data.T
                assert grad.shape == self.data.shape, 'DotBackward, grad.shape != data.shape'
                return grad
            t.grad_node.append(GRAD_NODE_FMT(self, DotBackward))

        if other.requires_grad:
            def DotBackward(grad):
                # d = self.data.reshape(-1, 1) if other.data.ndim == 1 else self.data.T
                grad = self.data.T @ grad
                assert grad.shape == other.data.shape, 'DotBackward, grad.shape != data.shape'
                return grad
            t.grad_node.append(GRAD_NODE_FMT(other, DotBackward))

        return t

    @staticmethod
    def astensor(data):
        if not isinstance(data, Tensor):
            data = Tensor(data)
        return data
