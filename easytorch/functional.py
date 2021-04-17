import numpy as np
import warnings
from easytorch import tensor


def tanh(inputs):
    data = np.tanh(inputs.data)
    requires_grad = inputs.requires_grad
    t = tensor.Tensor(data, requires_grad)
    t.is_leaf = False

    if inputs.requires_grad:
        def TanhBackward(grad):
            return grad * (1 - np.tanh(inputs.data) ** 2)
        t.grad_node.append(tensor.GRAD_NODE_FMT(inputs, TanhBackward))

    return t


def relu(inputs):
    data = np.maximum(0, inputs.data)
    requires_grad = inputs.requires_grad
    t = tensor.Tensor(data, requires_grad)
    t.is_leaf = False

    if inputs.requires_grad:
        def ReluBackward(grad):
            relu_prime = np.zeros_like(inputs.data)
            relu_prime[inputs.data > 0] = 1
            return grad * relu_prime

        t.grad_node.append(tensor.GRAD_NODE_FMT(inputs, ReluBackward))

    return t


def softmax(inputs, dim=0):
    raise NotImplementedError('There is a bug')
    def softmax_func(x):
        max_v = np.max(x)
        return np.e**(x - max_v) / np.sum(np.e**(x - max_v))
    assert inputs.data.ndim == 1 or (inputs.data.ndim == 2 and (inputs.data.shape[0] == 1 or inputs.data.shape[1] == 1))
    # data = np.apply_over_axes(softmax_func, dim, inputs.data)
    data = softmax_func(inputs.data)
    requires_grad = inputs.requires_grad
    t = tensor.Tensor(data, requires_grad)
    t.is_leaf = False

    if inputs.requires_grad:
        def SoftmaxBackward(grad):
            result = softmax_func(inputs.data)
            length = inputs.data.reshape(-1).shape[0]
            mat = np.zeros((length, length))
            for i in range(length):
                for j in range(length):
                    if i == j:
                        mat[i][j] = result[i]*(1 - result[i])
                    else:
                        mat[i][j] = result[i] * result[j]
            print('mat')
            print(mat)
            print('grad')
            print(grad)
            next_grad = mat @ grad
            return next_grad
        t.grad_node.append(tensor.GRAD_NODE_FMT(inputs, SoftmaxBackward))

    return t


def mse_loss(target_y, y):
    if y.shape != target_y.shape:
        warnings.warn('mse_loss, target size {} is different from input size {}, '
                      'this will likely lead to incorrect results due to broadcasting'.format(target_y.shape, y.shape))
    return ((y - target_y) * (y - target_y)).mean()


def l1_loss(target_y, y):
    if y.shape != target_y.shape:
        warnings.warn('mse_loss, target size {} is different from input size {}, '
                      'this will likely lead to incorrect results due to broadcasting'.format(target_y.shape, y.shape))
    return (y - target_y).abs().mean()
