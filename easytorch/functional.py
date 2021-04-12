import numpy as np
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


def mse_loss(y, target_y):
    return ((y - target_y) * (y - target_y)).mean()
