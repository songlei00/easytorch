from easytorch import tensor
import easytorch.functional as F
import abc


class Layer(metaclass=abc.ABCMeta):

    def __init__(self):
        self.params = []

    @abc.abstractmethod
    def forward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return self.params


class Linear(Layer):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = tensor.random(in_features, out_features)
        self.params.append(self.weight)
        if bias:
            self.bias = tensor.random(out_features)
            self.params.append(self.bias)
        else:
            self.bias = None

    def forward(self, x):
        y = x @ self.weight
        if self.bias:
            y += self.bias
        return y


class Sequential(Layer):

    def __init__(self, layers):
        super(Sequential, self).__init__()
        self.layers = layers
        for layer in layers:
            assert isinstance(layer, Layer)
            self.params.extend(layer.parameters())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ReLU(Layer):

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return F.relu(x)


class Tanh(Layer):

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        return F.tanh(x)
