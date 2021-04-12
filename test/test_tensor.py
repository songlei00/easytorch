import sys
sys.path.append('..')

import unittest
from easytorch.tensor import Tensor
from torch import tensor as torchTensor


def is_tensor_equal(leaves1, leaves2):
    ret = True
    for t1, t2 in zip(leaves1, leaves2):
        val_eq = ((t1.data - t2.detach().numpy()) < 1e-4).all()
        grad_eq = ((t1.grad - t2.grad.detach().numpy()) < 1e-4).all()
        requires_grad_eq = (t1.requires_grad == t2.requires_grad)
        ret = ret and val_eq and grad_eq and requires_grad_eq
    return ret


def print_leaves(leaves):
    print('-------------------------')
    for leaf in leaves:
        print(leaf.grad)
    print('-------------------------')


class TestTensor(unittest.TestCase):

    def run_test_case(self, case):
        leaves1 = case(Tensor)
        leaves2 = case(torchTensor)
        self.assertTrue(is_tensor_equal(leaves1, leaves2))

    def test_ops(self):
        def case_add(tensor):
            a = tensor([1., 2.], requires_grad=True)
            b = tensor([3., 4.], requires_grad=True)
            c = tensor([[5., 6.], [7., 8.]], requires_grad=True)
            leaves = [a, b, c]
            d = a + b + 10  # 相同尺寸的逐元素加法和标量加法
            d = d + c + c  # broadcast和两个操作数为同一个对象
            d = d.mean()
            d.backward()
            leaves = list(filter(lambda x: x.grad is not None, leaves))

            return leaves
        self.run_test_case(case_add)

        def case_sub(tensor):
            a = tensor([1., 2.], requires_grad=True)
            b = tensor([3., 4.], requires_grad=True)
            c = tensor([[5., 6.], [7., 8.]], requires_grad=True)
            leaves = [a, b, c]
            d = a - b - 10  # 相同尺寸的逐元素减法和标量减法
            d = d - c - c  # broadcast和两个操作数为同一个对象
            d = 100 - d
            d = d.sum()
            d.backward()
            leaves = list(filter(lambda x: x.grad is not None, leaves))

            return leaves
        self.run_test_case(case_sub)

        def case_mul(tensor):
            a = tensor([1., 2.], requires_grad=True)
            b = tensor([3., 4.], requires_grad=True)
            c = tensor([[5., 6.], [7., 8.]], requires_grad=True)
            leaves = [a, b, c]
            d = a * a
            d = d * c * c + a
            d = d.sum()
            d.backward()
            leaves = list(filter(lambda x: x.grad is not None, leaves))

            return leaves
        self.run_test_case(case_mul)

        def case1(tensor):
            a = tensor([1., 2.], requires_grad=True)
            b = tensor([3., 4.], requires_grad=True)
            c = tensor([[5., 6.], [7., 8.]], requires_grad=True)
            leaves = [a, b, c]

            d = 3*a + b + 1
            d = d * b
            d = d + 5*c / 20
            d = d.mean()
            d.backward()
            leaves = list(filter(lambda x: x.grad is not None, leaves))
            # print_leaves(leaves)

            return leaves
        self.run_test_case(case1)

    def test_dot(self):
        def case1(tensor):
            a = tensor([[1., 2.], [3., 4.]], requires_grad=True)
            b = tensor([[5., 6., 7.], [8., 9., 10.]], requires_grad=True)
            leaves = [a, b]
            c = a @ b
            c = c.sum()
            c.backward()
            return leaves
        self.run_test_case(case1)

        def case2(tensor):
            a = tensor([[1., 2.]], requires_grad=True)
            b = tensor([[5.], [8.]], requires_grad=True)
            leaves = [a, b]
            c = a @ b
            c = c.sum()
            c.backward()
            return leaves
        self.run_test_case(case2)

        def case3(tensor):
            a = tensor([1., 2.], requires_grad=True)
            b = tensor([3., 4.], requires_grad=True)
            leaves = [a, b]
            c = a @ b + b
            c = c.sum()
            c.backward()
            return leaves
        self.run_test_case(case3)

        def case4(tensor):
            a = tensor([1.], requires_grad=True)
            b = tensor([3.], requires_grad=True)
            leaves = [a, b]
            c = a @ b + b
            c = c.sum()
            c.backward()
            return leaves
        self.run_test_case(case4)

    def test_reshape(self):
        a = Tensor([1, 2])
        b = a.reshape(2, 1)
        a[0] = 10
        self.assertTrue(a.data[0], b.data[0][0])

        def case1(tensor):
            a = tensor([[1.], [2.]], requires_grad=True)
            b = tensor([3., 4.], requires_grad=True)
            leaves = [a, b]
            c = (2*a + 10).reshape(2)
            d = b * (c + 10)
            d = d.sum()
            d.backward()
            return leaves
        self.run_test_case(case1)

    def test_activation_func(self):
        def case_tanh(tensor):
            a = tensor([1., 2.], requires_grad=True)
            b = tensor([3., 4.], requires_grad=True)
            c = tensor([[5., 6.], [7., 8.]], requires_grad=True)
            leaves = [a, b, c]

            d = 3 * a + b + 1
            d = (d * b).tanh()
            d = (d + 5 * c / 20).tanh()
            d = d.mean()
            d.backward()
            leaves = list(filter(lambda x: x.grad is not None, leaves))

            return leaves
        self.run_test_case(case_tanh)

        def case_relu(tensor):
            a = tensor([1., 2.], requires_grad=True)
            b = tensor([3., 4.], requires_grad=True)
            c = tensor([[5., 6.], [7., 8.]], requires_grad=True)
            leaves = [a, b, c]

            d = 3 * a + b + 1
            d = (d * b).relu()
            d = (d + 5 * c / 20).relu()
            d = d.mean()
            d.backward()
            leaves = list(filter(lambda x: x.grad is not None, leaves))

            return leaves
        self.run_test_case(case_relu)

    def test_pow(self):
        def case(tensor):
            a = tensor([1., 2.], requires_grad=True)
            b = tensor([3., 4.], requires_grad=True)
            c = tensor([[5., 6.], [7., 8.]], requires_grad=True)
            leaves = [a, b, c]

            d = 3 * a.pow(5) + b + 1
            d = (d * b).pow(2).tanh()
            d = (d + 5 * c / 20).relu()
            d = d.mean()
            d.backward()
            leaves = list(filter(lambda x: x.grad is not None, leaves))

            return leaves
        self.run_test_case(case)

    def test_select(self):
        def case(tensor):
            a = tensor([1., 2.], requires_grad=True)
            b = tensor([3., 4.], requires_grad=True)
            c = tensor([[5., 6.], [7., 8.]], requires_grad=True)
            leaves = [a, b, c]
            d = a + b + c[0]
            d = d.mean()
            d.backward()
            leaves = list(filter(lambda x: x.grad is not None, leaves))

            return leaves
        self.run_test_case(case)


if __name__ == '__main__':
    unittest.main()
