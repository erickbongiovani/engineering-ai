from value import Value
import math
from torch import relu, torch, tensor

x = Value(2.0)
y = pow(x, 3)
y.backward()

print(x.grad)


def tang(x):
    t = math.tanh(x.data)
    saida = Value(t, children=[x])

    def _backward():
        derivada_local = 1.0 - (t ** 2)
        x.grad += derivada_local * saida.grad

    saida._backward = _backward
    return saida

x0 = Value(0.0)
y0 = tang(x0)
y0.backward()

print(x0.grad)

x1 = Value(2.0)
y1 = tang(x1)
y1.backward()
print(f'{x1.grad:.4f}')


w1 = 2.0
x1 = 3.0
w2 = 1.5
x2 = -1.0
b = 1.0

w1_pt = tensor(w1, requires_grad=True)
x1_pt = tensor(x1, requires_grad=True)
w2_pt = tensor(w2, requires_grad=True)
x2_pt = tensor(x2, requires_grad=True)
b_pt  = tensor(b, requires_grad=True)


y = relu(w1_pt * x1_pt + w2_pt * x2_pt + b_pt)
y.backward()

print(f"dy/dx1 -> PyTorch: {x1_pt.grad.item():5.1f} (esperado: w1 = {w1})")
print(f'dy/dw1 -> PyTorch: {w1_pt.grad.item():5.1f} (esperado: x1 = {x1}')
print(f"dy/dw2 -> PyTorch: {w2_pt.grad.item():5.1f} (esperado: x2 = {x2})")
print(f"dy/dx2 -> PyTorch: {x2_pt.grad.item():5.1f} (esperado: w2 = {w2})")
print(f"dy/db  -> PyTorch: {b_pt.grad.item():5.1f} (esperado: 1.0")