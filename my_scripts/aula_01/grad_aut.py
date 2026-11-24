import torch
from value import Value

x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)

a = x1 * x2
b = a + 1.0
y = torch.relu(b)
y.backward()

print(f'Pytorch dy/dx1 = {x1.grad.item()}')
print(f'Pytorch dy/dx2 = {x2.grad.item()}')

a = Value(2.0)
b = Value(-3)
c = Value(10.0)

f = ((a * b) + c).relu()
f.backward()

print(f'df/da = {a.grad}')
print(f'df/db = {b.grad}')
print(f'df/dc = {c.grad}')