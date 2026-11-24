import numpy as np

a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)

print(f'a + b = {a+b}')
print(f'a . b = {np.dot(a, b)}')
print(f'|a| = {np.linalg.norm(a):.4f}')
print(f'cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f})')

W = np.random.rand(2, 3) * 0.1
x = np.array([1, 0.5, -0.3])

print(f'Wx: {W @ x}')

v = np.array([1,2,3])
u = np.array([1,1,1])

proj = (np.dot(u, v) / (np.dot(v, v) * v))
print(f'Projecao: {proj}')

#DETRERMINANTE COM NUMPY
a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

det_a = np.linalg.det(a)
print(det_a)
