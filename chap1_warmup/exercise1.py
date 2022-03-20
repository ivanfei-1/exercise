import numpy as np
import matplotlib.pyplot as plt

a = np.array([4, 5, 6])

print(a.dtype)
print(a.shape)
print(a[0])

b = np.array([[4, 5, 6], [1, 2, 3]])

print(b.shape)

print(b[0, 0], b[0, 1], b[1, 1])

c = np.zeros((3, 3), dtype=int)

print(c)

d = np.ones((4, 5), dtype=int)

print(d)

e = np.identity(4)

print("e = \n", e)

f = np.random.rand(3, 2)

print(f)

# 5

g = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print(g)

print(g[2, 3], g[0, 0])

# 6

h = g[0:2, 2:4]

print("h=", h)

print(h[0, 0])

# 7

i = g[1:, ...]

print("i=", i)
print(i[0, -1])

# 8

j = np.array([[1, 2], [3, 4], [5, 6]])

print(j)

print(j[[0, 1, 2], [0, 1, 0]])

# 9

k = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

k_ = np.array([0, 2, 0, 1])

print(k[np.arange(4), k_])

# 10

k[np.arange(4), k_] += 10

print("k''=", k[np.arange(4), k_])

# 11

L = np.array([1, 2])

print(L.dtype)

# 12

m = np.array([1.0, 2.0])

print(m.dtype)

# 13

n = np.array([[1, 2], [3, 4]], dtype=np.float64)

o = np.array([[5, 6], [7, 8]], dtype=np.float64)

print(n + o)

print(np.add(n, o))

# 14

print(n - o)

print(np.subtract(n, o))

# 15

print("15", n * o)

print(np.multiply(n, o))

print(np.dot(n, o))

# 16

print(n / o)

print(np.divide(n, o))

# 17

print(np.sqrt(n))

# 18

print(n.dot(o))

print(o.dot(n))

# 19

print("n = ", n)

print(np.sum(n))

print(np.sum(n, axis=0))

print(np.sum(n, axis=1))

# 20

print(np.mean(n))

print(np.mean(n, axis=0))

print(np.mean(n, axis=1))

# 21

print(n.T)

# 22

print(np.exp(n))

# 23

print(n)

print(np.argmax(n))

print(np.argmax(n, axis=0))

print(np.argmax(n, axis=1))

# 24

p = np.arange(0, 100, 0.1)

q = p * p

# plt.plot(p, q)

# 25

r = np.arange(0, 3 * np.pi, 0.1)

s = np.sin(r)

t = np.cos(r)

plt.plot(r, s)

plt.plot(r, t)

plt.show()