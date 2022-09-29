import math
import numpy


# Romberg积分法
# 输入：积分上下限a和b, 迭代次数n, 精度e
def Romberg(a, b, n, e):
    res = numpy.zeros((n, n))   # 新建一个n阶全零矩阵
    res[0][0] = 0.5 * (b - a) * (f(a) + f(b))
    # print(res[0][0])
    for k in range(1, n, 1):
        fx = 0
        start = (b - a) / pow(2, k) + a
        times = pow(2, k) - 1
        for i in range(1, times + 1, 2):
            x = (start - a) * i + a
            fx = f(x) + fx
        # print("fx =", fx, "* (", b, "-", a, "/", pow(2, k - 1), ")")
        fx = fx * (b - a) / pow(2, k - 1)
        res[k][0] = 0.5 * (res[k - 1][0] + fx)
        for m in range(1, k + 1):
            res[k][m] = (pow(4, m) * res[k][m - 1] - res[k - 1][m - 1]) / (pow(4, m) - 1)
        if(abs(res[k][k] - res[k - 1][k - 1]) < e):
            return res
    return

print("-----T1(1)-----")
f = lambda x: pow(x, 2) * math.exp(x)
print("所以，T-数表为")
print(Romberg(0, 1, 5, pow(10, -6)))

print("-----T1(2)-----")
f = lambda x: math.exp(x) * math.sin(x)
print("所以，T-数表为")
print(Romberg(1, 3, 4, pow(10, -1)))

print("-----T1(3)-----")
f = lambda x: 4 / (1 + pow(x, 2))
print("所以，T-数表为")
print(Romberg(0, 1, 6, pow(10, -6)))

print("-----T1(4)-----")
f = lambda x: 1 / (1 + x)
print("所以，T-数表为")
print(Romberg(0, 1, 5, pow(10, -6)))
