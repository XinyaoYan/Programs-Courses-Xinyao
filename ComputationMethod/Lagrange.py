import math


# Lagrange插值多项式算法
# 输入：步长h, 插值区间下界start, 插值次数n, 插值点x0
def Lagrange(h, start, n, x0):
    y = 0.0
    k = 0
    x = []  # 定义一个空的list

    # 将所有插值点一次记录在x中
    for i in range(n + 1):
        x.append(float(format(start + i * h / n, '.8f')))
    # print(x)

    # 用插值多项式进行近似计算
    while(k <= n):
        l = 1.0
        for j in range(n + 1):
            if(j != k):
                # print("l =", l, " * (", x0, " -", x[j], ") / (", x[k], " -", x[j], ")")
                l = l * (x0 - x[j]) / (x[k] - x[j])
        # print(l)
        y = y + l * f(x[k])
        k = k + 1
    return y    # 返回近似值


# 打印题号
print("-----T1(1)-----")
# 定义f(x)，下同
f = lambda x: 1 / (1 + pow(x, 2))
print("当n=5时，Pn(x)在 x=0.75 , x=1.75 , x=2.75 , x=3.75 , x=4.75 处的函数值分别为")
print("              ", Lagrange(10, -5, 5, 0.75), Lagrange(10, -5, 5, 1.75), Lagrange(10, -5, 5, 2.75), Lagrange(10, -5, 5, 3.75), Lagrange(10, -5, 5, 4.75))
print("当n=10时，Pn(x)在 x=0.75 , x=1.75 , x=2.75 , x=3.75 , x=4.75 处的函数值分别为")
print("              ", Lagrange(10, -5, 10, 0.75), Lagrange(10, -5, 10, 1.75), Lagrange(10, -5, 10, 2.75), Lagrange(10, -5, 10, 3.75), Lagrange(10, -5, 10, 4.75))
print("当n=20时，Pn(x)在 x=0.75 , x=1.75 , x=2.75 , x=3.75 , x=4.75 处的函数值分别为")
print("              ", Lagrange(10, -5, 20, 0.75), Lagrange(10, -5, 20, 1.75), Lagrange(10, -5, 20, 2.75), Lagrange(10, -5, 20, 3.75), Lagrange(10, -5, 20, 4.75))

print("-----T1(2)-----")
f = lambda x: math.exp(x)
print("当n=5时，Pn(x)在 x=-0.95 , x=-0.05 , x=0.05 , x=0.95 处的函数值分别为")
print("              ", Lagrange(2, -1, 5, -0.95), Lagrange(2, -1, 5, -0.05), Lagrange(2, -1, 5, 0.05), Lagrange(2, -1, 5, 0.95))
print("当n=10时，Pn(x)在 x=-0.95 , x=-0.05 , x=0.05 , x=0.95 处的函数值分别为")
print("              ", Lagrange(2, -1, 10, -0.95), Lagrange(2, -1, 10, -0.05), Lagrange(2, -1, 10, 0.05), Lagrange(2, -1, 10, 0.95))
print("当n=20时，Pn(x)在 x=-0.95 , x=-0.05 , x=0.05 , x=0.95 处的函数值分别为")
print("              ", Lagrange(2, -1, 20, -0.95), Lagrange(2, -1, 20, -0.05), Lagrange(2, -1, 20, 0.05), Lagrange(2, -1, 20, 0.95))

print("-----T2(1)-----")
f = lambda x: 1 / (1 + pow(x, 2))    # 输入函数f(x)
print("当n=5时，Pn(x)在 x=-0.95 , x=-0.05 , x=0.05 , x=0.95 处的函数值分别为")
print("              ", Lagrange(2, -1, 5, -0.95), Lagrange(2, -1, 5, -0.05), Lagrange(2, -1, 5, 0.05), Lagrange(2, -1, 5, 0.95))
print("当n=10时，Pn(x)在 x=-0.95 , x=-0.05 , x=0.05 , x=0.95 处的函数值分别为")
print("              ", Lagrange(2, -1, 10, -0.95), Lagrange(2, -1, 10, -0.05), Lagrange(2, -1, 10, 0.05), Lagrange(2, -1, 10, 0.95))
print("当n=20时，Pn(x)在 x=-0.95 , x=-0.05 , x=0.05 , x=0.95 处的函数值分别为")
print("              ", Lagrange(2, -1, 20, -0.95), Lagrange(2, -1, 20, -0.05), Lagrange(2, -1, 20, 0.05), Lagrange(2, -1, 20, 0.95))

print("-----T2(2)-----")
f = lambda x: math.exp(x)
print("当n=5时，Pn(x)在 x=-4.75 , x=-0.25 , x=0.25 , x=4.75 处的函数值分别为")
print("              ", Lagrange(10, -5, 5, -4.75), Lagrange(10, -5, 5, -0.25), Lagrange(10, -5, 5, 0.25), Lagrange(10, -5, 5, 4.75))
print("当n=10时，Pn(x)在 x=-4.75 , x=-0.25 , x=0.25 , x=4.75 处的函数值分别为")
print("              ", Lagrange(10, -5, 10, -4.75), Lagrange(10, -5, 10, -0.25), Lagrange(10, -5, 10, 0.25), Lagrange(10, -5, 10, 4.75))
print("当n=20时，Pn(x)在 x=-4.75 , x=-0.25 , x=0.25 , x=4.75 处的函数值分别为")
print("              ", Lagrange(10, -5, 20, -4.75), Lagrange(10, -5, 20, -0.25), Lagrange(10, -5, 20, 0.25), Lagrange(10, -5, 20, 4.75))

f = lambda x: pow(x, 0.5)
res = lambda x0, x1, x2, x: f(x0) * (x - x1) * (x - x2) / (x0 - x1) / (x0 - x2) + f(x1) * (x - x0) * (x - x2) / (x1 - x0) / (x1 - x2) + f(x2) * (x - x1) * (x - x0) / (x2 - x1) / (x2 - x0)
print("-----T4(1)-----")
print("Pn(x)在 x=5 , x=50 , x=115 , x=185 处的函数值分别为")
print("       ", res(1, 4, 9, 5), res(1, 4, 9, 50), res(1, 4, 9, 115), res(1, 4, 9, 185))

print("-----T4(2)----")
print("Pn(x)在 x=5 , x=50 , x=115 , x=185 处的函数值分别为")
print("       ", res(36, 49, 64, 5), res(36, 49, 64, 50), res(36, 49, 64, 115), res(36, 49, 64, 185))

print("-----T4(3)----")
print("Pn(x)在 x=5 , x=50 , x=115 , x=185 处的函数值分别为")
print("       ", res(100, 121, 144, 5), res(100, 121, 144, 50), res(100, 121, 144, 115), res(100, 121, 144, 185))

print("-----T4(4)----")
print("Pn(x)在 x=5 , x=50 , x=115 , x=185 处的函数值分别为")
print("       ", res(169, 196, 225, 5), res(169, 196, 225, 50), res(169, 196, 225, 115), res(169, 196, 225, 185))





