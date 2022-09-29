import math


# RungeKutta算法
# 输入：定义域范围上下界a和b, y的初值alpha, N
def RungeKutta(a, b, alpha, N):
    x0 = a
    y0 = alpha
    h = (b - a) / N
    for n in range(N):
        K1 = h * f(x0, y0)
        K2 = h * f(x0 + h / 2, y0 + K1 / 2)
        K3 = h * f(x0 + h / 2, y0 + K2 / 2)
        K4 = h * f(x0 + h, y0 + K3)
        x1 = x0 + h
        y1 = y0 + 1 / 6 * (K1 + 2 * K2 + 2 * K3 + K4)

        # 打印结果
        print("x = ", float(format(x1, '.8f')), "y = ", float(format(y1, '.8f')))

        # 迭代
        x0 = x1
        y0 = y1

    return


# 打印题号
print("-----T1(1)-----")
# 定义f(x)，下同
f = lambda x, y: x + y
print("当 N = 5 时")
RungeKutta(0, 1, -1, 5)
print("当 N = 10 时")
RungeKutta(0, 1, -1, 10)
print("当 N = 20 时")
RungeKutta(0, 1, -1, 20)

print("-----T1(2)-----")
f = lambda x, y: -pow(y, 2)
print("当 N = 5 时")
RungeKutta(0, 1, 1, 5)
print("当 N = 10 时")
RungeKutta(0, 1, 1, 10)
print("当 N = 20 时")
RungeKutta(0, 1, 1, 20)

print("-----T2(1)-----")
f = lambda x, y: 2 / x * y + pow(x, 2) * math.exp(x)
print("当 N = 5 时")
RungeKutta(1, 3, 0, 5)
print("当 N = 10 时")
RungeKutta(1, 3, 0, 10)
print("当 N = 20 时")
RungeKutta(1, 3, 0, 20)

print("-----T2(2)-----")
f = lambda x, y: 1 / x * (pow(y, 2) + y)
print("当 N = 5 时")
RungeKutta(1, 3, -2, 5)
print("当 N = 10 时")	
RungeKutta(1, 3, -2, 10)
print("当 N = 20 时")
RungeKutta(1, 3, -2, 20)

print("-----T3(1)-----")
f = lambda x, y: -20 * (y - pow(x, 2)) + 2 * x
print("当 N = 5 时")
RungeKutta(0, 1, 1 / 3, 5)
print("当 N = 10 时")
RungeKutta(0, 1, 1 / 3, 10)
print("当 N = 20 时")
RungeKutta(0, 1, 1 / 3, 20)

print("-----T3(2)-----")
f = lambda x, y: -20 * y + 20 * math.sin(x) + math.cos(x)
print("当 N = 5 时")
RungeKutta(0, 1, 1, 5)
print("当 N = 10 时")
RungeKutta(0, 1, 1, 10)
print("当 N = 20 时")
RungeKutta(0, 1, 1, 20)

print("-----T3(3)-----")
f = lambda x, y: -20 * (y - math.exp(x) * math.sin(x)) + math.exp(x) * (math.sin(x) + math.cos(x))
print("当 N = 5 时")
RungeKutta(0, 1, 0, 5)
print("当 N = 10 时")
RungeKutta(0, 1, 0, 10)
print("当 N = 20 时")
RungeKutta(0, 1, 0, 20)

