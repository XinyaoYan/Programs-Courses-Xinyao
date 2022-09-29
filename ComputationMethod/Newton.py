import math


# Newton迭代法
# 输入：迭代次数n, 初值x0, 精度e1和e2
def Newton(n, x0, e1, e2):
    for i in range(1, n):
        F = f(x0)
        DF = df(x0)

        if(abs(F) < e1):
            print(x0)
            return
        if(abs(DF) < e2):
            print("error!")
            return

        x1 = x0 - F / DF
        Tol = abs(x1 - x0)

        # 比较是否满足精度要求
        if(Tol < e1):
            print(x1)
            return

        # 迭代
        x0 = x1
    print("error!")
    return


# 打印题号
print("-----T1(1)-----")
# 定义f(x)及其倒数df(x)，下同
f = lambda x: math.cos(x) - x
df = lambda x: - math.sin(x) - 1
Newton(10, 0.785398163, pow(10, -6), pow(10, -4))

print("-----T1(2)-----")
f = lambda x: math.exp(- x) - math.sin(x)
df = lambda x: - math.exp(- x) - math.cos(x)
Newton(10, 0.6, pow(10, -6), pow(10, -4))

print("-----T2(1)-----")
f = lambda x: x - math.exp(- x)
df = lambda x: 1 + math.exp(- x)
Newton(10, 0.6, pow(10, -6), pow(10, -4))

print("-----T2(2)-----")
f = lambda x: pow(x, 2) - 2 * x * math.exp(- x) + math.exp(-2 * x)
df = lambda x: 2 * x - 2 * math.exp(- x) + 2 * x * math.exp(- x) - 2 *  math.exp(-2 * x)
Newton(10, 0.5, pow(10, -6), pow(10, -4))