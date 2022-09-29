import math
import re


possible_a = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]
possible_b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]


"""
加密算法 encryption(a, b, m)
c = (am + b)(mod 26
"""
def encryption(a, b, m):
    c = []

    for each_char in m:
        if(each_char == ' ' or each_char == ',' or each_char == '.'):
            c.append(each_char)
        elif(ord(each_char) < 97):
            temp = (a * (ord(each_char) - 65) + b) % 26
            c.append(chr(temp + 97))
        else:
            temp = (a * (ord(each_char) - 97) + b) % 26
            c.append(chr(temp + 97))
    res = ''.join(c)
    return res


"""
求解a的逆元算法 inverse
"""
def inverse(a):
    inv = 1
    while ((a * inv) % 26) != 1:
        inv = inv + 1
    return inv


"""
解密算法 decryption
m = ((a^(-1))(c-b))(mod 26)
"""
def decryption(a, b, c):
    m = []      # 存放解密后的内容

    for i in range(len(c)):
        if(c[i] == ' ' or c[i] == ',' or c[i] == '.'):
            m.append(c[i])
        else:
            temp = ord(c[i]) - 97 - b
            m.append(chr((a * temp) % 26 + 97))
    res = ''.join(m)
    return res


"""
统计字频函数 statistic(str)
"""
def statistic(str):
    frequency = {}

    # 计算字频并存入字典probability
    for i in range(26):
        char = chr(i + 97)
        # 字频 = 某个字符出现的次数 / 字符串总长
        frequency[char] = str.count(char) / len(str)
    return frequency


"""
欧几里得相似度计算算法 similarity
sum((f[i] - fstd[i]) * (f[i] - fstd[i]))
"""
def similarity(x, y):
    sum = 0
    for key in x.keys():
        sum = sum + (x[key] - y[key]) * (x[key] - y[key])
    return sum


"""
暴力破解算法 force(c)
"""
def force(c):
    record = {}       # 记录欧几里得相似度

    # 自然语言字频
    reality_frequency = dict(a=0.082, b=0.015, c=0.028, d=0.043, e=0.127, f=0.022, g=0.020, h=0.061, i=0.070,
                        j=0.002, k=0.008, l=0.040, m=0.024, n=0.067, o=0.075, p=0.019, q=0.001, r=0.060,
                        s=0.063, t=0.090, u=0.028, v=0.010, w=0.023, x=0.001, y=0.020, z=0.001)

    # 遍历每一个可能的a和b
    for i in range(len(possible_a)):
        inv_a = inverse(possible_a[i])  # 求解逆元
        for j in range(len(possible_b)):
            # 解密，结果计入res
            res = decryption(inv_a, possible_b[j], c)

            # 统计res的字频
            res_frequency = statistic(res)

            # 计算欧几里得相似度并存入record
            ab = (possible_a[i], possible_b[j], res)
            record[ab] = similarity(reality_frequency, res_frequency)
        else:
            continue

    # 对欧几里得相似度排序
    order = dict(sorted(record.items(), key=lambda x: x[1], reverse=False))

    # 十个和自然语言最相似的结果
    for i in range(11):
        best_key = list(order)[i]
        best_similarity = order[best_key]
        fin_a = best_key[0]
        fin_b = best_key[1]
        fin_res = best_key[2]
        print("------rank", i, "------")
        print("if a = " + str(fin_a) + " and b = " + str(fin_b)
              + ", the plaintext after decryption is: " + fin_res)
        print("the similarity is " + str(best_similarity))


if __name__ == "__main__":
    # 输入密钥
    print("Please input the secret keys a, b: ")
    x, y = input().split()
    a = int(x)
    b = int(y)

    # 进行约束条件判断
    while math.gcd(a, 26) != 1:
        print("Error: wrong input!")
        print("Please input the secret keys a, b: ")
        x, y = input().split()
        a = int(x)
        b = int(y)

    # 待加密明文
    m1 = "life is a journey, not the destination, but the scenery along the should be and the mood at the view."
    m2 = "Time goes by so fast, people go in and out of your life. You must never miss the opportunity to tell " \
         "these people how much they mean to you."
    m3 = "Accept what was and what is, and you will have more positive energy to pursue what will be."

    # 进行加密操作
    print("\n------encryption------")
    c1 = encryption(a, b, m1)
    print("m1: The ciphertext is: ", c1)
    c2 = encryption(a, b, m2)
    print("m2: The ciphertext is: ", c2)
    c3 = encryption(a, b, m3)
    print("m3: The ciphertext is: ", c3)

    # 已知密钥解密
    print("\n------decryption(known key)------")
    inv_a = inverse(a)
    res1 = decryption(inv_a, b, c1)
    print("res1: The plaintext is: ", res1)
    res2 = decryption(inv_a, b, c2)
    print("res2: The plaintext is: ", res2)
    res3 = decryption(inv_a, b, c3)
    print("res3: The plaintext is: ", res3)

    # 待解密密文
    c = "ne jobm og ne elmx zekfg ox qeuxnkoxg kxf ruojf rzofamg ox iknmzg. jotm, weu aobm qm lzmgguzm," \
        " o kjge weu qozkyjm! fex'n rm ktzkof ne gheen k goxajm hezgm. ihkn kreun rmoxa kjexm kxf rzkbm?" \
        " weu ykx yzw kjj nhm ikw, run weu ykx'n rm kxazw. weu hkbm ne ae nhzeuah nhm fkwg ihmx xerefw" \
        " ykzmg kreun on ne imjyeqm klljkugm kxf tjeimzg."

    # 暴力破解
    print("\n------decryption(unknown key)------")
    print("Decrypt the ciphertext: ")
    force(c)
