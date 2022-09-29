import random
import string
import sys

sys.setrecursionlimit(100000)  # 设置迭代次数限制


class RSA(object):
    # 生成对应的转换list
    # 数字为00-09，a-z = 10-35，A-Z = 36-61
    list1 = [_ for _ in range(62)]
    list2 = []
    for i in range(10):
        list2.append(str(i))
    for _ in string.ascii_letters:
        list2.append(_)
    ctoi_map = dict(zip(list2, list1))
    itoc_map = dict(zip(list1, list2))

    # 文件读取
    def read(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            file.close()
        return content

    # 写入文件
    def write(self, filename, message):
        with open(filename, "w") as file:
            file.write(message)
            file.close()
        return 0

    def ctoi(self, char):
        for key in rsa.ctoi_map.keys():
            if char == key:
                if rsa.ctoi_map[key] < 10:
                    return "0" + str(rsa.ctoi_map[key])
                else:
                    return str(rsa.ctoi_map[key])
        return str(99)

    def itoc(self, num):
        for key in rsa.itoc_map.keys():
            if num == key:
                return rsa.itoc_map[key]
            if num == 99:
                return ""

    def itos(self, num):
        if num < 100:       # 防止0被吞掉
            return "0" + rsa.itoc(num)
        else:
            string = "0" * (4 - len(str(num))) + str(num)
            return rsa.itoc(int(string[0:2])) + rsa.itoc(int(string[2:4]))

    # 计算(a^t)(mod n)
    def operation(self, a, t, n):
        temp = 1
        binstr = bin(t)[2:][::-1]  # 通过切片去掉开头的0b，截取后面，然后反转
        for item in binstr:
            if item == '1':
                temp = (temp * a) % n
                a = pow(a, 2) % n
            elif item == '0':
                a = pow(a, 2) % n
        return temp

    # 加密
    def encrypt(self, plaintext, e, n):
        ciphertext = []
        message = []
        length = len(str(n))
        if len(plaintext) % 2 != 0:
            plaintext = plaintext + " "
        for i in range(0, len(plaintext), 2):
            message.append(int(self.ctoi(plaintext[i]) + self.ctoi(plaintext[i + 1])))
        for ele in message:
            ct = str(self.operation(ele, e, n))
            if len(ct) != length:
                ct = '0' * (length - len(ct)) + ct  # 每一个密文分组，长度不够，高位补0
            ciphertext.append(ct)
        return "".join(ciphertext)
    
    # 解密
    def decrypt(self, ciphertext, d, n):
        length = len(str(n))
        message = []
        plaintext = []
        for i in range(0, len(ciphertext), length):
            message.append(ciphertext[i: i + length])
        for ele in message:
            pt = self.operation(int(ele), d, n)
            plaintext.append(self.itos(pt))
        return "".join(plaintext)

    # 利用Miller-Rabin算法检验生成的奇数是否为素数
    def miller_rabin(self, n):
        # 把n-1写成(2^k)*m, 其中m是一个奇数
        t = n - 1
        k = 0
        while t % 2 == 0:
            t = t // 2
            k = k + 1

        a = random.randint(2, n)  # 随机选择一个整数a，满足1<=a<=n-1
        b = self.operation(a, t, n)  # b = (a^t)(mod n)
        if b == 1:
            return 1
        for i in range(k):
            if b == n - 1:
                return 1
            else:
                b = pow(b, 2 % n)
        return 0

    # 生成大素数，20次miller_rabin算法缩小出错的概率
    def big_prime(self):
        min = pow(10, 39)
        max = pow(10, 40)
        while 1:
            p = random.randrange(min, max, 1)
            for i in range(20):
                if self.miller_rabin(p) == 0:
                    break
                elif i == 19:
                    return p

    # 求最大公因子
    def gcd(self, a, b):
        if a % b == 0:
            return b
        else:
            return self.gcd(b, a % b)
    
    # 求逆元
    def extended_euclid(self, x, n):
        r0 = n
        r1 = x % n
        if r1 == 1:
            return 1
        else:
            s0 = 1
            s1 = 0
            t0 = 0
            t1 = 1
            while r0 % r1 != 0:
                q = r0 // r1
                r = r0 % r1
                r0 = r1
                r1 = r
                s = s0 - q * s1
                s0 = s1
                s1 = s
                t = t0 - q * t1
                t0 = t1
                t1 = t
                if r == 1:
                    return (t + n) % n

    # 生成公钥和私钥
    def build_key(self):
        p = self.big_prime()        # 产生大素数p
        q = self.big_prime()        # 产生大素数q
        n = p * q                   # 计算p和q的乘积n
        fn = (p - 1) * (q - 1)      # n的欧拉函数
    
        # 选择一个和fn互质的e
        e = 11014
        while 1:
            if self.gcd(e, fn) == 1:
                break
            e = e - 1
    
        # 扩展欧几里得算法求逆元
        d = self.extended_euclid(e, fn)
    
        # 输出p、q、n、d、e
        print("p:", p)
        print("q:", q)
        print("n:", n)
        print("d:", d)
        print("e:", e)

        # 写入文件
        self.write('n.txt', str(hex(n))[2:])
        self.write('e.txt', str(hex(e))[2:])
        self.write('d.txt', str(hex(d))[2:])
        print("The keys are separately written to n.txt, e.txt, d.txt")


if __name__ == "__main__":
    rsa = RSA()

    print(" > > > building keys.......... Success!")
    rsa.build_key()  # 生成公钥、密钥

    print("\n > > > reading in plaintext.......... Success!")
    plaintext = rsa.read("plaintext.txt")
    print("The plaintext is:", plaintext)
    plaintext = plaintext.replace(" ","").replace(",", "").replace(".", "").replace("-", "")

    print("\n > > > reading in keys.......... Success!")
    e = int(rsa.read("e.txt"), 16)
    n = int(rsa.read("n.txt"), 16)
    d = int(rsa.read("d.txt"), 16)
    print("Read keys from e.txt, n.txt and d.txt")

    """------ encryption by public key and decryption by private key ------"""
    print("\n > > > encryption(public key).......... Success!")
    ciphertext = rsa.encrypt(plaintext, e, n)
    print("The ciphertext is:", ciphertext)

    rsa.write("ciphertext.txt", ciphertext)
    print("The ciphertext is written to ciphertext.txt")

    print("\n > > > decryption(private key).......... Success!")
    result = rsa.decrypt(ciphertext, d, n)
    print("The plaintext is:", result)

    rsa.write("result.txt", result)
    print("The plaintext is written to result.txt")

    """------ encryption by private key and decryption by public key ------"""
    print("\n > > > encryption(private key).......... Success!")
    ciphertext = rsa.encrypt(plaintext, d, n)
    print("The ciphertext is:", ciphertext)

    rsa.write("ciphertext.txt", ciphertext)
    print("The ciphertext is written to ciphertext.txt")

    print("\n > > > decryption(public key).......... Success!")
    result = rsa.decrypt(ciphertext, e, n)
    print("The plaintext is:", result)

    rsa.write("result.txt", result)
    print("The plaintext is written to result.txt")