class AES:
    mix_c = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]

    mix_c_inv = [[0xe, 0xb, 0xd, 0x9], [0x9, 0xe, 0xb, 0xd], [0xd, 0x9, 0xe, 0xb], [0xb, 0xd, 0x9, 0xe]]

    RCon = [0x01000000, 0x02000000, 0x04000000, 0x08000000, 0x10000000, 0x20000000,
            0x40000000, 0x80000000, 0x1b000000, 0x36000000]

    s = [[0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76],
         [0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0],
         [0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15],
         [0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75],
         [0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84],
         [0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf],
         [0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8],
         [0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2],
         [0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73],
         [0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb],
         [0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79],
         [0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08],
         [0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a],
         [0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e],
         [0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf],
         [0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16]]

    s2 = [[0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb],
          [0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb],
          [0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e],
          [0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25],
          [0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92],
          [0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84],
          [0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06],
          [0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b],
          [0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73],
          [0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e],
          [0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b],
          [0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4],
          [0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f],
          [0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef],
          [0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61],
          [0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d]]

    # string to ascii
    def stoa(self, item):
        list = []
        for c in item:
            temp = hex(ord(c)).replace('0x', '')
            # 考虑到如果temp只有一位会影响后续解密
            if (len(temp) < 2):
                list.append("0" + temp)
            else:
                list.append(temp)
        return "".join(list)

    # ascii to string
    def atos(self, num):
        res = []
        for i in range(0, len(num), 2):
            temp = chr(int(num[i] + num[i + 1], 16))
            if(temp == '\r' or ord(temp) == 16):
                break
            else:
                res.append(chr(int(num[i] + num[i + 1], 16)))
        return "".join(res)

    # 10进制整型list -> 16进制string
    def normalization(self, list):
        ret = []
        for ele in list:
            if (ele < 16):
                ret.append(hex(ele).replace("0x", "0"))
            else:
                ret.append(hex(ele).replace("0x", ""))
        return "".join(ret)

    # 16bytes to int
    def btoi(cls, _16bytes):
        return int.from_bytes(_16bytes, byteorder='big')

    # int to 16bytes
    def itob(cls, num):
        return num.to_bytes(16, byteorder='big')

    # 打印ASCII码值
    def print_ascii(self, num):
        ascii = []
        for i in range(0, len(num), 2):
            ascii.append("0x" + num[i] + num[i + 1])
        print(ascii)

    # 字节替换
    def sub_bytes(self, state):
        return [self.s[i][j] for i, j in [(_ >> 4, _ & 0xF) for _ in state]]

    # 逆字节替换
    def sub_bytes_inv(self, state):
        return [self.s2[i][j] for i, j in [(_ >> 4, _ & 0xF) for _ in state]]

    # 行移位
    def shift_rows(self, S):
        return [S[0], S[5], S[10], S[15], S[4], S[9], S[14], S[3],
                S[8], S[13], S[2], S[7], S[12], S[1], S[6], S[11]]

    # 逆行移位
    def shift_rows_inv(self, S):
        return [S[0], S[13], S[10], S[7], S[4], S[1], S[14], S[11],
                S[8], S[5], S[2], S[15], S[12], S[9], S[6], S[3]]

    # 列混合
    def mix_cols(self, state):
        return self.matrix_mul(self.mix_c, state)

    # 逆列混合
    def mix_cols_inv(self, state):
        return self.matrix_mul(self.mix_c_inv, state)

    # 用于生成轮密钥的字移位
    def rot_word(self, _4byte_block):
        return ((_4byte_block & 0xffffff) << 8) + (_4byte_block >> 24)

    # 用于生成密钥的字节替换
    def sub_word(self, _4byte_block):
        result = 0
        for position in range(4):
            i = _4byte_block >> position * 8 + 4 & 0xf
            j = _4byte_block >> position * 8 & 0xf
            result ^= self.s[i][j] << position * 8
        return result

    # poly模多项式mod
    def mod(self, poly, mod=0b100011011):
        while poly.bit_length() > 8:
            poly ^= mod << poly.bit_length() - 9
        return poly

    # 多项式相乘
    def mul(self, poly1, poly2):
        result = 0
        for index in range(poly2.bit_length()):
            if poly2 & 1 << index:
                result ^= poly1 << index
        return result

    # 按位异或
    def xor(self, num1, num2):
        return num1 ^ num2

    # 用于列混合的矩阵相乘
    def matrix_mul(self, M1, M2):  # M1 = mix_c  M2 = state
        M = [0] * 16
        for row in range(4):
            for col in range(4):
                for round in range(4):
                    M[row + col * 4] ^= self.mul(M1[row][round], M2[round + col * 4])
                M[row + col * 4] = self.mod(M[row + col * 4])
        return M

    # 密钥扩展
    def extend_key(self, _16bytes_key):
        # w[0]~w[3]
        w = [_16bytes_key >> 96,
             _16bytes_key >> 64 & 0xFFFFFFFF,
             _16bytes_key >> 32 & 0xFFFFFFFF,
             _16bytes_key & 0xFFFFFFFF] + [0] * 40

        # w[4]~w[43]
        for i in range(4, 44):
            temp = w[i - 1]
            if not i % 4:
                temp = self.sub_word(self.rot_word(temp)) ^ self.RCon[i // 4 - 1]
            w[i] = w[i - 4] ^ temp

        # # 打印轮密钥
        # print("\n------round key generator------")
        # print("轮密钥为")
        # for i in range(44):
        #     print("w[", i, "] = ", '%#x' % w[i], end="\t")
        #     if i % 4 == 3:
        #         print()

        return [self.itob(
            sum([w[4 * i] << 96, w[4 * i + 1] << 64, w[4 * i + 2] << 32, w[4 * i + 3]]))
            for i in range(11)]

    # 轮密钥加
    def add_round(self, state, round_key, index):
        return [state[i] ^ round_key[index][i] for i in range(16)]

    # 更新向量
    def update(self, list):
        ret = []
        for ele in list:
            ret.append(str(ele))
        return int("".join(ret))

    # 加密算法
    def aes(self, plaintext, round_key):
        state = plaintext
        state = self.add_round(state, round_key, 0)
        for round in range(1, 10):
            state = self.sub_bytes(state)
            state = self.shift_rows(state)
            state = self.mix_cols(state)
            state = self.add_round(state, round_key, round)
            # print("state", round, ">>>>> ", [hex(_) for _ in state])
        state = self.sub_bytes(state)
        state = self.shift_rows(state)
        state = self.add_round(state, round_key, 10)
        # print("state", 10, ">>>>> ", [hex(_) for _ in state])
        return state

    # 解密算法
    def deaes(self, ciphertext, round_key):
        state = ciphertext
        state = self.add_round(state, round_key, 10)
        for round in range(1, 10):
            state = self.shift_rows_inv(state)
            state = self.sub_bytes_inv(state)
            state = self.add_round(state, round_key, 10 - round)
            state = self.mix_cols_inv(state)
        state = self.shift_rows_inv(state)
        state = self.sub_bytes_inv(state)
        state = self.add_round(state, round_key, 0)
        return state

    # 输入密钥并进行检查
    def check_key(self):
        print("------input key------")
        while 1:
            print("请输入16个字符的密钥: ")
            key = input()
            klen = len(key)
            if klen != 16:
                print("请输入16个字符的密钥,当前密钥的长度为", klen)
            else:
                print("你输入的密钥为: ", key)
                return key

    def check_iv(self):
        print("\n------input IV------")
        while 1:
            print("请输入16个字符的初始化向量: ")
            iv = input()
            ilen = len(iv)
            if ilen != 16:
                print("请输入16个字符的初始化向量,当前的长度为", ilen)
            else:
                print("你输入的初始化向量为: ", iv)
                return iv

    # 输入明文并检查
    def check_pt(self):
        print("\n------input plaintext------")
        print("请输入你的明文(明文字符长度必须为16的倍数): ")
        plaintext = input()
        print("你输入的明文为:", plaintext, "长度为", len(plaintext))

        # PKCS7 padding
        plaintext = self.PKCS7_padding(plaintext)
        print("PKCS7 padding...........finish")
        return plaintext

    def PKCS7_padding(self, s):
        num = 16 - (len(s) % 16)
        return s + num * chr(num)
    
    # 文件存储
    def store(self, content, filename):
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
            file.close()

    # 文件读取
    def read(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            file.close()
        return content

    # 加密过程
    def encryption(self, plaintext, iv, round_key):
        print("\n------encryption------")
        plaintext_list = []
        ciphertext_list = []
        plen = len(plaintext)
        vector = int(self.stoa(iv), 16)

        # 以16个为一组，拆解字符串
        for i in range(0, plen, 16):
            plaintext_list.append(plaintext[i:i + 16])

        # 处理字符串并加密
        for pt in plaintext_list:
            pt = int(self.stoa(pt), 16)             # 字符串 -> ascii码 -> 整型
            pt = self.xor(pt, vector)               # 明文和向量按bit异或
            pt = self.itob(pt)                      # 整型 -> 16字节
            ct = self.aes(pt, round_key)            # 加密的过程，ct是整型数组成的list
            ct = self.normalization(ct)             # list -> string
            vector = int(ct, 16)                    # 更新向量
            ciphertext_list.append(ct)

        # 打印ASCII码值
        print("ASCII: ciphertext = ", end="")
        self.print_ascii("".join(ciphertext_list))

        # 存储最终的密文
        store = self.atos("".join(ciphertext_list))
        self.store(store, 'ciphertext.txt')
        print("加密后的密文为", store, ", 密文已存入ciphertext.txt")

    # 解密过程
    def decryption(self, iv, round_key):
        print("\n------decryption------")

        # 获取密文
        print("从ciphertext.txt中获取到要解密的密文", end="")
        ciphertext = self.read('ciphertext.txt')
        print(ciphertext)

        plaintext_list = []
        ciphertext_list = []
        clen = len(ciphertext)
        vector = int(self.stoa(iv), 16)

        # 以16个为一组，拆解字符串
        for i in range(0, clen, 16):
            ciphertext_list.append(ciphertext[i:i + 16])

        for ct in ciphertext_list:
            ct = int(self.stoa(ct), 16)             # 字符串 -> 整型
            temp1 = ct                              # 记录该密文分组
            ct = self.itob(ct)                      # 整型 -> 16字节
            pt = self.deaes(ct, round_key)          # 解密
            pt = self.normalization(pt)             # list(10) -> string(16)
            temp = int(pt, 16)                      # string -> int
            pt = self.xor(temp, vector)             # 明文和向量按bit异或
            pt = hex(pt).replace("0x", "")
            vector = temp1                          # 更新向量
            plaintext_list.append(pt)

        # 打印ASCII码值
        print("ASCII: plaintext = ", end="")
        self.print_ascii("".join(plaintext_list))

        # 存储最终的明文
        store = self.atos("".join(plaintext_list))
        self.store(store, 'plaintext.txt')
        print("解密后的明文为", store, ", 明文已存入plaintext.txt")


if __name__ == '__main__':
    aes = AES()

    # 输入密钥并检查
    key = aes.check_key()

    # 输入初始化向量并检查
    iv = aes.check_iv()

    # 生成轮密钥
    key = int(aes.stoa(key), 16)
    round_key = aes.extend_key(key)

    # 输入明文并检查
    plaintext = aes.check_pt()

    # 进行加密
    aes.encryption(plaintext, iv, round_key)

    # 进行解密
    aes.decryption(iv, round_key)

