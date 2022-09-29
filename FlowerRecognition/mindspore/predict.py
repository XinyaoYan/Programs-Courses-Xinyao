import numpy as np
from PIL import Image
from easydict import EasyDict
from matplotlib import pyplot as plt
import mindspore  
from mindspore import Tensor, load_checkpoint
from mindspore.train import Model
from recognition_yxy import MyNet


# 配置参数
cfg = EasyDict({
    'channel': 3,                                   # 输入图像通道数
    'num_class': 6,                                 # 数据集类别数
    'dropout_ratio': 0.5                            # dropout比率
})

for i in range(1,7):
    img_path = './TestImages/test' + str(i) + '.jpg'
    
    # 读取图像
    img = Image.open(fp=img_path)
    
    # 调整大小
    temp = img.resize((200, 200))
   
    # 转换成numpy格式
    temp = np.array(temp)
    
    # 将HWC格式转化成CHW格式
    temp = temp.transpose(2, 0, 1)
    
    # 增加一个batch维度
    temp = np.expand_dims(temp, 0)
    
    # 将图像转成向量
    img_tensor = Tensor(temp, dtype=mindspore.float32)

    # 实例化网络结构
    net = MyNet(num_class=cfg.num_class,
                channel=cfg.channel,
                init_sigma=0.1)
    
    # 权重路径
    CKPT = './model/checkpoint-1_320.ckpt'
    
    # 网络加载权重
    load_checkpoint(CKPT, net=net)
    
    # 实例化模型
    model = Model(net)
    
    # 6类花分别对应的标签值
    class_names = {0: 'bee_balm', 1: 'blackberry_lily', 2: 'blanket_flower', 3: 'bougainvillea', 4: 'bromelia', 5: 'foxglove'}
    
    # 网络预测
    predictions = model.predict(img_tensor)
    
    # 将预测结果转成numpy格式
    predictions = predictions.asnumpy()
    
    # 获取预测结果最大值所对应的索引，根据索引获取类别名称
    label = class_names[np.argmax(predictions)]
    
    # 展示预测结果
    print("预测结果:{}".format(label))
    