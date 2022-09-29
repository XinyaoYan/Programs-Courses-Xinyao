# - * - coding: utf-8 - * -
# 导入模块
from easydict import EasyDict
from mindspore import context, dataset, nn, Tensor
from mindspore.common import dtype
from mindspore.train import Model
from mindspore.dataset.vision import Inter
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.common.initializer import TruncatedNormal
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
import numpy as np
from PIL import Image

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


# 配置参数
cfg = EasyDict({
    'train_path': './flower_photos/',   # 训练集路径
    'data_size': 10000,                 # 数据总量（张）
    'resize_width': 200,                # 图像宽
    'resize_height': 200,               # 图像高
    'rescale': 1.0 / 255.0,
    'shift': 0.0,
    'rescale_nml': 1 / 0.3081,
    'shift_nml': -1 * 0.1307 / 0.3081,
    'batch_size': 32,                   # 每个batch的大小
    'repeat_size': 20,                  # 扩充训练集的倍数
    'epoch_size': 1,                    # 训练模型次数
    'channel': 3,                       # 输入图像通道数
    'num_class': 6,                     # 数据集类别数
    'num_parallel_workers': 1,
    'learning_rate': 0.0001,            # 学习率
    'sigma': 0.01,                      # 权重初始化参数
    'save_checkpoint_steps': 1,         # 多少步保存一次模型
    'keep_checkpoint_max': 1,           # 最多保存多少个模型
    'output_directory': './model',      # 保存模型路径
    'output_prefix': "checkpoint"       # 保存模型文件名字
})


class DatasetGenerator:
    '''
    图片集合到数据集的映射
    '''
    def __init__(self):
        self.image = []
        self.label = []
        for i in range(1, 7):
            image = Image.open('./TestImages/test' + str(i) + '.jpg')
            image = np.array(image)
            self.image.append(image)
            self.label.append(i - 1)

    def __getitem__(self, index):
        return self.image[index], self.label[index]

    def __len__(self):
        return len(self.image)


def create_train_dataset(data_path, batch_size, repeat_size, num_parallel_workers):
    '''
    生成训练集
    '''
    # 读取训练集图片
    train_dataset = dataset.ImageFolderDataset(data_path, shuffle=True)
    # 定义所需要操作的map映射
    type_cast_op = C.TypeCast(dtype.int32)
    decode_op = CV.Decode()
    # 使用map映射函数，将数据操作应用到数据集
    train_dataset = train_dataset.map(operations=type_cast_op, 
                                      input_columns="label", 
                                      num_parallel_workers=num_parallel_workers)
    train_dataset = train_dataset.map(operations=decode_op,
                                      input_columns="image",
                                      num_parallel_workers=num_parallel_workers)
    # 继续预处理操作
    train_dataset = pretreat_dataset(train_dataset, num_parallel_workers)
    # 打乱数据并打包
    train_dataset = train_dataset.shuffle(buffer_size=cfg.data_size)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.repeat(count=repeat_size)
    return train_dataset


def create_test_dataset():
    '''
    生成训练集
    '''
    # 读取测试集图片
    test_dataset = dataset.GeneratorDataset(DatasetGenerator(), ["image", "label"], shuffle=False)
    # 完成预处理操作
    test_dataset = pretreat_dataset(test_dataset, num_parallel_workers=1)
    # 打包测试集
    test_dataset = test_dataset.batch(batch_size=1, drop_remainder=True)
    return test_dataset


def pretreat_dataset(ds, num_parallel_workers):
    '''
    完成creat_train_dataset和creat_test_dataset都需要的数据预处理操作
    '''
    # 定义所需要操作的map映射
    resize_op = CV.Resize((cfg.resize_height, cfg.resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(cfg.rescale_nml, cfg.shift_nml)
    rescale_op = CV.Rescale(cfg.rescale, cfg.shift)
    hwc2chw_op = CV.HWC2CHW()
    # 使用map映射函数，将数据操作应用到数据集
    ds = ds.map(operations=[resize_op, rescale_op, rescale_nml_op, hwc2chw_op],
                input_columns="image",
                num_parallel_workers=num_parallel_workers)
    return ds  # 返回处理过的数据集


class MyNet(nn.Cell):
    '''
    定义神经网络结构 LeNet-5
    '''
    def __init__(self, num_class, channel, init_sigma):
        super(MyNet, self).__init__()
        # 卷积层 - nn.Conv2d(in_channels, out_channels, kernel_size, ......)
        self.conv1 = nn.Conv2d(channel, 6, 5, weight_init=TruncatedNormal(sigma=init_sigma))
        self.conv2 = nn.Conv2d(6, 16, 5, weight_init=TruncatedNormal(sigma=init_sigma))
        # 激活层
        self.relu = nn.ReLU()
        # 池化层
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 平滑层
        self.flatten = nn.Flatten()
        # 连接层
        self.fc1 = nn.Dense(50 * 50 * 16, 1024, weight_init=TruncatedNormal(sigma=init_sigma))
        self.fc2 = nn.Dense(1024, 512, weight_init=TruncatedNormal(sigma=init_sigma))
        self.fc3 = nn.Dense(512, num_class, weight_init=TruncatedNormal(sigma=init_sigma))

    def construct(self, x):
        x = self.conv1(x)  # 200 * 200 * 6
        x = self.relu(x)
        x = self.max_pool(x)  # 100 * 100 * 6
        x = self.conv2(x)  # 100 * 100 * 16
        x = self.relu(x)
        x = self.max_pool(x)  # 50 * 50 * 16
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def train(model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """定义训练的方法"""
    # 获取训练集
    ds_train = create_train_dataset(data_path, 32, repeat_size, 1)
    # 训练模型
    model.train(epoch_size, ds_train, 
                callbacks=[ckpoint_cb, LossMonitor(20)],
                dataset_sink_mode=sink_mode)


def test(model):
    """定义验证的方法"""
    # 获取测试集
    ds_test = create_test_dataset()
    # 对于测试集中的每一张图片
    for data in ds_test.create_dict_iterator():
        # 预测结果
        predictions = model.predict(Tensor(data['image']))
        # 转换为numpy类型
        predictions = predictions.asnumpy()
        # 转换为对应的label
        class_names = {0: 'bee_balm',
                       1: 'blackberry_lily',
                       2: 'blanket_flower',
                       3: 'bougainvillea',
                       4: 'bromelia',
                       5: 'foxglove'}
        label = class_names[np.argmax(predictions)]
        # 打印结果
        print("预测结果:{}".format(predictions) + ", 即{}".format(label))


if __name__ == '__main__':
    # 神经网络模型
    print('>>> 定义网络结构......')
    net = MyNet(num_class=cfg.num_class,
                channel=cfg.channel,
                init_sigma=cfg.sigma)
    # 定义损失函数
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # 定义优化器
    net_opt = nn.Adam(params=net.trainable_params(),
                      learning_rate=cfg.learning_rate)
    # 设置模型保存参数
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    # 应用模型保存参数
    ckpoint_cb = ModelCheckpoint(prefix=cfg.output_prefix,
                                 directory=cfg.output_directory,
                                 config=config_ck)
    # 构建模型
    print('>>> 构建模型......')
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": nn.Accuracy()})
    # 训练模型
    print('>>> 训练模型......')
    train(model, cfg.epoch_size, cfg.train_path, cfg.repeat_size, ckpoint_cb, False)
    # 进行预测
    print('>>> 预测......')
    test(model)