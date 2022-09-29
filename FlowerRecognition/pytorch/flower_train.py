import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

BATCH_SIZE = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 数据处理

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

Labels = {'bee_balm': 0, 'blackberry_lily': 1, 'blanket_flower': 2, 'bougainvillea': 3, 'bromelia': 4, 'foxglove': 5}


class SeedlingData(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.test = test
        self.transforms = transforms

        if self.test:
            imgs = [os.path.join(root, img) for img in os.listdir(root)]
            self.imgs = imgs
        else:
            imgs_labels = [os.path.join(root, img) for img in os.listdir(root)]
            imgs = []
            for imglable in imgs_labels:
                for imgname in os.listdir(imglable):
                    imgpath = os.path.join(imglable, imgname)
                    imgs.append(imgpath)
            trainval_files, val_files = train_test_split(imgs, test_size=0.3, random_state=42)
            if train:
                self.imgs = trainval_files
            else:
                self.imgs = val_files

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        img_path = img_path.replace("\\", '/')
        if self.test:
            label = -1
        else:
            labelname = img_path.split('/')[-2]
            label = Labels[labelname]
        data = Image.open(img_path).convert('RGB')
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


dataset_train = SeedlingData('data/train', transforms=transform, train=True)
dataset_test = SeedlingData('data/train', transforms=transform_test, train=False)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()

        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64, 3)                    # 64 * 224 * 224
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 64 * 112 * 112

        self.conv2_1 = nn.Conv2d(64, 128, 3)                  # 128 * 112 * 112
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 128 * 56 * 56

        self.conv3_1 = nn.Conv2d(128, 256, 3)                   # 256 * 56 * 56
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1))   # pooling 256 * 28 * 28

        self.conv4_1 = nn.Conv2d(256, 512, 3)                  # 512 * 28 * 28
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 14 * 14

        self.conv5_1 = nn.Conv2d(512, 512, 3)                   # 512 * 14 * 14
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 7 * 7

        # view

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 6)
        # softmax 1 * 1 * 1000

    def forward(self, x):
        # x.size(0)即为batch_size
        in_size = x.size(0)

        out = self.conv1_1(x)  # 224
        out = F.relu(out)
        out = self.conv1_2(out)
        out = F.relu(out)
        out = self.maxpool1(out)  # 112

        out = self.conv2_1(out)  # 112
        out = F.relu(out)
        out = self.conv2_2(out)
        out = F.relu(out)
        out = self.maxpool2(out)  # 56

        out = self.conv3_1(out)  # 56
        out = F.relu(out)
        out = self.conv3_2(out)
        out = F.relu(out)
        out = self.conv3_3(out)
        out = F.relu(out)
        out = self.maxpool3(out)  # 28

        out = self.conv4_1(out)  # 28
        out = F.relu(out)
        out = self.conv4_2(out)
        out = F.relu(out)
        out = self.conv4_3(out)
        out = F.relu(out)
        out = self.maxpool4(out)  # 14

        out = self.conv5_1(out)  # 14
        out = F.relu(out)
        out = self.conv5_2(out)
        out = F.relu(out)
        out = self.conv5_3(out)
        out = F.relu(out)
        out = self.maxpool5(out)  # 7

        # 展平
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        out = F.log_softmax(out, dim=1)
        return out

net = VGG16()
net.to(device)  # 连接到设备

loss_function = nn.CrossEntropyLoss()                 # 损失函数，使用针对多类别的损失交叉熵函数
optimizer = optim.Adam(net.parameters(), lr=0.0001)   # 优化器，优化对象为网络所有的可训练参数，学习率设置为0.0001

best_acc = 0.0  # 定义最佳准确率，在训练过程中保存准确率最高的一次训练的模型
val_num = len(dataset_test)

for epoch in range(100):  # 设置epoch的轮数
    # train
    net.train()
    running_loss = 0.0  # 用于统计训练过程中的平均损失
    for step, data in enumerate(train_loader, start=0):  # 遍历数据集，返回每一批数据data以及data对应的step
        images, labels = data  # 将数据分为图像和标签
        optimizer.zero_grad()  # 清空之前的梯度信息
        outputs = net(images.to(device))  # 将输入的图片引入到网络，将训练图像指认到一个设备中，进行正向传播得到输出，
        loss = loss_function(outputs, labels.to(device))  # 将网络预测的值与真实的标签值进行对比，计算损失梯度
        loss.backward()  # 误差的反向传播
        optimizer.step()  # 通过优化器对每个结点的参数进行更新
        running_loss += loss.item()  # 将每次计算的loss累加到running_loss中
        rate = (step + 1) / len(train_loader)  # 计算训练进度，当前的步数除以训练一轮所需要的总的步数
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        # 打印训练进度
    print()

    # validation
    validate_loader =  test_loader
    net.eval()  # 关闭Dropout
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():  # torch.no_grad()之后的计算过程中，不计算每个结点的误差损失梯度
        for val_data in validate_loader:
            """
            遍历验证集；将数据划分为图片和对应的标签值
            对结点进行参数的更新；指认到设备上并传入网络得到输出
            """
            val_images, val_labels = val_data
            optimizer.zero_grad()
            outputs = net(val_images.to(device))
            """求得输出的最大值作为预测最有可能的类别"""
            predict_y = torch.max(outputs, dim=1)[1]
            """
            predict_y == val_labels.to(device) 预测类别与真实标签值的对比，相同为1，不同为0
            item()获取数据，将Tensor转换为数值，通过sum()加到acc中，求和可得预测正确的样本数
            """
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        """如果准确率大于历史最优的准确率，将当前值赋值给最优，并且保存当前的权重"""
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), 'net1000.pth')
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))  # 打印训练到第几轮，累加的平均误差，以及最优的准确率

print('Finished Training')

import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.autograd import Variable

classes = ('bee_balm', 'blackberry_lily', 'blanket_flower', 'bougainvillea', 'bromelia', 'foxglove')
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = torch.load("model.pth")
model = VGG16()
model.load_state_dict(torch.load("net.pth"))
model.eval()
model.to(DEVICE)

dataset_test =SeedlingData('data/test/', transform_test,test=True)
print(len(dataset_test))
# 对应文件夹的label

for index in range(len(dataset_test)):
    item = dataset_test[index]
    img, label = item
    img.unsqueeze_(0)
    data = Variable(img).to(DEVICE)
    output = model(data)
    _, pred = torch.max(output.data, 1)
    print('Image Name:{},predict:{}'.format(dataset_test.imgs[index], classes[pred.data.item()]))
    index += 1

