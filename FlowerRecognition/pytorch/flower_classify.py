from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

#1.预处理
def show_image(path):
    img = Image.open(path)
    img_arr = np.array(img)
    plt.figure(figsize=(5,5))
    plt.imshow(np.transpose(img_arr, (0, 1, 2)))


transformations = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

total_dataset = datasets.ImageFolder("./flower_photos", transform = transformations)
dataset_loader = DataLoader(dataset = total_dataset, batch_size = 100)
items = iter(dataset_loader)
image, label = items.next()
total_dataset.class_to_idx

#2.划分训练集和测试集
train_num = int(0.8 * len(total_dataset))
test_num = len(total_dataset) - train_num
train_dataset, test_dataset = random_split(total_dataset, [train_num, test_num])

train_dataset_loader = DataLoader(dataset = train_dataset, batch_size = 100)
test_dataset_loader = DataLoader(dataset = test_dataset, batch_size = 100)

import torch.nn as nn

#3.建立CNN模型
class flowerclassify(nn.Module):
    
    def __init__(self, num_classes=6):
        super(flowerclassify,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3,stride=1, padding=1)
        self.relu1 = nn.ReLU()       
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)        
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()   
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.lf = nn.Linear(in_features=16 * 16 * 24, out_features=num_classes)
    
    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)       
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = output.view(-1, 16 * 16 * 24)
       
        output = self.lf(output)

        return output

#4.导出训练日志
import logging
 
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

from torch.optim import Adam

cnnmodel = flowerclassify()
optimizer = Adam(cnnmodel.parameters())
loss_fn = nn.CrossEntropyLoss()

logger = get_logger('./exp.log')
#5.训练
def train_and_build(n_epoches):
    
    for epoch in range(n_epoches):
        cnnmodel.train()
        for i, (images, labels) in enumerate(train_dataset_loader):
            optimizer.zero_grad()
            outputs = cnnmodel(images)
            loss = loss_fn(outputs, labels) 
            loss.backward()
            optimizer.step()
        logger.info('Epoch:[{}]\t loss={:.5f}\t '.format(epoch ,loss))

train_and_build(100)
print('finish training!')

#6.测试集上运行
import torch

cnnmodel.eval()
torch.save(cnnmodel,'model.pkl')
torch.save(cnnmodel.state_dict(),'model_param.pkl')
test_acc_count = 0
for k, (test_images, test_labels) in enumerate(test_dataset_loader):
    test_outputs = cnnmodel(test_images)
    _, prediction = torch.max(test_outputs.data, 1)
    test_acc_count += torch.sum(prediction == test_labels.data).item()

test_accuracy = test_acc_count / len(test_dataset)

print(test_accuracy)

#7.进行预测
test_image1 = Image.open("./TestImages/test1.jpg")
test_image2 = Image.open("./TestImages/test2.jpg")
test_image3 = Image.open("./TestImages/test3.jpg")
test_image4 = Image.open("./TestImages/test4.jpg")
test_image5 = Image.open("./TestImages/test5.jpg")
test_image6 = Image.open("./TestImages/test6.jpg")
img = [test_image1,test_image2,test_image3,test_image4,test_image5,test_image6]
for i in img:
    test_image_tensor = transformations(i).float()
    test_image_tensor = test_image_tensor.unsqueeze_(0)
    output = cnnmodel(test_image_tensor)
    class_index = output.data.numpy().argmax()
    print(class_index)

