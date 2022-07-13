''' 
pytorch编写神经网络四步骤:
1. 准备数据集(将数据集转化为Tensor形式)
2. 构造神经网络
3. 构造优化器，损失
4. 训练，评估，测试
'''
from cProfile import label
from pickletools import optimize
from statistics import mode
import torch.nn
import torch.optim
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm

from back_propagation import forward
from mnist_linear import Cross_loss
import matplotlib.pyplot as plt

# 将PIL转化成Tensor向量
data_transform = transforms.Compose([transforms.ToTensor()])

# 准备数据集
class Cat_and_Dog_Dataset(Dataset):
    def __init__(self, filepath):
        self.images = []
        self.labels = []
        self.transform = data_transform
        for filename in tqdm(os.listdir(filepath)):
            image = Image.open(filepath + filename)
            image = image.resize((224,224))
            # 将图片转化为Tensor
            image = self.transform(image)
            self.images.append(image)
            if filename.split('_')[0] == 'cat':
                self.labels.append(0)
            elif filename.split('_')[0] == 'dog':
                self.labels.append(1)
            # 将标签转化为Tensor
        self.labels = torch.LongTensor(self.labels)
    
    # 构造迭代器
    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    # 迭代器长度
    def __len__(self):
        images = np.array(self.images)
        len = images.shape[0]
        return len

# 构造Inception block
class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.averag_pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1_1_24 = torch.nn.Conv2d(in_channels,24,kernel_size=1)
        self.conv1_1_16 = torch.nn.Conv2d(in_channels,16,kernel_size=1)
        self.conv5_5_24 = torch.nn.Conv2d(16,24,kernel_size=5,padding=2)
        self.conv3_3_24_1 = torch.nn.Conv2d(16,24,kernel_size=3, padding=1)
        self.conv3_3_24_2 = torch.nn.Conv2d(24,24,kernel_size=3,padding=1)
    
    def forward(self, x):
        x1 = self.averag_pool(x)
        x1 = self.conv1_1_24(x1)

        x2 = self.conv1_1_16(x)

        x3 = self.conv1_1_16(x)
        x3 = self.conv5_5_24(x3)

        x4 = self.conv1_1_16(x)
        x4 = self.conv3_3_24_1(x4)
        x4 = self.conv3_3_24_2(x4)

        outputs = [x1,x2,x3,x4]
        return torch.cat(outputs, dim=1)

# 构造自己的Net
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,10,kernel_size=3,padding=1)
        self.conv2 = torch.nn.Conv2d(88,20,kernel_size=3,padding=1)

        self.incept1 = InceptionA(in_channels=10)
        self.incept2 = InceptionA(in_channels=20)

        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.linear1 = torch.nn.Linear(275968,10)
    
    def forward(self,x):
        batchsize = x.size(0)
        # 3,224,224 -> 10,112,112
        x = self.maxpool(self.relu(self.conv1(x)))
        # 10,112,112 -> 88,112,112
        x = self.incept1(x)
        # 88,112,112 -> 20,56,56
        x = self.maxpool(self.relu(self.conv2(x)))
        # 20,64,64 -> 88,56,56
        x = self.incept2(x)
        # 类似shape，-1指将维度自动调整，转化向量进行全连接操作
        x = x.view(batchsize, -1)
        x = self.linear1(x)
        return x    

# 训练
def train(epoch, train_loader, model, Cross_loss, optimizer):
    print("epoch:",epoch+1)
    running_loss = 0
    for batch_index, data in enumerate(tqdm(train_loader),0):
        inputs, labels = data
        # 将数据转化为cuda格式
        inputs = inputs.cuda()
        labels = labels.cuda()
        y_pred = model(inputs)
        loss = Cross_loss(y_pred, labels)
        running_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("train loss:", (running_loss).item()/batch_index)
    torch.save(model.state_dict(), './model.pth')  

# 评估验证集的精确度
def val(val_loader, model, Cross_loss, optimizer):
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_index, data in enumerate(val_loader,0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            # 取维度最大
            _, predicts = torch.max(outputs,dim=1)

            total += labels.size(0)
            correct += (predicts==labels).sum().item()
        print("val acc:", correct/total)
        return correct/total

# 单张图片的测试
def test(imgpath):

    font={	'color': 'red',
		'size': 20,
		'family': 'Times New Roman',
    	'style':'italic'}

    o_img = Image.open(imgpath)
    o_img1 = o_img.resize((224,224))

    img = data_transform(o_img1)
    img = img.unsqueeze(0)
    img = img.cuda()
    print(img.shape)

    model = Net()
    model = model.cuda()
    model.load_state_dict(torch.load("model.pth")) 
    output = model(img)
    _, predict = torch.max(output,dim=1)
    if predict == 1:
        print("prediction: dog")
        plt.imshow(o_img)
        plt.text(0, -6.0, "prediction: dog", fontdict=font)
        plt.show()
    if predict == 0:
        print("prediction: cat")
        plt.imshow(o_img)
        plt.text(0, -6.0, "prediction: cat", fontdict=font)
        plt.show()

if __name__ == "__main__":
    # 加载训练集
    train_dataset = Cat_and_Dog_Dataset('cat_dog/train/')
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    # 加载验证集
    val_dataset = Cat_and_Dog_Dataset('cat_dog/val/')
    val_loader = DataLoader(dataset=val_dataset, batch_size=64)

    # 构造网络
    model = Net()
    model = model.cuda()

    # 优化器和损失函数（CrossEntropyLoss = Softmax + NLLloss）
    Cross_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20

    acc_list = []
    epoch_list = []

    for epoch in range(epochs):
        train(epoch, train_loader, model, Cross_loss, optimizer)
        acc = val(val_loader, model, Cross_loss, optimizer)
        acc_list.append(acc)
        epoch_list.append(epoch + 1)

    plt.plot(acc_list,epoch_list)
    plt.ylabel("ACC")
    plt.xlabel("Epoch")
    plt.show

    # 单张图片测试和可视化
    test('cat_dog/test/40.jpg')

    








             