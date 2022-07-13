from calendar import EPOCH
import imp
from pickletools import optimize
from turtle import Turtle, forward
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


# 准备数据集
batch_size = 64

# transform将图像PIL格式转化为Tensor格式，并且进行归一化和标准化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST('./', train=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST('./', train=False, transform=transform)
test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)

# 搭建网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(784,512)
        self.linear2 = torch.nn.Linear(512,256)
        self.linear3 = torch.nn.Linear(256,128)
        self.linear4 = torch.nn.Linear(128,64)
        self.linear5 = torch.nn.Linear(64,32)
        self.linear6 = torch.nn.Linear(32,10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # x.view 类似于reshape，其中-1指将维度进行自动调整
        x = x.view(-1, 784)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.relu(self.linear5(x))
        return self.linear6(x)

model = Net()
# 这里有个细节，CrossEntropyLoss中采用了softmax + NLLloss，因此在Net中不需要加入softmax
Cross_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epoch = 100

def train(epoch):
    print("epoch:", epoch)
    running_loss = 0
    # enumerate输出数据和相应索引
    for i,data in enumerate(train_loader, 0):
        # 其中input是一个64张图片的特征向量，label是64张图片对应的真实标签
        input, label = data
        y_pred = model(input)
        loss = Cross_loss(y_pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("loss:", loss.item())

def test():
    total = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            # 找到0~9中预测出的最大值的下标作为预测值
            _, predicts = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicts == labels).sum().item()
        print("acc=", correct/total)

if __name__ == "__main__":
    for epoch in range(10):
        train(epoch)
        test()



    

