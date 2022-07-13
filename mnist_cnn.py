from cProfile import label
from http.client import ImproperConnectionState
import imp
from pickletools import optimize
from statistics import mode
from turtle import down, forward
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from mnist_linear import Cross_loss

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST('./', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST('./', train=False, transform=transform, download=True) 
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size = 1)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size = 3)
        self.linear1 = torch.nn.Linear(576, 120)
        self.linear2 = torch.nn.Linear(120, 84)
        self.linear3 = torch.nn.Linear(84, 10)
        self.maxpool = torch.nn.MaxPool2d(kernel_size = 2)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        # [1,28,28] -> [6,28,28]
        x = self.conv1(x)
        # [6,28,28] -> [6,14,14]
        x = self.maxpool(x)
        x = self.relu(x)
        # [6,14,14] -> [16,12,12]
        x = self.conv2(x)
        # [16,12,12] -> [16,6,6]
        x = self.maxpool(x)
        x = self.relu(x)
        x = x.view(batch_size,-1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return self.linear3(x)

model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
Cross_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    print("epoch:", epoch + 1)
    running_loss = 0

    for batch_index,data in enumerate(train_loader):
        inputs, labels = data
        # 改为cuda类型输入
        inputs = inputs.to(device)
        labels = labels.to(device)
        y_pred = model(inputs)
        loss = Cross_loss(y_pred, labels)
        running_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("loss:", (running_loss/batch_index).item())

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            # 改为cuda类型输入
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicts = torch.max(outputs,dim=1)
            total += labels.size(0)
            correct += (predicts==labels).sum().item()
        print("acc:",correct/total)
        return correct/total

if __name__ == "__main__":
    accs = []
    epochs = []
    for epoch in range(10):
        train(epoch)
        acc = test()
        epochs.append(epoch)
        accs.append(acc)
    
    plt.plot(epochs,accs)
    plt.ylabel("ACC")
    plt.xlabel("Epoch")
    plt.show()
    
