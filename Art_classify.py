from cProfile import label
import imp
from pickletools import optimize
from statistics import mode
from unicodedata import name
from cv2 import split
from matplotlib import image
import numpy as np
from sklearn import datasets
import torch
import torch.nn
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import csv
import os
from tqdm import tqdm

from cnn import train
from mnist_linear import Cross_loss

data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

class Art_Dataset(Dataset):
    def __init__(self, img_path, label_path):
        self.transform = data_transform
        self.labels = []
        self.images = []
        with open(label_path) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                self.labels.append(int(row[1]))
        self.labels = torch.LongTensor(self.labels)

        for filename in tqdm(os.listdir(img_path)):
            image = Image.open(img_path + filename).convert('RGB')
            image = image.resize((224,224))
            image = self.transform(image)
            self.images.append(image)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        size = np.array(self.images).shape[0]
        return size

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
        self.linear1 = torch.nn.Linear(275968,49)
    
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

def train(epochs):
    model = Net()
    model.cuda()

    train_dataset = Art_Dataset('Art/train/','Art/train.csv')
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    print("Successful load dataset!")

    Cross_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    for epoch in range(epochs):
        print("epoch:", epoch + 1)
        total = 0
        correct = 0
        running_loss = 0
        for batch_index, data in enumerate(tqdm(train_loader), 0):
            inputs, labels = data

            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            _, predictions = torch.max(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total +=  labels.size(0)

            loss = Cross_loss(outputs, labels)
            running_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("train loss:", running_loss.item() / batch_index)
        print("train acc:", correct/total)
        torch.save(model.state_dict(), 'art_model.pth')

def name_list(namepath):
    names_list = []
    f = open(namepath, 'r')
    rows = f.readlines()
    for row in rows:
        if len(row.split(' ')) == 2:
            names_list.append(row.split(' ')[1])
        if len(row.split(' ')) == 3:
            names_list.append(row.split(' ')[1] + ' ' + row.split(' ')[2])
        if len(row.split(' ')) == 4:
            names_list.append(row.split(' ')[1] + ' ' + row.split(' ')[2] + ' ' + row.split(' ')[3])
    return names_list 

def test(imgpath):
    image1 = Image.open(imgpath)
    image = image1.resize((224,224))
    image = data_transform(image)
    image = image.unsqueeze(0)
    image = image.cuda()
    
    model = Net()
    model.cuda()
    model.load_state_dict(torch.load('art_model.pth'))
    
    outputs = model(image)
    _, prediction = torch.max(outputs,dim=1)
    
    names_list = name_list('Art/name_list.txt')
    print("predict result:", names_list[prediction])

    font={'color': 'red',
		'size': 20,
		'family': 'Times New Roman',
    	'style':'italic'}

    plt.imshow(image1)
    plt.text(0, -2.0, "prediction: " + names_list[prediction], fontdict=font)
    plt.show()
    

if __name__ == "__main__":
    # train(100)
    test('Art/train/13.jpg')


            
            
            



