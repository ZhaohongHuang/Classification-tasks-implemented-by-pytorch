from cProfile import label
from cgi import test
import enum
import imp
from pickletools import optimize
from statistics import mode
from turtle import forward
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class TrainDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter = ',', dtype = np.float32)
        self.x_data = xy[:,:-1]
        self.y_data = xy[:,[-1]]
        x_train, x_test, y_train, y_test = train_test_split(self.x_data, self.y_data, test_size = 0.3)
        self.len = x_train.shape[0]
        self.x_train = torch.from_numpy(x_train)
        self.y_train = torch.from_numpy(y_train)
        self.x_test = torch.from_numpy(x_test)
        self.y_test = torch.from_numpy(y_test)

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.len

    def achieve_test(self):
        return self.x_test, self.y_test


class Moudle(torch.nn.Module):
    def __init__(self):
        super(Moudle, self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,2)
        self.linear4 = torch.nn.Linear(2,1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        y = self.sigmoid(self.linear1(x))
        y = self.sigmoid(self.linear2(y))
        y = self.sigmoid(self.linear3(y))
        y = self.sigmoid(self.linear4(y))
        return y

def compile(model, X_test, Y_test):
    with torch.no_grad():
        Y_pred = model(X_test)
        Y_pred_labels = torch.where(Y_pred >= 0.5, torch.Tensor([1.0]), torch.Tensor([0.0]))
        return torch.eq(Y_pred_labels, Y_test).sum().item() / len(Y_test)

if __name__ == "__main__":
    dataset = TrainDataset('diabetes.csv')
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    X_test, Y_test = dataset.achieve_test()

    model = Moudle()
    BCE = torch.nn.BCELoss(size_average = True)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    epoch = 500
    for k in range(epoch):
        print("........" + str(k) + " epoch........")
        for i,data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = model(inputs)
            BCE_loss = BCE(y_pred, labels)
            optimizer.zero_grad()
            BCE_loss.backward()
            optimizer.step()
        print("loss:", BCE_loss.item())

    print("acc:", compile(model, X_test, Y_test))  

    print("input test_data:", X_test[0])
    print("predict result:", model(X_test[0]).item())
    print("GT:", Y_test[0].item())

    

        