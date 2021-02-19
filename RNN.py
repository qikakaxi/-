# 加载函数库
import torch
import torch.nn as nn
import torch .nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
def imshow(imp, title=None):
    print("imp.shape: ", imp.shape)
    imp = imp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    imp = std * imp + mean
    imp = np.clip(imp, 0, 1)
    plt.imshow(imp)
    #plt.show()
    if title is not None:
        plt.title(title)
    # plt.pause(10)
    plt.close()

# 1. 加载数据集
pipline = transforms.Compose([
    transforms.ToTensor()
])

trainset = datasets.MNIST(root="./data", train=True, download=False, transform=pipline)
testset = datasets.MNIST(root="./data", train=False, download=False, transform=pipline)

print(testset.data.shape)

# 2.定义超参数
BatchSize = 32
Epoch = 20

# 3.创建迭代对象
trainloader = DataLoader(trainset, batch_size=BatchSize, shuffle=True)
testloader = DataLoader(testset, batch_size=BatchSize, shuffle=True)

images, labels = next(iter(testloader))
out = torchvision.utils.make_grid(images)
print("images.shape: ", images.shape)

imshow(out)

class RNN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNN_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity="relu")
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(Device)
        out, hn = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out

input_dim = 28
hidden_dim = 100
layer_dim = 2
output_dim = 10
learningRate = 0.0001
model = RNN_Model(input_dim, hidden_dim, layer_dim, output_dim)
Device = "cpu"

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

# print(list(model.parameters()))

sequence_dim = 28
loss_list = []
accuracy_list = []
iteration_list = []

iter = 0

for epoch in range(Epoch):
    for i, (images, labels) in enumerate(trainloader):
        model.train()
        #print(type(images))
        images = images.view(-1, sequence_dim, input_dim).requires_grad_()
        labels = labels.to(Device)
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        iter = iter + 1
        print("iter: {} ".format(iter))
        if iter % 1000 == 0:
            model.eval()
            correct = 0.0
            total = 0.0
            for images, labels in testloader:
                images = images.view(-1, sequence_dim, input_dim).to(Device)
                output = model(images)
                _, predict = torch.max(output, dim=1)

                total = total + labels.size(0)

                correct = correct + (predict == labels).sum()
            accuracy = correct / total * 100
            iteration_list.append(iter)
            loss_list.append(loss.data)
            accuracy_list.append(accuracy)

            print("iter: {}, loss: {:.4f}, accuracy: {:.4f}, ".format(iter, loss.item(), accuracy))





