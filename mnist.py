
# 加载函数库
import torch
import torch.nn as nn
import torch .nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

# 定义超参数

BatchSize = 64
# device = torch.device("cuda" if torch.cuda.is_availale() else "cpu")
Device = "cpu"
Epoch = 50


# 定义神经网络
class Digit(nn.Module):
    def __init__(self):
        super(Digit, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)

        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):

        inputSize = x.size(0)

        x = self.conv1(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(inputSize, -1)

        x = self.fc1(x)
        x = self.fc2(x)

        output = F.softmax(x, dim=1)

        return output 

# 对图像处理
PipLine = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081, ))
])

# 下载数据集

trainSet = datasets.MNIST("data", train=True, download=True, transform=PipLine)
testSet = datasets.MNIST("data", train=False, download=True, transform=PipLine)

trainLoader = DataLoader(trainSet, batch_size=BatchSize, shuffle=True)
testLoader = DataLoader(testSet, batch_size=BatchSize, shuffle=True)

f = open("./data/MNIST/raw/train-images-idx3-ubyte", "rb")
file = f.read()

image = [int(str(item).encode('ascii'), 16) for item in file[16:16+784]]

imageNp = np.array(image, dtype=np.uint8).reshape(28, 28, 1)
cv2.imwrite("image.jpg", imageNp)
print(Device)

model = Digit().to(Device)
optimizer = optim.Adam(model.parameters())

def train_model(model, device, trainLoader, optimizer, epoch):
    model.train()

    for batch_index, (data, target) in enumerate (trainLoader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = F.cross_entropy(output, target)

        pred = output.max(1, keepdim=True)

        loss.backward()

        optimizer.step()

        if batch_index % 2000 == 0:
            print("train epoch: {}\t,train_loss: {:.6f}\t".format(epoch, loss.item()))

def test_model(model, device, testloader):

    model.eval()

    correct = 0.0

    test_loss = 0.0

    with torch.no_grad():
        for data, target in testloader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += F.cross_entropy(output, target).item()

            # pred = output.max(1, keepdim=True)[1]
            # pred = torch.max(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss = test_loss / len(testloader.dataset)
        print("test __ average : {:.4f}, Accuracy : {:.3f}\n".format(test_loss, 100 * correct / len(testloader.dataset)))

for epoch in range(1, 1+Epoch):

    train_model(model, Device, trainLoader, optimizer, epoch)
    test_model(model, Device, testLoader)






