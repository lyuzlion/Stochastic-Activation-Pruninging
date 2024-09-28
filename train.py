import requests
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
import math
import logging
import matplotlib.pyplot as plt
from models.ResNet import ResNet, ResidualUnit
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--typ', default='normal', type=str, help='sap or normal')
parser.add_argument('--frac', default=1.0, type=float, help='pruning frac')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# python train.py --resume
args = parser.parse_args()



# Define GPU devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Different levels of perturbation for the adversary
frac = 1.0


# Data transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def adjust_learning_rate(optimizer):
    global args
    args.lr = args.lr * 0.8
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


train_dataset = datasets.CIFAR10(root='../data/cifar10', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)

val_dataset = datasets.CIFAR10(root='../data/cifar10', train=False, download=True, transform=transform_test)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

# Hyperparameters
num_epochs = 150

# Training the model
model = ResNet(args.typ, args.frac).to(device)

model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())) # torch.nn.DataParallel 是 PyTorch 提供的一个包装器，它可以自动将模型复制到多个设备（GPU）上，并管理数据的拆分和合并。device_ids 参数指定了要使用的 GPU 设备的 ID 列表。range(torch.cuda.device_count()) 表示使用所有可用的 GPU 设备。
cudnn.benchmark = True # cudnn.benchmark 设置为 True 后，cuDNN 会在开始时花费一些时间来选择最适合当前硬件的算法，从而在后续的迭代中获得更快的速度。


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
valid_loss_min = np.Inf

for epoch in tqdm(range(num_epochs)):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    model.eval()
    valid_loss = 0.0
    tot = 0
    correct = 0
    for i, (images, labels) in enumerate(val_loader):
        images, labels = images.to(device), labels.to(device)
        tot += labels.size(0)
        outputs = model(images)
        loss = criterion(outputs, labels)

        valid_loss += loss.item() * images.size(0)
        y_pred = torch.argmax(outputs, dim=1)

        correct += (1 if labels[0].item() == y_pred else 0)
        
        
    print(f'\nEpoch [{epoch+1}/{num_epochs}], Loss: {valid_loss:.4f}, Accuracy: {correct / tot}')

    if (epoch + 1) % 10 == 0:
        adjust_learning_rate(optimizer)
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), 'checkpoint/trained_model_' + args.typ +'.pth')
        valid_loss_min = valid_loss

