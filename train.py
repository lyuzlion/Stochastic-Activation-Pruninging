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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--typ', default='normal', type=str, help='sap or normal')
parser.add_argument('--frac', default=1.0, type=float, help='pruning frac')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# python train.py --resume
args = parser.parse_args()



# Define GPU devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Different levels of perturbation for the adversary
epsilons = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
frac = 1.0


# Data transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


train_dataset = datasets.CIFAR10(root='../data/cifar10', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)

val_dataset = datasets.CIFAR10(root='../data/cifar10', train=False, download=True, transform=transform_test)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

# Hyperparameters
num_epochs = 150
learning_rate = args.lr

# Training the model
model = ResNet(args.typ, args.frac).to(device)

net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())) # torch.nn.DataParallel 是 PyTorch 提供的一个包装器，它可以自动将模型复制到多个设备（GPU）上，并管理数据的拆分和合并。device_ids 参数指定了要使用的 GPU 设备的 ID 列表。range(torch.cuda.device_count()) 表示使用所有可用的 GPU 设备。
cudnn.benchmark = True # cudnn.benchmark 设置为 True 后，cuDNN 会在开始时花费一些时间来选择最适合当前硬件的算法，从而在后续的迭代中获得更快的速度。


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
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
        
        
    print(f'\nEpoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {correct / tot}')
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), 'checkpoint/trained_model.pth')
        valid_loss_min = valid_loss


# Adversarial Iteration and Validation
# def adversarial_examples(model, images, labels, epsilon):
#     images = images.clone().detach().to(device)
#     images.requires_grad = True
    
#     outputs = model(images)
#     loss = criterion(outputs, labels)
#     model.zero_grad()
#     loss.backward()
    
#     data_grad = images.grad.data
#     perturbed_images = images + epsilon * data_grad.sign()
#     perturbed_images = torch.clamp(perturbed_images, 0, 1)
    
#     return perturbed_images

# def validate():
#     model.eval()
#     correct = [0] * len(epsilons)
#     total = [0] * len(epsilons)

#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)
#             for i, epsilon in enumerate(epsilons):
#                 perturbed_images = adversarial_examples(model, images, labels, epsilon)
#                 outputs = model(perturbed_images)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total[i] += labels.size(0)
#                 correct[i] += (predicted == labels).sum().item()

#     acc = [c / t for c, t in zip(correct, total)]
#     return acc

# # Validation accuracy
# acc = validate()

# # Plot results
# plt.plot(range(len(epsilons)), acc, marker='o')
# plt.xticks(range(len(epsilons)), epsilons)
# plt.xlabel('$\epsilon$')
# plt.ylabel('Accuracy')
# plt.ylim(0, 1)
# plt.title('Adversarial Accuracy')
# plt.show()