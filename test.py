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
# Adversarial Iteration and Validation

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
parser.add_argument('--typ', default='normal', type=str, help='sap or normal')
parser.add_argument('--frac', default=1.0, type=float, help='pruning frac')
# python train.py --resume
args = parser.parse_args()


epsilons = [0.0, 0.005, 0.01, 0.015]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



test_dataset = datasets.CIFAR10(root='../data/cifar10', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


def adversarial_examples(model, images, labels, epsilon): # 应用FGSM
    images = images.clone().detach().to(device)
    images.requires_grad = True
    
    outputs = model(images)
    # print(outputs, labels)
    loss = criterion(outputs, labels)

    init_pred = torch.argmax(outputs, dim=1)
    if not torch.equal(init_pred, labels):
        return images

    model.zero_grad()
    # print(loss)
    
    loss.backward()
    
    data_grad = images.grad.data
    perturbed_images = images + epsilon * data_grad.sign()
    perturbed_images = torch.clamp(perturbed_images, min=-1, max=1)
    
    return perturbed_images

def validate():
    model = ResNet(args.typ, args.frac)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())) # torch.nn.DataParallel 是 PyTorch 提供的一个包装器，它可以自动将模型复制到多个设备（GPU）上，并管理数据的拆分和合并。device_ids 参数指定了要使用的 GPU 设备的 ID 列表。range(torch.cuda.device_count()) 表示使用所有可用的 GPU 设备。
    cudnn.benchmark = True
    
    model.load_state_dict(torch.load('./checkpoint/trained_model_'+ args.typ + '.pth', weights_only=False))
    model = model.to(device) # 这里要先to一下
    model.eval()
    correct = [0] * len(epsilons)
    total = [0] * len(epsilons)

    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        for i, epsilon in enumerate(epsilons):
            perturbed_images = adversarial_examples(model, images, labels, epsilon)
            outputs = model(perturbed_images)
            predicted = torch.argmax(outputs, dim=1)
            total[i] += labels.size(0)
            correct[i] += (predicted == labels).sum().item()

    acc = [c / t for c, t in zip(correct, total)]
    return acc

# Validation accuracy
acc = validate()

# Plot results
plt.plot(range(len(epsilons)), acc, marker='o')
plt.xticks(range(len(epsilons)), epsilons)
plt.xlabel('$\epsilon$')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.title('Adversarial Accuracy')
plt.show()