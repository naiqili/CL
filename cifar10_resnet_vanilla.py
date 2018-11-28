import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split
from tensorboardX import SummaryWriter
from cifar10_model.resnet import *

import os

from datetime import datetime
now = datetime.now()
time_suffix = 'Vanilla_epoch150_' + now.strftime("%Y%m%d-%H%M%S")

dataroot = '~/dataset/'
b_size = 200
n_epochs = 150
lr = 0.01
best_loss = 100000
n_iter = 0
test_step = 100
model_file = './tmp/model1_%s.md' % time_suffix

device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer_train = SummaryWriter('./tmp/log_train_%s' % time_suffix)
writer_val = SummaryWriter('./tmp/log_test_%s' % time_suffix)

def test_validate(model, device, test_loader, test_valid='Test'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    writer_val.add_scalar('Loss', test_loss, n_iter)
    writer_val.add_scalar('Acc', acc, n_iter)
    print('\n{} set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        test_valid, test_loss, correct, len(test_loader.dataset), acc))
    return test_loss
    
def train(model, device, train_loader, test_loader, optimizer, epoch):
    global best_loss, patience, n_iter
    batch_idx = 0
    for data, target in train_loader:
        model.train()
        batch_idx += 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        acc = 100. * correct / b_size
        writer_train.add_scalar('Loss', loss.item(), n_iter)
        writer_train.add_scalar('Acc', acc, n_iter)
        n_iter += 1
        optimizer.step()
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        if n_iter % test_step == 0:
            print('Start testing...')
            test_loss = test_validate(model, device, test_loader)

def main():        
    dataroot = '~/dataset/'
    
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

    train_dataset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    
    train_size = len(train_dataset)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=b_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=b_size, shuffle=False, num_workers=2)

    model = ResNet18().to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        train(model, device, train_loader, test_loader, optimizer, epoch)          
        
main()
writer_train.close()
writer_val.close()