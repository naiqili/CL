import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.dataset import random_split
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import  WeightedRandomSampler

import os

'''
https://github.com/KellerJordan/ResNet-PyTorch-CIFAR10
'''

class ResNet(nn.Module):
    
    def __init__(self, n=7, res_option='A', use_dropout=False):
        super(ResNet, self).__init__()
        self.res_option = res_option
        self.use_dropout = use_dropout
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.layers1 = self._make_layer(n, 16, 16, 1)
        self.layers2 = self._make_layer(n, 32, 16, 2)
        self.layers3 = self._make_layer(n, 64, 32, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(64, 10)
    
    def _make_layer(self, layer_count, channels, channels_in, stride):
        return nn.Sequential(
            ResBlock(channels, channels_in, stride, res_option=self.res_option, use_dropout=self.use_dropout),
            *[ResBlock(channels) for _ in range(layer_count-1)])
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResBlock(nn.Module):
    
    def __init__(self, num_filters, channels_in=None, stride=1, res_option='A', use_dropout=False):
        super(ResBlock, self).__init__()
        
        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == num_filters:
            channels_in = num_filters
            self.projection = None
        else:
            if res_option == 'A':
                self.projection = IdentityPadding(num_filters, channels_in, stride)
            elif res_option == 'B':
                self.projection = ConvProjection(num_filters, channels_in, stride)
            elif res_option == 'C':
                self.projection = AvgPoolPadding(num_filters, channels_in, stride)
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        if self.use_dropout:
            self.dropout = nn.Dropout(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        out += residual
        out = self.relu2(out)
        return out


# various projection options to change number of filters in residual connection
# option A from paper
class IdentityPadding(nn.Module):
    def __init__(self, num_filters, channels_in, stride):
        super(IdentityPadding, self).__init__()
        # with kernel_size=1, max pooling is equivalent to identity mapping with stride
        self.identity = nn.MaxPool2d(1, stride=stride)
        self.num_zeros = num_filters - channels_in
    
    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out

# option B from paper
class ConvProjection(nn.Module):

    def __init__(self, num_filters, channels_in, stride):
        super(ResA, self).__init__()
        self.conv = nn.Conv2d(channels_in, num_filters, kernel_size=1, stride=stride)
    
    def forward(self, x):
        out = self.conv(x)
        return out

# experimental option C
class AvgPoolPadding(nn.Module):

    def __init__(self, num_filters, channels_in, stride):
        super(AvgPoolPadding, self).__init__()
        self.identity = nn.AvgPool2d(stride, stride=stride)
        self.num_zeros = num_filters - channels_in
    
    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out
		
lid_file = './tmp/train_lid_k100'
dataroot = '~/dataset/'
lid_res = torch.load(lid_file)
lid_vec = np.mean(lid_res, axis=1)
lid_idx = np.argsort(lid_vec)

train_dataset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transforms.ToTensor())
train_size = len(train_dataset)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dist = [[] for _ in range(10)]
label_to_idx = [[] for _ in range(10)]
for m in range(train_size):
    _, label = train_dataset[m]
    dist[label].append(lid_vec[m])
    label_to_idx[label].append(m)    
    
def _softmax(p, mask=None):
    ep = np.exp(p)
    if not mask is None:
        ep *= mask
    ep /= np.sum(ep)
    return ep
	
from datetime import datetime
now = datetime.now()
time_suffix = 'LID_epoch200_' + now.strftime("%Y%m%d-%H%M%S")

dataroot = '~/dataset/'
b_size = 200
n_epochs = 200
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
    
def train(model, device, train_loader_per_class, test_loader, optimizer, epoch):
    global best_loss, patience, n_iter
    batch_idx = 0
    for _ in range(train_size // b_size):
        loader_iters = map(iter, train_loader_per_class)
        datas, targets = [], []
        for train_loader in loader_iters:
            data, target = train_loader.next()
            datas.append(data)
            targets.append(target)
        #print(datas, targets)
        data, target = torch.cat(datas), torch.cat(targets)
        data, target = data.to(device), target.to(device)
        model.train()
        batch_idx += 1
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
                100. * batch_idx * b_size / train_size, loss.item()))
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
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=b_size, shuffle=False, num_workers=2)

    model = ResNet(9, res_option='A', use_dropout=True).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    #temp_list = [0.05, 0.1, 0.5, 1, 2, 5, 10]
    temp_list = np.arange(0.05, 0.5, 0.01).tolist() + np.arange(0.5, 10, 0.2).tolist() 
    for epoch in range(1, n_epochs + 1):
        if (epoch - 1) // 2 < len(temp_list):
            T = temp_list[(epoch - 1) // 2]
        else:
            T = temp_list[-1]
        print('Set temperature: %f' % T)
        train_loader_per_class = []
        for c in range(10):
            mask = np.zeros(train_size)
            mask[np.array(label_to_idx[c])] = 1
            a_vec = _softmax(-lid_vec / T, mask)
            sampler = WeightedRandomSampler(a_vec,\
                                            num_samples=b_size // 10,\
                                            replacement=False)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=b_size // 10, shuffle=False,
                                                    sampler=sampler)
            train_loader_per_class.append(train_loader)
        train(model, device, train_loader_per_class, test_loader, optimizer, epoch)          
        
main()
writer_train.close()
writer_val.close()