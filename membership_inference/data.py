import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os
import pandas as pd
import numpy as np
import random

from nn import *
from util import *

# helper for attack_dataset
def delta_inference(dataloader, target_path, shadow_path, label):
    debug("called delta_inference")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load everything
    info("starting shadow model inference on the dataset")
    debug("sending Conv2DNet model to {} device".format(device))
    model = Conv2DNet().to(device)
    PATH = './' + shadow_path
    para = torch.load(PATH)
    debug("loading state_dict into model")
    model.load_state_dict(para)

    results_s = []
    # inference
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # this is not normalized
            outputs = model(images)

            for o in outputs.data:
                results_s.append(o)
    info("starting target model inference on the dataset")
    debug("sending Conv2DNet model to {} device".format(device))
    model = Conv2DNet().to(device)
    PATH = './' + target_path
    para = torch.load(PATH)
    debug("loading state_dict into model")
    model.load_state_dict(para)

    results_t = []

    # inference
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # this is not normalized
            outputs = model(images)

            for o in outputs.data:
                results_t.append(o)

    debug("processing and returning outputs (delta)")
    min = np.inf
    max = -np.inf
    results = []
    for id, v in enumerate(results_s):
        delta = abs(v)-abs(results_t[id])
        results.append((delta,label))
        for nbr in delta:
            if nbr > max:
                max = nbr
            if nbr < min:
                min = nbr
    return (results, min, max)

def get_attack_data_loaders(member_data_loader, nonmember_data_loader, target_path, shadow_path):
    test_size = 0.1
    debug("test split size: {}".format(test_size))
    data_member, min1, max1  = delta_inference(member_data_loader, target_path, shadow_path, 1)
    data_nmember, min2, max2 = delta_inference(nonmember_data_loader, target_path, shadow_path, 0)

    min_value = min(min1,min2)
    max_value = max(max1,max2)

    # normalize all the data to 0.0-1.0 using the min and max
    total_range = max_value + abs(min_value)
    data_member = list(map(lambda x: ((x[0] + abs(min_value))/total_range, x[1]), data_member))
    data_nmember = list(map(lambda x: ((x[0] + abs(min_value))/total_range, x[1]), data_nmember))
    # shuffle the datasets
    random.shuffle(data_member)
    random.shuffle(data_nmember)
    # split

    train = pd.DataFrame(data_member[int(len(data_member)*0.1):] + data_nmember[int(len(data_member)*0.1):])
    test = pd.DataFrame(data_member[:int(len(data_member)*0.1)] + data_nmember[:int(len(data_nmember)*0.1)])
    # create the dataset and then dataloaders
    traindataset = CustomDataset(train)
    testdataset = CustomDataset(test)

    traindataloader = DataLoader(traindataset, batch_size=32,
                    shuffle=True, num_workers=2)

    testdataloader = DataLoader(testdataset, batch_size=32,
                    shuffle=True, num_workers=2)
    return traindataloader, testdataloader

class CustomDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label


def get_mnist_loaders():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5)) # because the MNIST has only 1 channel, so we set (0.5), (0.5).
                                            # if the dataset has three channel, like CIFAR-10, we set (0.5,0.5,0.5), (0.5,0.5,0.5)
         ])


    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)

    batch_size = 32 # 16 32 64 is ok, just depend your computer
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return (trainloader,testloader)

def get_cifar10_loaders():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # because the MNIST has only 1 channel, so we set (0.5), (0.5).
                                                            # if the dataset has three channel, like CIFAR-10, we set (0.5,0.5,0.5), (0.5,0.5,0.5)
         ])


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    batch_size = 32 # 16 32 64 is ok, just depend your computer
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return (trainloader,testloader)

def load_dataset(path):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5)) # because the MNIST has only 1 channel, so we set (0.5), (0.5).
         ])

    raise Exception("MIA/data.py:171  I dont know how your data looks like so please implement the dataset parsing yourself :D")
    # Please also adopt the transform to produce ONE channel, if u have multiple like in CIFAR please compress them to ONE channel.
    dataset = None


    batch_size = 32 # 16 32 64 is ok, just depend your computer
    dataloader = DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    return dataloader
