from functools import partial
from math import sqrt
import numpy as np
import torch
from torch import nn
import torch.functional as F
import torch.nn.functional as F
import random
from PIL import Image
from matplotlib import pyplot
import os
import torchvision.transforms as transforms
from random import sample

def seed_all(random_state):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)
    random.seed(random_state)

IMG_SIZE = 32
ONE_HOT_VECTOR_SIZE = 4

class ImageLoader:
  def __init__(self, data_paths = ["training", "testing"], train_percentage = .8):
    self.train_dataset = []
    self.test_dataset = []

    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize((IMG_SIZE, IMG_SIZE))
    ])

    temp_ds = []
    for data_path in data_paths:
    #   for folder_name in os.listdir(data_path):
        path = os.path.join(data_path)

        if not os.path.isdir(path):
          continue
        for image_file_name in os.listdir(path):
            image_path = os.path.join(path, image_file_name)
            image = Image.open(image_path)
            tokens = image_file_name.split('_')
            if len(tokens) == 4 and int(tokens[1]) <= 1 and int(tokens[2]) <= 3:
                gender = int(tokens[1])
                race = int(tokens[2])
                data = transform(image)
                temp_ds.append((data, (race, gender)))
    train_len = int(len(temp_ds) * train_percentage)
    random.shuffle(temp_ds)
    self.train_dataset = temp_ds[: train_len]
    self.test_dataset = temp_ds[train_len:]

class Conv2DNet(nn.Module):
    def __init__(self, dim, batch_size, num_of_classes):
        super(Conv2DNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.conv_params = [
            [3, 32, (3, 3), 1, 0, 0],
            [32, 64, (3, 3), 1, 0, 1],
        ]

        self.conv1 = nn.Conv2d(in_channels = self.conv_params[0][0], out_channels = self.conv_params[0][1], kernel_size = self.conv_params[0][2], stride = self.conv_params[0][3], padding = self.conv_params[0][4])
        self.conv2 = nn.Conv2d(in_channels = self.conv_params[1][0], out_channels = self.conv_params[1][1], kernel_size = self.conv_params[1][2], stride = self.conv_params[1][3], padding = self.conv_params[1][4])

        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        lc_ind = len(self.conv_params) - 1
        h, w = conv_out_shape(self.conv_params, h = dim[0], w = dim[1])

        self.fc1 = nn.Linear(h * w * self.conv_params[lc_ind][1], 128)
        self.fc2 = nn.Linear(128, num_of_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        emb = self.fc1(x)
        x = F.relu(emb)
        output = self.fc2(x)
        return output, emb

class Target(torch.nn.Module):
    def __init__(self, dim, batch_size, num_of_classes):
        super(Target, self).__init__()

        self.conv_params = [
            [3, 16, (3, 3), 1, 0, 1],
            [16, 32, (3, 3), 1, 0, 0],
            [32, 64, (3, 3), 1, 0, 1],
        ]
        self.layers = []
        lc_ind = len(self.conv_params) - 1
        drop_prob = 0.1
        self.number_of_fc = 2
        self.dropout = nn.Dropout(drop_prob)
        self.conv1 = nn.Conv2d(in_channels = self.conv_params[0][0], out_channels = self.conv_params[0][1], kernel_size = self.conv_params[0][2], stride = self.conv_params[0][3], padding = self.conv_params[0][4])
        if self.conv_params[0][5]:
            self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = self.conv_params[1][0], out_channels = self.conv_params[1][1], kernel_size = self.conv_params[1][2], stride = self.conv_params[1][3], padding = self.conv_params[1][4])
        if self.conv_params[1][5]:
            self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = self.conv_params[2][0], out_channels = self.conv_params[2][1], kernel_size = self.conv_params[2][2], stride = self.conv_params[2][3], padding = self.conv_params[2][4])
        if self.conv_params[2][5]:
            self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        h, w = conv_out_shape(self.conv_params, h = dim[0], w = dim[1])

        if self.number_of_fc == 2:
            # (h * w * self.conv_params[lc_ind][1]) // 2
            self.fc = nn.Linear(h * w * self.conv_params[lc_ind][1], 128)
            self.fc2 = nn.Linear(128, num_of_classes)
        else:
            self.fc = nn.Linear(h * w * self.conv_params[lc_ind][1], num_of_classes)

        self.sigmoid = nn.Sigmoid()
        self.log_soft_max = nn.LogSoftmax(dim=1)

    def forward(self, batch):
        output = self.conv1(batch)
        output = F.relu(output)
        if self.conv_params[0][5]:
          output = self.pool1(output)
        output = self.conv2(output)
        output = F.relu(output)
        if self.conv_params[1][5]:
          output = self.pool2(output)
        output = self.conv3(output)
        output = F.relu(output)
        if self.conv_params[2][5]:
          output = self.pool3(output)
        output = output.view(batch.shape[0], -1)
        embedding = self.fc(output)
        if self.number_of_fc == 2:
            embedding = F.relu(embedding)
            output = self.dropout(output)
            embedding = self.fc2(embedding)
        output = self.log_soft_max(embedding)
        return embedding, output

class Attack(torch.nn.Module):
    def __init__(self, dim, layer1_nodes, num_of_classes):
        super(Attack, self).__init__()
        self.fc = nn.Linear(layer1_nodes, layer1_nodes // 2)
        self.fc2 = nn.Linear(layer1_nodes // 2, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, num_of_classes)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        output = self.fc(batch)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.fc3(output)
        output = F.relu(output)
        output = self.fc4(output)

        return F.sigmoid(output).squeeze(dim=1)
        # return output


def create_data_loader(train_dataset, test_dataset, batch_size = 1, shuffle = False):
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle = shuffle, batch_size = batch_size)
    return (train_loader, test_loader)

def conv_out_shape(conv_params, h, w):
    from math import floor
    lc_ind = len(conv_params) - 1

    for ind, conv in enumerate(conv_params):

        h = int(floor(h - conv[2][0] + (2 * conv[4]) / conv[3])) + 1
        w = int(floor(w - conv[2][1] + (2 * conv[4]) / conv[3])) + 1
        if conv[5]: # if conv[*][5] is true, calculate for max-pool
            h = int(floor(h/2))
            w = int(floor(w/2))
        print(f"Output (h, w) after applying conv{ind + 1}: {(h, w)}")
    print(f"FC_1 layer nodes: {h * w * conv_params[lc_ind][1]}")
    return (h, w)

def idx_to_one_hot(idx):
    v = [0] * ONE_HOT_VECTOR_SIZE
    v[idx] = 1
    return torch.tensor(v)

def batch_one_hot(batch):
    # return torch.tensor(map(idx_to_one_hot, batch))
    return torch.nn.functional.one_hot(batch)

def softmax_to_one_hot(prob):
    return idx_to_one_hot(torch.argmax(prob))

def batch_softmax_to_one_hot(batch):
    return torch.nn.functional.one_hot(list(map(softmax_to_one_hot, batch)))
    # return torch.tensor(map(softmax_to_one_hot, batch))

def plot_losses(losses, filename = 'loss.png'):
  plt = pyplot
  plt.plot(losses)
  plt.title('Epoch Vs Loss')
  plt.xlabel("Number Of Epochs")
  plt.ylabel("Training Loss")
  plt.savefig(filename)

def normalize_weight(face):
    return (face - torch.min(face)) / (torch.max(face) - torch.min(face))
