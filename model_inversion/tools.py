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

IMG_SIZE = 64

class ImageLoader:
  def __init__(self, data_paths = ["training", "testing"], num_of_training_images = 9, num_of_labels = 40):
    self.labels = {}
    self.train_dataset = []
    self.test_dataset = []
    seq = [i for i in range(10)]

    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize((IMG_SIZE, IMG_SIZE))
    ])

    for data_path in data_paths:
      for folder_name in os.listdir(data_path):
        path = os.path.join(data_path, folder_name)

        temp_ds = []
        if not os.path.isdir(path):
          continue
        for image_file_name in os.listdir(path):
            image_path = os.path.join(path, image_file_name)
            image = Image.open(image_path)
            label = int(folder_name[1:]) - 1
            if label < num_of_labels:
              self.labels[folder_name] = label
              data = transform(image)
              temp_ds.append((data, label))
        samples = sample(seq, 10 - num_of_training_images)
        if len(temp_ds) > 0:
          for ind in seq:
              if ind in samples:
                  self.test_dataset.append(temp_ds[ind])
              else:
                  self.train_dataset.append(temp_ds[ind])

class Target(torch.nn.Module):
    def __init__(self, dim, batch_size, num_of_classes):
        super(Target, self).__init__()

        self.conv_params = [
            [1, 32, (7, 7), 1, 0, 1],
            [32, 64, (7, 7), 1, 0, 1],
            [64, 128, (7, 7), 1, 0, 1],
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
            self.fc = nn.Linear(h * w * self.conv_params[lc_ind][1], (h * w * self.conv_params[lc_ind][1]) // 2)
            self.fc2 = nn.Linear((h * w * self.conv_params[lc_ind][1]) // 2, num_of_classes)
        else:
            self.fc = nn.Linear(h * w * self.conv_params[lc_ind][1], num_of_classes)

        self.sigmoid = nn.Sigmoid()

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
        output = self.fc(output)
        if self.number_of_fc == 2:
            output = F.relu(output)
            output = self.dropout(output)
            output = self.fc2(output)
        return output



def create_data_loader(train_dataset, test_dataset, batch_size = 1, shuffle = True):
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle = shuffle, batch_size = batch_size)
    return (train_loader, test_loader)

def conv_out_shape(conv_params, h, w):
    from math import floor
    cnt = 0
    for ind, conv in enumerate(conv_params):
        h = int(floor(h - conv[2][0] + (2 * conv[4]) / conv[3])) + 1
        w = int(floor(w - conv[2][1] + (2 * conv[4]) / conv[3])) + 1
        if conv[5]: # if conv[*][5] is true, calculate for max-pool
            h = int(floor(h/2))
            w = int(floor(w/2))
        print(f"Output (h, w) after applying conv{ind + 1}: {(h, w)}")
    return (h, w)

def plot_losses(losses, filename = 'loss.png'):
  plt = pyplot
  plt.plot(losses)
  plt.title('Epoch Vs Loss')
  plt.xlabel("Number Of Epochs")
  plt.ylabel("Training Loss")
  plt.savefig(filename)

def normalize_weight(face):
    return (face - torch.min(face)) / (torch.max(face) - torch.min(face))

def plot_Inverted_face(face, meta, label, loss):
    face = face.view(IMG_SIZE, IMG_SIZE).detach()
    face = (face - torch.min(face)) / (torch.max(face) - torch.min(face))
    restore = transforms.Compose([
        transforms.ToPILImage()])
    face =  restore(face)
    pyplot.imshow(face)
    pyplot.title(f'Inverted Face for Label: {label} (loss: {loss:.5f})')
    pyplot.savefig(f'{meta}')

def save_model(model, filename):
  torch.save(model.state_dict(), filename)

def load_model(model, filename):
  model.load_state_dict(torch.load(filename), strict=False)