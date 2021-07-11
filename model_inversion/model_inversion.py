from functools import partial
from math import sqrt
import numpy as np
import timeit
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
from pathlib import Path

from tools import *

# To store in and import files from google drive

# from google.colab import drive
# drive.mount('/content/drive2/')

# end of google drive mount related code

cwd = os.getcwd()
# print(f'Current Directory: {cwd}')
data_file_location = os.path.join(cwd, 'faces/')
os.chdir(data_file_location)


# Set device to cpu or gpu
CUDA_ID = None
DEVICE = None
if torch.cuda.is_available():
    CUDA_ID = torch.cuda.current_device()
    DEVICE = torch.device('cuda')
    print("Running On GPU")
else:
    DEVICE = "cpu"
    print("No GPU")



# All the functions, methods and classes in this file have self-explanatory names.
# We have added comments where something is ambigous



seed_all(60065)



IMG_SIZE = 64


# ImageLoader loads all the images and randomli splits the images to training and test set.
# data_paths: all the paths collect images from
# num_of_training_images: number of images to use out of 10 images in training
# num_of_labels: number of labels to include for training and testing out of 40 labels
training_image_loader = ImageLoader(data_paths = ["training", "testing"], num_of_training_images = 9, num_of_labels = 40)

batch_size = 1
num_of_classes = len(training_image_loader.labels)
train_loader, test_loader = create_data_loader(training_image_loader.train_dataset, training_image_loader.test_dataset, batch_size=batch_size)

print(f'training data len: {len(train_loader.dataset)}\ntest data len: {len(test_loader.dataset)}')
print(f'image shape: {train_loader.dataset[0][0].shape}')




# Given a model and a data loader, trains the model using the data from the data loader
def train(model, train_loader, patience = 5, model_path = "model.pt", batch_size = 1, print_loss = True, print_step = 20, epochs = 50):
    print("\nTraining started")
    start = timeit.default_timer()

    model = model
    losses = []
    total_batches = len(train_loader)
    for iter in range(epochs):
        epoc_loss = 0
        batch_processed = 0
        model.train()
        for X, Y in train_loader:
            Y_pred = model(X)
            loss = criterion(Y_pred, Y)
            epoc_loss += loss.item()
            batch_processed += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if print_loss and ((batch_processed % print_step == 0) or batch_processed == total_batches):
                print(".", end='')

        losses.append(epoc_loss / batch_processed)
        if print_loss:
            print("\nAvg. loss at epoch " + str(iter + 1) + ": " + str(epoc_loss / batch_processed))
        
    stop = timeit.default_timer()
    
    print("Training finished.\nTraining Took " + str(round((stop - start) / 60, 2)) + " Minutes\n\n")

    return losses



learning_rate = 0.0001
target = Target((IMG_SIZE, IMG_SIZE), batch_size=batch_size, num_of_classes=num_of_classes)

optimizer = torch.optim.Adam(target.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()
train_losses = train(target, train_loader, batch_size = batch_size)

os.chdir('..')

plot_losses(train_losses)

def test(model, test_loader):
    model.eval()
    total = 0.
    correct = 0.
    test_labels = []
    for X, Y in test_loader:
        Y_pred = model(X)
        pred = torch.argmax(Y_pred)
        total += 1 if len(Y_pred.size()) == 0 else Y_pred.shape[0]
        test_labels.append(Y.item())
        correct += np.sum(pred.item() == Y.item())

    print(f'Tested {len(set(test_labels))} unique classes (faces)\nTest Accuracy: {(100. * (correct / total)):.2f} %')

test(target, test_loader)

def attack_criterion(x, target_model, label_index):
    y = F.softmax(target_model(x), dim = 1)
    return 1. - y[0][label_index]


def MI_FACE(target_model, label_index, img_size, lr, alpha, process):
    torch.manual_seed(label_index)
    
    x = torch.zeros(img_size)
    x.requires_grad = True
    target_model.eval()

    min_x = None
    min_loss = 1e9
    
    for i in range(alpha):

        loss = attack_criterion(x, target_model, label_index)
        target_model.zero_grad()
        loss.backward()
        x.data = x - lr * x.grad.data
        x.data = process(x.data)
        x.grad.zero_()
        loss = loss.item()

        if min_loss > loss:
            min_loss = loss
            min_x = x.data
    
    return min_x, min_loss


Path('inverted_faces').mkdir(parents = True, exist_ok = True)
os.chdir('inverted_faces')

# inverting class 21
print('\ninverting class #21')
img, loss = MI_FACE(target, 21, (1, 1, IMG_SIZE, IMG_SIZE), 100, 1000, normalize_weight)
plot_Inverted_face(img, f'21_loss_{loss:.9f}.png', 21, loss)

# inverting class 30
print('inverting class #30')
img, loss = MI_FACE(target, 30, (1, 1, IMG_SIZE, IMG_SIZE), 100, 1000, normalize_weight)
plot_Inverted_face(img, f'30_loss_{loss:.9f}.png', 30, loss)

# inverting class 32
print('inverting class #32')
img, loss = MI_FACE(target, 32, (1, 1, IMG_SIZE, IMG_SIZE), 100, 1000, normalize_weight)
plot_Inverted_face(img, f'32_loss_{loss:.9f}.png', 32, loss)

# inverting class 33
print('inverting class #33')
img, loss = MI_FACE(target, 33, (1, 1, IMG_SIZE, IMG_SIZE), 100, 1000, normalize_weight)
plot_Inverted_face(img, f'33_loss_{loss:.9f}.png', 33, loss)

# inverting class 38
print('inverting class #38')
img, loss = MI_FACE(target, 38, (1, 1, IMG_SIZE, IMG_SIZE), 100, 1000, normalize_weight)
plot_Inverted_face(img, f'38_loss_{loss:.9f}.png', 38, loss)