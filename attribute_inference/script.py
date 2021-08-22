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
import sys
import torchvision.transforms as transforms
from random import sample
from pathlib import Path

from tools import *

# To store in and import files from google drive

# from google.colab import drive
# drive.mount('/content/drive2/')

# end of google drive mount related code

def main(dataset_path, target_model_path, attack_model_path):
    # cwd = os.getcwd()
    # print(f'Current Directory: {cwd}')
    # data_file_location = os.path.join(cwd, 'attribute_inference/')
    # os.chdir(data_file_location)

    Path("attribute_inference/models").mkdir(parents = True, exist_ok = True)


    # All the functions, methods and classes in this file have self-explanatory names.
    # We have added comments where something is ambigous



    seed_all(60065)



    # IMG_SIZE = 50


    # ImageLoader loads all the images and randomli splits the images to training and test set.
    # data_paths: all the paths collect images from
    # num_of_training_images: number of images to use out of 10 images in training
    # num_of_labels: number of labels to include for training and testing out of 40 labels
    training_image_loader = ImageLoader(data_paths = [dataset_path], train_percentage = 0.8)

    batch_size = 16
    num_of_race_classes = 4
    train_loader, test_loader = create_data_loader(
        [(x, y[0]) for (x, y) in training_image_loader.train_dataset],
        [(x, y[0]) for (x, y) in training_image_loader.test_dataset],
        batch_size=batch_size)

    print(f'training data len: {len(train_loader.dataset)}\ntest data len: {len(test_loader.dataset)}')
    print(f'image shape: {train_loader.dataset[0][0].shape}')




    # Given a model and a data loader, trains the model using the data from the data loader
    def train(model, train_loader, patience = 5, model_path = "model.pt", batch_size = 1, print_loss = True, print_step = 20, epochs = 14):
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
                Y_pred, _ = model(X)
                # Y = batch_one_hot(Y)
                # print(Y_pred.shape, Y.shape)
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
    target = Conv2DNet((IMG_SIZE, IMG_SIZE), batch_size=batch_size, num_of_classes=num_of_race_classes)

    optimizer = torch.optim.Adam(target.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_losses = []

    if target_model_path:
        load_model(target, target_model_path)
    else:
        train_losses = train(target, train_loader, batch_size = batch_size)
        save_model(target, "attribute_inference/models/ai_target_model.pt")
        plot_losses(train_losses, filename='target_train_loss')


    def test(model, test_loader):
        model.eval()
        total = 0.
        correct = 0.
        for X, Y in test_loader:
            Y_pred, _ = model(X)
            Y_pred = F.log_softmax(Y_pred, dim=1)
            pred = torch.argmax(Y_pred, dim=1)
            total += 1 if len(Y_pred.size()) == 0 else Y_pred.shape[0]
            correct += torch.sum(pred == Y)

        print(f'Test Accuracy: {(100. * (correct / total)):.2f} %')

    test(target, test_loader)

    def attack_train(target, attack, train_loader, print_loss = True, print_step = 20, epochs = 8):
        print("\nAttack model training started")
        start = timeit.default_timer()

        target.eval()
        attack.train()
        losses = []
        total_batches = len(train_loader)
        for iter in range(epochs):
            epoc_loss = 0
            batch_processed = 0
            for X, Y in train_loader:
                _, emb = target(X)
                Y_pred = attack(emb)
                Y = Y.float()
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

        print("Attack model training finished.\nTraining Took " + str(round((stop - start) / 60, 2)) + " Minutes\n\n")

        return losses

    def attack_test(target, attack, test_loader):
        target.eval()
        attack.eval()
        total = 0.
        correct = 0.
        for X, Y in test_loader:
            _, emb = target(X)
            Y_pred= attack(emb)
            total += 1 if len(Y_pred.size()) == 0 else Y_pred.shape[0]
            # print(torch.round(Y_pred))
            correct += torch.sum(torch.round(Y_pred) == Y)

        print(f'Attack Model Test Accuracy: {(100. * (correct / total)):.2f} %')

    train_loader, test_loader = create_data_loader(
        [(x, y[1]) for (x, y) in training_image_loader.train_dataset],
        [(x, y[1]) for (x, y) in training_image_loader.test_dataset],
        batch_size=batch_size)

    learning_rate = 0.0001
    attack = Attack((IMG_SIZE, IMG_SIZE), layer1_nodes=128, num_of_classes=1)

    optimizer = torch.optim.Adam(attack.parameters(), lr = learning_rate)
    criterion = nn.BCELoss()
    attack_train_losses = []

    if attack_model_path:
        load_model(attack, attack_model_path)
    else:
        attack_train_losses = attack_train(target, attack, train_loader, epochs=50)
        save_model(attack, 'attribute_inference/models/ai_attack_model.pt')
        plot_losses(attack_train_losses, filename='attack_train_loss')


    attack_test(target, attack, test_loader)