# torch imports
import torch
import torch.optim as optim
# others
import numpy as np
# local
from util import *
from nn import *



def train_target(trainloaderTarget, testloaderTarget, target_path):
    debug("Called train_target")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info("Using {} device".format(device))
    debug("sending Conv2DNet model to {} device".format(device))
    model = Conv2DNet().to(device)

    epochs = 50
    debug("epochs = {}".format(epochs))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    min_valid_loss = np.inf

    debug("start epoch loop")
    for epoch in range(epochs):  # loop over the dataset multiple times
        debug("starting epoch {}".format(epoch+1))
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloaderTarget, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)


            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                info('[%d/%d, %5d] loss: %.4f' %
                    (epoch + 1, epochs, i + 1, running_loss / 200))
                running_loss = 0.0

        # validiation loss after each epoch, todo cut down the test data set to proper validiation set
        debug("starting valid_loss evaluation")
        model.eval()
        valid_loss = 0.0
        for i, data in enumerate(testloaderTarget, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward Pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Calculate Loss
            valid_loss += loss.item()
        info('V-Loss=%.4f' % (valid_loss/len(testloaderTarget)))

        if min_valid_loss > valid_loss:
            min_valid_loss = valid_loss
        else:
            warn("valid_loss did not improve for this epoch")

    info('Finished training target model')
    PATH = './' + target_path
    torch.save(model.state_dict(), PATH)
    debug("saved model at " + PATH)
