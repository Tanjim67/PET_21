import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2DNet(nn.Module):
    def __init__(self):
        super(Conv2DNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.fc1 = nn.Linear(9216, 128)# Why 9216? because in the forward funtion, this input of this layer has 9216 length, like [32, 9216], where 32 is the batch_size we set at line 18
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)  # change a matrix or feature map like [32, aa, bb, cc] to [32, aa X bb X cc]
        x = self.fc1(x)
        x = F.relu(x)
        output = self.fc2(x)
        #output = F.log_softmax(x, dim=1)  because we use nn.CrossEntropyLoss() as the loss function, it already include log_softmax
        return output


class AttackNet(nn.Module):
    def __init__(self):
        super(AttackNet, self).__init__()
        self.fc1 = nn.Linear(10, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        output = self.fc5(x)
        return output
