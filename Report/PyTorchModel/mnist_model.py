import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import parse_data
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
import csv

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(28*28, 2*7*7)
        self.fc1 = nn.Linear(2*7*7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = x.view(in_size, -1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)

use_cuda = torch.cuda.is_available()
#use_cuda = False
device = torch.device("cuda:0" if use_cuda else "cpu")

X_train, Y_train, X_test, Y_test = parse_data.loadData('mnist_train.csv', 'mnist_test.csv', device)

def finalTrainAndTest():
    start = timer()
    n_epochs = 100
    net = Net()
    net.cuda()    
    lrate = 0.01
    momen = 0.9
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lrate, momentum=0.0)
    #Test for number of epochs we found with above function
    for i in range(n_epochs):
        running_loss = 0.0

        batch_s = 1
        lb = 0
        ub = batch_s
        
        if i == 15:
            lrate = 1e-3
        elif i == 30:
            lrate = 1e-4
        elif i == 45:
            lrate = 1e-5

        for g in optimizer.param_groups:
            g['lr'] = lrate

        while ub <= len(X_train):
            optimizer.zero_grad() 
            output = net(X_train[lb: ub])
            loss = criterion(output, Y_train[lb: ub])
            loss.backward()
            optimizer.step()
            lb += batch_s
            ub += batch_s
            running_loss += loss.item()
        
        num_correct = 0
        val_guess = net(X_test)
        loss = criterion(val_guess, Y_test)

        for j in range(len(Y_test)):
            if torch.argmax(val_guess[j]) == Y_test[j]:
                num_correct += 1

        acc = (num_correct / len(Y_test))

        test_loss = 0.0 + loss.item()

        print("Epoch: " + str(i) + ": " + str(acc))
        print("Training loss: " + str((batch_s * running_loss) / len(X_train)))
        print("Test loss: " + str(test_loss))
        end = timer()
        print("Training time: " + str(end - start) + " seconds\n")

finalTrainAndTest()
