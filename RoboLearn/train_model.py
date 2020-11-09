import pickle

import Data_Loaders as dl
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np



def train_model(no_epochs):
    batch_size = 32
    data_loaders = dl.Data_Loaders(batch_size)
    model = Action_Conditioned_FF()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.model.parameters(), lr=0.001)

    test_losses = []
    train_losses = []

    for epoch_i in range(no_epochs):
        #print(f"epoch # {epoch_i + 1}")
        model.train()
        loss_test = model.evaluate(model, data_loaders.test_loader, loss_fn)
        print('test')
        print(loss_test)
        test_losses.append(loss_test)
        l=[]

        for idx, sample in enumerate(data_loaders.train_loader):  # sample['input'] and sample['label']
            input, label = sample['input'].float(), sample['label'].float()

            #forward step
            out = model.forward(input)
            loss_train = loss_fn(out, label.view(-1,1))
            l.append(loss_train)

            #backpropagation
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
        print('train')
        print(sum(l)/len(l))
        train_losses.append(sum(l) / len(l))

    torch.save(model.state_dict(), 'saved_model.pkl', _use_new_zipfile_serialization=False)



def createPlot(train_losses, test_losses):
    # create plotting variables from lists
    plotter_train=[(i,k) for i, k in enumerate(train_losses)]
    plotter_test=[(i,k) for i, k in enumerate(test_losses)]
    # plot training and testing losses
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('loss')

    ax.plot([x[0] for x in plotter_train], [x[1] for x in plotter_train], color='tab:blue')
    ax.plot([x[0] for x in plotter_test], [x[1] for x in plotter_test], color='tab:orange')
    plt.show()

    # save file to be used in goal_seekers.py
    torch.save(model.state_dict(), 'saved_model.pkl', _use_new_zipfile_serialization=False)



no_epochs = 150
train_model(no_epochs)
