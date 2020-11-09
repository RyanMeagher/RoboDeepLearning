import pickle

import torch
import torch.nn as nn


def createModArch():
    model = nn.Sequential(
        nn.Linear(6, 100),
        nn.Tanh(),
        nn.Dropout(0.2),
        nn.Linear(100, 50),
        nn.Tanh(),
        nn.Dropout(0.2),
        nn.Linear(50, 10),
        nn.Tanh(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )

    return model


class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        super().__init__()


        self.model = createModArch()

    # STUDENTS: __init__() must initiatize nn.Module and define your network's
    # custom architecture

    def forward(self, input):
        return self.model(input)

    def evaluate(self, model, test_loader, loss_function):
        losses = []
        scaler = pickle.load(open("saved/scaler.pkl", "rb"))
        # breaks the dataset into the batches so to find a loss over an
        # epoch we will enumerate over all batches and average scores
        for idx, sample in enumerate(test_loader):

            input, label =sample['input'].float(), sample['label'].float()
            out = model.forward(input.float())

            loss = loss_function(out, label.view(-1, 1))


            losses.append(loss)

        return sum(losses) / len(losses)

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()