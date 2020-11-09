import random

import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.full_data = np.genfromtxt('saved/training_data.csv', delimiter=',')

        self.norm_scaler = MinMaxScaler()
        self.norm_scaler_full = MinMaxScaler()
        self.norm_scaler_full.fit(self.full_data)
        self.std_scaler = StandardScaler()
        self.y = self.full_data[:,-1]
        self.X =self. full_data[:,:-1]
        self.X_norm = self.norm_scaler.fit_transform(self.X)
        self.X_std = self.std_scaler.fit_transform(self.X)  # fits and transforms
        pickle.dump(self.norm_scaler_full, open("saved/scaler.pkl", "wb"))  # save to normalize at inference

    def __len__(self):
        return self.X[:, 0].size

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        return {'input': self.X_norm[idx],
                'label': self.y[idx]}

8


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        n = len(self.nav_dataset)
        split_idx = int(n * .75)
        idx = list(range(n))
        random.shuffle(idx)

        train_idx, test_idx = idx[:split_idx], idx[split_idx:]

        train_set = data.Subset(self.nav_dataset, train_idx)
        test_set = data.Subset(self.nav_dataset, test_idx)

        self.train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        input, label = sample['input'], sample['label']


    for idx, sample in enumerate(data_loaders.test_loader):
        input, label = sample['input'], sample['label']




if __name__ == '__main__':
    main()


