import torch
import numpy as np
from torch.utils.data import Dataset

class MNISTGraphDataset(Dataset):
    def __init__(self, dataset_path, num_pix, train=True, intensities=True, num=-1):
        if train:
            dataset = np.loadtxt(dataset_path + 'mnist_train.csv', delimiter=',', dtype=np.float32)
        else:
            dataset = np.loadtxt(dataset_path + 'mnist_test.csv', delimiter=',', dtype=np.float32)

        print("MNIST CSV Loaded")

        if isinstance(num, list):
            map1 = list(map(lambda x: x in num, dataset[:, 0]))
            dataset = dataset[map1]
        elif num > -1:
            dataset = dataset[dataset[:, 0] == num]

        print(dataset.shape)

        X_pre = (dataset[:, 1:] - 127.5) / 255.0

        imrange = np.linspace(-0.5, 0.5, num=num_pix, endpoint=False)

        xs, ys = np.meshgrid(imrange, imrange)

        xs = xs.reshape(-1)
        ys = ys.reshape(-1)

        self.X = np.array(list(map(lambda x: np.array([xs, ys, x]).T, X_pre)))

        if not intensities:
            self.X = np.array(list(map(lambda x: x[x[-num_pix:, 2].argsort()][:, :2], self.X)))
        else:
            self.X = np.array(list(map(lambda x: x[x[-num_pix:,].argsort()][:], self.X)))

        self.X = torch.FloatTensor(self.X)

        # one-hot labels
        labels = dataset[:,0]
        y = []
        for i in range(len(labels)):
            one_hot = []
            for j in range(10):
                one_hot.append(0)
            num = int(labels[i])
            one_hot[num] = 1
            y.append(one_hot)
        self.Y = torch.tensor(y).to(torch.float32)

        print(self.X.shape)
        print("Data Processed")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])
