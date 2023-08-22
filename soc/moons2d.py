import torch
import numpy as np
from sklearn.datasets import make_moons
from torch.utils.data import Dataset

class Moons2D(Dataset):
    def __init__(self, nsamples, noise, uniform_noise=0, n_outliers=0, outlier_uniform_noise=0, div_factor=1, transform=None, center=False):
        self.nsamples = nsamples
        self.noise = noise
        X, labels = make_moons(n_samples=nsamples, noise=noise)

        if uniform_noise > 0:
            inds = np.arange(X.shape[0])
            if n_outliers > 0:
                np.random.shuffle(inds)

                r = outlier_uniform_noise * np.sqrt(np.random.rand(n_outliers))
                theta = np.random.rand(n_outliers) * 2 * np.pi
                X[inds[:n_outliers], 0] += r * np.cos(theta)
                X[inds[:n_outliers], 1] += r * np.sin(theta)

            r = uniform_noise * np.sqrt(np.random.rand(X.shape[0]-n_outliers))
            theta = np.random.rand(X.shape[0]-n_outliers) * 2 * np.pi
            X[inds[n_outliers:], 0] += r * np.cos(theta)
            X[inds[n_outliers:], 1] += r * np.sin(theta)

        if center:
            X[:, 0] += 1.0
            X[:, 1] += 0.5
            X[:, 0] /= 3.0
            X[:, 1] /= 1.5

        self.data = X
        self.labels = labels
        self.transform = transform
        self.div_factor = div_factor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        value = torch.Tensor(self.data[index])
        value /= self.div_factor
        return value, self.labels[index]