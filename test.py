import torch as pt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
from torch import nn
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pytorch_fit_module as pfm

# NN classification with pytorch
# lets make some data

n_samples = 1000
x, y = make_circles(n_samples=n_samples,
                    noise=0.03,
                    random_state=1990)
x_pt = pt.tensor(x, device='cpu', dtype=pt.float32)
y_pt = pt.tensor(y, device='cpu', dtype=pt.float32)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1990)
# X_train = pt.tensor(X_train, device='cpu', dtype=pt.float32)
# X_test = pt.tensor(X_test, device='cpu', dtype=pt.float32)
#
# y_train = pt.tensor(y_train, device='cpu', dtype=pt.float32)
# y_test = pt.tensor(y_test, device='cpu', dtype=pt.float32)


class Circlemodelv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=5)  # that means we use 5 neurons
        self.layer2 = nn.Linear(in_features=5, out_features=5)
        self.layer3 = nn.Linear(in_features=5, out_features=5)
        self.layer4 = nn.Linear(in_features=5, out_features=1)

        self.two_linear_layers = nn.Sequential(self.layer1,
                                               nn.ReLU(),
                                               self.layer2,
                                               nn.Tanh(),
                                               self.layer3,
                                               nn.ReLU(),
                                               self.layer4,
                                               nn.Sigmoid())

    def forward(self, x):
        return self.two_linear_layers(x)  # x -> layer1 -> layer2 -> output


model_5 = Circlemodelv1()

pfm.TrainPytorchNN(train_split=(X_train, y_train),
                   valid_split=(X_test, y_test),
                   n_class=2, model=model_5, loss='BCELoss', metrics=['BinaryAUROC', 'Accuracy'],
                   verbose=True, epochs=500, batch_sizes=15, n_batches=10, device='cpu', optimizer='Adam',
                   random_seed=42, print_every=500, learning_rate=0.01,
                   )

