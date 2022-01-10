import torch
import torchvision
from torch import nn, optim, autograd
from torch.nn import functional as F
import numpy as np
from sklearn.utils import shuffle as skshuffle
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, trange
import pytest
from ../LB_utils import * 
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
from laplace import Laplace

from load_not_MNIST import notMNIST

#### SETTINGS

def main():
    p = ArgumentParser()
    p.add_argument('-s', '--num_seeds', type=int, default=5)
    p.add_argument('-o', '--out_folder', type=str, default='../Experiments_results/')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cuda_status = torch.cuda.is_available()
    print("device: ", device)
    print("cuda status: ", cuda_status)

    ### define network
    class ConvNet(nn.Module):
        
        def __init__(self, num_classes=10):
            super(ConvNet, self).__init__()
            
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(1, 16, 5),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2,2),
                torch.nn.Conv2d(16, 32, 5),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2,2),
                torch.nn.Flatten(),
                torch.nn.Linear(4 * 4 * 32, num_classes)
            )
        def forward(self, x):
            out = self.net(x)
            return out

    BATCH_SIZE = 128

    ### load data
    MNIST_transform = torchvision.transforms.ToTensor()
    MNIST_train = torchvision.datasets.MNIST(
            '~/data/mnist',
            train=True,
            download=True,
            transform=MNIST_transform)

    MNIST_train_loader = torch.utils.data.dataloader.DataLoader(
        MNIST_train,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    MNIST_test = torchvision.datasets.MNIST(
            '~/data/mnist',
            train=False,
            download=False,
            transform=MNIST_transform)

    MNIST_test_loader = torch.utils.data.dataloader.DataLoader(
        MNIST_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    FMNIST_test = torchvision.datasets.FashionMNIST(
        '~/data/fmnist', train=False, download=True,
        transform=MNIST_transform)  

    FMNIST_test_loader = torch.utils.data.DataLoader(
        FMNIST_test,
        batch_size=BATCH_SIZE, shuffle=False)

    KMNIST_test = torchvision.datasets.KMNIST(
        '~/data/kmnist', train=False, download=True,
        transform=MNIST_transform)

    KMNIST_test_loader = torch.utils.data.DataLoader(
        KMNIST_test,
        batch_size=BATCH_SIZE, shuffle=False)

    root = os.path.expanduser('~/data')

    # Instantiating the notMNIST dataset class we created
    notMNIST_test = notMNIST(root=os.path.join(root, 'notMNIST_small'),
                                   transform=MNIST_transform)

    notMNIST_test_loader = torch.utils.data.dataloader.DataLoader(
                                dataset=notMNIST_test,
                                batch_size=BATCH_SIZE,
                                shuffle=False)

    ### load model
    MNIST_PATH = "../pretrained_weights/MNIST_pretrained_10_classes_last_layer.pth"

    mnist_model = ConvNet().to(device)
    print("loading model from: {}".format(MNIST_PATH))
    mnist_model.load_state_dict(torch.load(MNIST_PATH))
    mnist_model.eval()

    #### Experiments

    # set up storage


    # run experiments
    for s in range(args.num_seeds):
    
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        targets = MNIST_test.targets.numpy()
        targets_FMNIST = FMNIST_test.targets.numpy()
        targets_notMNIST = notMNIST_test.targets.numpy().astype(int)
        targets_KMNIST = KMNIST_test.targets.numpy()

        #TODO


    #### save results


#### RUN
if __name__ == '__main__':
    main()