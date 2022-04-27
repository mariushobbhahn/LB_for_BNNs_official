import torch
import torchvision
from torch import nn, optim, autograd
from torch.nn import functional as F
import numpy as np
from sklearn.utils import shuffle as skshuffle
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, trange
import pytest
from utils.LB_utils import *
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
from laplace import Laplace
import utils.scoring as scoring

from utils.load_not_MNIST import notMNIST
import argparse
from laplace.curvature import AsdlGGN
from laplace.curvature.backpack import BackPackGGN

#### SETTINGS

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-s', '--num_seeds', type=int, default=5)
    p.add_argument('-o', '--out_folder', type=str, default='../Experiments_results/')
    p.add_argument('-l', '--last_layer', type=bool, default=True)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cuda_status = torch.cuda.is_available()
    print("device: ", device)
    print("cuda status: ", cuda_status)
    print("last_layer: ", args.last_layer)
    print("all_layers: ", not args.last_layer)

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
    MNIST_PATH = os.getcwd() + "/pretrained_weights/MNIST_pretrained_10_classes_last_layer_s1.pth"

    mnist_model = ConvNet().to(device)
    print("loading model from: {}".format(MNIST_PATH))
    mnist_model.load_state_dict(torch.load(MNIST_PATH))
    mnist_model.eval()
    
    targets = MNIST_test.targets.numpy()
    targets_FMNIST = FMNIST_test.targets.numpy()
    targets_notMNIST = notMNIST_test.targets.numpy().astype(int)
    targets_KMNIST = KMNIST_test.targets.numpy()
    
    num_samples = 100

    #### Experiments

    # MAP
    MAP_MMC_in = []
    MAP_MMC_FMNIST = []
    MAP_MMC_notMNIST = []
    MAP_MMC_KMNIST = []
    MAP_AUROC_FMNIST = []
    MAP_AUROC_notMNIST = []
    MAP_AUROC_KMNIST = []
    MAP_NLL_in = []
    MAP_NLL_FMNIST = []
    MAP_NLL_notMNIST = []
    MAP_NLL_KMNIST = []
    MAP_ECE_in = []
    MAP_ECE_FMNIST = []
    MAP_ECE_notMNIST = []
    MAP_ECE_KMNIST = []
    MAP_Brier_in = []
    MAP_Brier_FMNIST = []
    MAP_Brier_notMNIST = []
    MAP_Brier_KMNIST = []
    
    # Diag samples
    Diag_samples_MMC_in = []
    Diag_samples_MMC_FMNIST = []
    Diag_samples_MMC_notMNIST = []
    Diag_samples_MMC_KMNIST = []
    Diag_samples_AUROC_FMNIST = []
    Diag_samples_AUROC_notMNIST = []
    Diag_samples_AUROC_KMNIST = []
    Diag_samples_NLL_in = []
    Diag_samples_NLL_FMNIST = []
    Diag_samples_NLL_notMNIST = []
    Diag_samples_NLL_KMNIST = []
    Diag_samples_ECE_in = []
    Diag_samples_ECE_FMNIST = []
    Diag_samples_ECE_notMNIST = []
    Diag_samples_ECE_KMNIST = []
    Diag_samples_Brier_in = []
    Diag_samples_Brier_FMNIST = []
    Diag_samples_Brier_notMNIST = []
    Diag_samples_Brier_KMNIST = []
    
    
    # KFAC samples
    KFAC_samples_MMC_in = []
    KFAC_samples_MMC_FMNIST = []
    KFAC_samples_MMC_notMNIST = []
    KFAC_samples_MMC_KMNIST = []
    KFAC_samples_AUROC_FMNIST = []
    KFAC_samples_AUROC_notMNIST = []
    KFAC_samples_AUROC_KMNIST = []
    KFAC_samples_NLL_in = []
    KFAC_samples_NLL_FMNIST = []
    KFAC_samples_NLL_notMNIST = []
    KFAC_samples_NLL_KMNIST = []
    KFAC_samples_ECE_in = []
    KFAC_samples_ECE_FMNIST = []
    KFAC_samples_ECE_notMNIST = []
    KFAC_samples_ECE_KMNIST = []
    KFAC_samples_Brier_in = []
    KFAC_samples_Brier_FMNIST = []
    KFAC_samples_Brier_notMNIST = []
    KFAC_samples_Brier_KMNIST = []

    # Diag LB
    Diag_LB_MMC_in = []
    Diag_LB_MMC_FMNIST = []
    Diag_LB_MMC_notMNIST = []
    Diag_LB_MMC_KMNIST = []
    Diag_LB_AUROC_FMNIST = []
    Diag_LB_AUROC_notMNIST = []
    Diag_LB_AUROC_KMNIST = []
    Diag_LB_NLL_in = []
    Diag_LB_NLL_FMNIST = []
    Diag_LB_NLL_notMNIST = []
    Diag_LB_NLL_KMNIST = []
    Diag_LB_ECE_in = []
    Diag_LB_ECE_FMNIST = []
    Diag_LB_ECE_notMNIST = []
    Diag_LB_ECE_KMNIST = []
    Diag_LB_Brier_in = []
    Diag_LB_Brier_FMNIST = []
    Diag_LB_Brier_notMNIST = []
    Diag_LB_Brier_KMNIST = []
    
    # KFAC LB
    KFAC_LB_MMC_in = []
    KFAC_LB_MMC_FMNIST = []
    KFAC_LB_MMC_notMNIST = []
    KFAC_LB_MMC_KMNIST = []
    KFAC_LB_AUROC_FMNIST = []
    KFAC_LB_AUROC_notMNIST = []
    KFAC_LB_AUROC_KMNIST = [] 
    KFAC_LB_NLL_in = []
    KFAC_LB_NLL_FMNIST = []
    KFAC_LB_NLL_notMNIST = []
    KFAC_LB_NLL_KMNIST = []
    KFAC_LB_ECE_in = []
    KFAC_LB_ECE_FMNIST = []
    KFAC_LB_ECE_notMNIST = []
    KFAC_LB_ECE_KMNIST = []
    KFAC_LB_Brier_in = []
    KFAC_LB_Brier_FMNIST = []
    KFAC_LB_Brier_notMNIST = []
    KFAC_LB_Brier_KMNIST = []

    # Diag LB normalized
    Diag_LB_norm_MMC_in = []
    Diag_LB_norm_MMC_FMNIST = []
    Diag_LB_norm_MMC_notMNIST = []
    Diag_LB_norm_MMC_KMNIST = []
    Diag_LB_norm_AUROC_FMNIST = []
    Diag_LB_norm_AUROC_notMNIST = []
    Diag_LB_norm_AUROC_KMNIST = []
    Diag_LB_norm_NLL_in = []
    Diag_LB_norm_NLL_FMNIST = []
    Diag_LB_norm_NLL_notMNIST = []
    Diag_LB_norm_NLL_KMNIST = []
    Diag_LB_norm_ECE_in = []
    Diag_LB_norm_ECE_FMNIST = []
    Diag_LB_norm_ECE_notMNIST = []
    Diag_LB_norm_ECE_KMNIST = []
    Diag_LB_norm_Brier_in = []
    Diag_LB_norm_Brier_FMNIST = []
    Diag_LB_norm_Brier_notMNIST = []
    Diag_LB_norm_Brier_KMNIST = []
    
    # KFAC LB normalized
    KFAC_LB_norm_MMC_in = []
    KFAC_LB_norm_MMC_FMNIST = []
    KFAC_LB_norm_MMC_notMNIST = []
    KFAC_LB_norm_MMC_KMNIST = []
    KFAC_LB_norm_AUROC_FMNIST = []
    KFAC_LB_norm_AUROC_notMNIST = []
    KFAC_LB_norm_AUROC_KMNIST = [] 
    KFAC_LB_norm_NLL_in = []
    KFAC_LB_norm_NLL_FMNIST = []
    KFAC_LB_norm_NLL_notMNIST = []
    KFAC_LB_norm_NLL_KMNIST = []
    KFAC_LB_norm_ECE_in = []
    KFAC_LB_norm_ECE_FMNIST = []
    KFAC_LB_norm_ECE_notMNIST = []
    KFAC_LB_norm_ECE_KMNIST = []
    KFAC_LB_norm_Brier_in = []
    KFAC_LB_norm_Brier_FMNIST = []
    KFAC_LB_norm_Brier_notMNIST = []
    KFAC_LB_norm_Brier_KMNIST = []
    
    # Diag PROBIT
    Diag_PROBIT_MMC_in = []
    Diag_PROBIT_MMC_FMNIST = []
    Diag_PROBIT_MMC_notMNIST = []
    Diag_PROBIT_MMC_KMNIST = []
    Diag_PROBIT_AUROC_FMNIST = []
    Diag_PROBIT_AUROC_notMNIST = []
    Diag_PROBIT_AUROC_KMNIST = []
    Diag_PROBIT_NLL_in = []
    Diag_PROBIT_NLL_FMNIST = []
    Diag_PROBIT_NLL_notMNIST = []
    Diag_PROBIT_NLL_KMNIST = []
    Diag_PROBIT_ECE_in = []
    Diag_PROBIT_ECE_FMNIST = []
    Diag_PROBIT_ECE_notMNIST = []
    Diag_PROBIT_ECE_KMNIST = []
    Diag_PROBIT_Brier_in = []
    Diag_PROBIT_Brier_FMNIST = []
    Diag_PROBIT_Brier_notMNIST = []
    Diag_PROBIT_Brier_KMNIST = []
    
    # KFAC PROBIT
    KFAC_PROBIT_MMC_in = []
    KFAC_PROBIT_MMC_FMNIST = []
    KFAC_PROBIT_MMC_notMNIST = []
    KFAC_PROBIT_MMC_KMNIST = []
    KFAC_PROBIT_AUROC_FMNIST = []
    KFAC_PROBIT_AUROC_notMNIST = []
    KFAC_PROBIT_AUROC_KMNIST = []
    KFAC_PROBIT_NLL_in = []
    KFAC_PROBIT_NLL_FMNIST = []
    KFAC_PROBIT_NLL_notMNIST = []
    KFAC_PROBIT_NLL_KMNIST = []
    KFAC_PROBIT_ECE_in = []
    KFAC_PROBIT_ECE_FMNIST = []
    KFAC_PROBIT_ECE_notMNIST = []
    KFAC_PROBIT_ECE_KMNIST = []
    KFAC_PROBIT_Brier_in = []
    KFAC_PROBIT_Brier_FMNIST = []
    KFAC_PROBIT_Brier_notMNIST = []
    KFAC_PROBIT_Brier_KMNIST = []
    
    """
    # Diag SODPP
    Diag_SODPP_MMC_in = []
    Diag_SODPP_MMC_FMNIST = []
    Diag_SODPP_MMC_notMNIST = []
    Diag_SODPP_MMC_KMNIST = []
    Diag_SODPP_AUROC_FMNIST = []
    Diag_SODPP_AUROC_notMNIST = []
    Diag_SODPP_AUROC_KMNIST = []
    Diag_SODPP_NLL_in = []
    Diag_SODPP_NLL_FMNIST = []
    Diag_SODPP_NLL_notMNIST = []
    Diag_SODPP_NLL_KMNIST = []
    Diag_SODPP_ECE_in = []
    Diag_SODPP_ECE_FMNIST = []
    Diag_SODPP_ECE_notMNIST = []
    Diag_SODPP_ECE_KMNIST = []
    
    # KFAC SODPP
    KFAC_SODPP_MMC_in = []
    KFAC_SODPP_MMC_FMNIST = []
    KFAC_SODPP_MMC_notMNIST = []
    KFAC_SODPP_MMC_KMNIST = []
    KFAC_SODPP_AUROC_FMNIST = []
    KFAC_SODPP_AUROC_notMNIST = []
    KFAC_SODPP_AUROC_KMNIST = []
    KFAC_SODPP_NLL_in = []
    KFAC_SODPP_NLL_FMNIST = []
    KFAC_SODPP_NLL_notMNIST = []
    KFAC_SODPP_NLL_KMNIST = []
    KFAC_SODPP_ECE_in = []
    KFAC_SODPP_ECE_FMNIST = []
    KFAC_SODPP_ECE_notMNIST = []
    KFAC_SODPP_ECE_KMNIST = []
    """
    
    # run experiments
    for s in range(args.num_seeds):

        print("seed: ", s)
    
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        la_diag = Laplace(mnist_model, 'classification', 
                     subset_of_weights='last_layer' if args.last_layer else "all", 
                     hessian_structure='diag',
                     backend=BackPackGGN if args.last_layer else AsdlGGN,
                     prior_precision=5e-4) # 5e-4 # Choose prior precision according to weight decay
        la_diag.fit(MNIST_train_loader)
        
        la_kron = Laplace(mnist_model, 'classification', 
                     subset_of_weights='last_layer' if args.last_layer else "all", 
                     hessian_structure='kron',
                     backend=BackPackGGN if args.last_layer else AsdlGGN,
                     prior_precision=5e-4) # 5e-4 # Choose prior precision according to weight decay
        la_kron.fit(MNIST_train_loader)

        #MAP estimates
        MNIST_test_in_MAP = predict_MAP(mnist_model, MNIST_test_loader, device=device).cpu().numpy()
        MNIST_test_out_FMNIST_MAP = predict_MAP(mnist_model, FMNIST_test_loader, device=device).cpu().numpy()
        MNIST_test_out_notMNIST_MAP = predict_MAP(mnist_model, notMNIST_test_loader, device=device).cpu().numpy()
        MNIST_test_out_KMNIST_MAP = predict_MAP(mnist_model, KMNIST_test_loader, device=device).cpu().numpy()
        
        acc_in_MAP, prob_correct_in_MAP, ent_in_MAP, MMC_in_MAP = get_in_dist_values(MNIST_test_in_MAP, targets)
        acc_out_FMNIST_MAP, prob_correct_out_FMNIST_MAP, ent_out_FMNIST_MAP, MMC_out_FMNIST_MAP, auroc_out_FMNIST_MAP = get_out_dist_values(MNIST_test_in_MAP, MNIST_test_out_FMNIST_MAP, targets_FMNIST)
        acc_out_notMNIST_MAP, prob_correct_out_notMNIST_MAP, ent_out_notMNIST_MAP, MMC_out_notMNIST_MAP, auroc_out_notMNIST_MAP = get_out_dist_values(MNIST_test_in_MAP, MNIST_test_out_notMNIST_MAP, targets_notMNIST)
        acc_out_KMNIST_MAP, prob_correct_out_KMNIST_MAP, ent_out_KMNIST_MAP, MMC_out_KMNIST_MAP, auroc_out_KMNIST_MAP = get_out_dist_values(MNIST_test_in_MAP, MNIST_test_out_KMNIST_MAP, targets_KMNIST)
        
        MAP_MMC_in.append(MMC_in_MAP)
        MAP_MMC_FMNIST.append(MMC_out_FMNIST_MAP)
        MAP_MMC_notMNIST.append(MMC_out_notMNIST_MAP)
        MAP_MMC_KMNIST.append(MMC_out_KMNIST_MAP)
        MAP_AUROC_FMNIST.append(auroc_out_FMNIST_MAP)
        MAP_AUROC_notMNIST.append(auroc_out_notMNIST_MAP)
        MAP_AUROC_KMNIST.append(auroc_out_KMNIST_MAP)
        MAP_NLL_in.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_in_MAP)).log_prob(torch.tensor(targets)).mean().item())
        MAP_NLL_FMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_FMNIST_MAP)).log_prob(torch.tensor(targets_FMNIST)).mean().item())
        MAP_NLL_notMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_notMNIST_MAP)).log_prob(torch.tensor(targets_notMNIST)).mean().item())
        MAP_NLL_KMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_KMNIST_MAP)).log_prob(torch.tensor(targets_KMNIST)).mean().item())
        MAP_ECE_in.append(scoring.expected_calibration_error(targets, MNIST_test_in_MAP))
        MAP_ECE_FMNIST.append(scoring.expected_calibration_error(targets_FMNIST, MNIST_test_out_FMNIST_MAP))
        MAP_ECE_notMNIST.append(scoring.expected_calibration_error(targets_notMNIST, MNIST_test_out_notMNIST_MAP))
        MAP_ECE_KMNIST.append(scoring.expected_calibration_error(targets_KMNIST, MNIST_test_out_KMNIST_MAP))
        MAP_Brier_in.append(get_brier(MNIST_test_in_MAP, targets, n_classes=10))
        MAP_Brier_FMNIST.append(get_brier(MNIST_test_out_FMNIST_MAP, targets_FMNIST, n_classes=10))
        MAP_Brier_notMNIST.append(get_brier(MNIST_test_out_notMNIST_MAP, targets_notMNIST, n_classes=10))
        MAP_Brier_KMNIST.append(get_brier(MNIST_test_out_KMNIST_MAP, targets_KMNIST, n_classes=10))


        #Diag samples
        MNIST_test_in_D = predict_samples(la_diag, MNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_FMNIST_D = predict_samples(la_diag, FMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_notMNIST_D = predict_samples(la_diag, notMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_KMNIST_D = predict_samples(la_diag, KMNIST_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_D, prob_correct_in_D, ent_in_D, MMC_in_D = get_in_dist_values(MNIST_test_in_D, targets)
        acc_out_FMNIST_D, prob_correct_out_FMNIST_D, ent_out_FMNIST_D, MMC_out_FMNIST_D, auroc_out_FMNIST_D = get_out_dist_values(MNIST_test_in_D, MNIST_test_out_FMNIST_D, targets_FMNIST)
        acc_out_notMNIST_D, prob_correct_out_notMNIST_D, ent_out_notMNIST_D, MMC_out_notMNIST_D, auroc_out_notMNIST_D = get_out_dist_values(MNIST_test_in_D, MNIST_test_out_notMNIST_D, targets_notMNIST)
        acc_out_KMNIST_D, prob_correct_out_KMNIST_D, ent_out_KMNIST_D, MMC_out_KMNIST_D, auroc_out_KMNIST_D = get_out_dist_values(MNIST_test_in_D, MNIST_test_out_KMNIST_D, targets_KMNIST)
        
        Diag_samples_MMC_in.append(MMC_in_D)
        Diag_samples_MMC_FMNIST.append(MMC_out_FMNIST_D)
        Diag_samples_MMC_notMNIST.append(MMC_out_notMNIST_D)
        Diag_samples_MMC_KMNIST.append(MMC_out_KMNIST_D)
        Diag_samples_AUROC_FMNIST.append(auroc_out_FMNIST_D)
        Diag_samples_AUROC_notMNIST.append(auroc_out_notMNIST_D)
        Diag_samples_AUROC_KMNIST.append(auroc_out_KMNIST_D)
        Diag_samples_NLL_in.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_in_D)).log_prob(torch.tensor(targets)).mean().item())
        Diag_samples_NLL_FMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_FMNIST_D)).log_prob(torch.tensor(targets_FMNIST)).mean().item())
        Diag_samples_NLL_notMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_notMNIST_D)).log_prob(torch.tensor(targets_notMNIST)).mean().item())
        Diag_samples_NLL_KMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_KMNIST_D)).log_prob(torch.tensor(targets_KMNIST)).mean().item())
        Diag_samples_ECE_in.append(scoring.expected_calibration_error(targets, MNIST_test_in_D))
        Diag_samples_ECE_FMNIST.append(scoring.expected_calibration_error(targets_FMNIST, MNIST_test_out_FMNIST_D))
        Diag_samples_ECE_notMNIST.append(scoring.expected_calibration_error(targets_notMNIST, MNIST_test_out_notMNIST_D))
        Diag_samples_ECE_KMNIST.append(scoring.expected_calibration_error(targets_KMNIST, MNIST_test_out_KMNIST_D))
        Diag_samples_Brier_in.append(get_brier(MNIST_test_in_D, targets, n_classes=10))
        Diag_samples_Brier_FMNIST.append(get_brier(MNIST_test_out_FMNIST_D, targets_FMNIST, n_classes=10))
        Diag_samples_Brier_notMNIST.append(get_brier(MNIST_test_out_notMNIST_D, targets_notMNIST, n_classes=10))
        Diag_samples_Brier_KMNIST.append(get_brier(MNIST_test_out_KMNIST_D, targets_KMNIST, n_classes=10))

    
            
        #KFAC samples
        MNIST_test_in_KFAC = predict_samples(la_kron, MNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_FMNIST_KFAC = predict_samples(la_kron, FMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_notMNIST_KFAC = predict_samples(la_kron, notMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_KMNIST_KFAC = predict_samples(la_kron, KMNIST_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_KFAC, prob_correct_in_KFAC, ent_in_KFAC, MMC_in_KFAC = get_in_dist_values(MNIST_test_in_KFAC, targets)
        acc_out_FMNIST_KFAC, prob_correct_out_FMNIST_KFAC, ent_out_FMNIST_KFAC, MMC_out_FMNIST_KFAC, auroc_out_FMNIST_KFAC = get_out_dist_values(MNIST_test_in_KFAC, MNIST_test_out_FMNIST_KFAC, targets_FMNIST)
        acc_out_notMNIST_KFAC, prob_correct_out_notMNIST_KFAC, ent_out_notMNIST_KFAC, MMC_out_notMNIST_KFAC, auroc_out_notMNIST_KFAC = get_out_dist_values(MNIST_test_in_KFAC, MNIST_test_out_notMNIST_KFAC, targets_notMNIST)
        acc_out_KMNIST_KFAC, prob_correct_out_KMNIST_KFAC, ent_out_KMNIST_KFAC, MMC_out_KMNIST_KFAC, auroc_out_KMNIST_KFAC = get_out_dist_values(MNIST_test_in_KFAC, MNIST_test_out_KMNIST_KFAC, targets_KMNIST)
        
        # KFAC samples
        KFAC_samples_MMC_in.append(MMC_in_KFAC)
        KFAC_samples_MMC_FMNIST.append(MMC_out_FMNIST_KFAC)
        KFAC_samples_MMC_notMNIST.append(MMC_out_notMNIST_KFAC)
        KFAC_samples_MMC_KMNIST.append(MMC_out_KMNIST_KFAC)
        KFAC_samples_AUROC_FMNIST.append(auroc_out_FMNIST_KFAC)
        KFAC_samples_AUROC_notMNIST.append(auroc_out_notMNIST_KFAC)
        KFAC_samples_AUROC_KMNIST.append(auroc_out_KMNIST_KFAC)
        KFAC_samples_NLL_in.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_in_KFAC)).log_prob(torch.tensor(targets)).mean().item())
        KFAC_samples_NLL_FMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_FMNIST_KFAC)).log_prob(torch.tensor(targets_FMNIST)).mean().item())
        KFAC_samples_NLL_notMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_notMNIST_KFAC)).log_prob(torch.tensor(targets_notMNIST)).mean().item())
        KFAC_samples_NLL_KMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_KMNIST_KFAC)).log_prob(torch.tensor(targets_KMNIST)).mean().item())
        KFAC_samples_ECE_in.append(scoring.expected_calibration_error(targets, MNIST_test_in_KFAC))
        KFAC_samples_ECE_FMNIST.append(scoring.expected_calibration_error(targets_FMNIST, MNIST_test_out_FMNIST_KFAC))
        KFAC_samples_ECE_notMNIST.append(scoring.expected_calibration_error(targets_notMNIST, MNIST_test_out_notMNIST_KFAC))
        KFAC_samples_ECE_KMNIST.append(scoring.expected_calibration_error(targets_KMNIST, MNIST_test_out_KMNIST_KFAC))
        KFAC_samples_Brier_in.append(get_brier(MNIST_test_in_KFAC, targets, n_classes=10))
        KFAC_samples_Brier_FMNIST.append(get_brier(MNIST_test_out_FMNIST_KFAC, targets_FMNIST, n_classes=10))
        KFAC_samples_Brier_notMNIST.append(get_brier(MNIST_test_out_notMNIST_KFAC, targets_notMNIST, n_classes=10))
        KFAC_samples_Brier_KMNIST.append(get_brier(MNIST_test_out_KMNIST_KFAC, targets_KMNIST, n_classes=10))


        
        #LB diag
        MNIST_test_in_LB_D = predict_LB(la_diag, MNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_FMNIST_LB_D = predict_LB(la_diag, FMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_notMNIST_LB_D = predict_LB(la_diag, notMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_KMNIST_LB_D = predict_LB(la_diag, KMNIST_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_LB_D, prob_correct_in_LB_D, ent_in_LB_D, MMC_in_LB_D = get_in_dist_values(MNIST_test_in_LB_D, targets)
        acc_out_FMNIST_LB_D, prob_correct_out_FMNIST_LB_D, ent_out_FMNIST_LB_D, MMC_out_FMNIST_LB_D, auroc_out_FMNIST_LB_D = get_out_dist_values(MNIST_test_in_LB_D, MNIST_test_out_FMNIST_LB_D, targets_FMNIST)
        acc_out_notMNIST_LB_D, prob_correct_out_notMNIST_LB_D, ent_out_notMNIST_LB_D, MMC_out_notMNIST_LB_D, auroc_out_notMNIST_LB_D = get_out_dist_values(MNIST_test_in_LB_D, MNIST_test_out_notMNIST_LB_D, targets_notMNIST)
        acc_out_KMNIST_LB_D, prob_correct_out_KMNIST_LB_D, ent_out_KMNIST_LB_D, MMC_out_KMNIST_LB_D, auroc_out_KMNIST_LB_D = get_out_dist_values(MNIST_test_in_LB_D, MNIST_test_out_KMNIST_LB_D, targets_KMNIST)
        
        Diag_LB_MMC_in.append(MMC_in_LB_D)
        Diag_LB_MMC_FMNIST.append(MMC_out_FMNIST_LB_D)
        Diag_LB_MMC_notMNIST.append(MMC_out_notMNIST_LB_D)
        Diag_LB_MMC_KMNIST.append(MMC_out_KMNIST_LB_D)
        Diag_LB_AUROC_FMNIST.append(auroc_out_FMNIST_LB_D)
        Diag_LB_AUROC_notMNIST.append(auroc_out_notMNIST_LB_D)
        Diag_LB_AUROC_KMNIST.append(auroc_out_KMNIST_LB_D)
        Diag_LB_NLL_in.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_in_LB_D)).log_prob(torch.tensor(targets)).mean().item())
        Diag_LB_NLL_FMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_FMNIST_LB_D)).log_prob(torch.tensor(targets_FMNIST)).mean().item())
        Diag_LB_NLL_notMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_notMNIST_LB_D)).log_prob(torch.tensor(targets_notMNIST)).mean().item())
        Diag_LB_NLL_KMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_KMNIST_LB_D)).log_prob(torch.tensor(targets_KMNIST)).mean().item())
        Diag_LB_ECE_in.append(scoring.expected_calibration_error(targets, MNIST_test_in_LB_D))
        Diag_LB_ECE_FMNIST.append(scoring.expected_calibration_error(targets_FMNIST, MNIST_test_out_FMNIST_LB_D))
        Diag_LB_ECE_notMNIST.append(scoring.expected_calibration_error(targets_notMNIST, MNIST_test_out_notMNIST_LB_D))
        Diag_LB_ECE_KMNIST.append(scoring.expected_calibration_error(targets_KMNIST, MNIST_test_out_KMNIST_LB_D))
        Diag_LB_Brier_in.append(get_brier(MNIST_test_in_LB_D, targets, n_classes=10))
        Diag_LB_Brier_FMNIST.append(get_brier(MNIST_test_out_FMNIST_LB_D, targets_FMNIST, n_classes=10))
        Diag_LB_Brier_notMNIST.append(get_brier(MNIST_test_out_notMNIST_LB_D, targets_notMNIST, n_classes=10))
        Diag_LB_Brier_KMNIST.append(get_brier(MNIST_test_out_KMNIST_LB_D, targets_KMNIST, n_classes=10))

        
        #LB KFAC
        MNIST_test_in_LB_KFAC = predict_LB(la_kron, MNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_FMNIST_LB_KFAC = predict_LB(la_kron, FMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_notMNIST_LB_KFAC = predict_LB(la_kron, notMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_KMNIST_LB_KFAC = predict_LB(la_kron, KMNIST_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_LB_KFAC, prob_correct_in_LB_KFAC, ent_in_LB_KFAC, MMC_in_LB_KFAC = get_in_dist_values(MNIST_test_in_LB_KFAC, targets)
        acc_out_FMNIST_LB_KFAC, prob_correct_out_FMNIST_LB_KFAC, ent_out_FMNIST_LB_KFAC, MMC_out_FMNIST_LB_KFAC, auroc_out_FMNIST_LB_KFAC = get_out_dist_values(MNIST_test_in_LB_KFAC, MNIST_test_out_FMNIST_LB_KFAC, targets_FMNIST)
        acc_out_notMNIST_LB_KFAC, prob_correct_out_notMNIST_LB_KFAC, ent_out_notMNIST_LB_KFAC, MMC_out_notMNIST_LB_KFAC, auroc_out_notMNIST_LB_KFAC = get_out_dist_values(MNIST_test_in_LB_KFAC, MNIST_test_out_notMNIST_LB_KFAC, targets_notMNIST)
        acc_out_KMNIST_LB_KFAC, prob_correct_out_KMNIST_LB_KFAC, ent_out_KMNIST_LB_KFAC, MMC_out_KMNIST_LB_KFAC, auroc_out_KMNIST_LB_KFAC = get_out_dist_values(MNIST_test_in_LB_KFAC, MNIST_test_out_KMNIST_LB_KFAC, targets_KMNIST)
        
        KFAC_LB_MMC_in.append(MMC_in_LB_KFAC)
        KFAC_LB_MMC_FMNIST.append(MMC_out_FMNIST_LB_KFAC)
        KFAC_LB_MMC_notMNIST.append(MMC_out_notMNIST_LB_KFAC)
        KFAC_LB_MMC_KMNIST.append(MMC_out_KMNIST_LB_KFAC)
        KFAC_LB_AUROC_FMNIST.append(auroc_out_FMNIST_LB_KFAC)
        KFAC_LB_AUROC_notMNIST.append(auroc_out_notMNIST_LB_KFAC)
        KFAC_LB_AUROC_KMNIST.append(auroc_out_KMNIST_LB_KFAC)
        KFAC_LB_NLL_in.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_in_LB_KFAC)).log_prob(torch.tensor(targets)).mean().item())
        KFAC_LB_NLL_FMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_FMNIST_LB_KFAC)).log_prob(torch.tensor(targets_FMNIST)).mean().item())
        KFAC_LB_NLL_notMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_notMNIST_LB_KFAC)).log_prob(torch.tensor(targets_notMNIST)).mean().item())
        KFAC_LB_NLL_KMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_KMNIST_LB_KFAC)).log_prob(torch.tensor(targets_KMNIST)).mean().item())
        KFAC_LB_ECE_in.append(scoring.expected_calibration_error(targets, MNIST_test_in_LB_KFAC))
        KFAC_LB_ECE_FMNIST.append(scoring.expected_calibration_error(targets_FMNIST, MNIST_test_out_FMNIST_LB_KFAC))
        KFAC_LB_ECE_notMNIST.append(scoring.expected_calibration_error(targets_notMNIST, MNIST_test_out_notMNIST_LB_KFAC))
        KFAC_LB_ECE_KMNIST.append(scoring.expected_calibration_error(targets_KMNIST, MNIST_test_out_KMNIST_LB_KFAC))
        KFAC_LB_Brier_in.append(get_brier(MNIST_test_in_LB_KFAC, targets, n_classes=10))
        KFAC_LB_Brier_FMNIST.append(get_brier(MNIST_test_out_FMNIST_LB_KFAC, targets_FMNIST, n_classes=10))
        KFAC_LB_Brier_notMNIST.append(get_brier(MNIST_test_out_notMNIST_LB_KFAC, targets_notMNIST, n_classes=10))
        KFAC_LB_Brier_KMNIST.append(get_brier(MNIST_test_out_KMNIST_LB_KFAC, targets_KMNIST, n_classes=10))


        #LB diag normalized
        MNIST_test_in_LB_Dn = predict_LB_norm(la_diag, MNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_FMNIST_LB_Dn = predict_LB_norm(la_diag, FMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_notMNIST_LB_Dn = predict_LB_norm(la_diag, notMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_KMNIST_LB_Dn = predict_LB_norm(la_diag, KMNIST_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_LB_D, prob_correct_in_LB_D, ent_in_LB_D, MMC_in_LB_D = get_in_dist_values(MNIST_test_in_LB_Dn, targets)
        acc_out_FMNIST_LB_D, prob_correct_out_FMNIST_LB_D, ent_out_FMNIST_LB_D, MMC_out_FMNIST_LB_D, auroc_out_FMNIST_LB_D = get_out_dist_values(MNIST_test_in_LB_Dn, MNIST_test_out_FMNIST_LB_Dn, targets_FMNIST)
        acc_out_notMNIST_LB_D, prob_correct_out_notMNIST_LB_D, ent_out_notMNIST_LB_D, MMC_out_notMNIST_LB_D, auroc_out_notMNIST_LB_D = get_out_dist_values(MNIST_test_in_LB_Dn, MNIST_test_out_notMNIST_LB_Dn, targets_notMNIST)
        acc_out_KMNIST_LB_D, prob_correct_out_KMNIST_LB_D, ent_out_KMNIST_LB_D, MMC_out_KMNIST_LB_D, auroc_out_KMNIST_LB_D = get_out_dist_values(MNIST_test_in_LB_Dn, MNIST_test_out_KMNIST_LB_Dn, targets_KMNIST)
        
        Diag_LB_norm_MMC_in.append(MMC_in_LB_D)
        Diag_LB_norm_MMC_FMNIST.append(MMC_out_FMNIST_LB_D)
        Diag_LB_norm_MMC_notMNIST.append(MMC_out_notMNIST_LB_D)
        Diag_LB_norm_MMC_KMNIST.append(MMC_out_KMNIST_LB_D)
        Diag_LB_norm_AUROC_FMNIST.append(auroc_out_FMNIST_LB_D)
        Diag_LB_norm_AUROC_notMNIST.append(auroc_out_notMNIST_LB_D)
        Diag_LB_norm_AUROC_KMNIST.append(auroc_out_KMNIST_LB_D)
        Diag_LB_norm_NLL_in.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_in_LB_Dn)).log_prob(torch.tensor(targets)).mean().item())
        Diag_LB_norm_NLL_FMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_FMNIST_LB_Dn)).log_prob(torch.tensor(targets_FMNIST)).mean().item())
        Diag_LB_norm_NLL_notMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_notMNIST_LB_Dn)).log_prob(torch.tensor(targets_notMNIST)).mean().item())
        Diag_LB_norm_NLL_KMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_KMNIST_LB_Dn)).log_prob(torch.tensor(targets_KMNIST)).mean().item())
        Diag_LB_norm_ECE_in.append(scoring.expected_calibration_error(targets, MNIST_test_in_LB_Dn))
        Diag_LB_norm_ECE_FMNIST.append(scoring.expected_calibration_error(targets_FMNIST, MNIST_test_out_FMNIST_LB_Dn))
        Diag_LB_norm_ECE_notMNIST.append(scoring.expected_calibration_error(targets_notMNIST, MNIST_test_out_notMNIST_LB_Dn))
        Diag_LB_norm_ECE_KMNIST.append(scoring.expected_calibration_error(targets_KMNIST, MNIST_test_out_KMNIST_LB_Dn))
        Diag_LB_norm_Brier_in.append(get_brier(MNIST_test_in_LB_Dn, targets, n_classes=10))
        Diag_LB_norm_Brier_FMNIST.append(get_brier(MNIST_test_out_FMNIST_LB_Dn, targets_FMNIST, n_classes=10))
        Diag_LB_norm_Brier_notMNIST.append(get_brier(MNIST_test_out_notMNIST_LB_Dn, targets_notMNIST, n_classes=10))
        Diag_LB_norm_Brier_KMNIST.append(get_brier(MNIST_test_out_KMNIST_LB_Dn, targets_KMNIST, n_classes=10))

        
        #LB KFAC normalized
        MNIST_test_in_LB_KFACn = predict_LB_norm(la_kron, MNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_FMNIST_LB_KFACn = predict_LB_norm(la_kron, FMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_notMNIST_LB_KFACn = predict_LB_norm(la_kron, notMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_KMNIST_LB_KFACn = predict_LB_norm(la_kron, KMNIST_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_LB_KFAC, prob_correct_in_LB_KFAC, ent_in_LB_KFAC, MMC_in_LB_KFACn = get_in_dist_values(MNIST_test_in_LB_KFACn, targets)
        acc_out_FMNIST_LB_KFAC, prob_correct_out_FMNIST_LB_KFAC, ent_out_FMNIST_LB_KFAC, MMC_out_FMNIST_LB_KFACn, auroc_out_FMNIST_LB_KFACn = get_out_dist_values(MNIST_test_in_LB_KFACn, MNIST_test_out_FMNIST_LB_KFACn, targets_FMNIST)
        acc_out_notMNIST_LB_KFAC, prob_correct_out_notMNIST_LB_KFAC, ent_out_notMNIST_LB_KFAC, MMC_out_notMNIST_LB_KFACn, auroc_out_notMNIST_LB_KFACn = get_out_dist_values(MNIST_test_in_LB_KFACn, MNIST_test_out_notMNIST_LB_KFACn, targets_notMNIST)
        acc_out_KMNIST_LB_KFAC, prob_correct_out_KMNIST_LB_KFAC, ent_out_KMNIST_LB_KFAC, MMC_out_KMNIST_LB_KFACn, auroc_out_KMNIST_LB_KFACn = get_out_dist_values(MNIST_test_in_LB_KFACn, MNIST_test_out_KMNIST_LB_KFACn, targets_KMNIST)
        
        KFAC_LB_norm_MMC_in.append(MMC_in_LB_KFACn)
        KFAC_LB_norm_MMC_FMNIST.append(MMC_out_FMNIST_LB_KFACn)
        KFAC_LB_norm_MMC_notMNIST.append(MMC_out_notMNIST_LB_KFACn)
        KFAC_LB_norm_MMC_KMNIST.append(MMC_out_KMNIST_LB_KFACn)
        KFAC_LB_norm_AUROC_FMNIST.append(auroc_out_FMNIST_LB_KFACn)
        KFAC_LB_norm_AUROC_notMNIST.append(auroc_out_notMNIST_LB_KFACn)
        KFAC_LB_norm_AUROC_KMNIST.append(auroc_out_KMNIST_LB_KFACn)
        KFAC_LB_norm_NLL_in.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_in_LB_KFACn)).log_prob(torch.tensor(targets)).mean().item())
        KFAC_LB_norm_NLL_FMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_FMNIST_LB_KFACn)).log_prob(torch.tensor(targets_FMNIST)).mean().item())
        KFAC_LB_norm_NLL_notMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_notMNIST_LB_KFACn)).log_prob(torch.tensor(targets_notMNIST)).mean().item())
        KFAC_LB_norm_NLL_KMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_KMNIST_LB_KFACn)).log_prob(torch.tensor(targets_KMNIST)).mean().item())
        KFAC_LB_norm_ECE_in.append(scoring.expected_calibration_error(targets, MNIST_test_in_LB_KFACn))
        KFAC_LB_norm_ECE_FMNIST.append(scoring.expected_calibration_error(targets_FMNIST, MNIST_test_out_FMNIST_LB_KFACn))
        KFAC_LB_norm_ECE_notMNIST.append(scoring.expected_calibration_error(targets_notMNIST, MNIST_test_out_notMNIST_LB_KFACn))
        KFAC_LB_norm_ECE_KMNIST.append(scoring.expected_calibration_error(targets_KMNIST, MNIST_test_out_KMNIST_LB_KFACn))
        KFAC_LB_norm_Brier_in.append(get_brier(MNIST_test_in_LB_KFACn, targets, n_classes=10))
        KFAC_LB_norm_Brier_FMNIST.append(get_brier(MNIST_test_out_FMNIST_LB_KFACn, targets_FMNIST, n_classes=10))
        KFAC_LB_norm_Brier_notMNIST.append(get_brier(MNIST_test_out_notMNIST_LB_KFACn, targets_notMNIST, n_classes=10))
        KFAC_LB_norm_Brier_KMNIST.append(get_brier(MNIST_test_out_KMNIST_LB_KFACn, targets_KMNIST, n_classes=10))
        

        
        #Probit diag
        MNIST_test_in_PROBIT_D = predict_probit(la_diag, MNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_FMNIST_PROBIT_D = predict_probit(la_diag, FMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_notMNIST_PROBIT_D = predict_probit(la_diag, notMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_KMNIST_PROBIT_D = predict_probit(la_diag, KMNIST_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_PROBIT, prob_correct_in_PROBIT, ent_in_PROBIT, MMC_in_PROBIT = get_in_dist_values(MNIST_test_in_PROBIT_D, targets)
        acc_out_FMNIST_PROBIT, prob_correct_out_FMNIST_PROBIT, ent_out_FMNIST_PROBIT, MMC_out_FMNIST_PROBIT, auroc_out_FMNIST_PROBIT = get_out_dist_values(MNIST_test_in_PROBIT_D, MNIST_test_out_FMNIST_PROBIT_D, targets_FMNIST)
        acc_out_notMNIST_PROBIT, prob_correct_out_notMNIST_PROBIT, ent_out_notMNIST_PROBIT, MMC_out_notMNIST_PROBIT, auroc_out_notMNIST_PROBIT = get_out_dist_values(MNIST_test_in_PROBIT_D, MNIST_test_out_notMNIST_PROBIT_D, targets_notMNIST)
        acc_out_KMNIST_PROBIT, prob_correct_out_KMNIST_PROBIT, ent_out_KMNIST_PROBIT, MMC_out_KMNIST_PROBIT, auroc_out_KMNIST_PROBIT = get_out_dist_values(MNIST_test_in_PROBIT_D, MNIST_test_out_KMNIST_PROBIT_D, targets_KMNIST)
            
        Diag_PROBIT_MMC_in.append(MMC_in_PROBIT)
        Diag_PROBIT_MMC_FMNIST.append(MMC_out_FMNIST_PROBIT)
        Diag_PROBIT_MMC_notMNIST.append(MMC_out_notMNIST_PROBIT)
        Diag_PROBIT_MMC_KMNIST.append(MMC_out_KMNIST_PROBIT)
        Diag_PROBIT_AUROC_FMNIST.append(auroc_out_FMNIST_PROBIT)
        Diag_PROBIT_AUROC_notMNIST.append(auroc_out_notMNIST_PROBIT)
        Diag_PROBIT_AUROC_KMNIST.append(auroc_out_KMNIST_PROBIT)
        Diag_PROBIT_NLL_in.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_in_PROBIT_D)).log_prob(torch.tensor(targets)).mean().item())
        Diag_PROBIT_NLL_FMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_FMNIST_PROBIT_D)).log_prob(torch.tensor(targets_FMNIST)).mean().item())
        Diag_PROBIT_NLL_notMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_notMNIST_PROBIT_D)).log_prob(torch.tensor(targets_notMNIST)).mean().item())
        Diag_PROBIT_NLL_KMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_KMNIST_PROBIT_D)).log_prob(torch.tensor(targets_KMNIST)).mean().item())
        Diag_PROBIT_ECE_in.append(scoring.expected_calibration_error(targets, MNIST_test_in_PROBIT_D))
        Diag_PROBIT_ECE_FMNIST.append(scoring.expected_calibration_error(targets_FMNIST, MNIST_test_out_FMNIST_PROBIT_D))
        Diag_PROBIT_ECE_notMNIST.append(scoring.expected_calibration_error(targets_notMNIST, MNIST_test_out_notMNIST_PROBIT_D))
        Diag_PROBIT_ECE_KMNIST.append(scoring.expected_calibration_error(targets_KMNIST, MNIST_test_out_KMNIST_PROBIT_D))
        Diag_PROBIT_Brier_in.append(get_brier(MNIST_test_in_PROBIT_D, targets, n_classes=10))
        Diag_PROBIT_Brier_FMNIST.append(get_brier(MNIST_test_out_FMNIST_PROBIT_D, targets_FMNIST, n_classes=10))
        Diag_PROBIT_Brier_notMNIST.append(get_brier(MNIST_test_out_notMNIST_PROBIT_D, targets_notMNIST, n_classes=10))
        Diag_PROBIT_Brier_KMNIST.append(get_brier(MNIST_test_out_KMNIST_PROBIT_D, targets_KMNIST, n_classes=10))
        

        
        #Probit KFAC
        MNIST_test_in_PROBIT_K = predict_probit(la_kron, MNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_FMNIST_PROBIT_K = predict_probit(la_kron, FMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_notMNIST_PROBIT_K = predict_probit(la_kron, notMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_KMNIST_PROBIT_K = predict_probit(la_kron, KMNIST_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_PROBIT, prob_correct_in_PROBIT, ent_in_PROBIT, MMC_in_PROBIT = get_in_dist_values(MNIST_test_in_PROBIT_K, targets)
        acc_out_FMNIST_PROBIT, prob_correct_out_FMNIST_PROBIT, ent_out_FMNIST_PROBIT, MMC_out_FMNIST_PROBIT, auroc_out_FMNIST_PROBIT = get_out_dist_values(MNIST_test_in_PROBIT_K, MNIST_test_out_FMNIST_PROBIT_K, targets_FMNIST)
        acc_out_notMNIST_PROBIT, prob_correct_out_notMNIST_PROBIT, ent_out_notMNIST_PROBIT, MMC_out_notMNIST_PROBIT, auroc_out_notMNIST_PROBIT = get_out_dist_values(MNIST_test_in_PROBIT_K, MNIST_test_out_notMNIST_PROBIT_K, targets_notMNIST)
        acc_out_KMNIST_PROBIT, prob_correct_out_KMNIST_PROBIT, ent_out_KMNIST_PROBIT, MMC_out_KMNIST_PROBIT, auroc_out_KMNIST_PROBIT = get_out_dist_values(MNIST_test_in_PROBIT_K, MNIST_test_out_KMNIST_PROBIT_K, targets_KMNIST)
        
        KFAC_PROBIT_MMC_in.append(MMC_in_PROBIT)
        KFAC_PROBIT_MMC_FMNIST.append(MMC_out_FMNIST_PROBIT)
        KFAC_PROBIT_MMC_notMNIST.append(MMC_out_notMNIST_PROBIT)
        KFAC_PROBIT_MMC_KMNIST.append(MMC_out_KMNIST_PROBIT)
        KFAC_PROBIT_AUROC_FMNIST.append(auroc_out_FMNIST_PROBIT)
        KFAC_PROBIT_AUROC_notMNIST.append(auroc_out_notMNIST_PROBIT)
        KFAC_PROBIT_AUROC_KMNIST.append(auroc_out_KMNIST_PROBIT)
        KFAC_PROBIT_NLL_in.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_in_PROBIT_K)).log_prob(torch.tensor(targets)).mean().item())
        KFAC_PROBIT_NLL_FMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_FMNIST_PROBIT_K)).log_prob(torch.tensor(targets_FMNIST)).mean().item())
        KFAC_PROBIT_NLL_notMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_notMNIST_PROBIT_K)).log_prob(torch.tensor(targets_notMNIST)).mean().item())
        KFAC_PROBIT_NLL_KMNIST.append(-torch.distributions.Categorical(torch.tensor(MNIST_test_out_KMNIST_PROBIT_K)).log_prob(torch.tensor(targets_KMNIST)).mean().item())
        KFAC_PROBIT_ECE_in.append(scoring.expected_calibration_error(targets, MNIST_test_in_PROBIT_K))
        KFAC_PROBIT_ECE_FMNIST.append(scoring.expected_calibration_error(targets_FMNIST, MNIST_test_out_FMNIST_PROBIT_K))
        KFAC_PROBIT_ECE_notMNIST.append(scoring.expected_calibration_error(targets_notMNIST, MNIST_test_out_notMNIST_PROBIT_K))
        KFAC_PROBIT_ECE_KMNIST.append(scoring.expected_calibration_error(targets_KMNIST, MNIST_test_out_KMNIST_PROBIT_K))
        KFAC_PROBIT_Brier_in.append(get_brier(MNIST_test_in_PROBIT_K, targets, n_classes=10))
        KFAC_PROBIT_Brier_FMNIST.append(get_brier(MNIST_test_out_FMNIST_PROBIT_K, targets_FMNIST, n_classes=10))
        KFAC_PROBIT_Brier_notMNIST.append(get_brier(MNIST_test_out_notMNIST_PROBIT_K, targets_notMNIST, n_classes=10))
        KFAC_PROBIT_Brier_KMNIST.append(get_brier(MNIST_test_out_KMNIST_PROBIT_K, targets_KMNIST, n_classes=10))
        
        
        """
        #SODPP diag
        MNIST_test_in_SODPP_D = predict_second_order_dpp(la_diag, MNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_FMNIST_SODPP_D = predict_second_order_dpp(la_diag, FMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_notMNIST_SODPP_D = predict_second_order_dpp(la_diag, notMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_KMNIST_SODPP_D = predict_second_order_dpp(la_diag, KMNIST_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_SODPP, prob_correct_in_SODPP, ent_in_SODPP, MMC_in_SODPP = get_in_dist_values(MNIST_test_in_SODPP_D, targets)
        acc_out_FMNIST_SODPP, prob_correct_out_FMNIST_SODPP, ent_out_FMNIST_SODPP, MMC_out_FMNIST_SODPP, auroc_out_FMNIST_SODPP = get_out_dist_values(MNIST_test_in_SODPP_D, MNIST_test_out_FMNIST_SODPP_D, targets_FMNIST)
        acc_out_notMNIST_SODPP, prob_correct_out_notMNIST_SODPP, ent_out_notMNIST_SODPP, MMC_out_notMNIST_SODPP, auroc_out_notMNIST_SODPP = get_out_dist_values(MNIST_test_in_SODPP_D, MNIST_test_out_notMNIST_SODPP_D, targets_notMNIST)
        acc_out_KMNIST_SODPP, prob_correct_out_KMNIST_SODPP, ent_out_KMNIST_SODPP, MMC_out_KMNIST_SODPP, auroc_out_KMNIST_SODPP = get_out_dist_values(MNIST_test_in_SODPP_D, MNIST_test_out_KMNIST_SODPP_D, targets_KMNIST)
        
        Diag_SODPP_MMC_in.append(MMC_in_SODPP)
        Diag_SODPP_MMC_FMNIST.append(MMC_out_FMNIST_SODPP)
        Diag_SODPP_MMC_notMNIST.append(MMC_out_notMNIST_SODPP)
        Diag_SODPP_MMC_KMNIST.append(MMC_out_KMNIST_SODPP)
        Diag_SODPP_AUROC_FMNIST.append(auroc_out_FMNIST_SODPP)
        Diag_SODPP_AUROC_notMNIST.append(auroc_out_notMNIST_SODPP)
        Diag_SODPP_AUROC_KMNIST.append(auroc_out_KMNIST_SODPP)
        
        #SODPP KFAC
        MNIST_test_in_SODPP_K = predict_second_order_dpp(la_kron, MNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_FMNIST_SODPP_K = predict_second_order_dpp(la_kron, FMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_notMNIST_SODPP_K = predict_second_order_dpp(la_kron, notMNIST_test_loader, timing=False, device=device).cpu().numpy()
        MNIST_test_out_KMNIST_SODPP_K = predict_second_order_dpp(la_kron, KMNIST_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_SODPP, prob_correct_in_SODPP, ent_in_SODPP, MMC_in_SODPP = get_in_dist_values(MNIST_test_in_SODPP_K, targets)
        acc_out_FMNIST_SODPP, prob_correct_out_FMNIST_SODPP, ent_out_FMNIST_SODPP, MMC_out_FMNIST_SODPP, auroc_out_FMNIST_SODPP = get_out_dist_values(MNIST_test_in_SODPP_K, MNIST_test_out_FMNIST_SODPP_K, targets_FMNIST)
        acc_out_notMNIST_SODPP, prob_correct_out_notMNIST_SODPP, ent_out_notMNIST_SODPP, MMC_out_notMNIST_SODPP, auroc_out_notMNIST_SODPP = get_out_dist_values(MNIST_test_in_SODPP_K, MNIST_test_out_notMNIST_SODPP_K, targets_notMNIST)
        acc_out_KMNIST_SODPP, prob_correct_out_KMNIST_SODPP, ent_out_KMNIST_SODPP, MMC_out_KMNIST_SODPP, auroc_out_KMNIST_SODPP = get_out_dist_values(MNIST_test_in_SODPP_K, MNIST_test_out_KMNIST_SODPP_K, targets_KMNIST)
        
        KFAC_SODPP_MMC_in.append(MMC_in_SODPP)
        KFAC_SODPP_MMC_FMNIST.append(MMC_out_FMNIST_SODPP)
        KFAC_SODPP_MMC_notMNIST.append(MMC_out_notMNIST_SODPP)
        KFAC_SODPP_MMC_KMNIST.append(MMC_out_KMNIST_SODPP)
        KFAC_SODPP_AUROC_FMNIST.append(auroc_out_FMNIST_SODPP)
        KFAC_SODPP_AUROC_notMNIST.append(auroc_out_notMNIST_SODPP)
        KFAC_SODPP_AUROC_KMNIST.append(auroc_out_KMNIST_SODPP)
        """
        
    #### save results
    results_dict = {
        # MAP
        'MAP_MMC_in':MAP_MMC_in,
        'MAP_MMC_FMNIST':MAP_MMC_FMNIST,
        'MAP_MMC_notMNIST':MAP_MMC_notMNIST,
        'MAP_MMC_KMNIST':MAP_MMC_KMNIST,
        'MAP_AUROC_FMNIST':MAP_AUROC_FMNIST,
        'MAP_AUROC_notMNIST':MAP_AUROC_notMNIST,
        'MAP_AUROC_KMNIST':MAP_AUROC_KMNIST,
        'MAP_NLL_in':MAP_NLL_in,
        'MAP_NLL_FMNIST':MAP_NLL_FMNIST,
        'MAP_NLL_notMNIST':MAP_NLL_notMNIST,
        'MAP_NLL_KMNIST':MAP_NLL_KMNIST,
        'MAP_ECE_in':MAP_ECE_in,
        'MAP_ECE_FMNIST':MAP_ECE_FMNIST,
        'MAP_ECE_notMNIST':MAP_ECE_notMNIST,
        'MAP_ECE_KMNIST':MAP_ECE_KMNIST,
        'MAP_Brier_in':MAP_Brier_in,
        'MAP_Brier_FMNIST':MAP_Brier_FMNIST,
        'MAP_Brier_notMNIST':MAP_Brier_notMNIST,
        'MAP_Brier_KMNIST':MAP_Brier_KMNIST,
        'Diag_samples_MMC_in':Diag_samples_MMC_in,
        'Diag_samples_MMC_FMNIST':Diag_samples_MMC_FMNIST,
        'Diag_samples_MMC_notMNIST':Diag_samples_MMC_notMNIST,
        'Diag_samples_MMC_KMNIST':Diag_samples_MMC_KMNIST,
        'Diag_samples_AUROC_FMNIST':Diag_samples_AUROC_FMNIST,
        'Diag_samples_AUROC_notMNIST':Diag_samples_AUROC_notMNIST,
        'Diag_samples_AUROC_KMNIST':Diag_samples_AUROC_KMNIST,
        'Diag_samples_NLL_in':Diag_samples_NLL_in,
        'Diag_samples_NLL_FMNIST':Diag_samples_NLL_FMNIST,
        'Diag_samples_NLL_notMNIST':Diag_samples_NLL_notMNIST,
        'Diag_samples_NLL_KMNIST':Diag_samples_NLL_KMNIST,
        'Diag_samples_ECE_in':Diag_samples_ECE_in,
        'Diag_samples_ECE_FMNIST':Diag_samples_ECE_FMNIST,
        'Diag_samples_ECE_notMNIST':Diag_samples_ECE_notMNIST,
        'Diag_samples_ECE_KMNIST':Diag_samples_ECE_KMNIST,
        'Diag_samples_Brier_in':Diag_samples_Brier_in,
        'Diag_samples_Brier_FMNIST':Diag_samples_Brier_FMNIST,
        'Diag_samples_Brier_notMNIST':Diag_samples_Brier_notMNIST,
        'Diag_samples_Brier_KMNIST':Diag_samples_Brier_KMNIST,
        'KFAC_samples_MMC_in':KFAC_samples_MMC_in,
        'KFAC_samples_MMC_FMNIST':KFAC_samples_MMC_FMNIST,
        'KFAC_samples_MMC_notMNIST':KFAC_samples_MMC_notMNIST,
        'KFAC_samples_MMC_KMNIST':KFAC_samples_MMC_KMNIST,
        'KFAC_samples_AUROC_FMNIST':KFAC_samples_AUROC_FMNIST,
        'KFAC_samples_AUROC_notMNIST':KFAC_samples_AUROC_notMNIST,
        'KFAC_samples_AUROC_KMNIST':KFAC_samples_AUROC_KMNIST,
        'KFAC_samples_NLL_in':KFAC_samples_NLL_in,
        'KFAC_samples_NLL_FMNIST':KFAC_samples_NLL_FMNIST,
        'KFAC_samples_NLL_notMNIST':KFAC_samples_NLL_notMNIST,
        'KFAC_samples_NLL_KMNIST':KFAC_samples_NLL_KMNIST,
        'KFAC_samples_ECE_in':KFAC_samples_ECE_in,
        'KFAC_samples_ECE_FMNIST':KFAC_samples_ECE_FMNIST,
        'KFAC_samples_ECE_notMNIST':KFAC_samples_ECE_notMNIST,
        'KFAC_samples_ECE_KMNIST':KFAC_samples_ECE_KMNIST,
        'KFAC_samples_Brier_in':KFAC_samples_Brier_in,
        'KFAC_samples_Brier_FMNIST':KFAC_samples_Brier_FMNIST,
        'KFAC_samples_Brier_notMNIST':KFAC_samples_Brier_notMNIST,
        'KFAC_samples_Brier_KMNIST':KFAC_samples_Brier_KMNIST,
        'Diag_LB_MMC_in':Diag_LB_MMC_in,
        'Diag_LB_MMC_FMNIST':Diag_LB_MMC_FMNIST,
        'Diag_LB_MMC_notMNIST':Diag_LB_MMC_notMNIST,
        'Diag_LB_MMC_KMNIST':Diag_LB_MMC_KMNIST,
        'Diag_LB_AUROC_FMNIST':Diag_LB_AUROC_FMNIST,
        'Diag_LB_AUROC_notMNIST':Diag_LB_AUROC_notMNIST,
        'Diag_LB_AUROC_KMNIST':Diag_LB_AUROC_KMNIST,
        'Diag_LB_NLL_in':Diag_LB_NLL_in,
        'Diag_LB_NLL_FMNIST':Diag_LB_NLL_FMNIST,
        'Diag_LB_NLL_notMNIST':Diag_LB_NLL_notMNIST,
        'Diag_LB_NLL_KMNIST':Diag_LB_NLL_KMNIST,
        'Diag_LB_ECE_in':Diag_LB_ECE_in,
        'Diag_LB_ECE_FMNIST':Diag_LB_ECE_FMNIST,
        'Diag_LB_ECE_notMNIST':Diag_LB_ECE_notMNIST,
        'Diag_LB_ECE_KMNIST':Diag_LB_ECE_KMNIST,
        'Diag_LB_Brier_in':Diag_LB_Brier_in,
        'Diag_LB_Brier_FMNIST':Diag_LB_Brier_FMNIST,
        'Diag_LB_Brier_notMNIST':Diag_LB_Brier_notMNIST,
        'Diag_LB_Brier_KMNIST':Diag_LB_Brier_KMNIST,
        'KFAC_LB_MMC_in':KFAC_LB_MMC_in,
        'KFAC_LB_MMC_FMNIST':KFAC_LB_MMC_FMNIST,
        'KFAC_LB_MMC_notMNIST':KFAC_LB_MMC_notMNIST,
        'KFAC_LB_MMC_KMNIST':KFAC_LB_MMC_KMNIST,
        'KFAC_LB_AUROC_FMNIST':KFAC_LB_AUROC_FMNIST,
        'KFAC_LB_AUROC_notMNIST':KFAC_LB_AUROC_notMNIST,
        'KFAC_LB_AUROC_KMNIST':KFAC_LB_AUROC_KMNIST,
        'KFAC_LB_NLL_in':KFAC_LB_NLL_in,
        'KFAC_LB_NLL_FMNIST':KFAC_LB_NLL_FMNIST,
        'KFAC_LB_NLL_notMNIST':KFAC_LB_NLL_notMNIST,
        'KFAC_LB_NLL_KMNIST':KFAC_LB_NLL_KMNIST,
        'KFAC_LB_ECE_in':KFAC_LB_ECE_in,
        'KFAC_LB_ECE_FMNIST':KFAC_LB_ECE_FMNIST,
        'KFAC_LB_ECE_notMNIST':KFAC_LB_ECE_notMNIST,
        'KFAC_LB_ECE_KMNIST':KFAC_LB_ECE_KMNIST,
        'KFAC_LB_Brier_in':KFAC_LB_Brier_in,
        'KFAC_LB_Brier_FMNIST':KFAC_LB_Brier_FMNIST,
        'KFAC_LB_Brier_notMNIST':KFAC_LB_Brier_notMNIST,
        'KFAC_LB_Brier_KMNIST':KFAC_LB_Brier_KMNIST,
        'Diag_LB_norm_MMC_in':Diag_LB_norm_MMC_in,
        'Diag_LB_norm_MMC_FMNIST':Diag_LB_norm_MMC_FMNIST,
        'Diag_LB_norm_MMC_notMNIST':Diag_LB_norm_MMC_notMNIST,
        'Diag_LB_norm_MMC_KMNIST':Diag_LB_norm_MMC_KMNIST,
        'Diag_LB_norm_AUROC_FMNIST':Diag_LB_norm_AUROC_FMNIST,
        'Diag_LB_norm_AUROC_notMNIST':Diag_LB_norm_AUROC_notMNIST,
        'Diag_LB_norm_AUROC_KMNIST':Diag_LB_norm_AUROC_KMNIST,
        'Diag_LB_norm_NLL_in':Diag_LB_norm_NLL_in,
        'Diag_LB_norm_NLL_FMNIST':Diag_LB_norm_NLL_FMNIST,
        'Diag_LB_norm_NLL_notMNIST':Diag_LB_norm_NLL_notMNIST,
        'Diag_LB_norm_NLL_KMNIST':Diag_LB_norm_NLL_KMNIST,
        'Diag_LB_norm_ECE_in':Diag_LB_norm_ECE_in,
        'Diag_LB_norm_ECE_FMNIST':Diag_LB_norm_ECE_FMNIST,
        'Diag_LB_norm_ECE_notMNIST':Diag_LB_norm_ECE_notMNIST,
        'Diag_LB_norm_ECE_KMNIST':Diag_LB_norm_ECE_KMNIST,
        'Diag_LB_norm_Brier_in':Diag_LB_norm_Brier_in,
        'Diag_LB_norm_Brier_FMNIST':Diag_LB_norm_Brier_FMNIST,
        'Diag_LB_norm_Brier_notMNIST':Diag_LB_norm_Brier_notMNIST,
        'Diag_LB_norm_Brier_KMNIST':Diag_LB_norm_Brier_KMNIST,
        'KFAC_LB_norm_MMC_in':KFAC_LB_norm_MMC_in,
        'KFAC_LB_norm_MMC_FMNIST':KFAC_LB_norm_MMC_FMNIST,
        'KFAC_LB_norm_MMC_notMNIST':KFAC_LB_norm_MMC_notMNIST,
        'KFAC_LB_norm_MMC_KMNIST':KFAC_LB_norm_MMC_KMNIST,
        'KFAC_LB_norm_AUROC_FMNIST':KFAC_LB_norm_AUROC_FMNIST,
        'KFAC_LB_norm_AUROC_notMNIST':KFAC_LB_norm_AUROC_notMNIST,
        'KFAC_LB_norm_AUROC_KMNIST':KFAC_LB_norm_AUROC_KMNIST,
        'KFAC_LB_norm_NLL_in':KFAC_LB_norm_NLL_in,
        'KFAC_LB_norm_NLL_FMNIST':KFAC_LB_norm_NLL_FMNIST,
        'KFAC_LB_norm_NLL_notMNIST':KFAC_LB_norm_NLL_notMNIST,
        'KFAC_LB_norm_NLL_KMNIST':KFAC_LB_norm_NLL_KMNIST,
        'KFAC_LB_norm_ECE_in':KFAC_LB_norm_ECE_in,
        'KFAC_LB_norm_ECE_FMNIST':KFAC_LB_norm_ECE_FMNIST,
        'KFAC_LB_norm_ECE_notMNIST':KFAC_LB_norm_ECE_notMNIST,
        'KFAC_LB_norm_ECE_KMNIST':KFAC_LB_norm_ECE_KMNIST,
        'KFAC_LB_norm_Brier_in':KFAC_LB_norm_Brier_in,
        'KFAC_LB_norm_Brier_FMNIST':KFAC_LB_norm_Brier_FMNIST,
        'KFAC_LB_norm_Brier_notMNIST':KFAC_LB_norm_Brier_notMNIST,
        'KFAC_LB_norm_Brier_KMNIST':KFAC_LB_norm_Brier_KMNIST,
        'Diag_PROBIT_MMC_in':Diag_PROBIT_MMC_in,
        'Diag_PROBIT_MMC_FMNIST':Diag_PROBIT_MMC_FMNIST,
        'Diag_PROBIT_MMC_notMNIST':Diag_PROBIT_MMC_notMNIST,
        'Diag_PROBIT_MMC_KMNIST':Diag_PROBIT_MMC_KMNIST,
        'Diag_PROBIT_AUROC_FMNIST':Diag_PROBIT_AUROC_FMNIST,
        'Diag_PROBIT_AUROC_notMNIST':Diag_PROBIT_AUROC_notMNIST,
        'Diag_PROBIT_AUROC_KMNIST':Diag_PROBIT_AUROC_KMNIST,
        'Diag_PROBIT_NLL_in':Diag_PROBIT_NLL_in,
        'Diag_PROBIT_NLL_FMNIST':Diag_PROBIT_NLL_FMNIST,
        'Diag_PROBIT_NLL_notMNIST':Diag_PROBIT_NLL_notMNIST,
        'Diag_PROBIT_NLL_KMNIST':Diag_PROBIT_NLL_KMNIST,
        'Diag_PROBIT_ECE_in':Diag_PROBIT_ECE_in,
        'Diag_PROBIT_ECE_FMNIST':Diag_PROBIT_ECE_FMNIST,
        'Diag_PROBIT_ECE_notMNIST':Diag_PROBIT_ECE_notMNIST,
        'Diag_PROBIT_ECE_KMNIST':Diag_PROBIT_ECE_KMNIST,
        'Diag_PROBIT_Brier_in':Diag_PROBIT_Brier_in,
        'Diag_PROBIT_Brier_FMNIST':Diag_PROBIT_Brier_FMNIST,
        'Diag_PROBIT_Brier_notMNIST':Diag_PROBIT_Brier_notMNIST,
        'Diag_PROBIT_Brier_KMNIST':Diag_PROBIT_Brier_KMNIST,
        'KFAC_PROBIT_MMC_in':KFAC_PROBIT_MMC_in,
        'KFAC_PROBIT_MMC_FMNIST':KFAC_PROBIT_MMC_FMNIST,
        'KFAC_PROBIT_MMC_notMNIST':KFAC_PROBIT_MMC_notMNIST,
        'KFAC_PROBIT_MMC_KMNIST':KFAC_PROBIT_MMC_KMNIST,
        'KFAC_PROBIT_AUROC_FMNIST':KFAC_PROBIT_AUROC_FMNIST,
        'KFAC_PROBIT_AUROC_notMNIST':KFAC_PROBIT_AUROC_notMNIST,
        'KFAC_PROBIT_AUROC_KMNIST':KFAC_PROBIT_AUROC_KMNIST,
        'KFAC_PROBIT_NLL_in':KFAC_PROBIT_NLL_in,
        'KFAC_PROBIT_NLL_FMNIST':KFAC_PROBIT_NLL_FMNIST,
        'KFAC_PROBIT_NLL_notMNIST':KFAC_PROBIT_NLL_notMNIST,
        'KFAC_PROBIT_NLL_KMNIST':KFAC_PROBIT_NLL_KMNIST,
        'KFAC_PROBIT_ECE_in':KFAC_PROBIT_ECE_in,
        'KFAC_PROBIT_ECE_FMNIST':KFAC_PROBIT_ECE_FMNIST,
        'KFAC_PROBIT_ECE_notMNIST':KFAC_PROBIT_ECE_notMNIST,
        'KFAC_PROBIT_ECE_KMNIST':KFAC_PROBIT_ECE_KMNIST,
        'KFAC_PROBIT_Brier_in':KFAC_PROBIT_Brier_in,
        'KFAC_PROBIT_Brier_FMNIST':KFAC_PROBIT_Brier_FMNIST,
        'KFAC_PROBIT_Brier_notMNIST':KFAC_PROBIT_Brier_notMNIST,
        'KFAC_PROBIT_Brier_KMNIST':KFAC_PROBIT_Brier_KMNIST,
        #'Diag_SODPP_MMC_in':Diag_SODPP_MMC_in,
        #'Diag_SODPP_MMC_FMNIST':Diag_SODPP_MMC_FMNIST,
        #'Diag_SODPP_MMC_notMNIST':Diag_SODPP_MMC_notMNIST,
        #'Diag_SODPP_MMC_KMNIST':Diag_SODPP_MMC_KMNIST,
        #'Diag_SODPP_AUROC_FMNIST':Diag_SODPP_AUROC_FMNIST,
        #'Diag_SODPP_AUROC_notMNIST':Diag_SODPP_AUROC_notMNIST,
        #'Diag_SODPP_AUROC_KMNIST':Diag_SODPP_AUROC_KMNIST,
        #'KFAC_SODPP_MMC_in':KFAC_SODPP_MMC_in,
        #'KFAC_SODPP_MMC_FMNIST':KFAC_SODPP_MMC_FMNIST,
        #'KFAC_SODPP_MMC_notMNIST':KFAC_SODPP_MMC_notMNIST,
        #'KFAC_SODPP_MMC_KMNIST':KFAC_SODPP_MMC_KMNIST,
        #'KFAC_SODPP_AUROC_FMNIST':KFAC_SODPP_AUROC_FMNIST,
        #'KFAC_SODPP_AUROC_notMNIST':KFAC_SODPP_AUROC_notMNIST,
        #'KFAC_SODPP_AUROC_KMNIST':KFAC_SODPP_AUROC_KMNIST
    }
    results_df = pd.DataFrame(results_dict)
    RESULTS_PATH = os.getcwd() + "/Experiment_results/MNIST_results{}.csv".format("_all_layers" if not args.last_layer else "")
    print("saving at: ", RESULTS_PATH)
    results_df.to_csv(RESULTS_PATH)


#### RUN
if __name__ == '__main__':
    main()