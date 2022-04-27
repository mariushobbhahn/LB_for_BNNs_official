import torch
import torchvision
from torch import nn, optim, autograd
from torch.nn import functional as F
import numpy as np
from sklearn.utils import shuffle as skshuffle
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, trange
import pytest
from torchvision import transforms
from utils.LB_utils import *
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
from laplace import Laplace
import utils.scoring as scoring

from utils.load_not_MNIST import notMNIST
import argparse
import resnet18_v2

#### SETTINGS

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-s', '--num_seeds', type=int, default=5)
    p.add_argument('-o', '--out_folder', type=str, default='../Experiments_results/')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cuda_status = torch.cuda.is_available()
    print("device: ", device)
    print("cuda status: ", cuda_status)


    BATCH_SIZE = 128
    ### load data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    #~/data/cifar10
    CIFAR10_trainset = torchvision.datasets.CIFAR10(root='~/data/cifar10', train=True, download=True, transform=transform_train)
    CIFAR10_train_loader = torch.utils.data.DataLoader(CIFAR10_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    #~/data/cifar10
    CIFAR10_testset = torchvision.datasets.CIFAR10(root='~/data/cifar10', train=False, download=True, transform=transform_test)
    CIFAR10_test_loader = torch.utils.data.DataLoader(CIFAR10_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    CIFAR100_test = torchvision.datasets.CIFAR100(root='~/data/cifar100', train=False,
                                       download=True, transform=transform_test)
    CIFAR100_test_loader = torch.utils.data.DataLoader(CIFAR100_test, batch_size=BATCH_SIZE,
                                         shuffle=False)

    test_data_SVHN = torchvision.datasets.SVHN('~/data/SVHN', split='test',
                             download=False, transform=transform_test)
    SVHN_test_loader = torch.utils.data.DataLoader(test_data_SVHN, batch_size=BATCH_SIZE)


    ### load model
    CIFAR10_PATH = os.getcwd() + "/pretrained_weights/CIFAR10_resnet18_best_s1.pth"
    CIFAR10_model = resnet18_v2.ResNet18().to(device)
    print("loading model from: {}".format(CIFAR10_PATH))
    CIFAR10_model.load_state_dict(torch.load(CIFAR10_PATH))
    CIFAR10_model.eval()
    
    targets_CIFAR10 = np.array(CIFAR10_testset.targets)
    targets_CIFAR100 = torch.tensor(CIFAR100_test.targets).clip(0, 9).numpy()
    targets_SVHN = []
    for x,y in SVHN_test_loader:
        targets_SVHN.append(y)
    targets_SVHN = torch.cat(targets_SVHN).numpy()
        
    num_samples = 100

    #### Experiments

    # MAP
    MAP_MMC_in = []
    MAP_MMC_CIFAR100 = []
    MAP_MMC_SVHN = []
    MAP_AUROC_CIFAR100 = []
    MAP_AUROC_SVHN = []
    MAP_NLL_in = []
    MAP_NLL_CIFAR100 = []
    MAP_NLL_SVHN = []
    MAP_ECE_in = []
    MAP_ECE_CIFAR100 = []
    MAP_ECE_SVHN = []
    MAP_Brier_in = []
    MAP_Brier_CIFAR100 = []
    MAP_Brier_SVHN = []
    
    # Diag samples
    Diag_samples_MMC_in = []
    Diag_samples_MMC_CIFAR100 = []
    Diag_samples_MMC_SVHN = []
    Diag_samples_AUROC_CIFAR100 = []
    Diag_samples_AUROC_SVHN = []
    Diag_samples_NLL_in = []
    Diag_samples_NLL_CIFAR100 = []
    Diag_samples_NLL_SVHN = []
    Diag_samples_ECE_in = []
    Diag_samples_ECE_CIFAR100 = []
    Diag_samples_ECE_SVHN = []
    Diag_samples_Brier_in = []
    Diag_samples_Brier_CIFAR100 = []
    Diag_samples_Brier_SVHN = []

    # KFAC samples
    KFAC_samples_MMC_in = []
    KFAC_samples_MMC_CIFAR100 = []
    KFAC_samples_MMC_SVHN = []
    KFAC_samples_AUROC_CIFAR100 = []
    KFAC_samples_AUROC_SVHN = []
    KFAC_samples_NLL_in = []
    KFAC_samples_NLL_CIFAR100 = []
    KFAC_samples_NLL_SVHN = []
    KFAC_samples_ECE_in = []
    KFAC_samples_ECE_CIFAR100 = []
    KFAC_samples_ECE_SVHN = []
    KFAC_samples_Brier_in = []
    KFAC_samples_Brier_CIFAR100 = []
    KFAC_samples_Brier_SVHN = []

    # Diag LB
    Diag_LB_MMC_in = []
    Diag_LB_MMC_CIFAR100 = []
    Diag_LB_MMC_SVHN = []
    Diag_LB_AUROC_CIFAR100 = []
    Diag_LB_AUROC_SVHN = []
    Diag_LB_NLL_in = []
    Diag_LB_NLL_CIFAR100 = []
    Diag_LB_NLL_SVHN = []
    Diag_LB_ECE_in = []
    Diag_LB_ECE_CIFAR100 = []
    Diag_LB_ECE_SVHN = []
    Diag_LB_Brier_in = []
    Diag_LB_Brier_CIFAR100 = []
    Diag_LB_Brier_SVHN = []
    
    # KFAC LB
    KFAC_LB_MMC_in = []
    KFAC_LB_MMC_CIFAR100 = []
    KFAC_LB_MMC_SVHN = []
    KFAC_LB_AUROC_CIFAR100 = []
    KFAC_LB_AUROC_SVHN = []
    KFAC_LB_NLL_in = []
    KFAC_LB_NLL_CIFAR100 = []
    KFAC_LB_NLL_SVHN = []
    KFAC_LB_ECE_in = []
    KFAC_LB_ECE_CIFAR100 = []
    KFAC_LB_ECE_SVHN = []
    KFAC_LB_Brier_in = []
    KFAC_LB_Brier_CIFAR100 = []
    KFAC_LB_Brier_SVHN = []

    # Diag LB normalized
    Diag_LB_norm_MMC_in = []
    Diag_LB_norm_MMC_CIFAR100 = []
    Diag_LB_norm_MMC_SVHN = []
    Diag_LB_norm_AUROC_CIFAR100 = []
    Diag_LB_norm_AUROC_SVHN = []
    Diag_LB_norm_NLL_in = []
    Diag_LB_norm_NLL_CIFAR100 = []
    Diag_LB_norm_NLL_SVHN = []
    Diag_LB_norm_ECE_in = []
    Diag_LB_norm_ECE_CIFAR100 = []
    Diag_LB_norm_ECE_SVHN = []
    Diag_LB_norm_Brier_in = []
    Diag_LB_norm_Brier_CIFAR100 = []
    Diag_LB_norm_Brier_SVHN = []
    
    # KFAC LB normalized
    KFAC_LB_norm_MMC_in = []
    KFAC_LB_norm_MMC_CIFAR100 = []
    KFAC_LB_norm_MMC_SVHN = []
    KFAC_LB_norm_AUROC_CIFAR100 = []
    KFAC_LB_norm_AUROC_SVHN = []
    KFAC_LB_norm_NLL_in = []
    KFAC_LB_norm_NLL_CIFAR100 = []
    KFAC_LB_norm_NLL_SVHN = []
    KFAC_LB_norm_ECE_in = []
    KFAC_LB_norm_ECE_CIFAR100 = []
    KFAC_LB_norm_ECE_SVHN = []
    KFAC_LB_norm_Brier_in = []
    KFAC_LB_norm_Brier_CIFAR100 = []
    KFAC_LB_norm_Brier_SVHN = []
    
    # Diag PROBIT
    Diag_PROBIT_MMC_in = []
    Diag_PROBIT_MMC_CIFAR100 = []
    Diag_PROBIT_MMC_SVHN = []
    Diag_PROBIT_AUROC_CIFAR100 = []
    Diag_PROBIT_AUROC_SVHN = []
    Diag_PROBIT_NLL_in = []
    Diag_PROBIT_NLL_CIFAR100 = []
    Diag_PROBIT_NLL_SVHN = []
    Diag_PROBIT_ECE_in = []
    Diag_PROBIT_ECE_CIFAR100 = []
    Diag_PROBIT_ECE_SVHN = []
    Diag_PROBIT_Brier_in = []
    Diag_PROBIT_Brier_CIFAR100 = []
    Diag_PROBIT_Brier_SVHN = []
    
    # KFAC PROBIT
    KFAC_PROBIT_MMC_in = []
    KFAC_PROBIT_MMC_CIFAR100 = []
    KFAC_PROBIT_MMC_SVHN = []
    KFAC_PROBIT_AUROC_CIFAR100 = []
    KFAC_PROBIT_AUROC_SVHN = []
    KFAC_PROBIT_NLL_in = []
    KFAC_PROBIT_NLL_CIFAR100 = []
    KFAC_PROBIT_NLL_SVHN = []
    KFAC_PROBIT_ECE_in = []
    KFAC_PROBIT_ECE_CIFAR100 = []
    KFAC_PROBIT_ECE_SVHN = []
    KFAC_PROBIT_Brier_in = []
    KFAC_PROBIT_Brier_CIFAR100 = []
    KFAC_PROBIT_Brier_SVHN = []
    
    """
    # Diag SODPP
    Diag_SODPP_MMC_in = []
    Diag_SODPP_MMC_CIFAR100 = []
    Diag_SODPP_MMC_SVHN = []
    Diag_SODPP_AUROC_CIFAR100 = []
    Diag_SODPP_AUROC_SVHN = []
    
    # KFAC SODPP
    KFAC_SODPP_MMC_in = []
    KFAC_SODPP_MMC_CIFAR100 = []
    KFAC_SODPP_MMC_SVHN = []
    KFAC_SODPP_AUROC_CIFAR100 = []
    KFAC_SODPP_AUROC_SVHN = []
    """
    
    # run experiments
    for s in range(args.num_seeds):

        print("seed: ", s)
    
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        la_diag = Laplace(CIFAR10_model, 'classification', 
                     subset_of_weights='last_layer', 
                     hessian_structure='diag',
                     prior_precision=1) # 5e-4 # Choose prior precision according to weight decay
        la_diag.fit(CIFAR10_train_loader)
        
        la_kron = Laplace(CIFAR10_model, 'classification', 
                     subset_of_weights='last_layer', 
                     hessian_structure='kron',
                     prior_precision=1e-1) # 5e-4 # Choose prior precision according to weight decay
        la_kron.fit(CIFAR10_train_loader)

        #MAP estimates
        CIFAR10_test_in_MAP = predict_MAP(CIFAR10_model, CIFAR10_test_loader, device=device).cpu().numpy()
        CIFAR10_test_out_CIFAR100_MAP = predict_MAP(CIFAR10_model, CIFAR100_test_loader, device=device).cpu().numpy()
        CIFAR10_test_out_SVHN_MAP = predict_MAP(CIFAR10_model, SVHN_test_loader, device=device).cpu().numpy()
        
        acc_in_MAP, prob_correct_in_MAP, ent_in_MAP, MMC_in_MAP = get_in_dist_values(CIFAR10_test_in_MAP, targets_CIFAR10)
        acc_out_CIFAR100_MAP, prob_correct_out_CIFAR100_MAP, ent_out_CIFAR100, MMC_out_CIFAR100_MAP, auroc_out_CIFAR100_MAP = get_out_dist_values(CIFAR10_test_in_MAP, CIFAR10_test_out_CIFAR100_MAP, targets_CIFAR100)
        acc_out_SVHN_MAP, prob_correct_out_SVHN_MAP, ent_out_SVHN, MMC_out_SVHN_MAP, auroc_out_SVHN_MAP = get_out_dist_values(CIFAR10_test_in_MAP, CIFAR10_test_out_SVHN_MAP, targets_SVHN)
        
        MAP_MMC_in.append(MMC_in_MAP)
        MAP_MMC_CIFAR100.append(MMC_out_CIFAR100_MAP)
        MAP_MMC_SVHN.append(MMC_out_SVHN_MAP)
        MAP_AUROC_CIFAR100.append(auroc_out_CIFAR100_MAP)
        MAP_AUROC_SVHN.append(auroc_out_SVHN_MAP)
        MAP_NLL_in.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_in_MAP)).log_prob(torch.tensor(targets_CIFAR10)).mean().item())
        MAP_NLL_CIFAR100.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_out_CIFAR100_MAP)).log_prob(torch.tensor(targets_CIFAR100)).mean().item())
        MAP_NLL_SVHN.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_out_SVHN_MAP)).log_prob(torch.tensor(targets_SVHN)).mean().item())
        MAP_ECE_in.append(scoring.expected_calibration_error(targets_CIFAR10, CIFAR10_test_in_MAP))
        MAP_ECE_CIFAR100.append(scoring.expected_calibration_error(targets_CIFAR100, CIFAR10_test_out_CIFAR100_MAP))
        MAP_ECE_SVHN.append(scoring.expected_calibration_error(targets_SVHN, CIFAR10_test_out_SVHN_MAP))
        MAP_Brier_in.append(get_brier(CIFAR10_test_in_MAP, targets_CIFAR10, n_classes=10))
        MAP_Brier_CIFAR100.append(get_brier(CIFAR10_test_out_CIFAR100_MAP, targets_CIFAR100, n_classes=10))
        MAP_Brier_SVHN.append(get_brier(CIFAR10_test_out_SVHN_MAP, targets_SVHN, n_classes=10))


        #Diag samples
        CIFAR10_test_in_D = predict_samples(la_diag, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_CIFAR100_D = predict_samples(la_diag, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_SVHN_D = predict_samples(la_diag, SVHN_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_D, prob_correct_in_D, ent_in_D, MMC_in_D = get_in_dist_values(CIFAR10_test_in_D, targets_CIFAR10)
        acc_out_CIFAR100_D, prob_correct_out_CIFAR100_D, ent_out_CIFAR100_D, MMC_out_CIFAR100_D, auroc_out_CIFAR100_D = get_out_dist_values(CIFAR10_test_in_D, CIFAR10_test_out_CIFAR100_D, targets_CIFAR100)
        acc_out_SVHN_D, prob_correct_out_SVHN_D, ent_out_SVHN_D, MMC_out_SVHN_D, auroc_out_SVHN_D = get_out_dist_values(CIFAR10_test_in_D, CIFAR10_test_out_SVHN_D, targets_SVHN)
        
        Diag_samples_MMC_in.append(MMC_in_D)
        Diag_samples_MMC_CIFAR100.append(MMC_out_CIFAR100_D)
        Diag_samples_MMC_SVHN.append(MMC_out_SVHN_D)
        Diag_samples_AUROC_CIFAR100.append(auroc_out_CIFAR100_D)
        Diag_samples_AUROC_SVHN.append(auroc_out_SVHN_D)
        Diag_samples_NLL_in.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_in_D)).log_prob(torch.tensor(targets_CIFAR10)).mean().item())
        Diag_samples_NLL_CIFAR100.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_out_CIFAR100_D)).log_prob(torch.tensor(targets_CIFAR100)).mean().item())
        Diag_samples_NLL_SVHN.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_out_SVHN_D)).log_prob(torch.tensor(targets_SVHN)).mean().item())
        Diag_samples_ECE_in.append(scoring.expected_calibration_error(targets_CIFAR10, CIFAR10_test_in_D))
        Diag_samples_ECE_CIFAR100.append(scoring.expected_calibration_error(targets_CIFAR100, CIFAR10_test_out_CIFAR100_D))
        Diag_samples_ECE_SVHN.append(scoring.expected_calibration_error(targets_SVHN, CIFAR10_test_out_SVHN_D))
        Diag_samples_Brier_in.append(get_brier(CIFAR10_test_in_D, targets_CIFAR10, n_classes=10))
        Diag_samples_Brier_CIFAR100.append(get_brier(CIFAR10_test_out_CIFAR100_D, targets_CIFAR100, n_classes=10))
        Diag_samples_Brier_SVHN.append(get_brier(CIFAR10_test_out_SVHN_D, targets_SVHN, n_classes=10))

        
        #KFAC samples
        CIFAR10_test_in_KFAC = predict_samples(la_kron, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_CIFAR100_KFAC = predict_samples(la_kron, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_SVHN_KFAC = predict_samples(la_kron, SVHN_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_K, prob_correct_in_K, ent_in_K, MMC_in_K = get_in_dist_values(CIFAR10_test_in_KFAC, targets_CIFAR10)
        acc_out_CIFAR100_K, prob_correct_out_CIFAR100_K, ent_out_CIFAR100_K, MMC_out_CIFAR100_K, auroc_out_CIFAR100_K = get_out_dist_values(CIFAR10_test_in_KFAC, CIFAR10_test_out_CIFAR100_KFAC, targets_CIFAR100)
        acc_out_SVHN_K, prob_correct_out_SVHN_K, ent_out_SVHN_K, MMC_out_SVHN_K, auroc_out_SVHN_K = get_out_dist_values(CIFAR10_test_in_KFAC, CIFAR10_test_out_SVHN_KFAC, targets_SVHN)

        # KFAC samples
        KFAC_samples_MMC_in.append(MMC_in_K)
        KFAC_samples_MMC_CIFAR100.append(MMC_out_CIFAR100_K)
        KFAC_samples_MMC_SVHN.append(MMC_out_SVHN_K)
        KFAC_samples_AUROC_CIFAR100.append(auroc_out_CIFAR100_K)
        KFAC_samples_AUROC_SVHN.append(auroc_out_SVHN_K)
        KFAC_samples_NLL_in.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_in_KFAC)).log_prob(torch.tensor(targets_CIFAR10)).mean().item())
        KFAC_samples_NLL_CIFAR100.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_out_CIFAR100_KFAC)).log_prob(torch.tensor(targets_CIFAR100)).mean().item())
        KFAC_samples_NLL_SVHN.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_out_SVHN_KFAC)).log_prob(torch.tensor(targets_SVHN)).mean().item())
        KFAC_samples_ECE_in.append(scoring.expected_calibration_error(targets_CIFAR10, CIFAR10_test_in_KFAC))
        KFAC_samples_ECE_CIFAR100.append(scoring.expected_calibration_error(targets_CIFAR100, CIFAR10_test_out_CIFAR100_KFAC))
        KFAC_samples_ECE_SVHN.append(scoring.expected_calibration_error(targets_SVHN, CIFAR10_test_out_SVHN_KFAC))
        KFAC_samples_Brier_in.append(get_brier(CIFAR10_test_in_KFAC, targets_CIFAR10, n_classes=10))
        KFAC_samples_Brier_CIFAR100.append(get_brier(CIFAR10_test_out_CIFAR100_KFAC, targets_CIFAR100, n_classes=10))
        KFAC_samples_Brier_SVHN.append(get_brier(CIFAR10_test_out_SVHN_KFAC, targets_SVHN, n_classes=10))
        
        
        #LB diag
        CIFAR10_test_in_LB_D = predict_LB(la_diag, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_CIFAR100_LB_D = predict_LB(la_diag, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_SVHN_LB_D = predict_LB(la_diag, SVHN_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_LB, prob_correct_in_LB, ent_in_LB, MMC_in_LB = get_in_dist_values(CIFAR10_test_in_LB_D, targets_CIFAR10)
        acc_out_CIFAR100_LB, prob_correct_out_CIFAR100_LB, ent_out_CIFAR100_LB, MMC_out_CIFAR100_LB, auroc_out_CIFAR100_LB = get_out_dist_values(CIFAR10_test_in_LB_D, CIFAR10_test_out_CIFAR100_LB_D, targets_CIFAR100)
        acc_out_SVHN_LB, prob_correct_out_SVHN_LB, ent_out_SVHN_LB, MMC_out_SVHN_LB, auroc_out_SVHN_LB = get_out_dist_values(CIFAR10_test_in_LB_D, CIFAR10_test_out_SVHN_LB_D, targets_SVHN)

        Diag_LB_MMC_in.append(MMC_in_LB)
        Diag_LB_MMC_CIFAR100.append(MMC_out_CIFAR100_LB)
        Diag_LB_MMC_SVHN.append(MMC_out_SVHN_LB)
        Diag_LB_AUROC_CIFAR100.append(auroc_out_CIFAR100_LB)
        Diag_LB_AUROC_SVHN.append(auroc_out_SVHN_LB)
        Diag_LB_NLL_in.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_in_LB_D)).log_prob(torch.tensor(targets_CIFAR10)).mean().item())
        Diag_LB_NLL_CIFAR100.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_out_CIFAR100_LB_D)).log_prob(torch.tensor(targets_CIFAR100)).mean().item())
        Diag_LB_NLL_SVHN.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_out_SVHN_LB_D)).log_prob(torch.tensor(targets_SVHN)).mean().item())
        Diag_LB_ECE_in.append(scoring.expected_calibration_error(targets_CIFAR10, CIFAR10_test_in_LB_D))
        Diag_LB_ECE_CIFAR100.append(scoring.expected_calibration_error(targets_CIFAR100, CIFAR10_test_out_CIFAR100_LB_D))
        Diag_LB_ECE_SVHN.append(scoring.expected_calibration_error(targets_SVHN, CIFAR10_test_out_SVHN_LB_D))
        Diag_LB_Brier_in.append(get_brier(CIFAR10_test_in_LB_D, targets_CIFAR10, n_classes=10))
        Diag_LB_Brier_CIFAR100.append(get_brier(CIFAR10_test_out_CIFAR100_LB_D, targets_CIFAR100, n_classes=10))
        Diag_LB_Brier_SVHN.append(get_brier(CIFAR10_test_out_SVHN_LB_D, targets_SVHN, n_classes=10))
        

        
        #LB KFAC
        CIFAR10_test_in_LB_KFAC = predict_LB(la_kron, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_CIFAR100_LB_KFAC = predict_LB(la_kron, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_SVHN_LB_KFAC = predict_LB(la_kron, SVHN_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_LB_KFAC, prob_correct_in_LB_KFAC, ent_in_LB_KFAC, MMC_in_LB_KFAC = get_in_dist_values(CIFAR10_test_in_LB_KFAC, targets_CIFAR10)
        acc_out_CIFAR100_LB_KFAC, prob_correct_out_CIFAR100_LB_KFAC, ent_out_CIFAR100_LB_KFAC, MMC_out_CIFAR100_LB_KFAC, auroc_out_CIFAR100_LB_KFAC = get_out_dist_values(CIFAR10_test_in_LB_KFAC, CIFAR10_test_out_CIFAR100_LB_KFAC, targets_CIFAR100)
        acc_out_SVHN_LB_KFAC, prob_correct_out_SVHN_LB_KFAC, ent_out_SVHN_LB_KFAC, MMC_out_SVHN_LB_KFAC, auroc_out_SVHN_LB_KFAC = get_out_dist_values(CIFAR10_test_in_LB_KFAC, CIFAR10_test_out_SVHN_LB_KFAC, targets_SVHN)
        
        KFAC_LB_MMC_in.append(MMC_in_LB_KFAC)
        KFAC_LB_MMC_CIFAR100.append(MMC_out_CIFAR100_LB_KFAC)
        KFAC_LB_MMC_SVHN.append(MMC_out_SVHN_LB_KFAC)
        KFAC_LB_AUROC_CIFAR100.append(auroc_out_CIFAR100_LB_KFAC)
        KFAC_LB_AUROC_SVHN.append(auroc_out_SVHN_LB_KFAC)
        KFAC_LB_NLL_in.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_in_LB_KFAC)).log_prob(torch.tensor(targets_CIFAR10)).mean().item())
        KFAC_LB_NLL_CIFAR100.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_out_CIFAR100_LB_KFAC)).log_prob(torch.tensor(targets_CIFAR100)).mean().item())
        KFAC_LB_NLL_SVHN.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_out_SVHN_LB_KFAC)).log_prob(torch.tensor(targets_SVHN)).mean().item())
        KFAC_LB_ECE_in.append(scoring.expected_calibration_error(targets_CIFAR10, CIFAR10_test_in_LB_KFAC))
        KFAC_LB_ECE_CIFAR100.append(scoring.expected_calibration_error(targets_CIFAR100, CIFAR10_test_out_CIFAR100_LB_KFAC))
        KFAC_LB_ECE_SVHN.append(scoring.expected_calibration_error(targets_SVHN, CIFAR10_test_out_SVHN_LB_KFAC))
        KFAC_LB_Brier_in.append(get_brier(CIFAR10_test_in_LB_KFAC, targets_CIFAR10, n_classes=10))
        KFAC_LB_Brier_CIFAR100.append(get_brier(CIFAR10_test_out_CIFAR100_LB_KFAC, targets_CIFAR100, n_classes=10))
        KFAC_LB_Brier_SVHN.append(get_brier(CIFAR10_test_out_SVHN_LB_KFAC, targets_SVHN, n_classes=10))
        


        #LB diag normalized
        CIFAR10_test_in_LB_Dn = predict_LB_norm(la_diag, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_CIFAR100_LB_Dn = predict_LB_norm(la_diag, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_SVHN_LB_Dn = predict_LB_norm(la_diag, SVHN_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_LB, prob_correct_in_LB, ent_in_LB, MMC_in_LB = get_in_dist_values(CIFAR10_test_in_LB_Dn, targets_CIFAR10)
        acc_out_CIFAR100_LB, prob_correct_out_CIFAR100_LB, ent_out_CIFAR100_LB, MMC_out_CIFAR100_LB, auroc_out_CIFAR100_LB = get_out_dist_values(CIFAR10_test_in_LB_Dn, CIFAR10_test_out_CIFAR100_LB_Dn, targets_CIFAR100)
        acc_out_SVHN_LB, prob_correct_out_SVHN_LB, ent_out_SVHN_LB, MMC_out_SVHN_LB, auroc_out_SVHN_LB = get_out_dist_values(CIFAR10_test_in_LB_Dn, CIFAR10_test_out_SVHN_LB_Dn, targets_SVHN)

        Diag_LB_norm_MMC_in.append(MMC_in_LB)
        Diag_LB_norm_MMC_CIFAR100.append(MMC_out_CIFAR100_LB)
        Diag_LB_norm_MMC_SVHN.append(MMC_out_SVHN_LB)
        Diag_LB_norm_AUROC_CIFAR100.append(auroc_out_CIFAR100_LB)
        Diag_LB_norm_AUROC_SVHN.append(auroc_out_SVHN_LB)
        Diag_LB_norm_NLL_in.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_in_LB_Dn)).log_prob(torch.tensor(targets_CIFAR10)).mean().item())
        Diag_LB_norm_NLL_CIFAR100.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_out_CIFAR100_LB_Dn)).log_prob(torch.tensor(targets_CIFAR100)).mean().item())
        Diag_LB_norm_NLL_SVHN.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_out_SVHN_LB_Dn)).log_prob(torch.tensor(targets_SVHN)).mean().item())
        Diag_LB_norm_ECE_in.append(scoring.expected_calibration_error(targets_CIFAR10, CIFAR10_test_in_LB_Dn))
        Diag_LB_norm_ECE_CIFAR100.append(scoring.expected_calibration_error(targets_CIFAR100, CIFAR10_test_out_CIFAR100_LB_Dn))
        Diag_LB_norm_ECE_SVHN.append(scoring.expected_calibration_error(targets_SVHN, CIFAR10_test_out_SVHN_LB_Dn))
        Diag_LB_norm_Brier_in.append(get_brier(CIFAR10_test_in_LB_Dn, targets_CIFAR10, n_classes=10))
        Diag_LB_norm_Brier_CIFAR100.append(get_brier(CIFAR10_test_out_CIFAR100_LB_Dn, targets_CIFAR100, n_classes=10))
        Diag_LB_norm_Brier_SVHN.append(get_brier(CIFAR10_test_out_SVHN_LB_Dn, targets_SVHN, n_classes=10))
        
       

        #LB KFAC normalized
        CIFAR10_test_in_LB_KFACn = predict_LB_norm(la_kron, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_CIFAR100_LB_KFACn = predict_LB_norm(la_kron, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_SVHN_LB_KFACn = predict_LB_norm(la_kron, SVHN_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_LB_KFAC, prob_correct_in_LB_KFAC, ent_in_LB_KFAC, MMC_in_LB_KFAC = get_in_dist_values(CIFAR10_test_in_LB_KFACn, targets_CIFAR10)
        acc_out_CIFAR100_LB_KFAC, prob_correct_out_CIFAR100_LB_KFAC, ent_out_CIFAR100_LB_KFAC, MMC_out_CIFAR100_LB_KFAC, auroc_out_CIFAR100_LB_KFAC = get_out_dist_values(CIFAR10_test_in_LB_KFACn, CIFAR10_test_out_CIFAR100_LB_KFACn, targets_CIFAR100)
        acc_out_SVHN_LB_KFAC, prob_correct_out_SVHN_LB_KFAC, ent_out_SVHN_LB_KFAC, MMC_out_SVHN_LB_KFAC, auroc_out_SVHN_LB_KFAC = get_out_dist_values(CIFAR10_test_in_LB_KFACn, CIFAR10_test_out_SVHN_LB_KFACn, targets_SVHN)
        
        KFAC_LB_norm_MMC_in.append(MMC_in_LB_KFAC)
        KFAC_LB_norm_MMC_CIFAR100.append(MMC_out_CIFAR100_LB_KFAC)
        KFAC_LB_norm_MMC_SVHN.append(MMC_out_SVHN_LB_KFAC)
        KFAC_LB_norm_AUROC_CIFAR100.append(auroc_out_CIFAR100_LB_KFAC)
        KFAC_LB_norm_AUROC_SVHN.append(auroc_out_SVHN_LB_KFAC)
        KFAC_LB_norm_NLL_in.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_in_LB_KFACn)).log_prob(torch.tensor(targets_CIFAR10)).mean().item())
        KFAC_LB_norm_NLL_CIFAR100.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_out_CIFAR100_LB_KFACn)).log_prob(torch.tensor(targets_CIFAR100)).mean().item())
        KFAC_LB_norm_NLL_SVHN.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_out_SVHN_LB_KFACn)).log_prob(torch.tensor(targets_SVHN)).mean().item())
        KFAC_LB_norm_ECE_in.append(scoring.expected_calibration_error(targets_CIFAR10, CIFAR10_test_in_LB_KFACn))
        KFAC_LB_norm_ECE_CIFAR100.append(scoring.expected_calibration_error(targets_CIFAR100, CIFAR10_test_out_CIFAR100_LB_KFACn))
        KFAC_LB_norm_ECE_SVHN.append(scoring.expected_calibration_error(targets_SVHN, CIFAR10_test_out_SVHN_LB_KFACn))
        KFAC_LB_norm_Brier_in.append(get_brier(CIFAR10_test_in_LB_KFAC, targets_CIFAR10, n_classes=10))
        KFAC_LB_norm_Brier_CIFAR100.append(get_brier(CIFAR10_test_out_CIFAR100_LB_KFAC, targets_CIFAR100, n_classes=10))
        KFAC_LB_norm_Brier_SVHN.append(get_brier(CIFAR10_test_out_SVHN_LB_KFAC, targets_SVHN, n_classes=10))
        


        #Extended MacKay diag
        CIFAR10_test_in_PROBIT_D = predict_extended_MacKay(la_diag, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_CIFAR100_PROBIT_D = predict_extended_MacKay(la_diag, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_SVHN_PROBIT_D = predict_extended_MacKay(la_diag, SVHN_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_PROBIT, prob_correct_in_PROBIT, ent_in_PROBIT, MMC_in_PROBIT = get_in_dist_values(CIFAR10_test_in_PROBIT_D, targets_CIFAR10)
        acc_out_CIFAR100_PROBIT, prob_correct_out_CIFAR100_PROBIT, ent_out_CIFAR100_PROBIT, MMC_out_CIFAR100_PROBIT, auroc_out_CIFAR100_PROBIT = get_out_dist_values(CIFAR10_test_in_PROBIT_D, CIFAR10_test_out_CIFAR100_PROBIT_D, targets_CIFAR100)
        acc_out_SVHN_PROBIT, prob_correct_out_SVHN_PROBIT, ent_out_SVHN_PROBIT, MMC_out_SVHN_PROBIT, auroc_out_SVHN_PROBIT = get_out_dist_values(CIFAR10_test_in_PROBIT_D, CIFAR10_test_out_SVHN_PROBIT_D, targets_SVHN)
        
        Diag_PROBIT_MMC_in.append(MMC_in_PROBIT)
        Diag_PROBIT_MMC_CIFAR100.append(MMC_out_CIFAR100_PROBIT)
        Diag_PROBIT_MMC_SVHN.append(MMC_out_SVHN_PROBIT)
        Diag_PROBIT_AUROC_CIFAR100.append(auroc_out_CIFAR100_PROBIT)
        Diag_PROBIT_AUROC_SVHN.append(auroc_out_SVHN_PROBIT)
        Diag_PROBIT_NLL_in.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_in_PROBIT_D)).log_prob(torch.tensor(targets_CIFAR10)).mean().item())
        Diag_PROBIT_NLL_CIFAR100.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_out_CIFAR100_PROBIT_D)).log_prob(torch.tensor(targets_CIFAR100)).mean().item())
        Diag_PROBIT_NLL_SVHN.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_out_SVHN_PROBIT_D)).log_prob(torch.tensor(targets_SVHN)).mean().item())
        Diag_PROBIT_ECE_in.append(scoring.expected_calibration_error(targets_CIFAR10, CIFAR10_test_in_PROBIT_D))
        Diag_PROBIT_ECE_CIFAR100.append(scoring.expected_calibration_error(targets_CIFAR100, CIFAR10_test_out_CIFAR100_PROBIT_D))
        Diag_PROBIT_ECE_SVHN.append(scoring.expected_calibration_error(targets_SVHN, CIFAR10_test_out_SVHN_PROBIT_D))
        Diag_PROBIT_Brier_in.append(get_brier(CIFAR10_test_in_PROBIT_D, targets_CIFAR10, n_classes=10))
        Diag_PROBIT_Brier_CIFAR100.append(get_brier(CIFAR10_test_out_CIFAR100_PROBIT_D, targets_CIFAR100, n_classes=10))
        Diag_PROBIT_Brier_SVHN.append(get_brier(CIFAR10_test_out_SVHN_PROBIT_D, targets_SVHN, n_classes=10))

        
        #Extended MacKay KFAC
        CIFAR10_test_in_PROBIT_K = predict_extended_MacKay(la_kron, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_CIFAR100_PROBIT_K = predict_extended_MacKay(la_kron, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_SVHN_PROBIT_K = predict_extended_MacKay(la_kron, SVHN_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_PROBIT, prob_correct_in_PROBIT, ent_in_PROBIT, MMC_in_PROBIT = get_in_dist_values(CIFAR10_test_in_PROBIT_K, targets_CIFAR10)
        acc_out_CIFAR100_PROBIT, prob_correct_out_CIFAR100_PROBIT, ent_out_CIFAR100_PROBIT, MMC_out_CIFAR100_PROBIT, auroc_out_CIFAR100_PROBIT = get_out_dist_values(CIFAR10_test_in_PROBIT_K, CIFAR10_test_out_CIFAR100_PROBIT_K, targets_CIFAR100)
        acc_out_SVHN_PROBIT, prob_correct_out_SVHN_PROBIT, ent_out_SVHN_PROBIT, MMC_out_SVHN_PROBIT, auroc_out_SVHN_PROBIT = get_out_dist_values(CIFAR10_test_in_PROBIT_K, CIFAR10_test_out_SVHN_PROBIT_K, targets_SVHN)
        
        KFAC_PROBIT_MMC_in.append(MMC_in_PROBIT)
        KFAC_PROBIT_MMC_CIFAR100.append(MMC_out_CIFAR100_PROBIT)
        KFAC_PROBIT_MMC_SVHN.append(MMC_out_SVHN_PROBIT)
        KFAC_PROBIT_AUROC_CIFAR100.append(auroc_out_CIFAR100_PROBIT)
        KFAC_PROBIT_AUROC_SVHN.append(auroc_out_SVHN_PROBIT)
        KFAC_PROBIT_NLL_in.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_in_PROBIT_K)).log_prob(torch.tensor(targets_CIFAR10)).mean().item())
        KFAC_PROBIT_NLL_CIFAR100.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_out_CIFAR100_PROBIT_K)).log_prob(torch.tensor(targets_CIFAR100)).mean().item())
        KFAC_PROBIT_NLL_SVHN.append(-torch.distributions.Categorical(torch.tensor(CIFAR10_test_out_SVHN_PROBIT_K)).log_prob(torch.tensor(targets_SVHN)).mean().item())
        KFAC_PROBIT_ECE_in.append(scoring.expected_calibration_error(targets_CIFAR10, CIFAR10_test_in_PROBIT_K))
        KFAC_PROBIT_ECE_CIFAR100.append(scoring.expected_calibration_error(targets_CIFAR100, CIFAR10_test_out_CIFAR100_PROBIT_K))
        KFAC_PROBIT_ECE_SVHN.append(scoring.expected_calibration_error(targets_SVHN, CIFAR10_test_out_SVHN_PROBIT_K))
        KFAC_PROBIT_Brier_in.append(get_brier(CIFAR10_test_in_PROBIT_K, targets_CIFAR10, n_classes=10))
        KFAC_PROBIT_Brier_CIFAR100.append(get_brier(CIFAR10_test_out_CIFAR100_PROBIT_K, targets_CIFAR100, n_classes=10))
        KFAC_PROBIT_Brier_SVHN.append(get_brier(CIFAR10_test_out_SVHN_PROBIT_K, targets_SVHN, n_classes=10))
        

        
        """
        #SODPP diag
        CIFAR10_test_in_SODPP_D = predict_second_order_dpp(la_diag, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_CIFAR100_SODPP_D = predict_second_order_dpp(la_diag, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_SVHN_SODPP_D = predict_second_order_dpp(la_diag, SVHN_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_SODPP, prob_correct_in_SODPP, ent_in_SODPP, MMC_in_SODPP = get_in_dist_values(CIFAR10_test_in_SODPP_D, targets_CIFAR10)
        acc_out_CIFAR100_SODPP, prob_correct_out_CIFAR100_SODPP, ent_out_CIFAR100_SODPP, MMC_out_CIFAR100_SODPP, auroc_out_CIFAR100_SODPP = get_out_dist_values(CIFAR10_test_in_SODPP_D, CIFAR10_test_out_CIFAR100_SODPP_D, targets_CIFAR100)
        acc_out_SVHN_SODPP, prob_correct_out_SVHN_SODPP, ent_out_SVHN_SODPP, MMC_out_SVHN_SODPP, auroc_out_SVHN_SODPP = get_out_dist_values(CIFAR10_test_in_SODPP_D, CIFAR10_test_out_SVHN_SODPP_D, targets_SVHN)
        
        Diag_SODPP_MMC_in.append(MMC_in_SODPP)
        Diag_SODPP_MMC_CIFAR100.append(MMC_out_CIFAR100_SODPP)
        Diag_SODPP_MMC_SVHN.append(MMC_out_SVHN_SODPP)
        Diag_SODPP_AUROC_CIFAR100.append(auroc_out_CIFAR100_SODPP)
        Diag_SODPP_AUROC_SVHN.append(auroc_out_SVHN_SODPP)
        
        #SODPP KFAC
        CIFAR10_test_in_SODPP_K = predict_second_order_dpp(la_kron, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_CIFAR100_SODPP_K = predict_second_order_dpp(la_kron, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR10_test_out_SVHN_SODPP_K = predict_second_order_dpp(la_kron, SVHN_test_loader, timing=False, device=device).cpu().numpy()

        acc_in_SODPP, prob_correct_in_SODPP, ent_in_SODPP, MMC_in_SODPP = get_in_dist_values(CIFAR10_test_in_SODPP_K, targets_CIFAR10)
        acc_out_CIFAR100_SODPP, prob_correct_out_CIFAR100_SODPP, ent_out_CIFAR100_SODPP, MMC_out_CIFAR100_SODPP, auroc_out_CIFAR100_SODPP = get_out_dist_values(CIFAR10_test_in_SODPP_K, CIFAR10_test_out_CIFAR100_SODPP_K, targets_CIFAR100)
        acc_out_SVHN_SODPP, prob_correct_out_SVHN_SODPP, ent_out_SVHN_SODPP, MMC_out_SVHN_SODPP, auroc_out_SVHN_SODPP = get_out_dist_values(CIFAR10_test_in_SODPP_K, CIFAR10_test_out_SVHN_SODPP_K, targets_SVHN)

        KFAC_SODPP_MMC_in.append(MMC_in_SODPP)
        KFAC_SODPP_MMC_CIFAR100.append(MMC_out_CIFAR100_SODPP)
        KFAC_SODPP_MMC_SVHN.append(MMC_out_SVHN_SODPP)
        KFAC_SODPP_AUROC_CIFAR100.append(auroc_out_CIFAR100_SODPP)
        KFAC_SODPP_AUROC_SVHN.append(auroc_out_SVHN_SODPP)
        """
        
    #### save results
    results_dict = {
        'MAP_MMC_in':MAP_MMC_in,
        'MAP_MMC_CIFAR100':MAP_MMC_CIFAR100,
        'MAP_MMC_SVHN':MAP_MMC_SVHN,
        'MAP_AUROC_CIFAR100':MAP_AUROC_CIFAR100,
        'MAP_AUROC_SVHN':MAP_AUROC_SVHN,
        'MAP_NLL_in':MAP_NLL_in,
        'MAP_NLL_CIFAR100':MAP_NLL_CIFAR100,
        'MAP_NLL_SVHN':MAP_NLL_SVHN,
        'MAP_ECE_in':MAP_ECE_in,
        'MAP_ECE_CIFAR100':MAP_ECE_CIFAR100,
        'MAP_ECE_SVHN':MAP_ECE_SVHN,
        'MAP_Brier_in':MAP_Brier_in,
        'MAP_Brier_CIFAR100':MAP_Brier_CIFAR100,
        'MAP_Brier_SVHN':MAP_Brier_SVHN,
        'Diag_samples_MMC_in':Diag_samples_MMC_in,
        'Diag_samples_MMC_CIFAR100':Diag_samples_MMC_CIFAR100,
        'Diag_samples_MMC_SVHN':Diag_samples_MMC_SVHN,
        'Diag_samples_AUROC_CIFAR100':Diag_samples_AUROC_CIFAR100,
        'Diag_samples_AUROC_SVHN':Diag_samples_AUROC_SVHN,
        'Diag_samples_NLL_in':Diag_samples_NLL_in,
        'Diag_samples_NLL_CIFAR100':Diag_samples_NLL_CIFAR100,
        'Diag_samples_NLL_SVHN':Diag_samples_NLL_SVHN,
        'Diag_samples_ECE_in':Diag_samples_ECE_in,
        'Diag_samples_ECE_CIFAR100':Diag_samples_ECE_CIFAR100,
        'Diag_samples_ECE_SVHN':Diag_samples_ECE_SVHN,
        'Diag_samples_Brier_in':Diag_samples_Brier_in,
        'Diag_samples_Brier_CIFAR100':Diag_samples_Brier_CIFAR100,
        'Diag_samples_Brier_SVHN':Diag_samples_Brier_SVHN,
        'KFAC_samples_MMC_in':KFAC_samples_MMC_in,
        'KFAC_samples_MMC_CIFAR100':KFAC_samples_MMC_CIFAR100,
        'KFAC_samples_MMC_SVHN':KFAC_samples_MMC_SVHN,
        'KFAC_samples_AUROC_CIFAR100':KFAC_samples_AUROC_CIFAR100,
        'KFAC_samples_AUROC_SVHN':KFAC_samples_AUROC_SVHN,
        'KFAC_samples_NLL_in':KFAC_samples_NLL_in,
        'KFAC_samples_NLL_CIFAR100':KFAC_samples_NLL_CIFAR100,
        'KFAC_samples_NLL_SVHN':KFAC_samples_NLL_SVHN,
        'KFAC_samples_ECE_in':KFAC_samples_ECE_in,
        'KFAC_samples_ECE_CIFAR100':KFAC_samples_ECE_CIFAR100,
        'KFAC_samples_ECE_SVHN':KFAC_samples_ECE_SVHN,
        'KFAC_samples_Brier_in':KFAC_samples_Brier_in,
        'KFAC_samples_Brier_CIFAR100':KFAC_samples_Brier_CIFAR100,
        'KFAC_samples_Brier_SVHN':KFAC_samples_Brier_SVHN,
        'Diag_LB_MMC_in':Diag_LB_MMC_in,
        'Diag_LB_MMC_CIFAR100':Diag_LB_MMC_CIFAR100,
        'Diag_LB_MMC_SVHN':Diag_LB_MMC_SVHN,
        'Diag_LB_AUROC_CIFAR100':Diag_LB_AUROC_CIFAR100,
        'Diag_LB_AUROC_SVHN':Diag_LB_AUROC_SVHN,
        'Diag_LB_NLL_in':Diag_LB_NLL_in,
        'Diag_LB_NLL_CIFAR100':Diag_LB_NLL_CIFAR100,
        'Diag_LB_NLL_SVHN':Diag_LB_NLL_SVHN,
        'Diag_LB_ECE_in':Diag_LB_ECE_in,
        'Diag_LB_ECE_CIFAR100':Diag_LB_ECE_CIFAR100,
        'Diag_LB_ECE_SVHN':Diag_LB_ECE_SVHN,
        'Diag_LB_Brier_in':Diag_LB_Brier_in,
        'Diag_LB_Brier_CIFAR100':Diag_LB_Brier_CIFAR100,
        'Diag_LB_Brier_SVHN':Diag_LB_Brier_SVHN,
        'KFAC_LB_MMC_in':KFAC_LB_MMC_in,
        'KFAC_LB_MMC_CIFAR100':KFAC_LB_MMC_CIFAR100,
        'KFAC_LB_MMC_SVHN':KFAC_LB_MMC_SVHN,
        'KFAC_LB_AUROC_CIFAR100':KFAC_LB_AUROC_CIFAR100,
        'KFAC_LB_AUROC_SVHN':KFAC_LB_AUROC_SVHN,
        'KFAC_LB_NLL_in':KFAC_LB_NLL_in,
        'KFAC_LB_NLL_CIFAR100':KFAC_LB_NLL_CIFAR100,
        'KFAC_LB_NLL_SVHN':KFAC_LB_NLL_SVHN,
        'KFAC_LB_ECE_in':KFAC_LB_ECE_in,
        'KFAC_LB_ECE_CIFAR100':KFAC_LB_ECE_CIFAR100,
        'KFAC_LB_ECE_SVHN':KFAC_LB_ECE_SVHN,
        'KFAC_LB_Brier_in':KFAC_LB_Brier_in,
        'KFAC_LB_Brier_CIFAR100':KFAC_LB_Brier_CIFAR100,
        'KFAC_LB_Brier_SVHN':KFAC_LB_Brier_SVHN,
        'Diag_LB_norm_MMC_in':Diag_LB_norm_MMC_in,
        'Diag_LB_norm_MMC_CIFAR100':Diag_LB_norm_MMC_CIFAR100,
        'Diag_LB_norm_MMC_SVHN':Diag_LB_norm_MMC_SVHN,
        'Diag_LB_norm_AUROC_CIFAR100':Diag_LB_norm_AUROC_CIFAR100,
        'Diag_LB_norm_AUROC_SVHN':Diag_LB_norm_AUROC_SVHN,
        'Diag_LB_norm_NLL_in':Diag_LB_norm_NLL_in,
        'Diag_LB_norm_NLL_CIFAR100':Diag_LB_norm_NLL_CIFAR100,
        'Diag_LB_norm_NLL_SVHN':Diag_LB_norm_NLL_SVHN,
        'Diag_LB_norm_ECE_in':Diag_LB_norm_ECE_in,
        'Diag_LB_norm_ECE_CIFAR100':Diag_LB_norm_ECE_CIFAR100,
        'Diag_LB_norm_ECE_SVHN':Diag_LB_norm_ECE_SVHN,
        'Diag_LB_norm_Brier_in':Diag_LB_norm_Brier_in,
        'Diag_LB_norm_Brier_CIFAR100':Diag_LB_norm_Brier_CIFAR100,
        'Diag_LB_norm_Brier_SVHN':Diag_LB_norm_Brier_SVHN,
        'KFAC_LB_norm_MMC_in':KFAC_LB_norm_MMC_in,
        'KFAC_LB_norm_MMC_CIFAR100':KFAC_LB_norm_MMC_CIFAR100,
        'KFAC_LB_norm_MMC_SVHN':KFAC_LB_norm_MMC_SVHN,
        'KFAC_LB_norm_AUROC_CIFAR100':KFAC_LB_norm_AUROC_CIFAR100,
        'KFAC_LB_norm_AUROC_SVHN':KFAC_LB_norm_AUROC_SVHN,
        'KFAC_LB_norm_NLL_in':KFAC_LB_norm_NLL_in,
        'KFAC_LB_norm_NLL_CIFAR100':KFAC_LB_norm_NLL_CIFAR100,
        'KFAC_LB_norm_NLL_SVHN':KFAC_LB_norm_NLL_SVHN,
        'KFAC_LB_norm_ECE_in':KFAC_LB_norm_ECE_in,
        'KFAC_LB_norm_ECE_CIFAR100':KFAC_LB_norm_ECE_CIFAR100,
        'KFAC_LB_norm_ECE_SVHN':KFAC_LB_norm_ECE_SVHN,
        'KFAC_LB_norm_Brier_in':KFAC_LB_norm_Brier_in,
        'KFAC_LB_norm_Brier_CIFAR100':KFAC_LB_norm_Brier_CIFAR100,
        'KFAC_LB_norm_Brier_SVHN':KFAC_LB_norm_Brier_SVHN,
        'Diag_PROBIT_MMC_in':Diag_PROBIT_MMC_in,
        'Diag_PROBIT_MMC_CIFAR100':Diag_PROBIT_MMC_CIFAR100,
        'Diag_PROBIT_MMC_SVHN':Diag_PROBIT_MMC_SVHN,
        'Diag_PROBIT_AUROC_CIFAR100':Diag_PROBIT_AUROC_CIFAR100,
        'Diag_PROBIT_AUROC_SVHN':Diag_PROBIT_AUROC_SVHN,
        'Diag_PROBIT_NLL_in':Diag_PROBIT_NLL_in,
        'Diag_PROBIT_NLL_CIFAR100':Diag_PROBIT_NLL_CIFAR100,
        'Diag_PROBIT_NLL_SVHN':Diag_PROBIT_NLL_SVHN,
        'Diag_PROBIT_ECE_in':Diag_PROBIT_ECE_in,
        'Diag_PROBIT_ECE_CIFAR100':Diag_PROBIT_ECE_CIFAR100,
        'Diag_PROBIT_ECE_SVHN':Diag_PROBIT_ECE_SVHN,
        'Diag_PROBIT_Brier_in':Diag_PROBIT_Brier_in,
        'Diag_PROBIT_Brier_CIFAR100':Diag_PROBIT_Brier_CIFAR100,
        'Diag_PROBIT_Brier_SVHN':Diag_PROBIT_Brier_SVHN,
        'KFAC_PROBIT_MMC_in':KFAC_PROBIT_MMC_in,
        'KFAC_PROBIT_MMC_CIFAR100':KFAC_PROBIT_MMC_CIFAR100,
        'KFAC_PROBIT_MMC_SVHN':KFAC_PROBIT_MMC_SVHN,
        'KFAC_PROBIT_AUROC_CIFAR100':KFAC_PROBIT_AUROC_CIFAR100,
        'KFAC_PROBIT_AUROC_SVHN':KFAC_PROBIT_AUROC_SVHN,
        'KFAC_PROBIT_NLL_in':KFAC_PROBIT_NLL_in,
        'KFAC_PROBIT_NLL_CIFAR100':KFAC_PROBIT_NLL_CIFAR100,
        'KFAC_PROBIT_NLL_SVHN':KFAC_PROBIT_NLL_SVHN,
        'KFAC_PROBIT_ECE_in':KFAC_PROBIT_ECE_in,
        'KFAC_PROBIT_ECE_CIFAR100':KFAC_PROBIT_ECE_CIFAR100,
        'KFAC_PROBIT_ECE_SVHN':KFAC_PROBIT_ECE_SVHN,
        'KFAC_PROBIT_Brier_in':KFAC_PROBIT_Brier_in,
        'KFAC_PROBIT_Brier_CIFAR100':KFAC_PROBIT_Brier_CIFAR100,
        'KFAC_PROBIT_Brier_SVHN':KFAC_PROBIT_Brier_SVHN,
        #'Diag_SODPP_MMC_in':Diag_SODPP_MMC_in,
        #'Diag_SODPP_MMC_CIFAR100':Diag_SODPP_MMC_CIFAR100,
        #'Diag_SODPP_MMC_SVHN':Diag_SODPP_MMC_SVHN,
        #'Diag_SODPP_AUROC_CIFAR100':Diag_SODPP_AUROC_CIFAR100,
        #'Diag_SODPP_AUROC_SVHN':Diag_SODPP_AUROC_SVHN,
        #'KFAC_SODPP_MMC_in':KFAC_SODPP_MMC_in,
        #'KFAC_SODPP_MMC_CIFAR100':KFAC_SODPP_MMC_CIFAR100,
        #'KFAC_SODPP_MMC_SVHN':KFAC_SODPP_MMC_SVHN,
        #'KFAC_SODPP_AUROC_CIFAR100':KFAC_SODPP_AUROC_CIFAR100,
        #'KFAC_SODPP_AUROC_SVHN':KFAC_SODPP_AUROC_SVHN
    }
    results_df = pd.DataFrame(results_dict)
    RESULTS_PATH = os.getcwd() + "/Experiment_results/CIFAR10_results.csv"
    print("saving at: ", RESULTS_PATH)
    results_df.to_csv(RESULTS_PATH)


#### RUN
if __name__ == '__main__':
    main()