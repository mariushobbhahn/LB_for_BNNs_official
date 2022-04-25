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


    BATCH_SIZE = 32
    ### load data
    #load in CIFAR100

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        #transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
        #transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    ])


    #'~/data/cifar100'
    CIFAR100_train = torchvision.datasets.CIFAR100(root='~/data/cifar100', train=True,
                                           download=True, transform=transform_train)
    CIFAR100_train_loader = torch.utils.data.DataLoader(CIFAR100_train, batch_size=BATCH_SIZE,
                                             shuffle=True)
    #'~/data/cifar100'
    CIFAR100_test = torchvision.datasets.CIFAR100(root='~/data/cifar100', train=False,
                                           download=True, transform=transform_test)
    CIFAR100_test_loader = torch.utils.data.DataLoader(CIFAR100_test, batch_size=BATCH_SIZE,
                                             shuffle=False)

    CIFAR10_testset = torchvision.datasets.CIFAR10(root='~/data/cifar10', train=False, download=True, transform=transform_test)
    CIFAR10_test_loader = torch.utils.data.DataLoader(CIFAR10_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


    test_data_SVHN = torchvision.datasets.SVHN('~/data/SVHN', split='test',
                             download=False, transform=transform_test)
    SVHN_test_loader = torch.utils.data.DataLoader(test_data_SVHN, batch_size=BATCH_SIZE)


    ### load model
    CIFAR100_PATH = os.getcwd() + "/pretrained_weights/CIFAR100_resnet18_pretrained.pt"
    CIFAR100_model = resnet18_v2.ResNet18(num_classes=100).to(device)
    print("loading model from: {}".format(CIFAR100_PATH))
    CIFAR100_model.load_state_dict(torch.load(CIFAR100_PATH))
    CIFAR100_model.eval()
    
    targets_CIFAR10 = CIFAR10_testset.targets
    targets_CIFAR100 = CIFAR100_test.targets
    targets_SVHN = []
    for x,y in SVHN_test_loader:
        targets_SVHN.append(y)
    targets_SVHN = torch.cat(targets_SVHN).numpy()
        
    num_samples = 100

    #### Experiments

    # MAP
    MAP_MMC_in = []
    MAP_MMC_CIFAR10 = []
    MAP_MMC_SVHN = []
    MAP_AUROC_CIFAR10 = []
    MAP_AUROC_SVHN = []
    MAP_NLL_in = []
    MAP_NLL_CIFAR10 = []
    MAP_NLL_SVHN = []
    MAP_ECE_in = []
    MAP_ECE_CIFAR10 = []
    MAP_ECE_SVHN = []
    
    # Diag samples
    Diag_samples_MMC_in = []
    Diag_samples_MMC_CIFAR10 = []
    Diag_samples_MMC_SVHN = []
    Diag_samples_AUROC_CIFAR10 = []
    Diag_samples_AUROC_SVHN = []
    Diag_samples_NLL_in = []
    Diag_samples_NLL_CIFAR10 = []
    Diag_samples_NLL_SVHN = []
    Diag_samples_ECE_in = []
    Diag_samples_ECE_CIFAR10 = []
    Diag_samples_ECE_SVHN = []

    # KFAC samples
    KFAC_samples_MMC_in = []
    KFAC_samples_MMC_CIFAR10 = []
    KFAC_samples_MMC_SVHN = []
    KFAC_samples_AUROC_CIFAR10 = []
    KFAC_samples_AUROC_SVHN = []
    KFAC_samples_NLL_in = []
    KFAC_samples_NLL_CIFAR10 = []
    KFAC_samples_NLL_SVHN = []
    KFAC_samples_ECE_in = []
    KFAC_samples_ECE_CIFAR10 = []
    KFAC_samples_ECE_SVHN = []

    # Diag LB
    Diag_LB_MMC_in = []
    Diag_LB_MMC_CIFAR10 = []
    Diag_LB_MMC_SVHN = []
    Diag_LB_AUROC_CIFAR10 = []
    Diag_LB_AUROC_SVHN = []
    Diag_LB_NLL_in = []
    Diag_LB_NLL_CIFAR10 = []
    Diag_LB_NLL_SVHN = []
    Diag_LB_ECE_in = []
    Diag_LB_ECE_CIFAR10 = []
    Diag_LB_ECE_SVHN = []
    
    # KFAC LB
    KFAC_LB_MMC_in = []
    KFAC_LB_MMC_CIFAR10 = []
    KFAC_LB_MMC_SVHN = []
    KFAC_LB_AUROC_CIFAR10 = []
    KFAC_LB_AUROC_SVHN = []
    KFAC_LB_NLL_in = []
    KFAC_LB_NLL_CIFAR10 = []
    KFAC_LB_NLL_SVHN = []
    KFAC_LB_ECE_in = []
    KFAC_LB_ECE_CIFAR10 = []
    KFAC_LB_ECE_SVHN = []

    # Diag LB normalized
    Diag_LB_norm_MMC_in = []
    Diag_LB_norm_MMC_CIFAR10 = []
    Diag_LB_norm_MMC_SVHN = []
    Diag_LB_norm_AUROC_CIFAR10 = []
    Diag_LB_norm_AUROC_SVHN = []
    Diag_LB_norm_NLL_in = []
    Diag_LB_norm_NLL_CIFAR10 = []
    Diag_LB_norm_NLL_SVHN = []
    Diag_LB_norm_ECE_in = []
    Diag_LB_norm_ECE_CIFAR10 = []
    Diag_LB_norm_ECE_SVHN = []
    
    # KFAC LB normalized
    KFAC_LB_norm_MMC_in = []
    KFAC_LB_norm_MMC_CIFAR10 = []
    KFAC_LB_norm_MMC_SVHN = []
    KFAC_LB_norm_AUROC_CIFAR10 = []
    KFAC_LB_norm_AUROC_SVHN = []
    KFAC_LB_norm_NLL_in = []
    KFAC_LB_norm_NLL_CIFAR10 = []
    KFAC_LB_norm_NLL_SVHN = []
    KFAC_LB_norm_ECE_in = []
    KFAC_LB_norm_ECE_CIFAR10 = []
    KFAC_LB_norm_ECE_SVHN = []
    
    # Diag PROBIT
    Diag_PROBIT_MMC_in = []
    Diag_PROBIT_MMC_CIFAR10 = []
    Diag_PROBIT_MMC_SVHN = []
    Diag_PROBIT_AUROC_CIFAR10 = []
    Diag_PROBIT_AUROC_SVHN = []
    Diag_PROBIT_NLL_in = []
    Diag_PROBIT_NLL_CIFAR10 = []
    Diag_PROBIT_NLL_SVHN = []
    Diag_PROBIT_ECE_in = []
    Diag_PROBIT_ECE_CIFAR10 = []
    Diag_PROBIT_ECE_SVHN = []
    
    # KFAC PROBIT
    KFAC_PROBIT_MMC_in = []
    KFAC_PROBIT_MMC_CIFAR10 = []
    KFAC_PROBIT_MMC_SVHN = []
    KFAC_PROBIT_AUROC_CIFAR10 = []
    KFAC_PROBIT_AUROC_SVHN = []
    KFAC_PROBIT_NLL_in = []
    KFAC_PROBIT_NLL_CIFAR10 = []
    KFAC_PROBIT_NLL_SVHN = []
    KFAC_PROBIT_ECE_in = []
    KFAC_PROBIT_ECE_CIFAR10 = []
    KFAC_PROBIT_ECE_SVHN = []
    
    """
    # Diag SODPP
    Diag_SODPP_MMC_in = []
    Diag_SODPP_MMC_CIFAR10 = []
    Diag_SODPP_MMC_SVHN = []
    Diag_SODPP_AUROC_CIFAR10 = []
    Diag_SODPP_AUROC_SVHN = []
    
    # KFAC SODPP
    KFAC_SODPP_MMC_in = []
    KFAC_SODPP_MMC_CIFAR10 = []
    KFAC_SODPP_MMC_SVHN = []
    KFAC_SODPP_AUROC_CIFAR10 = []
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
        
        la_diag = Laplace(CIFAR100_model, 'classification', 
                     subset_of_weights='last_layer', 
                     hessian_structure='diag',
                     prior_precision=5e+1) # 5e-4 # Choose prior precision according to weight decay
        la_diag.fit(CIFAR100_train_loader)

        la_kron = Laplace(CIFAR100_model, 'classification', 
                     subset_of_weights='last_layer', 
                     hessian_structure='kron',
                     prior_precision=5e+0) # 5e-4 # Choose prior precision according to weight decay
        la_kron.fit(CIFAR100_train_loader)

        #MAP estimates
        CIFAR100_test_in_MAP = predict_MAP(CIFAR100_model, CIFAR100_test_loader, device=device).cpu().numpy()
        CIFAR100_test_out_CIFAR10_MAP = predict_MAP(CIFAR100_model, CIFAR10_test_loader, device=device).cpu().numpy()
        CIFAR100_test_out_SVHN_MAP = predict_MAP(CIFAR100_model, SVHN_test_loader, device=device).cpu().numpy()
        
        acc_in_MAP, prob_correct_in_MAP, ent_in_MAP, MMC_in_MAP = get_in_dist_values(CIFAR100_test_in_MAP, targets_CIFAR100)
        acc_out_CIFAR10_MAP, prob_correct_out_CIFAR10_MAP, ent_out_CIFAR10, MMC_out_CIFAR10_MAP, auroc_out_CIFAR10_MAP = get_out_dist_values(CIFAR100_test_in_MAP, CIFAR100_test_out_CIFAR10_MAP, targets_CIFAR10)
        acc_out_SVHN_MAP, prob_correct_out_SVHN_MAP, ent_out_SVHN, MMC_out_SVHN_MAP, auroc_out_SVHN_MAP = get_out_dist_values(CIFAR100_test_in_MAP, CIFAR100_test_out_SVHN_MAP, targets_SVHN)
        
        MAP_MMC_in.append(MMC_in_MAP)
        MAP_MMC_CIFAR10.append(MMC_out_CIFAR10_MAP)
        MAP_MMC_SVHN.append(MMC_out_SVHN_MAP)
        MAP_AUROC_CIFAR10.append(auroc_out_CIFAR10_MAP)
        MAP_AUROC_SVHN.append(auroc_out_SVHN_MAP)
        MAP_NLL_in.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_in_MAP)).log_prob(torch.tensor(targets_CIFAR100)).mean().item())
        MAP_NLL_CIFAR10.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_out_CIFAR10_MAP)).log_prob(torch.tensor(targets_CIFAR10)).mean().item())
        MAP_NLL_SVHN.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_out_SVHN_MAP)).log_prob(torch.tensor(targets_SVHN)).mean().item())
        MAP_ECE_in.append(scoring.expected_calibration_error(targets_CIFAR100, CIFAR100_test_in_MAP))
        MAP_ECE_CIFAR10.append(scoring.expected_calibration_error(targets_CIFAR10, CIFAR100_test_out_CIFAR10_MAP))
        MAP_ECE_SVHN.append(scoring.expected_calibration_error(targets_SVHN, CIFAR100_test_out_SVHN_MAP))


        #Diag samples
        CIFAR100_test_in_D = predict_samples(la_diag, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_CIFAR10_D = predict_samples(la_diag, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_SVHN_D = predict_samples(la_diag, SVHN_test_loader, timing=False, device=device).cpu().numpy()

        acc_in_D, prob_correct_in_D, ent_in_D, MMC_in_D = get_in_dist_values(CIFAR100_test_in_D, targets_CIFAR100)
        acc_out_CIFAR10_D, prob_correct_out_CIFAR10_D, ent_out_CIFAR10_D, MMC_out_CIFAR10_D, auroc_out_CIFAR10_D = get_out_dist_values(CIFAR100_test_in_D, CIFAR100_test_out_CIFAR10_D, targets_CIFAR10)
        acc_out_SVHN_D, prob_correct_out_SVHN_D, ent_out_SVHN_D, MMC_out_SVHN_D, auroc_out_SVHN_D = get_out_dist_values(CIFAR100_test_in_D, CIFAR100_test_out_SVHN_D, targets_SVHN)

        Diag_samples_MMC_in.append(MMC_in_D)
        Diag_samples_MMC_CIFAR10.append(MMC_out_CIFAR10_D)
        Diag_samples_MMC_SVHN.append(MMC_out_SVHN_D)
        Diag_samples_AUROC_CIFAR10.append(auroc_out_CIFAR10_D)
        Diag_samples_AUROC_SVHN.append(auroc_out_SVHN_D)
        Diag_samples_NLL_in.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_in_D)).log_prob(torch.tensor(targets_CIFAR100)).mean().item())
        Diag_samples_NLL_CIFAR10.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_out_CIFAR10_D)).log_prob(torch.tensor(targets_CIFAR10)).mean().item())
        Diag_samples_NLL_SVHN.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_out_SVHN_D)).log_prob(torch.tensor(targets_SVHN)).mean().item())
        Diag_samples_ECE_in.append(scoring.expected_calibration_error(targets_CIFAR100, CIFAR100_test_in_D))
        Diag_samples_ECE_CIFAR10.append(scoring.expected_calibration_error(targets_CIFAR10, CIFAR100_test_out_CIFAR10_D))
        Diag_samples_ECE_SVHN.append(scoring.expected_calibration_error(targets_SVHN, CIFAR100_test_out_SVHN_D))

        
        #KFAC samples
        CIFAR100_test_in_K = predict_samples(la_kron, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_CIFAR10_K = predict_samples(la_kron, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_SVHN_K = predict_samples(la_kron, SVHN_test_loader, timing=False, device=device).cpu().numpy()

        acc_in_K, prob_correct_in_K, ent_in_K, MMC_in_K = get_in_dist_values(CIFAR100_test_in_K, targets_CIFAR100)
        acc_out_CIFAR10_K, prob_correct_out_CIFAR10_K, ent_out_CIFAR10_K, MMC_out_CIFAR10_K, auroc_out_CIFAR10_K = get_out_dist_values(CIFAR100_test_in_K, CIFAR100_test_out_CIFAR10_K, targets_CIFAR10)
        acc_out_SVHN_K, prob_correct_out_SVHN_K, ent_out_SVHN_K, MMC_out_SVHN_K, auroc_out_SVHN_K = get_out_dist_values(CIFAR100_test_in_K, CIFAR100_test_out_SVHN_K, targets_SVHN)

        # KFAC samples
        KFAC_samples_MMC_in.append(MMC_in_K)
        KFAC_samples_MMC_CIFAR10.append(MMC_out_CIFAR10_K)
        KFAC_samples_MMC_SVHN.append(MMC_out_SVHN_K)
        KFAC_samples_AUROC_CIFAR10.append(auroc_out_CIFAR10_K)
        KFAC_samples_AUROC_SVHN.append(auroc_out_SVHN_K)
        KFAC_samples_NLL_in.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_in_K)).log_prob(torch.tensor(targets_CIFAR100)).mean().item())
        KFAC_samples_NLL_CIFAR10.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_out_CIFAR10_K)).log_prob(torch.tensor(targets_CIFAR10)).mean().item())
        KFAC_samples_NLL_SVHN.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_out_SVHN_K)).log_prob(torch.tensor(targets_SVHN)).mean().item())
        KFAC_samples_ECE_in.append(scoring.expected_calibration_error(targets_CIFAR100, CIFAR100_test_in_K))
        KFAC_samples_ECE_CIFAR10.append(scoring.expected_calibration_error(targets_CIFAR10, CIFAR100_test_out_CIFAR10_K))
        KFAC_samples_ECE_SVHN.append(scoring.expected_calibration_error(targets_SVHN, CIFAR100_test_out_SVHN_K))

        
        #LB diag
        CIFAR100_test_in_LB_D = predict_LB(la_diag, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_CIFAR10_LB_D = predict_LB(la_diag, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_SVHN_LB_D = predict_LB(la_diag, SVHN_test_loader, timing=False, device=device).cpu().numpy()

        acc_in_LB, prob_correct_in_LB, ent_in_LB, MMC_in_LB = get_in_dist_values(CIFAR100_test_in_LB_D, targets_CIFAR100)
        acc_out_CIFAR10_LB, prob_correct_out_CIFAR10_LB, ent_out_CIFAR10_LB, MMC_out_CIFAR10_LB, auroc_out_CIFAR10_LB = get_out_dist_values(CIFAR100_test_in_LB_D, CIFAR100_test_out_CIFAR10_LB_D, targets_CIFAR10)
        acc_out_SVHN_LB, prob_correct_out_SVHN_LB, ent_out_SVHN_LB, MMC_out_SVHN_LB, auroc_out_SVHN_LB = get_out_dist_values(CIFAR100_test_in_LB_D, CIFAR100_test_out_SVHN_LB_D, targets_SVHN)

        Diag_LB_MMC_in.append(MMC_in_LB)
        Diag_LB_MMC_CIFAR10.append(MMC_out_CIFAR10_LB)
        Diag_LB_MMC_SVHN.append(MMC_out_SVHN_LB)
        Diag_LB_AUROC_CIFAR10.append(auroc_out_CIFAR10_LB)
        Diag_LB_AUROC_SVHN.append(auroc_out_SVHN_LB)
        Diag_LB_NLL_in.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_in_LB_D)).log_prob(torch.tensor(targets_CIFAR100)).mean().item())
        Diag_LB_NLL_CIFAR10.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_out_CIFAR10_LB_D)).log_prob(torch.tensor(targets_CIFAR10)).mean().item())
        Diag_LB_NLL_SVHN.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_out_SVHN_LB_D)).log_prob(torch.tensor(targets_SVHN)).mean().item())
        Diag_LB_ECE_in.append(scoring.expected_calibration_error(targets_CIFAR100, CIFAR100_test_in_LB_D))
        Diag_LB_ECE_CIFAR10.append(scoring.expected_calibration_error(targets_CIFAR10, CIFAR100_test_out_CIFAR10_LB_D))
        Diag_LB_ECE_SVHN.append(scoring.expected_calibration_error(targets_SVHN, CIFAR100_test_out_SVHN_LB_D))

        
        #LB KFAC
        CIFAR100_test_in_LB_K = predict_LB(la_kron, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_CIFAR10_LB_K = predict_LB(la_kron, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_SVHN_LB_K = predict_LB(la_kron, SVHN_test_loader, timing=False, device=device).cpu().numpy()

        acc_in_LB_KFAC, prob_correct_in_LB_KFAC, ent_in_LB_KFAC, MMC_in_LB_KFAC = get_in_dist_values(CIFAR100_test_in_LB_K, targets_CIFAR100)
        acc_out_CIFAR10_LB_KFAC, prob_correct_out_CIFAR10_LB_KFAC, ent_out_CIFAR10_LB_KFAC, MMC_out_CIFAR10_LB_KFAC, auroc_out_CIFAR10_LB_KFAC = get_out_dist_values(CIFAR100_test_in_LB_K, CIFAR100_test_out_CIFAR10_LB_K, targets_CIFAR10)
        acc_out_SVHN_LB_KFAC, prob_correct_out_SVHN_LB_KFAC, ent_out_SVHN_LB_KFAC, MMC_out_SVHN_LB_KFAC, auroc_out_SVHN_LB_KFAC = get_out_dist_values(CIFAR100_test_in_LB_K, CIFAR100_test_out_SVHN_LB_K, targets_SVHN)

        KFAC_LB_MMC_in.append(MMC_in_LB_KFAC)
        KFAC_LB_MMC_CIFAR10.append(MMC_out_CIFAR10_LB_KFAC)
        KFAC_LB_MMC_SVHN.append(MMC_out_SVHN_LB_KFAC)
        KFAC_LB_AUROC_CIFAR10.append(auroc_out_CIFAR10_LB_KFAC)
        KFAC_LB_AUROC_SVHN.append(auroc_out_SVHN_LB_KFAC)
        KFAC_LB_NLL_in.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_in_LB_K)).log_prob(torch.tensor(targets_CIFAR100)).mean().item())
        KFAC_LB_NLL_CIFAR10.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_out_CIFAR10_LB_K)).log_prob(torch.tensor(targets_CIFAR10)).mean().item())
        KFAC_LB_NLL_SVHN.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_out_SVHN_LB_K)).log_prob(torch.tensor(targets_SVHN)).mean().item())
        KFAC_LB_ECE_in.append(scoring.expected_calibration_error(targets_CIFAR100, CIFAR100_test_in_LB_K))
        KFAC_LB_ECE_CIFAR10.append(scoring.expected_calibration_error(targets_CIFAR10, CIFAR100_test_out_CIFAR10_LB_K))
        KFAC_LB_ECE_SVHN.append(scoring.expected_calibration_error(targets_SVHN, CIFAR100_test_out_SVHN_LB_K))


        #LB diag norm
        CIFAR100_test_in_LB_Dn = predict_LB_norm(la_diag, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_CIFAR10_LB_Dn = predict_LB_norm(la_diag, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_SVHN_LB_Dn = predict_LB_norm(la_diag, SVHN_test_loader, timing=False, device=device).cpu().numpy()

        acc_in_LB, prob_correct_in_LB, ent_in_LB, MMC_in_LB = get_in_dist_values(CIFAR100_test_in_LB_Dn, targets_CIFAR100)
        acc_out_CIFAR10_LB, prob_correct_out_CIFAR10_LB, ent_out_CIFAR10_LB, MMC_out_CIFAR10_LB, auroc_out_CIFAR10_LB = get_out_dist_values(CIFAR100_test_in_LB_Dn, CIFAR100_test_out_CIFAR10_LB_Dn, targets_CIFAR10)
        acc_out_SVHN_LB, prob_correct_out_SVHN_LB, ent_out_SVHN_LB, MMC_out_SVHN_LB, auroc_out_SVHN_LB = get_out_dist_values(CIFAR100_test_in_LB_Dn, CIFAR100_test_out_SVHN_LB_Dn, targets_SVHN)

        Diag_LB_norm_MMC_in.append(MMC_in_LB)
        Diag_LB_norm_MMC_CIFAR10.append(MMC_out_CIFAR10_LB)
        Diag_LB_norm_MMC_SVHN.append(MMC_out_SVHN_LB)
        Diag_LB_norm_AUROC_CIFAR10.append(auroc_out_CIFAR10_LB)
        Diag_LB_norm_AUROC_SVHN.append(auroc_out_SVHN_LB)
        Diag_LB_norm_NLL_in.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_in_LB_Dn)).log_prob(torch.tensor(targets_CIFAR100)).mean().item())
        Diag_LB_norm_NLL_CIFAR10.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_out_CIFAR10_LB_Dn)).log_prob(torch.tensor(targets_CIFAR10)).mean().item())
        Diag_LB_norm_NLL_SVHN.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_out_SVHN_LB_Dn)).log_prob(torch.tensor(targets_SVHN)).mean().item())
        Diag_LB_norm_ECE_in.append(scoring.expected_calibration_error(targets_CIFAR100, CIFAR100_test_in_LB_Dn))
        Diag_LB_norm_ECE_CIFAR10.append(scoring.expected_calibration_error(targets_CIFAR10, CIFAR100_test_out_CIFAR10_LB_Dn))
        Diag_LB_norm_ECE_SVHN.append(scoring.expected_calibration_error(targets_SVHN, CIFAR100_test_out_SVHN_LB_Dn))

        

        #LB KFAC norm
        CIFAR100_test_in_LB_Kn = predict_LB_norm(la_kron, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_CIFAR10_LB_Kn = predict_LB_norm(la_kron, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_SVHN_LB_Kn = predict_LB_norm(la_kron, SVHN_test_loader, timing=False, device=device).cpu().numpy()

        acc_in_LB_KFAC, prob_correct_in_LB_KFAC, ent_in_LB_KFAC, MMC_in_LB_KFAC = get_in_dist_values(CIFAR100_test_in_LB_Kn, targets_CIFAR100)
        acc_out_CIFAR10_LB_KFAC, prob_correct_out_CIFAR10_LB_KFAC, ent_out_CIFAR10_LB_KFAC, MMC_out_CIFAR10_LB_KFAC, auroc_out_CIFAR10_LB_KFAC = get_out_dist_values(CIFAR100_test_in_LB_Kn, CIFAR100_test_out_CIFAR10_LB_Kn, targets_CIFAR10)
        acc_out_SVHN_LB_KFAC, prob_correct_out_SVHN_LB_KFAC, ent_out_SVHN_LB_KFAC, MMC_out_SVHN_LB_KFAC, auroc_out_SVHN_LB_KFAC = get_out_dist_values(CIFAR100_test_in_LB_Kn, CIFAR100_test_out_SVHN_LB_Kn, targets_SVHN)

        KFAC_LB_norm_MMC_in.append(MMC_in_LB_KFAC)
        KFAC_LB_norm_MMC_CIFAR10.append(MMC_out_CIFAR10_LB_KFAC)
        KFAC_LB_norm_MMC_SVHN.append(MMC_out_SVHN_LB_KFAC)
        KFAC_LB_norm_AUROC_CIFAR10.append(auroc_out_CIFAR10_LB_KFAC)
        KFAC_LB_norm_AUROC_SVHN.append(auroc_out_SVHN_LB_KFAC)
        KFAC_LB_norm_NLL_in.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_in_LB_Kn)).log_prob(torch.tensor(targets_CIFAR100)).mean().item())
        KFAC_LB_norm_NLL_CIFAR10.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_out_CIFAR10_LB_Kn)).log_prob(torch.tensor(targets_CIFAR10)).mean().item())
        KFAC_LB_norm_NLL_SVHN.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_out_SVHN_LB_Kn)).log_prob(torch.tensor(targets_SVHN)).mean().item())
        KFAC_LB_norm_ECE_in.append(scoring.expected_calibration_error(targets_CIFAR100, CIFAR100_test_in_LB_Kn))
        KFAC_LB_norm_ECE_CIFAR10.append(scoring.expected_calibration_error(targets_CIFAR10, CIFAR100_test_out_CIFAR10_LB_Kn))
        KFAC_LB_norm_ECE_SVHN.append(scoring.expected_calibration_error(targets_SVHN, CIFAR100_test_out_SVHN_LB_Kn))


        #PROBIT diag
        CIFAR100_test_in_PROBIT_D = predict_probit(la_diag, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_CIFAR10_PROBIT_D = predict_probit(la_diag, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_SVHN_PROBIT_D = predict_probit(la_diag, SVHN_test_loader, timing=False, device=device).cpu().numpy()

        acc_in_PROBIT, prob_correct_in_PROBIT, ent_in_PROBIT, MMC_in_PROBIT = get_in_dist_values(CIFAR100_test_in_PROBIT_D, targets_CIFAR100)
        acc_out_CIFAR10_PROBIT, prob_correct_out_CIFAR10_PROBIT, ent_out_CIFAR10_PROBIT, MMC_out_CIFAR10_PROBIT, auroc_out_CIFAR10_PROBIT = get_out_dist_values(CIFAR100_test_in_PROBIT_D, CIFAR100_test_out_CIFAR10_PROBIT_D, targets_CIFAR10)
        acc_out_SVHN_PROBIT, prob_correct_out_SVHN_PROBIT, ent_out_SVHN_PROBIT, MMC_out_SVHN_PROBIT, auroc_out_SVHN_PROBIT = get_out_dist_values(CIFAR100_test_in_PROBIT_D, CIFAR100_test_out_SVHN_PROBIT_D, targets_SVHN)

        Diag_PROBIT_MMC_in.append(MMC_in_PROBIT)
        Diag_PROBIT_MMC_CIFAR10.append(MMC_out_CIFAR10_PROBIT)
        Diag_PROBIT_MMC_SVHN.append(MMC_out_SVHN_PROBIT)
        Diag_PROBIT_AUROC_CIFAR10.append(auroc_out_CIFAR10_PROBIT)
        Diag_PROBIT_AUROC_SVHN.append(auroc_out_SVHN_PROBIT)
        Diag_PROBIT_NLL_in.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_in_PROBIT_D)).log_prob(torch.tensor(targets_CIFAR100)).mean().item())
        Diag_PROBIT_NLL_CIFAR10.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_out_CIFAR10_PROBIT_D)).log_prob(torch.tensor(targets_CIFAR10)).mean().item())
        Diag_PROBIT_NLL_SVHN.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_out_SVHN_PROBIT_D)).log_prob(torch.tensor(targets_SVHN)).mean().item())
        Diag_PROBIT_ECE_in.append(scoring.expected_calibration_error(targets_CIFAR100, CIFAR100_test_in_PROBIT_D))
        Diag_PROBIT_ECE_CIFAR10.append(scoring.expected_calibration_error(targets_CIFAR10, CIFAR100_test_out_CIFAR10_PROBIT_D))
        Diag_PROBIT_ECE_SVHN.append(scoring.expected_calibration_error(targets_SVHN, CIFAR100_test_out_SVHN_PROBIT_D))

        
        #PROBIT KFAC
        CIFAR100_test_in_PROBIT_K = predict_extended_MacKay(la_kron, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_CIFAR10_PROBIT_K = predict_extended_MacKay(la_kron, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_SVHN_PROBIT_K = predict_extended_MacKay(la_kron, SVHN_test_loader, timing=False, device=device).cpu().numpy()

        acc_in_PROBIT, prob_correct_in_PROBIT, ent_in_PROBIT, MMC_in_PROBIT = get_in_dist_values(CIFAR100_test_in_PROBIT_K, targets_CIFAR100)
        acc_out_CIFAR10_PROBIT, prob_correct_out_CIFAR10_PROBIT, ent_out_CIFAR10_PROBIT, MMC_out_CIFAR10_PROBIT, auroc_out_CIFAR10_PROBIT = get_out_dist_values(CIFAR100_test_in_PROBIT_K, CIFAR100_test_out_CIFAR10_PROBIT_K, targets_CIFAR10)
        acc_out_SVHN_PROBIT, prob_correct_out_SVHN_PROBIT, ent_out_SVHN_PROBIT, MMC_out_SVHN_PROBIT, auroc_out_SVHN_PROBIT = get_out_dist_values(CIFAR100_test_in_PROBIT_K, CIFAR100_test_out_SVHN_PROBIT_K, targets_SVHN)

        KFAC_PROBIT_MMC_in.append(MMC_in_PROBIT)
        KFAC_PROBIT_MMC_CIFAR10.append(MMC_out_CIFAR10_PROBIT)
        KFAC_PROBIT_MMC_SVHN.append(MMC_out_SVHN_PROBIT)
        KFAC_PROBIT_AUROC_CIFAR10.append(auroc_out_CIFAR10_PROBIT)
        KFAC_PROBIT_AUROC_SVHN.append(auroc_out_SVHN_PROBIT)
        KFAC_PROBIT_NLL_in.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_in_PROBIT_K)).log_prob(torch.tensor(targets_CIFAR100)).mean().item())
        KFAC_PROBIT_NLL_CIFAR10.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_out_CIFAR10_PROBIT_K)).log_prob(torch.tensor(targets_CIFAR10)).mean().item())
        KFAC_PROBIT_NLL_SVHN.append(-torch.distributions.Categorical(torch.tensor(CIFAR100_test_out_SVHN_PROBIT_K)).log_prob(torch.tensor(targets_SVHN)).mean().item())
        KFAC_PROBIT_ECE_in.append(scoring.expected_calibration_error(targets_CIFAR100, CIFAR100_test_in_PROBIT_K))
        KFAC_PROBIT_ECE_CIFAR10.append(scoring.expected_calibration_error(targets_CIFAR10, CIFAR100_test_out_CIFAR10_PROBIT_K))
        KFAC_PROBIT_ECE_SVHN.append(scoring.expected_calibration_error(targets_SVHN, CIFAR100_test_out_SVHN_PROBIT_K))

        
        """
        #SODPP diag
        CIFAR100_test_in_SODPP_D = predict_second_order_dpp(la_diag, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_CIFAR10_SODPP_D = predict_second_order_dpp(la_diag, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_SVHN_SODPP_D = predict_second_order_dpp(la_diag, SVHN_test_loader, timing=False, device=device).cpu().numpy()

        acc_in_SODPP, prob_correct_in_SODPP, ent_in_SODPP, MMC_in_SODPP = get_in_dist_values(CIFAR100_test_in_SODPP_D, targets_CIFAR100)
        acc_out_CIFAR10_SODPP, prob_correct_out_CIFAR10_SODPP, ent_out_CIFAR10_SODPP, MMC_out_CIFAR10_SODPP, auroc_out_CIFAR10_SODPP = get_out_dist_values(CIFAR100_test_in_SODPP_D, CIFAR100_test_out_CIFAR10_SODPP_D, targets_CIFAR10)
        acc_out_SVHN_SODPP, prob_correct_out_SVHN_SODPP, ent_out_SVHN_SODPP, MMC_out_SVHN_SODPP, auroc_out_SVHN_SODPP = get_out_dist_values(CIFAR100_test_in_SODPP_D, CIFAR100_test_out_SVHN_SODPP_D, targets_SVHN)

        Diag_SODPP_MMC_in.append(MMC_in_SODPP)
        Diag_SODPP_MMC_CIFAR10.append(MMC_out_CIFAR10_SODPP)
        Diag_SODPP_MMC_SVHN.append(MMC_out_SVHN_SODPP)
        Diag_SODPP_AUROC_CIFAR10.append(auroc_out_CIFAR10_SODPP)
        Diag_SODPP_AUROC_SVHN.append(auroc_out_SVHN_SODPP)
        
        #SODPP KFAC
        CIFAR100_test_in_SODPP_K = predict_second_order_dpp(la_diag, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_CIFAR10_SODPP_K = predict_second_order_dpp(la_diag, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        CIFAR100_test_out_SVHN_SODPP_K = predict_second_order_dpp(la_diag, SVHN_test_loader, timing=False, device=device).cpu().numpy()

        acc_in_SODPP, prob_correct_in_SODPP, ent_in_SODPP, MMC_in_SODPP = get_in_dist_values(CIFAR100_test_in_SODPP_K, targets_CIFAR100)
        acc_out_CIFAR10_SODPP, prob_correct_out_CIFAR10_SODPP, ent_out_CIFAR10_SODPP, MMC_out_CIFAR10_SODPP, auroc_out_CIFAR10_SODPP = get_out_dist_values(CIFAR100_test_in_SODPP_K, CIFAR100_test_out_CIFAR10_SODPP_K, targets_CIFAR10)
        acc_out_SVHN_SODPP, prob_correct_out_SVHN_SODPP, ent_out_SVHN_SODPP, MMC_out_SVHN_SODPP, auroc_out_SVHN_SODPP = get_out_dist_values(CIFAR100_test_in_SODPP_K, CIFAR100_test_out_SVHN_SODPP_K, targets_SVHN)
        
        KFAC_SODPP_MMC_in.append(MMC_in_SODPP)
        KFAC_SODPP_MMC_CIFAR10.append(MMC_out_CIFAR10_SODPP)
        KFAC_SODPP_MMC_SVHN.append(MMC_out_SVHN_SODPP)
        KFAC_SODPP_AUROC_CIFAR10.append(auroc_out_CIFAR10_SODPP)
        KFAC_SODPP_AUROC_SVHN.append(auroc_out_SVHN_SODPP)
        """
        
    #### save results
    results_dict = {
        'MAP_MMC_in':MAP_MMC_in,
        'MAP_MMC_CIFAR10':MAP_MMC_CIFAR10,
        'MAP_MMC_SVHN':MAP_MMC_SVHN,
        'MAP_AUROC_CIFAR10':MAP_AUROC_CIFAR10,
        'MAP_AUROC_SVHN':MAP_AUROC_SVHN,
        'MAP_NLL_in':MAP_NLL_in,
        'MAP_NLL_CIFAR10':MAP_NLL_CIFAR10,
        'MAP_NLL_SVHN':MAP_NLL_SVHN,
        'MAP_ECE_in':MAP_ECE_in,
        'MAP_ECE_CIFAR10':MAP_ECE_CIFAR10,
        'MAP_ECE_SVHN':MAP_ECE_SVHN,
        'Diag_samples_MMC_in':Diag_samples_MMC_in,
        'Diag_samples_MMC_CIFAR10':Diag_samples_MMC_CIFAR10,
        'Diag_samples_MMC_SVHN':Diag_samples_MMC_SVHN,
        'Diag_samples_AUROC_CIFAR10':Diag_samples_AUROC_CIFAR10,
        'Diag_samples_AUROC_SVHN':Diag_samples_AUROC_SVHN,
        'Diag_samples_NLL_in':Diag_samples_NLL_in,
        'Diag_samples_NLL_CIFAR10':Diag_samples_NLL_CIFAR10,
        'Diag_samples_NLL_SVHN':Diag_samples_NLL_SVHN,
        'Diag_samples_ECE_in':Diag_samples_ECE_in,
        'Diag_samples_ECE_CIFAR10':Diag_samples_ECE_CIFAR10,
        'Diag_samples_ECE_SVHN':Diag_samples_ECE_SVHN,
        'KFAC_samples_MMC_in':KFAC_samples_MMC_in,
        'KFAC_samples_MMC_CIFAR10':KFAC_samples_MMC_CIFAR10,
        'KFAC_samples_MMC_SVHN':KFAC_samples_MMC_SVHN,
        'KFAC_samples_AUROC_CIFAR10':KFAC_samples_AUROC_CIFAR10,
        'KFAC_samples_AUROC_SVHN':KFAC_samples_AUROC_SVHN,
        'KFAC_samples_NLL_in':KFAC_samples_NLL_in,
        'KFAC_samples_NLL_CIFAR10':KFAC_samples_NLL_CIFAR10,
        'KFAC_samples_NLL_SVHN':KFAC_samples_NLL_SVHN,
        'KFAC_samples_ECE_in':KFAC_samples_ECE_in,
        'KFAC_samples_ECE_CIFAR10':KFAC_samples_ECE_CIFAR10,
        'KFAC_samples_ECE_SVHN':KFAC_samples_ECE_SVHN,
        'Diag_LB_MMC_in':Diag_LB_MMC_in,
        'Diag_LB_MMC_CIFAR10':Diag_LB_MMC_CIFAR10,
        'Diag_LB_MMC_SVHN':Diag_LB_MMC_SVHN,
        'Diag_LB_AUROC_CIFAR10':Diag_LB_AUROC_CIFAR10,
        'Diag_LB_AUROC_SVHN':Diag_LB_AUROC_SVHN,
        'Diag_LB_NLL_in':Diag_LB_NLL_in,
        'Diag_LB_NLL_CIFAR10':Diag_LB_NLL_CIFAR10,
        'Diag_LB_NLL_SVHN':Diag_LB_NLL_SVHN,
        'Diag_LB_ECE_in':Diag_LB_ECE_in,
        'Diag_LB_ECE_CIFAR10':Diag_LB_ECE_CIFAR10,
        'Diag_LB_ECE_SVHN':Diag_LB_ECE_SVHN,
        'KFAC_LB_MMC_in':KFAC_LB_MMC_in,
        'KFAC_LB_MMC_CIFAR10':KFAC_LB_MMC_CIFAR10,
        'KFAC_LB_MMC_SVHN':KFAC_LB_MMC_SVHN,
        'KFAC_LB_AUROC_CIFAR10':KFAC_LB_AUROC_CIFAR10,
        'KFAC_LB_AUROC_SVHN':KFAC_LB_AUROC_SVHN,
        'KFAC_LB_NLL_in':KFAC_LB_NLL_in,
        'KFAC_LB_NLL_CIFAR10':KFAC_LB_NLL_CIFAR10,
        'KFAC_LB_NLL_SVHN':KFAC_LB_NLL_SVHN,
        'KFAC_LB_ECE_in':KFAC_LB_ECE_in,
        'KFAC_LB_ECE_CIFAR10':KFAC_LB_ECE_CIFAR10,
        'KFAC_LB_ECE_SVHN':KFAC_LB_ECE_SVHN,
        'Diag_LB_norm_MMC_in':Diag_LB_norm_MMC_in,
        'Diag_LB_norm_MMC_CIFAR10':Diag_LB_norm_MMC_CIFAR10,
        'Diag_LB_norm_MMC_SVHN':Diag_LB_norm_MMC_SVHN,
        'Diag_LB_norm_AUROC_CIFAR10':Diag_LB_norm_AUROC_CIFAR10,
        'Diag_LB_norm_AUROC_SVHN':Diag_LB_norm_AUROC_SVHN,
        'Diag_LB_norm_NLL_in':Diag_LB_norm_NLL_in,
        'Diag_LB_norm_NLL_CIFAR10':Diag_LB_norm_NLL_CIFAR10,
        'Diag_LB_norm_NLL_SVHN':Diag_LB_norm_NLL_SVHN,
        'Diag_LB_norm_ECE_in':Diag_LB_norm_ECE_in,
        'Diag_LB_norm_ECE_CIFAR10':Diag_LB_norm_ECE_CIFAR10,
        'Diag_LB_norm_ECE_SVHN':Diag_LB_norm_ECE_SVHN,
        'KFAC_LB_norm_MMC_in':KFAC_LB_norm_MMC_in,
        'KFAC_LB_norm_MMC_CIFAR10':KFAC_LB_norm_MMC_CIFAR10,
        'KFAC_LB_norm_MMC_SVHN':KFAC_LB_norm_MMC_SVHN,
        'KFAC_LB_norm_AUROC_CIFAR10':KFAC_LB_norm_AUROC_CIFAR10,
        'KFAC_LB_norm_AUROC_SVHN':KFAC_LB_norm_AUROC_SVHN,
        'KFAC_LB_norm_NLL_in':KFAC_LB_norm_NLL_in,
        'KFAC_LB_norm_NLL_CIFAR10':KFAC_LB_norm_NLL_CIFAR10,
        'KFAC_LB_norm_NLL_SVHN':KFAC_LB_norm_NLL_SVHN,
        'KFAC_LB_norm_ECE_in':KFAC_LB_norm_ECE_in,
        'KFAC_LB_norm_ECE_CIFAR10':KFAC_LB_norm_ECE_CIFAR10,
        'KFAC_LB_norm_ECE_SVHN':KFAC_LB_norm_ECE_SVHN,
        'Diag_PROBIT_MMC_in':Diag_PROBIT_MMC_in,
        'Diag_PROBIT_MMC_CIFAR10':Diag_PROBIT_MMC_CIFAR10,
        'Diag_PROBIT_MMC_SVHN':Diag_PROBIT_MMC_SVHN,
        'Diag_PROBIT_AUROC_CIFAR10':Diag_PROBIT_AUROC_CIFAR10,
        'Diag_PROBIT_AUROC_SVHN':Diag_PROBIT_AUROC_SVHN,
        'Diag_PROBIT_NLL_in':Diag_PROBIT_NLL_in,
        'Diag_PROBIT_NLL_CIFAR10':Diag_PROBIT_NLL_CIFAR10,
        'Diag_PROBIT_NLL_SVHN':Diag_PROBIT_NLL_SVHN,
        'Diag_PROBIT_ECE_in':Diag_PROBIT_ECE_in,
        'Diag_PROBIT_ECE_CIFAR10':Diag_PROBIT_ECE_CIFAR10,
        'Diag_PROBIT_ECE_SVHN':Diag_PROBIT_ECE_SVHN,
        'KFAC_PROBIT_MMC_in':KFAC_PROBIT_MMC_in,
        'KFAC_PROBIT_MMC_CIFAR10':KFAC_PROBIT_MMC_CIFAR10,
        'KFAC_PROBIT_MMC_SVHN':KFAC_PROBIT_MMC_SVHN,
        'KFAC_PROBIT_AUROC_CIFAR10':KFAC_PROBIT_AUROC_CIFAR10,
        'KFAC_PROBIT_AUROC_SVHN':KFAC_PROBIT_AUROC_SVHN,
        'KFAC_PROBIT_NLL_in':KFAC_PROBIT_NLL_in,
        'KFAC_PROBIT_NLL_CIFAR10':KFAC_PROBIT_NLL_CIFAR10,
        'KFAC_PROBIT_NLL_SVHN':KFAC_PROBIT_NLL_SVHN,
        'KFAC_PROBIT_ECE_in':KFAC_PROBIT_ECE_in,
        'KFAC_PROBIT_ECE_CIFAR10':KFAC_PROBIT_ECE_CIFAR10,
        'KFAC_PROBIT_ECE_SVHN':KFAC_PROBIT_ECE_SVHN
        #'Diag_SODPP_MMC_in':Diag_SODPP_MMC_in,
        #'Diag_SODPP_MMC_CIFAR10':Diag_SODPP_MMC_CIFAR10,
        #'Diag_SODPP_MMC_SVHN':Diag_SODPP_MMC_SVHN,
        #'Diag_SODPP_AUROC_CIFAR10':Diag_SODPP_AUROC_CIFAR10,
        #'Diag_SODPP_AUROC_SVHN':Diag_SODPP_AUROC_SVHN,
        #'KFAC_SODPP_MMC_in':KFAC_SODPP_MMC_in,
        #'KFAC_SODPP_MMC_CIFAR10':KFAC_SODPP_MMC_CIFAR10,
        #'KFAC_SODPP_MMC_SVHN':KFAC_SODPP_MMC_SVHN,
        #'KFAC_SODPP_AUROC_CIFAR10':KFAC_SODPP_AUROC_CIFAR10,
        #'KFAC_SODPP_AUROC_SVHN':KFAC_SODPP_AUROC_SVHN
    }
    results_df = pd.DataFrame(results_dict)
    RESULTS_PATH = os.getcwd() + "/Experiment_results/CIFAR100_results.csv"
    print("saving at: ", RESULTS_PATH)
    results_df.to_csv(RESULTS_PATH)


#### RUN
if __name__ == '__main__':
    main()