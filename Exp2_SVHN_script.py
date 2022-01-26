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

from utils.load_not_MNIST import notMNIST
import argparse
import torchvision.models as models

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
    transform_base = [transforms.ToTensor()]

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        ] + transform_base)

    transform_test = transforms.Compose(transform_base)
    transform_train = transforms.RandomChoice([transform_train, transform_test])

    train_data_CIFAR10 = torchvision.datasets.SVHN('~/data/SVHN', split='train',
                             download=True, transform=transform_train)

    test_data_SVHN = torchvision.datasets.SVHN('~/data/SVHN', split='test',
                                 download=True, transform=transform_test)

    SVHN_train_loader = torch.utils.data.DataLoader(test_data_SVHN, batch_size=BATCH_SIZE)
    SVHN_test_loader = torch.utils.data.DataLoader(test_data_SVHN, batch_size=BATCH_SIZE)


    #~/data/cifar10
    CIFAR10_testset = torchvision.datasets.CIFAR10(root='~/data/cifar10', train=False, download=True, transform=transform_test)
    CIFAR10_test_loader = torch.utils.data.DataLoader(CIFAR10_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    CIFAR100_test = torchvision.datasets.CIFAR100(root='~/data/cifar100', train=False,
                                       download=True, transform=transform_test)
    CIFAR100_test_loader = torch.utils.data.DataLoader(CIFAR100_test, batch_size=BATCH_SIZE,
                                         shuffle=False)


    ### load model
    SVHN_PATH = os.getcwd() + "/pretrained_weights/SVHN_resnet18_best_s1.pth"
    SVHN_model = models.resnet18(pretrained=False, num_classes=10).to(device)
    print("loading model from: {}".format(SVHN_PATH))
    SVHN_model.load_state_dict(torch.load(SVHN_PATH))
    SVHN_model.eval()
    
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
    MAP_MMC_CIFAR100 = []
    MAP_MMC_CIFAR10 = []
    MAP_AUROC_CIFAR100 = []
    MAP_AUROC_CIFAR10 = []
    
    # Diag samples
    Diag_samples_MMC_in = []
    Diag_samples_MMC_CIFAR100 = []
    Diag_samples_MMC_CIFAR10 = []
    Diag_samples_AUROC_CIFAR100 = []
    Diag_samples_AUROC_CIFAR10 = []

    # KFAC samples
    KFAC_samples_MMC_in = []
    KFAC_samples_MMC_CIFAR100 = []
    KFAC_samples_MMC_CIFAR10 = []
    KFAC_samples_AUROC_CIFAR100 = []
    KFAC_samples_AUROC_CIFAR10 = []

    # Diag LB
    Diag_LB_MMC_in = []
    Diag_LB_MMC_CIFAR100 = []
    Diag_LB_MMC_CIFAR10 = []
    Diag_LB_AUROC_CIFAR100 = []
    Diag_LB_AUROC_CIFAR10 = []
    
    # KFAC LB
    KFAC_LB_MMC_in = []
    KFAC_LB_MMC_CIFAR100 = []
    KFAC_LB_MMC_CIFAR10 = []
    KFAC_LB_AUROC_CIFAR100 = []
    KFAC_LB_AUROC_CIFAR10 = []
    
    # Diag LB norm
    Diag_LB_norm_MMC_in = []
    Diag_LB_norm_MMC_CIFAR100 = []
    Diag_LB_norm_MMC_CIFAR10 = []
    Diag_LB_norm_AUROC_CIFAR100 = []
    Diag_LB_norm_AUROC_CIFAR10 = []
    
    # KFAC LB norm
    KFAC_LB_norm_MMC_in = []
    KFAC_LB_norm_MMC_CIFAR100 = []
    KFAC_LB_norm_MMC_CIFAR10 = []
    KFAC_LB_norm_AUROC_CIFAR100 = []
    KFAC_LB_norm_AUROC_CIFAR10 = []

    # Diag EMK
    Diag_EMK_MMC_in = []
    Diag_EMK_MMC_CIFAR100 = []
    Diag_EMK_MMC_CIFAR10 = []
    Diag_EMK_AUROC_CIFAR100 = []
    Diag_EMK_AUROC_CIFAR10 = []
    
    # KFAC EMK
    KFAC_EMK_MMC_in = []
    KFAC_EMK_MMC_CIFAR100 = []
    KFAC_EMK_MMC_CIFAR10 = []
    KFAC_EMK_AUROC_CIFAR100 = []
    KFAC_EMK_AUROC_CIFAR10 = []
    
    # Diag SODPP
    Diag_SODPP_MMC_in = []
    Diag_SODPP_MMC_CIFAR100 = []
    Diag_SODPP_MMC_CIFAR10 = []
    Diag_SODPP_AUROC_CIFAR100 = []
    Diag_SODPP_AUROC_CIFAR10 = []
    
    # KFAC SODPP
    KFAC_SODPP_MMC_in = []
    KFAC_SODPP_MMC_CIFAR100 = []
    KFAC_SODPP_MMC_CIFAR10 = []
    KFAC_SODPP_AUROC_CIFAR100 = []
    KFAC_SODPP_AUROC_CIFAR10 = []
    
    # run experiments
    for s in range(args.num_seeds):

        print("seed: ", s)
    
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        la_diag = Laplace(SVHN_model, 'classification', 
                     subset_of_weights='last_layer', 
                     hessian_structure='diag',
                     prior_precision=1e+1) # 5e-4 # Choose prior precision such that the MMC for inclass is slightly lower than for MAP
        la_diag.fit(SVHN_train_loader)
        
        la_kron = Laplace(SVHN_model, 'classification', 
                     subset_of_weights='last_layer', 
                     hessian_structure='kron',
                     prior_precision=5e-1) # 5e-4 # Choose prior precision according to weight decay
        la_kron.fit(SVHN_train_loader)

        #MAP estimates
        SVHN_test_in_MAP = predict_MAP(SVHN_model, SVHN_test_loader, device=device).cpu().numpy()
        SVHN_test_out_CIFAR100_MAP = predict_MAP(SVHN_model, CIFAR100_test_loader, device=device).cpu().numpy()
        SVHN_test_out_CIFAR10_MAP = predict_MAP(SVHN_model, CIFAR10_test_loader, device=device).cpu().numpy()
        
        acc_in_MAP, prob_correct_in_MAP, ent_in_MAP, MMC_in_MAP = get_in_dist_values(SVHN_test_in_MAP, targets_SVHN)
        acc_out_CIFAR100_MAP, prob_correct_out_CIFAR100_MAP, ent_out_CIFAR100, MMC_out_CIFAR100_MAP, auroc_out_CIFAR100_MAP = get_out_dist_values(SVHN_test_in_MAP, SVHN_test_out_CIFAR100_MAP, targets_CIFAR100)
        acc_out_CIFAR10_MAP, prob_correct_out_CIFAR10_MAP, ent_out_CIFAR10, MMC_out_CIFAR10_MAP, auroc_out_CIFAR10_MAP = get_out_dist_values(SVHN_test_in_MAP, SVHN_test_out_CIFAR10_MAP, targets_CIFAR10)
        
        MAP_MMC_in.append(MMC_in_MAP)
        MAP_MMC_CIFAR100.append(MMC_out_CIFAR100_MAP)
        MAP_MMC_CIFAR10.append(MMC_out_CIFAR10_MAP)
        MAP_AUROC_CIFAR100.append(auroc_out_CIFAR100_MAP)
        MAP_AUROC_CIFAR10.append(auroc_out_CIFAR10_MAP)

        #Diag samples
        SVHN_test_in_D = predict_samples(la_diag, SVHN_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR100_D = predict_samples(la_diag, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR10_D = predict_samples(la_diag, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_D, prob_correct_in_D, ent_in_D, MMC_in_D = get_in_dist_values(SVHN_test_in_D, targets_SVHN)
        acc_out_CIFAR100_D, prob_correct_out_CIFAR100_D, ent_out_CIFAR100_D, MMC_out_CIFAR100_D, auroc_out_CIFAR100_D = get_out_dist_values(SVHN_test_in_D, SVHN_test_out_CIFAR100_D, targets_CIFAR100)
        acc_out_CIFAR10_D, prob_correct_out_CIFAR10_D, ent_out_CIFAR10_D, MMC_out_CIFAR10_D, auroc_out_CIFAR10_D = get_out_dist_values(SVHN_test_in_D, SVHN_test_out_CIFAR10_D, targets_CIFAR10)

        Diag_samples_MMC_in.append(MMC_in_D)
        Diag_samples_MMC_CIFAR100.append(MMC_out_CIFAR100_D)
        Diag_samples_MMC_CIFAR10.append(MMC_out_CIFAR10_D)
        Diag_samples_AUROC_CIFAR100.append(auroc_out_CIFAR100_D)
        Diag_samples_AUROC_CIFAR10.append(auroc_out_CIFAR10_D)
        
        #KFAC samples
        SVHN_test_in_K = predict_samples(la_kron, SVHN_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR100_K = predict_samples(la_kron, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR10_K = predict_samples(la_kron, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()
        
        acc_in_K, prob_correct_in_K, ent_in_K, MMC_in_K = get_in_dist_values(SVHN_test_in_K, targets_SVHN)
        acc_out_CIFAR100_K, prob_correct_out_CIFAR100_K, ent_out_CIFAR100_K, MMC_out_CIFAR100_K, auroc_out_CIFAR100_K = get_out_dist_values(SVHN_test_in_K, SVHN_test_out_CIFAR100_K, targets_CIFAR100)
        acc_out_CIFAR10_K, prob_correct_out_CIFAR10_K, ent_out_CIFAR10_K, MMC_out_CIFAR10_K, auroc_out_CIFAR10_K = get_out_dist_values(SVHN_test_in_K, SVHN_test_out_CIFAR10_K, targets_CIFAR10)

        # KFAC samples
        KFAC_samples_MMC_in.append(MMC_in_K)
        KFAC_samples_MMC_CIFAR100.append(MMC_out_CIFAR100_K)
        KFAC_samples_MMC_CIFAR10.append(MMC_out_CIFAR10_K)
        KFAC_samples_AUROC_CIFAR100.append(auroc_out_CIFAR100_K)
        KFAC_samples_AUROC_CIFAR10.append(auroc_out_CIFAR10_K)
        
        #LB diag
        SVHN_test_in_LB_D = predict_LB(la_diag, SVHN_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR100_LB_D = predict_LB(la_diag, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR10_LB_D = predict_LB(la_diag, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()

        acc_in_LB, prob_correct_in_LB, ent_in_LB, MMC_in_LB = get_in_dist_values(SVHN_test_in_LB_D, targets_SVHN)
        acc_out_CIFAR100_LB, prob_correct_out_CIFAR100_LB, ent_out_CIFAR100_LB, MMC_out_CIFAR100_LB, auroc_out_CIFAR100_LB = get_out_dist_values(SVHN_test_in_LB_D, SVHN_test_out_CIFAR100_LB_D, targets_CIFAR100)
        acc_out_CIFAR10_LB, prob_correct_out_CIFAR10_LB, ent_out_CIFAR10_LB, MMC_out_CIFAR10_LB, auroc_out_CIFAR10_LB = get_out_dist_values(SVHN_test_in_LB_D, SVHN_test_out_CIFAR10_LB_D, targets_CIFAR10)

        Diag_LB_MMC_in.append(MMC_in_LB)
        Diag_LB_MMC_CIFAR100.append(MMC_out_CIFAR100_LB)
        Diag_LB_MMC_CIFAR10.append(MMC_out_CIFAR10_LB)
        Diag_LB_AUROC_CIFAR100.append(auroc_out_CIFAR100_LB)
        Diag_LB_AUROC_CIFAR10.append(auroc_out_CIFAR10_LB)
        
        #LB KFAC
        SVHN_test_in_LB_KFAC = predict_LB(la_kron, SVHN_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR100_LB_KFAC = predict_LB(la_kron, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR10_LB_KFAC = predict_LB(la_kron, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()

        acc_in_LB_KFAC, prob_correct_in_LB_KFAC, ent_in_LB_KFAC, MMC_in_LB_KFAC = get_in_dist_values(SVHN_test_in_LB_KFAC, targets_SVHN)
        acc_out_CIFAR100_LB_KFAC, prob_correct_out_CIFAR100_LB_KFAC, ent_out_CIFAR100_LB_KFAC, MMC_out_CIFAR100_LB_KFAC, auroc_out_CIFAR100_LB_KFAC = get_out_dist_values(SVHN_test_in_LB_KFAC, SVHN_test_out_CIFAR100_LB_KFAC, targets_CIFAR100)
        acc_out_CIFAR10_LB_KFAC, prob_correct_out_CIFAR10_LB_KFAC, ent_out_CIFAR10_LB_KFAC, MMC_out_CIFAR10_LB_KFAC, auroc_out_CIFAR10_LB_KFAC = get_out_dist_values(SVHN_test_in_LB_KFAC, SVHN_test_out_CIFAR10_LB_KFAC, targets_CIFAR10)

        KFAC_LB_MMC_in.append(MMC_in_LB_KFAC)
        KFAC_LB_MMC_CIFAR100.append(MMC_out_CIFAR100_LB_KFAC)
        KFAC_LB_MMC_CIFAR10.append(MMC_out_CIFAR10_LB_KFAC)
        KFAC_LB_AUROC_CIFAR100.append(auroc_out_CIFAR100_LB_KFAC)
        KFAC_LB_AUROC_CIFAR10.append(auroc_out_CIFAR10_LB_KFAC)

        #LB diag norm
        SVHN_test_in_LB_Dn = predict_LB_norm(la_diag, SVHN_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR100_LB_Dn = predict_LB_norm(la_diag, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR10_LB_Dn = predict_LB_norm(la_diag, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()

        acc_in_LB, prob_correct_in_LB, ent_in_LB, MMC_in_LB = get_in_dist_values(SVHN_test_in_LB_Dn, targets_SVHN)
        acc_out_CIFAR100_LB, prob_correct_out_CIFAR100_LB, ent_out_CIFAR100_LB, MMC_out_CIFAR100_LB, auroc_out_CIFAR100_LB = get_out_dist_values(SVHN_test_in_LB_Dn, SVHN_test_out_CIFAR100_LB_Dn, targets_CIFAR100)
        acc_out_CIFAR10_LB, prob_correct_out_CIFAR10_LB, ent_out_CIFAR10_LB, MMC_out_CIFAR10_LB, auroc_out_CIFAR10_LB = get_out_dist_values(SVHN_test_in_LB_Dn, SVHN_test_out_CIFAR10_LB_Dn, targets_CIFAR10)

        Diag_LB_norm_MMC_in.append(MMC_in_LB)
        Diag_LB_norm_MMC_CIFAR100.append(MMC_out_CIFAR100_LB)
        Diag_LB_norm_MMC_CIFAR10.append(MMC_out_CIFAR10_LB)
        Diag_LB_norm_AUROC_CIFAR100.append(auroc_out_CIFAR100_LB)
        Diag_LB_norm_AUROC_CIFAR10.append(auroc_out_CIFAR10_LB)
        
        #LB KFAC norm
        SVHN_test_in_LB_KFACn = predict_LB_norm(la_kron, SVHN_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR100_LB_KFACn = predict_LB_norm(la_kron, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR10_LB_KFACn = predict_LB_norm(la_kron, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()

        acc_in_LB_KFAC, prob_correct_in_LB_KFAC, ent_in_LB_KFAC, MMC_in_LB_KFAC = get_in_dist_values(SVHN_test_in_LB_KFACn, targets_SVHN)
        acc_out_CIFAR100_LB_KFAC, prob_correct_out_CIFAR100_LB_KFAC, ent_out_CIFAR100_LB_KFAC, MMC_out_CIFAR100_LB_KFAC, auroc_out_CIFAR100_LB_KFAC = get_out_dist_values(SVHN_test_in_LB_KFACn, SVHN_test_out_CIFAR100_LB_KFACn, targets_CIFAR100)
        acc_out_CIFAR10_LB_KFAC, prob_correct_out_CIFAR10_LB_KFAC, ent_out_CIFAR10_LB_KFAC, MMC_out_CIFAR10_LB_KFAC, auroc_out_CIFAR10_LB_KFAC = get_out_dist_values(SVHN_test_in_LB_KFACn, SVHN_test_out_CIFAR10_LB_KFACn, targets_CIFAR10)

        KFAC_LB_norm_MMC_in.append(MMC_in_LB_KFAC)
        KFAC_LB_norm_MMC_CIFAR100.append(MMC_out_CIFAR100_LB_KFAC)
        KFAC_LB_norm_MMC_CIFAR10.append(MMC_out_CIFAR10_LB_KFAC)
        KFAC_LB_norm_AUROC_CIFAR100.append(auroc_out_CIFAR100_LB_KFAC)
        KFAC_LB_norm_AUROC_CIFAR10.append(auroc_out_CIFAR10_LB_KFAC)

        #Extended MacKay diag
        SVHN_test_in_EMK_D = predict_extended_MacKay(la_diag, SVHN_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR100_EMK_D = predict_extended_MacKay(la_diag, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR10_EMK_D = predict_extended_MacKay(la_diag, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()

        acc_in_EMK, prob_correct_in_EMK, ent_in_EMK, MMC_in_EMK = get_in_dist_values(SVHN_test_in_EMK_D, targets_SVHN)
        acc_out_CIFAR100_EMK, prob_correct_out_CIFAR100_EMK, ent_out_CIFAR100_EMK, MMC_out_CIFAR100_EMK, auroc_out_CIFAR100_EMK = get_out_dist_values(SVHN_test_in_EMK_D, SVHN_test_out_CIFAR100_EMK_D, targets_CIFAR100)
        acc_out_CIFAR10_EMK, prob_correct_out_CIFAR10_EMK, ent_out_CIFAR10_EMK, MMC_out_CIFAR10_EMK, auroc_out_CIFAR10_EMK = get_out_dist_values(SVHN_test_in_EMK_D, SVHN_test_out_CIFAR10_EMK_D, targets_CIFAR10)

        Diag_EMK_MMC_in.append(MMC_in_EMK)
        Diag_EMK_MMC_CIFAR100.append(MMC_out_CIFAR100_EMK)
        Diag_EMK_MMC_CIFAR10.append(MMC_out_CIFAR10_EMK)
        Diag_EMK_AUROC_CIFAR100.append(auroc_out_CIFAR100_EMK)
        Diag_EMK_AUROC_CIFAR10.append(auroc_out_CIFAR10_EMK)
        
        #Extended MacKay KFAC
        SVHN_test_in_EMK_K = predict_extended_MacKay(la_kron, SVHN_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR100_EMK_K = predict_extended_MacKay(la_kron, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR10_EMK_K = predict_extended_MacKay(la_kron, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()

        acc_in_EMK, prob_correct_in_EMK, ent_in_EMK, MMC_in_EMK = get_in_dist_values(SVHN_test_in_EMK_K, targets_SVHN)
        acc_out_CIFAR100_EMK, prob_correct_out_CIFAR100_EMK, ent_out_CIFAR100_EMK, MMC_out_CIFAR100_EMK, auroc_out_CIFAR100_EMK = get_out_dist_values(SVHN_test_in_EMK_K, SVHN_test_out_CIFAR100_EMK_K, targets_CIFAR100)
        acc_out_CIFAR10_EMK, prob_correct_out_CIFAR10_EMK, ent_out_CIFAR10_EMK, MMC_out_CIFAR10_EMK, auroc_out_CIFAR10_EMK = get_out_dist_values(SVHN_test_in_EMK_K, SVHN_test_out_CIFAR10_EMK_K, targets_CIFAR10)

        KFAC_EMK_MMC_in.append(MMC_in_EMK)
        KFAC_EMK_MMC_CIFAR100.append(MMC_out_CIFAR100_EMK)
        KFAC_EMK_MMC_CIFAR10.append(MMC_out_CIFAR10_EMK)
        KFAC_EMK_AUROC_CIFAR100.append(auroc_out_CIFAR100_EMK)
        KFAC_EMK_AUROC_CIFAR10.append(auroc_out_CIFAR10_EMK)
        
        #SODPP diag
        SVHN_test_in_SODPP_D = predict_second_order_dpp(la_diag, SVHN_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR100_SODPP_D = predict_second_order_dpp(la_diag, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR10_SODPP_D = predict_second_order_dpp(la_diag, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()

        acc_in_SODPP, prob_correct_in_SODPP, ent_in_SODPP, MMC_in_SODPP = get_in_dist_values(SVHN_test_in_SODPP_D, targets_SVHN)
        acc_out_CIFAR100_SODPP, prob_correct_out_CIFAR100_SODPP, ent_out_CIFAR100_SODPP, MMC_out_CIFAR100_SODPP, auroc_out_CIFAR100_SODPP = get_out_dist_values(SVHN_test_in_SODPP_D, SVHN_test_out_CIFAR100_SODPP_D, targets_CIFAR100)
        acc_out_CIFAR10_SODPP, prob_correct_out_CIFAR10_SODPP, ent_out_CIFAR10_SODPP, MMC_out_CIFAR10_SODPP, auroc_out_CIFAR10_SODPP = get_out_dist_values(SVHN_test_in_SODPP_D, SVHN_test_out_CIFAR10_SODPP_D, targets_CIFAR10)

        Diag_SODPP_MMC_in.append(MMC_in_SODPP)
        Diag_SODPP_MMC_CIFAR100.append(MMC_out_CIFAR100_SODPP)
        Diag_SODPP_MMC_CIFAR10.append(MMC_out_CIFAR10_SODPP)
        Diag_SODPP_AUROC_CIFAR100.append(auroc_out_CIFAR100_SODPP)
        Diag_SODPP_AUROC_CIFAR10.append(auroc_out_CIFAR10_SODPP)
        
        #SODPP KFAC
        SVHN_test_in_SODPP_K = predict_second_order_dpp(la_kron, SVHN_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR100_SODPP_K = predict_second_order_dpp(la_kron, CIFAR100_test_loader, timing=False, device=device).cpu().numpy()
        SVHN_test_out_CIFAR10_SODPP_K = predict_second_order_dpp(la_kron, CIFAR10_test_loader, timing=False, device=device).cpu().numpy()

        acc_in_SODPP, prob_correct_in_SODPP, ent_in_SODPP, MMC_in_SODPP = get_in_dist_values(SVHN_test_in_SODPP_K, targets_SVHN)
        acc_out_CIFAR100_SODPP, prob_correct_out_CIFAR100_SODPP, ent_out_CIFAR100_SODPP, MMC_out_CIFAR100_SODPP, auroc_out_CIFAR100_SODPP = get_out_dist_values(SVHN_test_in_SODPP_K, SVHN_test_out_CIFAR100_SODPP_K, targets_CIFAR100)
        acc_out_CIFAR10_SODPP, prob_correct_out_CIFAR10_SODPP, ent_out_CIFAR10_SODPP, MMC_out_CIFAR10_SODPP, auroc_out_CIFAR10_SODPP = get_out_dist_values(SVHN_test_in_SODPP_K, SVHN_test_out_CIFAR10_SODPP_K, targets_CIFAR10)

        KFAC_SODPP_MMC_in.append(MMC_in_SODPP)
        KFAC_SODPP_MMC_CIFAR100.append(MMC_out_CIFAR100_SODPP)
        KFAC_SODPP_MMC_CIFAR10.append(MMC_out_CIFAR10_SODPP)
        KFAC_SODPP_AUROC_CIFAR100.append(auroc_out_CIFAR100_SODPP)
        KFAC_SODPP_AUROC_CIFAR10.append(auroc_out_CIFAR10_SODPP)
        
    #### save results
    results_dict = {
        'MAP_MMC_in':MAP_MMC_in,
        'MAP_MMC_CIFAR100':MAP_MMC_CIFAR100,
        'MAP_MMC_CIFAR10':MAP_MMC_CIFAR10,
        'MAP_AUROC_CIFAR100':MAP_AUROC_CIFAR100,
        'MAP_AUROC_CIFAR10':MAP_AUROC_CIFAR10,
        'Diag_samples_MMC_in':Diag_samples_MMC_in,
        'Diag_samples_MMC_CIFAR100':Diag_samples_MMC_CIFAR100,
        'Diag_samples_MMC_CIFAR10':Diag_samples_MMC_CIFAR10,
        'Diag_samples_AUROC_CIFAR100':Diag_samples_AUROC_CIFAR100,
        'Diag_samples_AUROC_CIFAR10':Diag_samples_AUROC_CIFAR10,
        'KFAC_samples_MMC_in':KFAC_samples_MMC_in,
        'KFAC_samples_MMC_CIFAR100':KFAC_samples_MMC_CIFAR100,
        'KFAC_samples_MMC_CIFAR10':KFAC_samples_MMC_CIFAR10,
        'KFAC_samples_AUROC_CIFAR100':KFAC_samples_AUROC_CIFAR100,
        'KFAC_samples_AUROC_CIFAR10':KFAC_samples_AUROC_CIFAR10,
        'Diag_LB_MMC_in':Diag_LB_MMC_in,
        'Diag_LB_MMC_CIFAR100':Diag_LB_MMC_CIFAR100,
        'Diag_LB_MMC_CIFAR10':Diag_LB_MMC_CIFAR10,
        'Diag_LB_AUROC_CIFAR100':Diag_LB_AUROC_CIFAR100,
        'Diag_LB_AUROC_CIFAR10':Diag_LB_AUROC_CIFAR10,
        'KFAC_LB_MMC_in':KFAC_LB_MMC_in,
        'KFAC_LB_MMC_CIFAR100':KFAC_LB_MMC_CIFAR100,
        'KFAC_LB_MMC_CIFAR10':KFAC_LB_MMC_CIFAR10,
        'KFAC_LB_AUROC_CIFAR100':KFAC_LB_AUROC_CIFAR100,
        'KFAC_LB_AUROC_CIFAR10':KFAC_LB_AUROC_CIFAR10,
        'Diag_LB_norm_MMC_in':Diag_LB_norm_MMC_in,
        'Diag_LB_norm_MMC_CIFAR100':Diag_LB_norm_MMC_CIFAR100,
        'Diag_LB_norm_MMC_CIFAR10':Diag_LB_norm_MMC_CIFAR10,
        'Diag_LB_norm_AUROC_CIFAR100':Diag_LB_norm_AUROC_CIFAR100,
        'Diag_LB_norm_AUROC_CIFAR10':Diag_LB_norm_AUROC_CIFAR10,
        'KFAC_LB_norm_MMC_in':KFAC_LB_norm_MMC_in,
        'KFAC_LB_norm_MMC_CIFAR100':KFAC_LB_norm_MMC_CIFAR100,
        'KFAC_LB_norm_MMC_CIFAR10':KFAC_LB_norm_MMC_CIFAR10,
        'KFAC_LB_norm_AUROC_CIFAR100':KFAC_LB_norm_AUROC_CIFAR100,
        'KFAC_LB_norm_AUROC_CIFAR10':KFAC_LB_norm_AUROC_CIFAR10,
        'Diag_EMK_MMC_in':Diag_EMK_MMC_in,
        'Diag_EMK_MMC_CIFAR100':Diag_EMK_MMC_CIFAR100,
        'Diag_EMK_MMC_CIFAR10':Diag_EMK_MMC_CIFAR10,
        'Diag_EMK_AUROC_CIFAR100':Diag_EMK_AUROC_CIFAR100,
        'Diag_EMK_AUROC_CIFAR10':Diag_EMK_AUROC_CIFAR10,
        'KFAC_EMK_MMC_in':KFAC_EMK_MMC_in,
        'KFAC_EMK_MMC_CIFAR100':KFAC_EMK_MMC_CIFAR100,
        'KFAC_EMK_MMC_CIFAR10':KFAC_EMK_MMC_CIFAR10,
        'KFAC_EMK_AUROC_CIFAR100':KFAC_EMK_AUROC_CIFAR100,
        'KFAC_EMK_AUROC_CIFAR10':KFAC_EMK_AUROC_CIFAR10,
        'Diag_SODPP_MMC_in':Diag_SODPP_MMC_in,
        'Diag_SODPP_MMC_CIFAR100':Diag_SODPP_MMC_CIFAR100,
        'Diag_SODPP_MMC_CIFAR10':Diag_SODPP_MMC_CIFAR10,
        'Diag_SODPP_AUROC_CIFAR100':Diag_SODPP_AUROC_CIFAR100,
        'Diag_SODPP_AUROC_CIFAR10':Diag_SODPP_AUROC_CIFAR10,
        'KFAC_SODPP_MMC_in':KFAC_SODPP_MMC_in,
        'KFAC_SODPP_MMC_CIFAR100':KFAC_SODPP_MMC_CIFAR100,
        'KFAC_SODPP_MMC_CIFAR10':KFAC_SODPP_MMC_CIFAR10,
        'KFAC_SODPP_AUROC_CIFAR100':KFAC_SODPP_AUROC_CIFAR100,
        'KFAC_SODPP_AUROC_CIFAR10':KFAC_SODPP_AUROC_CIFAR10
    }
    results_df = pd.DataFrame(results_dict)
    RESULTS_PATH = os.getcwd() + "/Experiment_results/SVHN_results.csv"
    print("saving at: ", RESULTS_PATH)
    results_df.to_csv(RESULTS_PATH)


#### RUN
if __name__ == '__main__':
    main()