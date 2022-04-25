import torch
import torchvision
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from backpack import backpack, extend
from backpack.extensions import KFAC, DiagHessian, DiagGGNMC
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import time

"""####### general utils #########"""

def get_accuracy(output, targets):
    """Helper function to print the accuracy"""
    predictions = output.argmax(dim=1, keepdim=True).view_as(targets)
    return predictions.eq(targets).float().mean().item()

def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#non-Bayesian estimate
@torch.no_grad()
def predict_MAP(model, test_loader, device='cuda'):
    py = []

    for batch_idx, (x, y) in enumerate(test_loader):

        x, y = x.to(device), y.to(device)

        py_ = torch.softmax(model(x), 1)

        py.append(py_)
    return torch.cat(py, dim=0)


#sampling of last-layer
@torch.no_grad()
def predict_samples(laplace_model, data_loader, num_samples=100, timing=False, device='cuda'):
    
    py = []
    t0 = time.process_time()
    for x, _ in data_loader:
        x = x.to(device)
        py.append(laplace_model(x, link_approx='mc', n_samples=100).detach())
    t1 = time.process_time()
    if timing:
        print("time: ", t1 - t0)
    return(torch.cat(py, dim=0))

# apply Laplace Bridge
@torch.no_grad()
def predict_LB(laplace_model, data_loader, timing=False, device='cuda'):
    
    py = []
    t0 = time.process_time()
    for x, _ in data_loader:
        x = x.to(device)
        py.append(laplace_model(x, link_approx='bridge').detach())
    t1 = time.process_time()
    if timing:
        print("time: ", t1 - t0)
    return(torch.cat(py, dim=0))

# probit approximation
@torch.no_grad()
def predict_probit(laplace_model, data_loader, timing=False, device='cuda'):
    
    py = []
    t0 = time.process_time()
    for x, _ in data_loader:
        x = x.to(device)
        py.append(laplace_model(x, link_approx='probit').detach())
    t1 = time.process_time()
    if timing:
        print("time: ", t1 - t0)
    return(torch.cat(py, dim=0))

# apply Laplace Bridge
@torch.no_grad()
def predict_LB_norm(laplace_model, data_loader, timing=False, device='cuda'):
    
    py = []
    t0 = time.process_time()
    for x, _ in data_loader:
        x = x.to(device)
        py.append(laplace_model(x, link_approx='bridge_norm').detach())
    t1 = time.process_time()
    if timing:
        print("time: ", t1 - t0)
    return(torch.cat(py, dim=0))

# apply Laplace Bridge
@torch.no_grad()
def predict_LB_alphas(laplace_model, data_loader, timing=False, device='cuda'):
    
    alphas = []
    t0 = time.process_time()
    for x, _ in data_loader:
        x = x.to(device)
        alphas.append(laplace_model(x, link_approx='bridge_alphas').detach())
    t1 = time.process_time()
    if timing:
        print("time: ", t1 - t0)
    return(torch.cat(py, dim=0))

# apply extended MacKay
@torch.no_grad()
def predict_extended_MacKay(laplace_model, data_loader, timing=False, device='cuda'):
    
    py = []
    t0 = time.process_time()
    for x, _ in data_loader:
        x = x.to(device)
        py.append(laplace_model(x, link_approx='extended_mackay').detach())
    t1 = time.process_time()
    if timing:
        print("time: ", t1 - t0)
    return(torch.cat(py, dim=0))

# apply second-order delta posterior predictive
@torch.no_grad()
def predict_second_order_dpp(laplace_model, data_loader, timing=False, device='cuda'):
    
    py = []
    t0 = time.process_time()
    for x, _ in data_loader:
        x = x.to(device)
        py.append(laplace_model(x, link_approx='second_order_dpp').detach())
    t1 = time.process_time()
    if timing:
        print("time: ", t1 - t0)
    return(torch.cat(py, dim=0))

#### alternative approaches
"""############ Functions related to calculating and printing the results ############"""

######## get all the data for in and out of dist samples

def get_fpr95(py_in, py_out):
    conf_in, conf_out = py_in.max(1), py_out.max(1)
    tpr = 95
    perc = np.percentile(conf_in, 100-tpr)
    fp = np.sum(conf_out >=  perc)
    fpr = np.sum(conf_out >=  perc)/len(conf_out)
    return fpr, perc


def get_in_dist_values(py_in, targets):
    acc_in = np.mean(np.argmax(py_in, 1) == targets)
    prob_correct = py_in[targets].mean()
    average_entropy = -np.sum(py_in*np.log(py_in+1e-8), axis=1).mean()
    MMC = py_in.max(1).mean()
    return(acc_in, prob_correct, average_entropy, MMC)

def get_out_dist_values(py_in, py_out, targets):
    average_entropy = -np.sum(py_out*np.log(py_out+1e-8), axis=1).mean()
    acc_out = np.mean(np.argmax(py_out, 1) == targets)
    if max(targets) > len(py_in[0]):
        targets = np.array(targets)
        targets[targets >= len(py_in[0])] = 0
    prob_correct = py_out[targets].mean()
    labels = np.zeros(len(py_in)+len(py_out), dtype='int32')
    labels[:len(py_in)] = 1
    examples = np.concatenate([py_in.max(1), py_out.max(1)])
    auroc = roc_auc_score(labels, examples)
    MMC = py_out.max(1).mean()
    return(acc_out, prob_correct, average_entropy, MMC, auroc)

def print_in_dist_values(acc_in, prob_correct, average_entropy, MMC, train='mnist', method='KFAC'):

    print(f'[In, {method}, {train}] Accuracy: {acc_in:.3f}; average entropy: {average_entropy:.3f}; \
    MMC: {MMC:.3f}; Prob @ correct: {prob_correct:.3f}')


def print_out_dist_values(acc_out, prob_correct, average_entropy, MMC, auroc, train='mnist', test='FMNIST', method='KFAC'):

    print(f'[Out-{test}, {method}, {train}] Accuracy: {acc_out:.3f}; Average entropy: {average_entropy:.3f};\
    MMC: {MMC:.3f}; AUROC: {auroc:.3f}; Prob @ correct: {prob_correct:.3f}')
