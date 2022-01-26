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

"""####### Functions related to the Laplace Bridge transform ###########"""

def get_Gaussian_output(x, mu_w, mu_b, sigma_w, sigma_b):
    #get the distributions per class
    batch_size = x.size(0)
    num_classes = mu_b.size(0)
    
    # get mu batch
    mu_w_batch = mu_w.repeat(batch_size, 1, 1)
    mu_b_batch = mu_b.repeat(batch_size, 1)
    mu_batch = torch.bmm(x.view(batch_size, 1, -1), mu_w_batch).view(batch_size, -1) + mu_b_batch
    
    #get sigma batch
    sigma_w_batch = sigma_w.repeat(batch_size, 1, 1)
    sigma_b_batch = sigma_b.repeat(batch_size, 1)
    sigmas_diag = torch.zeros(batch_size, num_classes, device='cuda')
    for j in range(num_classes):
        h1 = x * sigma_w_batch[:, j]
        helper = torch.matmul(h1.view(batch_size, 1, -1), x.view(batch_size, -1, 1))
        helper = helper.view(-1) + sigma_b_batch[:,j]
        sigmas_diag[:,j] = helper
        
    sigma_batch = torch.stack([torch.diag(x) for x in sigmas_diag])

    return(mu_batch, sigma_batch)

def get_alpha_from_Normal(mu, Sigma):
    batch_size, K = mu.size(0), mu.size(-1)
    Sigma_d = torch.diagonal(Sigma, dim1=1, dim2=2)
    sum_exp = torch.sum(torch.exp(-1*mu), dim=1).view(-1,1)
    alpha = 1/Sigma_d * (1 - 2/K + torch.exp(mu)/K**2 * sum_exp)
    
    return(alpha)


"""########## Functions related to the acquisition of second order information ###########"""

###########Diag-Hessian

def Diag_second_order(model, train_loader, var0 = 10, device='cpu'):

    W = list(model.parameters())[-2]
    b = list(model.parameters())[-1]
    m, n = W.shape
    print("n: {} inputs to linear layer with m: {} classes".format(n, m))
    lossfunc = torch.nn.CrossEntropyLoss()

    tau = 1/var0
    eps = 10e-6

    extend(lossfunc, debug=False)
    extend(model.linear, debug=False)

    with backpack(DiagHessian()):

        max_len = len(train_loader)
        weights_cov = torch.zeros(max_len, m, n, device=device)
        biases_cov = torch.zeros(max_len, m, device=device)

        for batch_idx, (x, y) in enumerate(train_loader):

            if device == 'cuda':
                x, y = x.cuda(), y.cuda()

            model.zero_grad()
            lossfunc(model(x), y).backward()

            with torch.no_grad():
                # Hessian of weight
                W_ = W.diag_h
                b_ = b.diag_h
                
                #add_prior: since it will be flattened later we can just add the prior like that
                W_ += tau * torch.ones(W_.size(), device=device)
                b_ += tau * torch.ones(b_.size(), device=device)

                W_inv = 1/W_
                b_inv = 1/b_
                
            weights_cov[batch_idx] = W_inv
            biases_cov[batch_idx] = b_inv

            print("Batch: {}/{}".format(batch_idx, max_len))

        print(len(weights_cov))
        C_W = torch.mean(weights_cov, dim=0)
        C_b = torch.mean(biases_cov, dim=0)

    # Predictive distribution
    with torch.no_grad():
        M_W_post = W.t()
        M_b_post = b

        C_W_post = C_W
        C_b_post = C_b
        
    print("M_W_post size: ", M_W_post.size())
    print("M_b_post size: ", M_b_post.size())
    print("C_W_post size: ", C_W_post.size())
    print("C_b_post size: ", C_b_post.size())

    return(M_W_post, M_b_post, C_W_post, C_b_post)

###########Laplace-KF
#=============================================================================================

def KFLP_second_order(model, train_loader, var0 = 10, device='cpu'):

    W = list(model.parameters())[-2]
    b = list(model.parameters())[-1]
    m, n = W.shape
    lossfunc = torch.nn.CrossEntropyLoss()

    tau = 1/var0
    batch_size=128

    extend(lossfunc, debug=False)
    extend(model.linear, debug=False)

    with backpack(KFAC()):
        U, V = torch.zeros(m, m, device=device), torch.zeros(n, n, device=device)
        B = torch.zeros(m, m, device=device)

        max_len = len(train_loader)
        for batch_idx, (x, y) in enumerate(train_loader):

            if device == 'cuda':
                x, y = x.cuda(), y.cuda()

            model.zero_grad()
            lossfunc(model(x), y).backward()

            with torch.no_grad():
                # Hessian of weight
                U_, V_ = W.kfac
                B_ = b.kfac[0]

                #U_ += np.sqrt(tau)*torch.eye(m, device=device)
                #V_ += np.sqrt(tau)*torch.eye(n, device=device)
                #B_ += tau*torch.eye(m, device=device)
                
                U_ = np.sqrt(batch_size)*U_ + np.sqrt(tau)*torch.eye(m, device=device)
                V_ = np.sqrt(batch_size)*V_ + np.sqrt(tau)*torch.eye(n, device=device)
                B_ = batch_size*B_ + tau*torch.eye(m, device=device)

                rho = min(1-1/(batch_idx+1), 0.95)

                U = rho*U + (1-rho)*U_
                V = rho*V + (1-rho)*V_
                B = rho*B + (1-rho)*B_

            print("Batch: {}/{}".format(batch_idx, max_len))


    # Predictive distribution
    with torch.no_grad():
        M_W_post = W.t()
        M_b_post = b

        # Covariances for Laplace
        U_post = torch.inverse(U)
        V_post = torch.inverse(V)
        B_post = torch.inverse(B)

    print("M_W_post size: ", M_W_post.size())
    print("M_b_post size: ", M_b_post.size())
    print("U_post size: ", U_post.size())
    print("V_post size: ", V_post.size())
    print("B_post size: ", B_post.size())

    return(M_W_post, M_b_post, U_post, V_post, B_post)

"""########## Functions related to predicting the distribution over the outputs ############"""

#non-Bayesian estimate
@torch.no_grad()
def predict_MAP(model, test_loader, num_samples=1, device='cuda'):
    py = []

    max_len = int(np.ceil(len(test_loader.dataset)/len(test_loader)))
    for batch_idx, (x, y) in enumerate(test_loader):

        x, y = x.to(device), y.to(device)

        py_ = 0
        for _ in range(num_samples):
            py_ += torch.softmax(model(x), 1)
        py_ /= num_samples

        py.append(py_)
    return torch.cat(py, dim=0)


#diagonal sampling of last-layer
@torch.no_grad()
def predict_diagonal_sampling(model, test_loader, M_W_post, M_b_post, C_W_post, C_b_post, n_samples, verbose=False, cuda=False, timing=False):
    py = []
    max_len = len(test_loader)
    if timing:
        time_sum = 0

    for batch_idx, (x, y) in enumerate(test_loader):

        if cuda:
            x, y = x.cuda(), y.cuda()

        phi = model.features(x)

        mu, Sigma = get_Gaussian_output(phi, M_W_post, M_b_post, C_W_post, C_b_post)
        #print("mu size: ", mu.size())
        #print("sigma size: ", Sigma.size())

        post_pred = MultivariateNormal(mu, Sigma)

        # MC-integral
        t0 = time.time()
        py_ = 0

        for _ in range(n_samples):
            f_s = post_pred.rsample()
            py_ += torch.softmax(f_s, 1)

        py_ /= n_samples
        py_ = py_.detach()

        py.append(py_)
        t1 = time.time()
        if timing:
            time_sum += (t1 - t0)

        if verbose:
            print("Batch: {}/{}".format(batch_idx, max_len))

    if timing: print("time used for sampling with {} samples: {}".format(n_samples, time_sum))
    
    return torch.cat(py, dim=0)

#KFAC sampling of last-layer
@torch.no_grad()
def predict_KFAC_sampling(model, test_loader, M_W_post, M_b_post, U_post, V_post, B_post, n_samples, timing=False, verbose=False, cuda=False):
    py = []
    max_len = len(test_loader)
    if timing:
        time_sum = 0

    for batch_idx, (x, y) in enumerate(test_loader):

        if cuda:
            x, y = x.cuda(), y.cuda()

        phi = model.features(x).detach()

        mu_pred = phi @ M_W_post + M_b_post
        Cov_pred = torch.diag(phi @ V_post @ phi.t()).reshape(-1, 1, 1) * U_post.unsqueeze(0) + B_post.unsqueeze(0)

        post_pred = MultivariateNormal(mu_pred, Cov_pred)

        # MC-integral
        t0 = time.time()
        py_ = 0

        for _ in range(n_samples):
            f_s = post_pred.rsample()
            py_ += torch.softmax(f_s, 1)

        py_ /= n_samples
        py_ = py_.detach()

        py.append(py_)
        t1 = time.time()
        if timing:
            time_sum += (t1 - t0)


        if verbose:
            print("Batch: {}/{}".format(batch_idx, max_len))
            
    if timing: print("time used for sampling with {} samples: {}".format(n_samples, time_sum))

    return torch.cat(py, dim=0)



# Make prediction for the Laplace Bridge
@torch.no_grad()
def predict_LB(model, test_loader, M_W_post, M_b_post, C_W_post, C_b_post, verbose=False, cuda=False, timing=False):
    alphas = []
    if timing:
        time_sum_fw = 0
        time_sum_lb = 0

    max_len = len(test_loader)
    
    for batch_idx, (x, y) in enumerate(test_loader):
        
        if cuda:
            x, y = x.cuda(), y.cuda()
        
        t0_fw = time.time()
        phi = model.features(x)

        mu_pred, Cov_pred = get_Gaussian_output(phi, M_W_post, M_b_post, C_W_post, C_b_post)
        t1_fw = time.time()
        
        t0_lb = time.time()
        alpha = get_alpha_from_Normal(mu_pred, Cov_pred).detach()
        t1_lb = time.time()
        if timing:
            time_sum_fw += (t1_fw - t0_fw)
            time_sum_lb += (t1_lb - t0_lb)

        alphas.append(alpha)

        if verbose:
            print("Batch: {}/{}".format(batch_idx, max_len))

    if timing:
        print("total time used for forward pass: {:.05f}".format(time_sum_fw))
        print("total time used for Laplace Bridge: {:.05f}".format(time_sum_lb))
    
    return(torch.cat(alphas, dim = 0))
    
# predict the Laplace Bridge from a KFAC approximation of the Gaussian
@torch.no_grad()
def predict_LB_KFAC(model, test_loader, M_W_post, M_b_post, U_post, V_post, B_post, timing=False, verbose=False, cuda=False):
    alphas = []
    max_len = len(test_loader)
    if timing:
        time_sum = 0

    for batch_idx, (x, y) in enumerate(test_loader):

        if cuda:
            x, y = x.cuda(), y.cuda()

        phi = model.features(x).detach()

        mu_pred = phi @ M_W_post + M_b_post
        Cov_pred = torch.diag(phi @ V_post @ phi.t()).reshape(-1, 1, 1) * U_post.unsqueeze(0) + B_post.unsqueeze(0)

        t0 = time.time()
        alpha = get_alpha_from_Normal(mu_pred, Cov_pred).detach()
        t1 = time.time()
        if timing:
            time_sum += (t1-t0)

        alphas.append(alpha)

        if verbose:
            print("Batch: {}/{}".format(batch_idx, max_len))

    if timing:
        print("total time used for transform: {:.05f}".format(time_sum))
    
    return(torch.cat(alphas, dim = 0))


# predict the extended MacKay approach
def tau_(v):
    return(1/torch.sqrt(1 + torch.Tensor([np.pi]).cuda() * v/8))

def extended_MacKay(mu, Sigma):
    variances = Sigma.diagonal(dim1=1, dim2=2)
    assert(mu.size() == variances.size())
    x_ = torch.exp(tau_(variances).cuda() * mu)
    x = x_ / x_.sum(1).view(-1,1)
    assert(torch.allclose(x.sum(1), torch.ones(x.size(0)).cuda()))
    return(x)

def predict_extended_MacKay(model, test_loader, M_W_post, M_b_post, C_W_post, C_b_post, verbose=False, cuda=False, timing=False):
    py = []
    max_len = len(test_loader)
    if timing:
        time_sum_fw = 0
        time_sum_emk = 0

    for batch_idx, (x, y) in enumerate(test_loader):

        if cuda:
            x, y = x.cuda(), y.cuda()

        t0_fw = time.time()
        phi = model.features(x).detach()

        mu, Sigma = get_Gaussian_output(phi, M_W_post, M_b_post, C_W_post, C_b_post)
        t1_fw = time.time()

        #print("mu, Sigma: ", mu.size(), Sigma.size())

        # Extended MacKay
        t0_emk = time.time()
        py_ = 0

        py_ = extended_MacKay(mu, Sigma).detach()

        py.append(py_)
        t1_emk = time.time()
        if timing:
            time_sum_fw += (t1_fw - t0_fw)
            time_sum_emk += (t1_emk - t0_emk)

        if verbose:
            print("Batch: {}/{}".format(batch_idx, max_len))

    if timing: 
        print("time used for forward pass: {}".format(time_sum_fw))
        print("time used for Extended MacKay Approach: {}".format(time_sum_emk))
    
    return torch.cat(py, dim=0)

# predict second-order delta posterior predictive
def SODPP(mu, Sigma):
    batch_size = mu.size(0)
    exp_ = torch.exp(mu)
    sm = (exp_ / exp_.sum(1).view(-1,1)).view(batch_size, -1, 1)
    variances = Sigma.diagonal(dim1=1, dim2=2)
    
    two = torch.bmm(Sigma, sm)
    one = torch.bmm(sm.view(batch_size, 1, -1), two)
    three = 0.5 * variances - 0.5 * torch.bmm(sm.view(batch_size, 1, -1), variances.view(batch_size, -1, 1)).view(batch_size, -1)

    R_ = 1 + one.view(batch_size, -1) - two.view(batch_size, -1) + three.view(batch_size, -1)
    R = sm.view(batch_size, -1) * R_
    return(R)

def predict_SODPP(model, test_loader, M_W_post, M_b_post, C_W_post, C_b_post, verbose=False, cuda=False, timing=False):
    py = []
    max_len = len(test_loader)
    if timing:
        time_sum_fw = 0
        time_sum_SODPP = 0

    for batch_idx, (x, y) in enumerate(test_loader):

        if cuda:
            x, y = x.cuda(), y.cuda()

        t0_fw = time.time()
        phi = model.features(x).detach()

        mu, Sigma = get_Gaussian_output(phi, M_W_post, M_b_post, C_W_post, C_b_post)
        t1_fw = time.time()

        #print("mu, Sigma: ", mu.size(), Sigma.size())

        # Second oder delta posterior predictive
        t0_SODPP = time.time()
        py_ = 0

        py_ = SODPP(mu, Sigma).detach()

        py.append(py_)
        t1_SODPP = time.time()
        if timing:
            time_sum_fw += (t1_fw - t0_fw)
            time_sum_SODPP += (t1_SODPP - t0_SODPP)

        if verbose:
            print("Batch: {}/{}".format(batch_idx, max_len))

    if timing: 
        print("time used for forward pass: {}".format(time_sum_fw))
        print("time used for Second order delta posterior predictive: {}".format(time_sum_SODPP))
    
    return torch.cat(py, dim=0)

"""############ Functions related to calculating and printing the results ############"""

######## get all the data for in and out of dist samples

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
