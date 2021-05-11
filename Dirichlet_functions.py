from scipy.special import digamma, loggamma
import torch

def beta_function(alpha):
    return(np.exp(np.sum([loggamma(a_i) for a_i in alpha]) - loggamma(np.sum(alpha))))

def alphas_norm(alphas):
    alphas = np.array(alphas)
    return(alphas/alphas.sum(axis=1).reshape(-1,1))

def alphas_variance(alphas):
    alphas = np.array(alphas)
    norm = alphas_norm(alphas)
    nom = norm * (1 - norm)
    den = alphas.sum(axis=1).reshape(-1,1) + 1
    return(nom/den)

def log_beta_function(alpha):
    return(np.sum([loggamma(a_i) for a_i in alpha]) - loggamma(np.sum(alpha)))

def alphas_entropy(alphas):
    K = len(alphas[0])
    alphas = np.array(alphas)
    entropy = []
    for x in alphas:
        B = log_beta_function(x)
        alpha_0 = np.sum(x)
        C = (alpha_0 - K)*digamma(alpha_0)
        D = np.sum((x-1)*digamma(x))
        entropy.append(B + C - D)
    
    return(np.array(entropy))
        

def alphas_log_prob(alphas):
    alphas = np.array(alphas)
    dig_sum = digamma(alphas.sum(axis=1).reshape(-1,1))
    log_prob = digamma(alphas) - dig_sum
    return(log_prob)

def auroc_entropy(alphas_in, alphas_out):
    
    entropy_in = alphas_entropy(alphas_in)
    entropy_out = alphas_entropy(alphas_out)
    labels = np.zeros(len(entropy_in)+len(entropy_out), dtype='int32')
    labels[:len(entropy_in)] = 1
    examples = np.concatenate([entropy_in, entropy_out])
    auroc_ent = roc_auc_score(labels, examples)
    return(auroc_ent)

def auroc_variance(alphas_in, alphas_out, method='mean'):
    
    if method=='mean':
        variance_in = alphas_variance(alphas_in).mean(1)
        variance_out = alphas_variance(alphas_out).mean(1)
    elif method=='max':
        variance_in = alphas_variance(alphas_in).max(1)
        variance_out = alphas_variance(alphas_out).max(1)
    labels = np.zeros(len(variance_in)+len(variance_out), dtype='int32')
    labels[:len(variance_in)] = 1
    examples = np.concatenate([variance_in, variance_out])
    auroc_ent = roc_auc_score(labels, examples)
    return(auroc_ent)
