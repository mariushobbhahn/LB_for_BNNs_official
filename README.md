# LB for BNNs

This is the official GitHub repository for the paper "Fast Predictive Uncertainty For Classification Bayesian Deep Networks" by me (Marius Hobbhahn), Agustinus Kristiadi, and Philipp Hennig. The paper can be found on arxiv at: https://arxiv.org/abs/2003.01227. I have also written a <a href='https://www.mariushobbhahn.com/2020-11-03-LB_for_BNNs/'>blog post</a>, which is an easy-to-understand summary of the paper.

All files are supposed to be stand-alone notebooks so you can look at different experiments independent of each other.

### Experiment 1

This is the preliminary experiment for better intuition that has been moved to Appendix D. It shows that the Laplace Bridge has meaningfully different uncertainty between in-distribution and out-of-distribution samples for a small MNIST setting.

### Experiment 2

The OOD experiment from the main paper compares the mean of the Dirichlet from the Laplace Bridge to the MC integral of 1000 samples.
Each notebook additionally contains the experiments for the comparisons of the Laplace Bridge with other methods to approximate the integral in a fast fashion in appendix D.

If you want to play around with the different methods yourself, you can simply comment the call of the train() method back in and start running the notebook.
The prior variances (called var0 in the code) of the Gaussians have been chosen such that their in-dataset MMC differs only around 5% from that of the MAP estimate but you can choose different ones as well if you wish.

### Experiment 3

The timing experiment for the main paper is done partly here. The comparison and contextualization for the timing of the forward pass vs. that of the LB is done in the respective cells of the CIFAR10 notebook in experiment 2.

### Experiment 4

The experiment on ImageNet is conducted here. If you don't have the training and test data for ImageNet yourself, you can't validate the experiment and, unfortunately, I am not allowed to share them for legal (copyright) reasons.

### Applying the Laplace Bridge to other use cases

If you want to apply the Laplace Bridge to your own datasets this repository should include all necessary components. Applying a last-layer Laplace approximation to a network is very simple since most classifiers have a linear last layer. There are many references in the original paper showing that a last-layer Laplace approximation is nearly as good as a full layer one at the fraction of a cost. so don't worry about losing too much quality of your prediction. Using the code in LB_utils.py you can easily create the approximations of the Hessian to your last layer and use them for your predictions. I would generally recommend using the CIFAR10 experiments as a guiding notebook if you can't quite follow what's going on - they are commented on in more detail.

## Final Words

I'm always looking for feedback. If you have problems understanding the content don't hesitate to contact me.
If you have any further questions or suggestions about this repo you can write me at marius.hobbhahn[at]gmail.com
