# SMI-experiments

The project explores Sliced Mutual Information (SMI or aSMI/1SMI), showcasing its implementation and experimental results on synthetic data and the MNIST dataset using Convolutional Neural Networks (CNN). SMi is a scalable measure for estimating mutual information (MI) in high-dimensional settings. Traditional MI estimation struggles with high-dimensional data due to the curse of dimensionality. SMI addresses this by averaging MI estimations over one-dimensional random projections (slicing technique), providing computational efficiency while retaining key properties of MI. The experiments demonstrate the effective, i.e. when values of SMI are well-correlated with the real MI values, input vector dimensionality for estimation (~< 30, but it also depends on the entropy estimator params). 

I explicitly note that **SMI is not Mutual Information**, but a disparate metric that has some of MI properties (see [1]). Also SMI uses the same procedure as the regular MI estimation, but it does random projections of the input random vectors. One can think of this slicing technique as the data "preprocessing", which actually loses information bacause the projections are random and 1-dimensional.  

This repository contains the implementation and experiments for the Sliced Mutual Information [1]. It builds upon the [VanessB/Information-v3](https://github.com/VanessB/Information-v3) repository and its mutinfo library [2], thus you should install it beforehand. If you have some issues with installation, try installing from the fork [Pqlet/Information-v3](https://github.com/Pqlet/Information-v3.git).

The `Synthetic_tests_SMI.ipynb` notebook measures sets experiments on the synthetic data, i.e. with known MI (Figures 1 and 2).

The MNIST notebooks `MNIST-SMI.ipynb` and `MNIST-AEMI.ipynb` measure SMI and AE+MI on MNIST classification task (for Figure 3).

The `Plotting_MI_SMI.ipynb` notebook plots Figure 3.

## Results 

The synthetic experiments are conducted with the following parameters of the SMI and WKL estimator (Berrett et al. [2017]): #Projections is 2k, #Samples is 3k, number of neighbours in Weighted Kozachenko-Leonenko estimator (WKL) k = 5.

### Figure 1: Experiments with estimating MI of synthetic data. Base MI estimation (blue) is $\hat I(X,Y)$, and SMI estimation (orange) is $\hat I_{SMI}(X,Y)$.
![Figure 1](https://github.com/Pqlet/SMI-experiments/blob/main/images/comparison_runs.png)
The correlation value is computed between the real MI values (red) and the corresponding estimated values.

### Figure 2: Correlation of real $I$ and estimated $\hat{I}$ MI values over dimensionality of input vectors (averaged over 10 runs) on synthetic data.
![Figure 2](https://github.com/Pqlet/SMI-experiments/blob/main/images/correlation_comparison_multiseed.png)
Base estimator is the blue line, and the sliced version is orange. The raise of correlation on dimensionality of 75 is due to the high variance of the results, i.e. more runs for averaging are required to get a more precise picture.

### Figure 3: Information Plane of a CNN on MNIST

![Figure 3](https://github.com/Pqlet/SMI-experiments/blob/main/images/mnist_comparison.png)

This figure shows the Information Plane during training, i.e. MI(L;Y) over MI(X;L), where L is the flattened hidden representation of a layer. MI between representations compressed (both X and L) with autoencoder (top) and SMI between uncompressed representations (bottom) during training a CNN classifier on MNIST dataset. SMI parameters: \#Projections is $1k$, \#Samples is $10k$. WKL MI estimator is employed with $k=25$. Overall, the experiment with training classification CNN and measuring SMI took more than 7 hours, which is times more than AE+MI estimation. However, this might be the inefficiency of my implementation. 

## Implementation of the SMI
The computation of SMI can be divided into 4 steps:
1. Sample m random vectors from a multivariate normal distribution.
2. Take Q matrix from QR decomposition of sampled vectors to get vectors uniformly distributed on a sphere.
3. Take the dot product between sampled projection directions and input vectors.
4. Calculate the MI between both projections of input vectors and take the average over m random projections.

The implementation in python:
```python
import numpy as np

def sample_spherical(dim, n_projections):
    sampled_vectors = np.array([]).reshape(0,dim)
    while len(sampled_vectors) < n_projections:
        vec = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), size=dim) # (num_vec, dim)
        vec = np.linalg.qr(vec).Q
        sampled_vectors = np.vstack((sampled_vectors, vec))
    return sampled_vectors[:n_projections] # (num_vec, dim)
    
class smi_compressor():
    def __init__(self, dim, n_projections):
        self.theta = sample_spherical(dim=dim, n_projections=n_projections) # (n_projections, dim)
        
    def __call__(self, X):
        # getting projections
        X_compressed = np.dot(self.theta, X.T)
        return X_compressed # m x n
```

The sampling is done in the while loop due to the restrictions on the Q matrix size (it becomes square of the smaller size of `vec`).

## Conclusion 

Even the dimensionalities of 20 − 30 can already be considered "high-dimensional" for SMI, being rather small in modern Deep Learning. MI estimation with the autoencoder compression and direct MI estimation [2] is possible for simple datasets like MNIST, where the compression to the dimensionalities of less than 10 is doable. Slicing technique, in its turn, would allow to estimate the information measure for more complex datasets, where compression to the d < 10 is detrimental. The drawback of using slicing with MI estimation would be more computations and not exactly MI estimation, but a disparate metric that is similar to and highly correlates with it.


## References
[1] Z. Goldfeld and K. Greenewald, “Sliced Mutual Information: A Scalable Measure of Statistical Dependence.” arXiv, Oct. 18, 2021. doi: 10.48550/arXiv.2110.05279.

[2] I. Butakov, A. Tolmachev, S. Malanchuk, A. Neopryatnaya, A. Frolov, and K. Andreev, “Information Bottleneck Analysis of Deep Neural Networks via Lossy Compression.” arXiv, May 13, 2023. doi: 10.48550/arXiv.2305.08013.

