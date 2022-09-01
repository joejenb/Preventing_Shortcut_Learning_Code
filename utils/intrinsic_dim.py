import torch
from intrinsics_dimension import twonn_pytorch

def intrinsic_dim(vectors, sample_size=128):
    '''
    Where m is the number of vectors and n is the dimension of each vector

    Takes as input either:
        -   Matrix of size (m, n)
        -   Tensor of size (b, n, h=w, h=w), with m = b * h * w
    '''
    n = vectors.size()[0]
    if vectors.dim() > 2:
        vectors = vectors.view(n, -1)

    vector_sample = vectors[:sample_size]
    return twonn_pytorch(vector_sample, return_xy=True)
