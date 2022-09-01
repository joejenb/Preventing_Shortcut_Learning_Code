from random import sample
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import numpy as np
import torch
import rcca
#import utils.rcca as rcca
import utils.cca as cca

from utils.kpca import KPCA


class SVCCA:
    def __init__(self, k="all", kernel_type="linear", transform_corrs='square', cca_reg=0.001, eps=1e-8, z_transform=True, sample_size=512, device="cpu"):
        self.device = device
        self.k = k
        self.z_transform = z_transform
        self.transform_corrs = transform_corrs
        self.cca_reg = cca_reg
        self.sample_size = sample_size
        self.eps = eps
        if kernel_type == "non_linear":
            self.kernel_type = "sigmoid"
        else:
            self.kernel_type = "linear"

    def to_vectors(self, X):
        n = X.size()[0]
        
        if self.k != "all":
            X = X.amax(dim=(-2, -1), keepdim=True)

        X = X.view(n, -1)

        vector_sample = X[:self.sample_size]
        #if self.k != "all":
        #    vector_sample = X[:, :self.k]

        return vector_sample
    
    def kernel_pca(self, Xc):
        Z = Xc.t()
        Zt = Xc

        ZtZ = Zt.matmul(Z)
        eigVals, eigVecs = torch.linalg.eigh(ZtZ) #the eigvecs are sorted in ascending eigenvalue order.

        V = ZtZ.matmul(eigVecs.detach())#.mul(inv_eig_vals)
        return V


    def corr_cca(self, X_proj, Y_proj, transform, is_reduced):

        vx = X_proj - torch.mean(X_proj, dim=0, keepdim=True)
        vy = Y_proj - torch.mean(Y_proj, dim=0, keepdim=True)
    
        # If projections from PyRCCA, fact = 2.0
        _vx_denom = torch.sum(vx ** 2, dim=0)
        _vy_denom = torch.sum(vy ** 2, dim=0)

        denom_eps = torch.ones_like(_vx_denom) * self.eps

        vx_denom = torch.sqrt(torch.maximum(_vx_denom, denom_eps))
        vy_denom = torch.sqrt(torch.maximum(_vy_denom, denom_eps))

        
        vx_denom = torch.maximum(torch.sqrt(_vx_denom), denom_eps)
        vy_denom = torch.maximum(torch.sqrt(_vy_denom), denom_eps)
        denominator = vx_denom * vy_denom
        nominator = torch.sum(vx * vy, dim=0)
        
        corrs = nominator / denominator
            
        assert len(corrs.shape) == 1, f"corrs has shape {corrs.shape} but must be a 1-D tensor, \nwhere element i corresponds to the correlation of layers' projections on the i-th CCA component."    

        if transform == 'square':
            corrs = torch.square(corrs)
        elif transform == 'abs':
            corrs = torch.abs(corrs)
            
        if is_reduced:
            corrs = corrs.mean()

        return corrs

    def svcca(self, X, Y):
        Xb = X.size()[1]
        Yb = Y.size()[1]

        X = self.to_vectors(X)
        Y = self.to_vectors(Y)
        #print(X.size(), Y.size())


        n = X.size()[0]
        X_centered = X - X.mean(dim=0, keepdim=True)
        Y_centered = Y - Y.mean(dim=0, keepdim=True)

        X = self.kernel_pca(X_centered)[:, :Xb]#X[:, :128]#
        Y = self.kernel_pca(Y_centered)[:, :Yb]#Y[:, :128]#

        n_components = min([X.shape[0], X.shape[1], Y.shape[1]]) // 3

        if self.z_transform:
            X_eps = torch.ones_like(X) * self.eps
            Y_eps = torch.ones_like(Y) * self.eps
            X = (X - X.mean(axis=0, keepdim=True)) / torch.maximum(X.std(axis=0, keepdim=True), X_eps)
            Y = (Y - Y.mean(axis=0, keepdim=True)) / torch.maximum(Y.std(axis=0, keepdim=True), Y_eps)
        
        Xd = X.detach().cpu().numpy()
        Yd = Y.detach().cpu().numpy()

        cca = rcca.CCA(kernelcca = False, reg = self.cca_reg, numCC = 2, verbose=False)#, device=Xd.device)
        cca.train([Xd, Yd])#, Xd.device)

        cancorrs = cca.cancorrs
        ws_1 = torch.from_numpy(cca.ws[0]).type(X.type())#cca.ws[0].detach()
        ws_2 = torch.from_numpy(cca.ws[1]).type(Y.type())#cca.ws[1].detach()
        X_proj = torch.matmul(X, ws_1)
        Y_proj = torch.matmul(Y, ws_2)
        
        return self.corr_cca(X_proj, Y_proj, transform=self.transform_corrs, is_reduced=True)
    
    def calculate_SVCCA(self, X, Y, scaling=True):
        measure_losses = dict()
        scaled_losses = dict()
        for block_num, Xn in X.items():
            svcca_score = self.svcca(Xn, Y[block_num])
            measure_losses[block_num] = svcca_score.detach()
            #Change so that is scaled by depth -> then change loss function so that scales as a function of epochs
            scaled_losses[block_num] = svcca_score
        return measure_losses, scaled_losses


class PCCA:
    def __init__(self, k="all", kernel_type="sigmoid", sample_size=768, device="cpu"):
        self.device = device
        self.sample_size = sample_size
        self.log = torch.log
        self.k = k

    def to_vectors(self, X):
        n = X.size()[0]
        
        X = X.view(n, -1)

        vector_sample = X[:self.sample_size]
        return vector_sample
    

    def kernel_pca(self, Xc):
        Z = Xc.t()
        Zt = Xc

        ZtZ = Zt.matmul(Z)
        eigVals, eigVecs = torch.linalg.eigh(ZtZ) #the eigvecs are sorted in ascending eigenvalue order.

        V = ZtZ.matmul(eigVecs.detach())#.mul(inv_eig_vals)
        return V
    
    def pcca(self, X, Y):
        Xb = X.size()[1]
        Yb = Y.size()[1]

        X = self.to_vectors(X)
        Y = self.to_vectors(Y)

        n = X.size()[0]
        X_centered = X - X.mean(dim=0, keepdim=True)
        Y_centered = Y - Y.mean(dim=0, keepdim=True)


        Xc = self.kernel_pca(X_centered)[:, :Xb]#X[:, :128]#
        Yc = self.kernel_pca(Y_centered)[:, :Yb]#Y[:, :128]#

        #Columns are variables
        Xc_mean = Xc.mean(dim=0, keepdim=True)
        Yc_mean = Yc.mean(dim=0, keepdim=True)

        #dim n, num_var
        #Xc_dev = (((Xc - Xc_mean) ** 2) + 0.00000001).sqrt()
        #Yc_dev = (((Yc - Yc_mean) ** 2) + 0.00000001).sqrt()
        Xc_dev = (Xc - Xc_mean)
        Yc_dev = (Yc - Yc_mean)
        
        #1, num_var
        Xc_var = (Xc_dev ** 2).sum(dim=0, keepdim=True) / (n - 1)
        Yc_var = (Yc_dev ** 2).sum(dim=0, keepdim=True) / (n - 1)
        
        #num_var_x, num_var_y
        XcYc_cov = Xc_dev.t().matmul(Yc_dev) / (n-1)

        #num_var_x, num_var_y
        XcYc_corr = XcYc_cov.mul(1 / torch.sqrt(Yc_var)).mul(1 / torch.sqrt(Xc_var).t())

        return XcYc_corr #XcYc_corr.max()

    def center_2d(self, X):
        X = X - X.mean(dim=0, keepdim=True)
        X = X - X.mean(dim=1, keepdim=True)
        return X

    def calculate_PCCA(self, X, Y, scaling=True):
        measure_losses = dict()
        scaled_losses = dict()
        for block_num, Xn in X.items():
            vals = self.pcca(Xn.detach(), Y[block_num])

            top_k = torch.topk(vals.flatten().detach(), 20).values
            #print("\n", top_k)
            measure_losses[block_num] = top_k.mean()
            #Change so that is scaled by depth -> then change loss function so that scales as a function of epochs
            #scaled_losses[block_num] = torch.mean(-2*self.log((1.05 - non_lin_vals)))#torch.mean(-self.log(1.1 - lin_vals)) + 
            scaled_losses[block_num] = torch.mean(torch.amax(vals, dim=0))# + torch.mean(torch.amax(non_lin_vals, dim=0)))
            #scaled_losses[block_num] = 7 * (torch.mean(torch.amax(vals, dim=0)))# + torch.mean(torch.amax(non_lin_vals, dim=0)))
            #scaled_losses[block_num] = 0.01 * torch.mean(torch.amax(lin_vals, dim=0))
        return measure_losses, scaled_losses


class R2CCA:
    def __init__(self, kernel_type="linear", sample_size=32, device="cpu"):
        super(R2CCA, self).__init__()
        self.device = device
        self.sample_size = sample_size
            
    def to_vectors(self, X):
        n = X.size()[0]

        if X.dim() > 2:
            #X = X.permute(0, 2, 3, 1).contiguous()
            X = X.view(n, -1)

        vector_sample = X[:self.sample_size]
        return vector_sample
    
    def calculate_R2CCA(self, X, Y):

        measure_losses = dict()
        scaled_losses = dict()
        for block_num, Xn in X.items():
            Xn = self.to_vectors(Xn)
            Yn = self.to_vectors(Y[block_num])


            #X_centered = Xn - Xn.mean(dim=0, keepdim=True)
            #Y_centered = Yn - Yn.mean(dim=0, keepdim=True)
            #qx, _ = torch.linalg.qr(X_centered)#, mode="complete")
            #qy, _ = torch.linalg.qr(Y_centered)#, mode="complete")

            qx, _ = torch.linalg.qr(Xn)#, mode="complete")
            qy, _ = torch.linalg.qr(Yn)#, mode="complete")

            #qx, sx, _ = torch.linalg.svd(X_centered, full_matrices=False)#, mode="complete")
            #qy, sy, _ = torch.linalg.svd(Y_centered, full_matrices=False)#, mode="complete")

            #qxs = qx.matmul(sx)
            #qys = qy.matmul(sy)

            #qxs_centered = qxs - qxs.mean(dim=0, keepdim=True)
            #qys_centered = qys - qys.mean(dim=0, keepdim=True)
            qxs_centered = qx - qx.mean(dim=0, keepdim=True)
            qys_centered = qy - qy.mean(dim=0, keepdim=True)

            #print(qxs_centered.size(), qys_centered.size())

            r2cca_score = torch.linalg.norm(qxs_centered.T.matmul(qys_centered), ord="fro") ** 2 / min(qxs_centered.shape[1], qys_centered.shape[1])

            measure_losses[block_num] = r2cca_score.detach()
            scaled_losses[block_num] = r2cca_score

        return measure_losses, scaled_losses


class CKA:
    '''
    Originally based on: https://github.com/jayroxis/CKA-similarity

    Assume that a representation is an embedding vector -> all of which are output of same
    function but for different inputs
    '''
    def __init__(self, kernel_type="linear", sample_size=768, device="cpu"):
        super(CKA, self).__init__()
        self.device = device
        self.sample_size = sample_size
        if kernel_type == "non_linear":
            self.kernel_type = "rbf"
            self.cka = self.kernel_CKA
        else:
            self.kernel_type = "linear"
            self.cka = self.linear_CKA

    def to_vectors(self, X):
        n = X.size()[0]

        if X.dim() > 2:
            #X = X.permute(0, 2, 3, 1).contiguous()
            X = X.view(n, -1)

        #indices = torch.randperm(len(X))[:self.sample_size]
        vector_sample = X[:self.sample_size]
        return vector_sample

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = torch.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))
        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)

    def calculate_CKA(self, X, Y, scaling=True):
        measure_losses = dict()
        scaled_losses = dict()
        for block_num, Xn in X.items():
            Xn = self.to_vectors(Xn)
            Yn = self.to_vectors(Y[block_num])
            cka_score = self.cka(Xn, Yn)
            measure_losses[block_num] = cka_score.detach()
            scaled_losses[block_num] = cka_score
        return measure_losses, scaled_losses


class SimilarityMeasure:

    def __init__(self, use_measure_loss=False, num_blocks=3, measure_type="cka", kernel_type="linear", device="cpu"):
        #Are all multiples of 2 for 32 by 32 
        self.measures = dict()
        self.num_blocks = num_blocks
        self.measure_type = measure_type
        self.use_measure_loss = use_measure_loss
        self.device = device
        self.cka_measure = CKA(kernel_type="linear", sample_size=32, device=self.device).calculate_CKA
        self.r2cca_measure = R2CCA(sample_size=128, device=self.device).calculate_R2CCA

        if measure_type == "cka":
            self.measure = CKA(kernel_type=kernel_type, device=device).calculate_CKA
        elif measure_type == "svcca" :
            self.measure = SVCCA(kernel_type=kernel_type, device=device).calculate_SVCCA
        elif measure_type == "pcca":
            self.measure = PCCA(kernel_type=kernel_type, device=device).calculate_PCCA

    def calculate_similarity(self, layer_outputs, scaling=True):
        inputs = {block_num: layer_outputs[block_num] for block_num in range(self.num_blocks)}
        output = {block_num: layer_outputs[self.num_blocks] for block_num in range(self.num_blocks)}
        measure_losses, scaled_losses = self.measure(inputs, output, scaling)
        #scaled_losses = {val: 3 * scaled_losses[val] / ((2 * val) + 1) for val in scaled_losses.keys()}
        #scaled_losses = {val: scaled_losses[val] / ((2 * val) + 1) for val in scaled_losses.keys()}
        return measure_losses, scaled_losses
    
    def similarity(self, layer_outputs):
        if not self.use_measure_loss:
            with torch.no_grad():
                return self.calculate_similarity(layer_outputs)
        else:
                return self.calculate_similarity(layer_outputs)

    def calculate_identity_similarity(self, layer_outputs):
        measure_losses, _ = self.measure(layer_outputs, layer_outputs)
        return measure_losses

    def calculate_within_layer_similarity(self, layer_outputs):
        cka_measure, r2cca_measure = [], []
        activations = {block_num: layer_outputs[block_num] for block_num in range(len(layer_outputs))}
        for activation in layer_outputs:
            cmp_activations = {block_num: activation for block_num in range(len(layer_outputs))}
            cka_measure_scores, _ = self.cka_measure(activations, cmp_activations)
            r2cca_measure_scores, _ = self.r2cca_measure(activations, cmp_activations)
            cka_measure.append(list(cka_measure_scores.values()))
            r2cca_measure.append(list(r2cca_measure_scores.values()))
        return torch.Tensor(cka_measure).to(self.device), torch.Tensor(r2cca_measure).to(self.device)

    def calculate_between_layer_similarity(self, layer_outputs_1, layer_outputs_2):
        cka_measure, r2cca_measure = [], []
        for activation_1 in layer_outputs_1:
            cmp_activations = {block_num: activation_1 for block_num in range(len(layer_outputs_2))}
            cka_measure_scores, _ = self.cka_measure(layer_outputs_2, cmp_activations)
            r2cca_measure_scores, _ = self.r2cca_measure(layer_outputs_2, cmp_activations)
            cka_measure.append(list(cka_measure_scores.values()))
            r2cca_measure.append(list(r2cca_measure_scores.values()))
        return torch.Tensor(cka_measure).to(self.device), torch.Tensor(r2cca_measure).to(self.device)

