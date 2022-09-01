import h5py
import joblib
import torch
from scipy.linalg import eigh
import numpy as np

'''
Need ws, train and cancorrs
'''

class _CCABase(object):
    def __init__(
            self,
            numCV=None,
            reg=None,
            regs=None,
            numCC=None,
            numCCs=None,
            kernelcca=True,
            ktype=None,
            verbose=False,
            select=0.2,
            cutoff=1e-15,
            gausigma=1.0,
            degree=2,
            device="cpu"
    ):
        self.numCV = numCV
        self.reg = reg
        self.regs = regs
        self.numCC = numCC
        self.numCCs = numCCs
        self.kernelcca = kernelcca
        self.ktype = ktype
        self.cutoff = cutoff
        self.select = select
        self.gausigma = gausigma
        self.degree = degree
        if self.kernelcca and self.ktype is None:
            self.ktype = "linear"
        self.verbose = verbose
        self.device = device

    def train(self, data, device):
        if self.verbose:
            print(
                "Training CCA, kernel = %s, regularization = %0.4f, "
                "%d components" % (self.ktype, self.reg, self.numCC)
            )

        comps = kcca(
            data,
            self.reg,
            self.numCC,
            kernelcca=self.kernelcca,
            ktype=self.ktype,
            gausigma=self.gausigma,
            degree=self.degree,
            device=device
        )
        self.cancorrs, self.ws, self.comps = recon(
            data, comps, kernelcca=self.kernelcca, device=device
        )
        if len(data) == 2:
            self.cancorrs = self.cancorrs[torch.nonzero(self.cancorrs)]
        return self

class CCA(_CCABase):
    """Attributes:
        reg (float): regularization parameter. Default is 0.1.
        numCC (int): number of canonical dimensions to keep. Default is 10.
        kernelcca (bool): kernel or non-kernel CCA. Default is True.
        ktype (string): type of kernel used if kernelcca is True.
                        Value can be 'linear' (default) or 'gaussian'.
        verbose (bool): default is True.
    Returns:
        ws (list): canonical weights
        comps (list): canonical components
        cancorrs (list): correlations of the canonical components
                         on the training dataset
        corrs (list): correlations on the validation dataset
        preds (list): predictions on the validation dataset
        ev (list): explained variance for each canonical dimension
    """

    def __init__(
            self, reg=0.0, numCC=10, kernelcca=True, ktype=None, verbose=True, cutoff=1e-15, device="cpu"
    ):
        super(CCA, self).__init__(
            reg=reg,
            numCC=numCC,
            kernelcca=kernelcca,
            ktype=ktype,
            verbose=verbose,
            cutoff=cutoff,
            device=device
        )

    def train(self, data, device):
        return super(CCA, self).train(data, device)


def kcca(
        data, reg=0.0, numCC=None, kernelcca=True, ktype="linear", gausigma=1.0, degree=2, device="cpu"
):
    """Set up and solve the kernel CCA eigenproblem"""
    if kernelcca:
        kernel = [
            _make_kernel(d, ktype=ktype, gausigma=gausigma, degree=degree) for d in data
        ]
    else:
        kernel = [d.T for d in data]

    nDs = len(kernel)
    nFs = [k.shape[0] for k in kernel]
    numCC = min([k.shape[0] for k in kernel]) if numCC is None else numCC

    # Get the auto- and cross-covariance matrices

    crosscovs = [torch.matmul(ki, kj.T) for ki in kernel for kj in kernel]

    # Allocate left-hand side (LH) and right-hand side (RH):
    n = sum(nFs)
    LH = torch.zeros((n, n), device=device)
    RH = torch.zeros((n, n), device=device)

    # Fill the left and right sides of the eigenvalue problem
    for i in range(nDs):
        RH[
        sum(nFs[:i]): sum(nFs[: i + 1]), sum(nFs[:i]): sum(nFs[: i + 1])
        ] = crosscovs[i * (nDs + 1)] + reg * torch.eye(nFs[i], device=device)

        for j in range(nDs):
            if i != j:
                LH[
                sum(nFs[:j]): sum(nFs[: j + 1]), sum(nFs[:i]): sum(nFs[: i + 1])
                ] = crosscovs[nDs * j + i]

    LH_np = ((LH + LH.T) / 2.0).cpu().detach().numpy()
    RH_np = ((RH + RH.T) / 2.0).cpu().detach().numpy()

    LH = ((LH + LH.T) / 2.0).unsqueeze(dim=0)#.cpu().detach().numpy()
    RH = ((RH + RH.T) / 2.0).unsqueeze(dim=0)#.cpu().detach().numpy()

    maxCC = LH_np.shape[0]

    #LH_RH = torch.stack((LH, RH), dim=0)
    #r, Vs = torch.linalg.eigh(LH_RH)
    r, Vs = torch.lobpcg(A=LH, B=RH, k=numCC)
    r, Vs = r.squeeze(), Vs.squeeze()
    #r_np, Vs_np = eigh(LH_np, RH_np, eigvals=(maxCC - numCC, maxCC - 1))

    #r, Vs = r[maxCC - numCC : maxCC], Vs[maxCC - numCC : maxCC]

    r[torch.isnan(r)] = 0
    rindex = torch.argsort(r)

    #r_np[np.isnan(r_np)] = 0
    #rindex_np = np.argsort(r_np)

    #print("org", rindex_np, "\n", rindex, "\n")

    #rindex_np = rindex_np[::-1]
    #rindex = rindex[:-1]

    #print("flipped", rindex_np, "\n", rindex, "\n")

    comp = []
    #print(Vs_np, "\n", Vs, "\n")
    Vs = Vs[:, rindex]
    #Vs_np = Vs_np[:, rindex_np]
    #print(Vs_np.shape, Vs.size())
    #print(Vs_np, "\n", Vs, "\n")

    #r, Vs = torch.from_numpy(r).to(device), torch.from_numpy(Vs).to(device)
    for i in range(nDs):
        comp.append(Vs[sum(nFs[:i]): sum(nFs[: i + 1]), :numCC])
    return comp


def recon(data, comp, corronly=False, kernelcca=True, device="cpu"):
    # Get canonical variates and CCs
    if kernelcca:
        ws = _listdot(data, comp)
    else:
        ws = comp
    ccomp = _listdot([d.T for d in data], ws)
    corrs = _listcorr(ccomp, device)
    if corronly:
        return corrs
    else:
        return corrs, ws, ccomp


def _demean(d):
    return d - d.mean(0)


def _listdot(d1, d2):
    return [torch.matmul(x[0].T, x[1]) for x in zip(d1, d2)]


def _listcorr(a, device="cpu"):
    """Returns pairwise row correlations for all items in array as a list of matrices"""
    corrs = torch.zeros((a[0].shape[1], len(a), len(a)), device=device)
    for i in range(len(a)):
        for j in range(len(a)):
            if j > i:
                tmp_corrs = []
                for (ai, aj) in zip(a[i].T, a[j].T):
                    ai_aj = torch.stack((ai, aj), dim=0)
                    corr_coef = torch.nan_to_num(torch.corrcoef(ai_aj)[0, 1])
                    tmp_corrs.append(corr_coef)

                corrs[:, i, j] = torch.Tensor(tmp_corrs).to(device)

                '''corrs[:, i, j] = [
                    torch.nan_to_num(torch.corrcoef(ai, aj)[0, 1])
                    for (ai, aj) in zip(a[i].T, a[j].T)
                ]'''
    return corrs


def _make_kernel(d, normalize=True, ktype="linear", gausigma=1.0, degree=2):
    """Makes a kernel for data d
    If ktype is 'linear', the kernel is a linear inner product
    If ktype is 'gaussian', the kernel is a Gaussian kernel, sigma = gausigma
    If ktype is 'poly', the kernel is a polynomial kernel with degree=degree
    """
    d = torch.nan_to_num(d)
    cd = _demean(d)
    if ktype == "linear":
        kernel = torch.matmul(cd, cd.T)
    elif ktype == "gaussian":
        from scipy.spatial.distance import pdist, squareform

        pairwise_dists = squareform(pdist(d, "euclidean"))
        kernel = torch.exp((-(pairwise_dists ** 2)) / (2 * gausigma ** 2))
    elif ktype == "poly":
        kernel = torch.matmul(cd, cd.T) ** degree
    kernel = (kernel + kernel.T) / 2.0
    if normalize:
        kernel = kernel / torch.linalg.eigvalsh(kernel).max()
    return kernel