import torch


class KPCA:

    def __init__(self, gamma=10, dims=100, mode='gaussian', device="cpu"):
        '''
        X is the necessary input. The data.
        gamma will be the user defined value that will be used in the kernel functions. The default is 3.
        dims will be the number of dimensions of the final output (basically the number of components to be picked). The default is 1.
        mode has three options 'gaussian', 'polynomial', 'hyperbolic tangent' which will be the kernel function to be used. The default is gaussian.
        '''

        #First the kernel function picked by the user is defined. Vectors need to be input in np.mat type
        self.gamma = gamma
        self.dims = dims
        self.device = device
        self.phi = getattr(self, mode)

    def gaussian(self, x1, x2):
        return (float(torch.exp(-self.gamma*((x1 - x2).dot((x1 - x2).T))))) #gaussian. (vectors are rather inconvenient in python, so instead of xTx for inner product we need to calculate xxT)

    def linear(self, XXt):
        return XXt

    def polynomial(self, XXt):
        return (1 + XXt)**self.gamma

    def sigmoid(self, XXt):
        gamma = 0.005
        return torch.tanh((gamma * XXt) + 0.5)
    
    def hyperbolic_tangent(self, x1, x2):
        return (float(torch.tanh(x1.dot(x2.T) + self.gamma))) #hyperbolic tangent
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def components(self, X):
        '''
        Kernel=[]
        for xi in X.T:
            row=[]
            for xj in X.T:
                kf = self.phi(xi, xj)
                row.append(kf)
            Kernel.append(row)
        kernel = torch.tensor(Kernel).to(self.device)
        '''

        #Xk = self.phi(X)

        #XkXkt = Xk.matmul(Xk.t())
        #kernel = X
        #kernel = self.centering(kernel)

        # Centering the symmetric NxN kernel matrix.
        '''
        N = kernel.shape[0]
        one_n = torch.ones((N, N), device=self.device) / N
        kernel = kernel - one_n.matmul(kernel) - kernel.matmul(one_n) + one_n.matmul(kernel).matmul(one_n) #centering
        '''

        eigVals, eigVecs = torch.linalg.eigh(XkXkt) #the eigvecs are sorted in ascending eigenvalue order.
        #y = eigVecs[:, -self.dims:].t() #user defined dims, since the order is reversed, we pick principal components from the last columns instead of the first
        return XkXkt, eigVals.detach(), eigVecs.detach()


'''
class KPCA:
    def __init__(self, X, kernel, d, device="cpu"):
        """
        KPCA object
        Parameters
        ----------
        
        X: dxn matrix
        kernel: kernel function from kernel class
        d: number of principal components to be chosen
        """
        self.X = X
        self.kernel = kernel 
        self.d = d
        self.device = device
    
    def _is_pos_semidef(self, x):
        return torch.all(x >= 0)

    def __kernel_matrix(self):
        """
        Compute kernel matrix
        Output:
        
        K: nxn matrix
        """
        K = []
        r, c = self.X.shape
        for fil in range(c):
            k_aux = []
            for col in range(c):
                k_aux.append(self.kernel(self.X[:, fil], self.X[:, col]))
            K.append(k_aux)
        K = torch.Tensor(K).to(self.device)
        # Centering K
        ones = torch.ones(K.shape, device=self.device)/c
        K = K - ones@K - K@ones + ones@K@ones
        return K
    
    def __descomp(self):
        """
        Decomposition of K
        Output:
        
        tuplas_eig: List of ordered tuples by singular 
                    values; (singular_value, eigenvector)
        """
        self.K = self.__kernel_matrix()
        eigval, eigvec = torch.linalg.eig(self.K)
        if not self._is_pos_semidef(eigval):
            print("La matriz K no es semidefinida positiva")
        # Normalize eigenvectors and compute singular values of K
        tuplas_eig = [(torch.sqrt(eigval[i]), eigvec[:,i]/torch.sqrt(eigval[i]) ) for i in range(len(eigval))]
        tuplas_eig.sort(key=lambda x: x[0], reverse=True)
        return tuplas_eig
    
    def project(self):
        """
        Compute scores
        Output:
        
        scores: T = sigma * V_d^t
        """
        self.tuplas_eig = self.__descomp()
        tuplas_eig_dim = self.tuplas_eig[:self.d]
        self.sigma = torch.diag([i[0] for i in tuplas_eig_dim]).to(self.device)
        self.v = torch.Tensor([list(j[1]) for j in tuplas_eig_dim]).to(self.device).T
        self.sigma = torch.real_if_close(self.sigma, tol=1)
        self.v = torch.real_if_close(self.v, tol=1)
        self.scores = self.sigma @ self.v.T
        return self.scores
'''