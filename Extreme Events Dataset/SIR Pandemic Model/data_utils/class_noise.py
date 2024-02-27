import numpy as np
from scipy.interpolate import interp1d

class Noise:

    def __init__(self, domain, sigma=0.1, ell=4.0):
        self.ti = domain[0]
        self.tf = domain[1]
        self.tl = domain[1] - domain[0]
        self.R = self.get_covariance(sigma, ell)
        self.lam, self.phi = self.kle(self.R)

    def get_covariance(self, sigma, ell):
        m = 500 + 1
        self.t = np.linspace(self.ti, self.tf, m)
        self.dt = self.tl/(m-1)
        R = np.zeros([m, m])
        for i in range(m):
            for j in range(m):
                tau = self.t[j] - self.t[i]
                R[i,j] = sigma*np.exp(-tau**2/(2*ell**2)) 
        return R*self.dt

    def kle(self, R):
        lam, phi = np.linalg.eigh(R)
        phi = phi/np.sqrt(self.dt)
        idx = lam.argsort()[::-1]
        lam = lam[idx]
        phi = phi[:,idx]
        return lam, phi

    def get_eigenvalues(self, trunc=None):
        return self.lam[0:trunc]

    def get_eigenvectors(self, trunc=None):
        return self.phi[:,0:trunc]

    def get_sample(self, xi):
        nRV = np.asarray(xi).shape[0] 
        phi_trunc = self.phi[:,0:nRV]
        lam_trunc = self.lam[0:nRV]
        lam_sqrtm = np.diag(np.sqrt(lam_trunc))
        sample = np.dot(phi_trunc, np.dot(lam_sqrtm, xi))
        return sample

    def get_sample_interp(self, xi):
        sample = self.get_sample(xi.ravel())
        sample_int = interp1d(self.t, sample, kind='cubic')
        return sample_int