import numpy as np
from GPy.util.linalg import pdinv
import pyDOE as doe
from scipy.stats import uniform
from .utils import process_parameters

class Inputs():
    """A class for input definition.

    Parameters
    ----------
    domain : list
        Domain definition.  Must be of the form 
            [ [x1min, x1max], [x2min, x2max], ... ]

    Attributes
    ----------
    domain : see Parameters
    input_dim : int
        Dimensionality of the input space.

    """

    def __init__(self, domain):
        self.domain = domain
        self.input_dim = len(domain)

    def draw_samples(self, n_samples, sample_method):
        """Generate samples over the input space.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate. If sample_method="grd",
            number of grid points along each dimension. Otherwise, 
            total number of samples.
        sample_method : {"lhs", "uni", "grd", "pdf"}
            Method to generate the samples. 
                - "lhs": Latin Hypercube Sampling
                - "uni": Uniform Sampling
                - "grd": Grid Sampling
                - "pdf": Sample according to `self.pdf`
                
        Returns
        -------
        pts : array
            Array of samples, size (n_samples, n_dim)

        """
        sample_method = sample_method.lower()

        if sample_method == "lhs":
            pts = doe.lhs(self.input_dim, samples=n_samples)
            pts = self.rescale_samples(pts, self.domain)

        elif sample_method == "uni":
            pts = np.random.rand(n_samples, self.input_dim)
            pts = self.rescale_samples(pts, self.domain)

        elif sample_method == "grd":
            ngrid = n_samples
            grd = np.mgrid[ [slice(bd[0], bd[1], ngrid*1j) \
                             for bd in self.domain] ]
            pts = grd.T.reshape(-1, self.input_dim)

        elif sample_method == "pdf":
            pts = self.rvs(n_samples=n_samples)

        return pts

    @staticmethod
    def rescale_samples(x, domain):
        """Rescale samples from [0,1]^d to actual domain."""
        for i in range(x.shape[1]):
            bd = domain[i]
            x[:,i] = x[:,i]*(bd[1]-bd[0]) + bd[0]
        return x

    def pdf(self, x):
        """Definition of the input pdf p(x).

        Parameters
        ----------
        x : array
            Query points. Should be of size (n_pts, n_dim)

        Returns
        -------
        pdf : array
            Density function at the query points.

        """
        raise NotImplementedError 


    def pdf_jac(self, x):
        """Gradients of the input pdf p(x) w.r.t. x.

        Parameters
        ----------
        x : array
            Query points. Should be of size (n_pts, n_dim)

        Returns
        -------
        jac : array
            Gradient of input pdf at query points.

        """
        raise NotImplementedError

    def rvs(self, n_samples):
        """Draw random samples according to `self.pdf`.

        Parameters
        ----------
        n_samples : int
            Number of samples.

	Returns
        -------
        res : array
            Random samples.

        """
        raise NotImplementedError
    
    
class GaussianInputs(Inputs):
    """A class for Multivariate Gaussian inputs.

    Parameters
    ----------
    domain : see parent class (Inputs)
    mu : array_like
         Mean of the distribution
    cov : array_like
         Covariance matrix of the distribution

    Attributes
    ----------
    domain, input_dim : see parent class (Inputs)
    mu, cov : see Parameters
    inv : array
        Inverse of the covariance matrix.
    constant : float
        Normalization constant.

    """

    def __init__(self, domain, mu, cov):
        super().__init__(domain)
        self.mu, self.cov = process_parameters(self.input_dim, mu, cov)
        self.inv, _, _, ld, = pdinv(self.cov)
        self.constant = -0.5*(self.input_dim * np.log(2*np.pi) + ld)

    def pdf(self, x):
        d = x - self.mu
        lnpdf = self.constant - 0.5 * np.sum(d * np.dot(d, self.inv), 1)
        return np.exp(lnpdf)

    def pdf_jac(self, x):
        const = np.dot(x - self.mu, self.inv) 
        pdf = self.pdf(x)
        return - const * pdf[:,None]

    def rvs(self, n_samples):
        res = np.random.multivariate_normal(self.mu, self.cov, n_samples)
        return res