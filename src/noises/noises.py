import math
import warnings

import numpy as np
import scipy as sp
import scipy.special
import scipy.stats
import torch
from scipy.stats import beta, binom, gamma, laplace, norm, betaprime
import torch.distributions as D

from .utils import (atanh, diffmethod_table, get_radii_from_convex_table,
                    get_radii_from_table, lvsetmethod_table, make_or_load,
                    plexp, relu, sample_l1_sphere, sample_l2_sphere,
                    sample_linf_sphere, wfun)


class Noise(object):
    
    __adv__ = []

    def __init__(self, dim, sigma=None, lambd=None, device='cpu'):
        self.dim = dim
        self.device = device
        if lambd is None and sigma is not None:
            self.sigma = sigma
            self.lambd = self.get_lambd(sigma)
        elif sigma is None and lambd is not None:
            self.lambd = lambd
            self.sigma = self.get_sigma(lambd)
        else:
            raise ValueError('Please give exactly one of sigma or lambd')

    def _sigma(self):
        '''Calculates the sigma if lambd = 1
        '''
        raise NotImplementedError()
    def get_sigma(self, lambd=None):
        '''Calculates the sigma given lambd
        '''
        if lambd is None:
            lambd = self.lambd
        return lambd * self._sigma()

    def get_lambd(self, sigma=None):
        '''Calculates the lambd given sigma
        '''
        if sigma is None:
            sigma = self.sigma
        return sigma / self._sigma()

    def sample(self, x):
        '''Apply noise to x'''
        raise NotImplementedError()

    def certify(self, prob_lb, adv):
        raise NotImplementedError()

    def _certify_lp_convert(self, prob_lb, adv, warn=True):
        if adv in self.__adv__:
            cert = getattr(self, f'certify_l{adv}')
            return cert(prob_lb)
        else:
            r = {}
            for a in self.__adv__:
                cert = getattr(self, f'certify_l{a}')
                ppen = self.dim ** (1/a - 1/adv) if adv > a else 1
                r[a] = cert(prob_lb) / ppen
            if warn:                
                lpstr = ', '.join([f'l{p}' for p in self.__adv__])
                warnings.warn(f'No direct robustness guarantee for l{adv}; '
                              f'converting {lpstr} radii to l{adv}.')
            if len(r) == 1:
                return list(r.values())[0]
            else:
                radii = list(r.values())
                out = torch.max(radii.pop(), radii.pop())
                while len(radii) > 0:
                    c = radii.pop()
                    out = torch.max(out, c)
                return out

    def certify_l1(self, prob_lb):
        return self.certify(prob_lb, adv=1)

    def certify_l2(self, prob_lb):
        return self.certify(prob_lb, adv=2)

    def certify_linf(self, prob_lb):
        return self.certify(prob_lb, adv=np.inf)


class Uniform(Noise):
    '''Uniform noise on [-lambda, lambda]^dim
    '''

    __adv__ = [1, np.inf]

    def __init__(self, dim, sigma=None, lambd=None, device='cpu'):
        super().__init__(dim, sigma, lambd, device)

    def __str__(self):
        return f"Uniform(dim={self.dim}, lambd={self.lambd}, sigma={self.sigma})"

    def plotstr(self):
        return 'Uniform'

    def tabstr(self, adv):
        return f'unif_{adv}_d{self.dim}'

    def _sigma(self):
        return 3 ** -0.5

    def sample(self, x):
        return (torch.rand_like(x, device=self.device) - 0.5) * 2 * self.lambd + x

    def certify(self, prob_lb, adv):
        return self._certify_lp_convert(prob_lb, adv, warn=True)

    def certify_l1(self, prob_lb):
        return 2 * self.lambd * (prob_lb - 0.5)

    def certify_linf(self, prob_lb):
        return 2 * self.lambd * (1 - (1.5 - prob_lb) ** (1 / self.dim))


class Gaussian(Noise):
    '''Isotropic Gaussian noise
    '''

    __adv__ = [2]

    def __init__(self, dim, sigma=None, lambd=None, device='cpu'):
        super().__init__(dim, sigma, lambd, device)
        self.norm_dist = D.Normal(loc=torch.tensor(0., device=device),
                                scale=torch.tensor(self.lambd, device=device))

    def __str__(self):
        return f"Gaussian(dim={self.dim}, lambd={self.lambd}, sigma={self.sigma})"
    
    def plotstr(self):
        return "Gaussian"

    def tabstr(self, adv):
        return f'gaussian_{adv}_d{self.dim}'

    def _sigma(self):
        return 1

    def sample(self, x):
        return torch.randn_like(x) * self.lambd + x

    def certify(self, prob_lb, adv):
        return self._certify_lp_convert(prob_lb, adv, warn=False)

    def certify_l2(self, prob_lb):
        return self.norm_dist.icdf(prob_lb)


class Laplace(Noise):
    '''Isotropic Laplace noise
    '''

    __adv__ = [1, np.inf]

    def __init__(self, dim, sigma=None, lambd=None, device='cpu'):
        super().__init__(dim, sigma, lambd, device)
        self.laplace_dist = D.Laplace(loc=torch.tensor(0.0, device=device),
                                    scale=torch.tensor(self.lambd, device=device))
        self.linf_radii = self.linf_rho = self._linf_table_info = None

    def __str__(self):
        return f"Laplace(dim={self.dim}, lambd={self.lambd}, sigma={self.sigma})"

    def plotstr(self):
        return "Laplace"

    def tabstr(self, adv):
        return f'laplace_{adv}_d{self.dim}'

    def _sigma(self):
        return 2 ** 0.5

    def sample(self, x):
        return self.laplace_dist.sample(x.shape) + x

    def certify(self, prob_lb, adv):
        return self._certify_lp_convert(prob_lb, adv)

    def certify_l1(self, prob_lb):
        return -self.lambd * (torch.log(2 * (1 - prob_lb)))

    def certify_linf(self, prob_lb, mode='approx',
                    inc=0.001, grid_type='radius', upper=3, save=True):
       
        if mode == 'approx':
            return self.lambd * D.Normal(0, 1).icdf(prob_lb) / self.dim ** 0.5
        elif mode == 'integrate':
            table_info = dict(inc=inc, grid_type=grid_type, upper=upper)
            if self.linf_rho is None or self._linf_table_info != table_info:
                self.make_linf_table(inc, grid_type, upper, save)
                self._table_info = table_info
            return self.lambd * get_radii_from_convex_table(
                            self.linf_rho, self.linf_radii, prob_lb
            )
        else:
            raise ValueError(f'Unrecognized mode "{mode}"')

    def Phi_linf(self, prob):
        def phi(c, d):
            return binom(d, 0.5).sf((c+d)/2)
        def phiinv(p, d):
            return 2 * binom(d, 0.5).isf(p) - d
        d = self.dim
        c = phiinv(prob, d)
        pp = phi(c, d)
        return c * (prob - pp) + d * phi(c - 1/2, d-1) - d * phi(c, d)

    def _certify_linf_integrate(self, rho):
        return sp.integrate.quad(lambda p: 1/self.Phi_linf(p),
                                1 - rho, 1/2)[0]
    def make_linf_table(self, inc=0.001, grid_type='radius', upper=3, save=True,
                                loc='tables'):
        
        
        self.linf_rho, self.linf_radii = make_or_load(
                        self.tabstr('linf'), self._make_linf_table, inc=inc,
                        grid_type=grid_type, upper=upper, save=save, loc=loc)
        return self.linf_rho, self.linf_radii

    def _make_linf_table(self, inc=0.001, grid_type='radius', upper=3):
        return diffmethod_table(self.Phi_linf, f=norm.cdf,
                    inc=inc, grid_type=grid_type, upper=upper)







