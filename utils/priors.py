import torch


"""
Prior for the weights of the model
"""

class DirichletPrior:
    def __init__(self, concentration):
        self._dist = torch.distributions.Dirichlet(concentration)
        
    def log_prob(self, x):
        return self._dist.log_prob(x)
    
"""
Priors for the means of the model
"""
    
class GaussianPrior:
    def __init__(self, dim, mean, stddev):
        self._dim = dim
        self._mean = mean
        self._stddev = stddev
        self._dist = torch.distributions.Normal(self._mean, self._stddev) 

    def log_prob(self, x):
        log_prob = self._dist.log_prob(x)
        return log_prob.sum(dim=-1)
    
class SoftballPrior:
    def __init__(self, dim, radius, a=1):
        self.dim = dim
        self.radius = radius
        self.a = a
        self.norm = torch.lgamma(torch.tensor(1 + dim * 0.5)) - dim * (
            torch.log(torch.tensor(radius)) + 0.5 * torch.log(torch.tensor(torch.pi))
        )

    def log_prob(self, x):
        return self.norm - torch.log(
            1 + torch.exp(self.a * (torch.norm(x, dim=-1) / self.radius - 1))
        )
    
"""
Priors for the covariance matrix of the model
"""
    
       
class WishartPrior:
    def __init__(self, df, scale, covariance_type):
        self.df = df
        self.scale = scale
        self.covariance_type = covariance_type
        if covariance_type in ['full', 'tied']:
            self._dist = torch.distributions.Wishart(df, scale)
        elif covariance_type == 'diagonal':
            self._dist = torch.distributions.Gamma(df / 2, scale.diag() / 2)
        elif covariance_type == 'spherical':
            self._dist = torch.distributions.Gamma(df / 2, scale / 2)

    def log_prob(self, x):
        if self.covariance_type == 'full':
            return self._dist.log_prob(x)
        elif self.covariance_type == 'tied':
            return self._dist.log_prob(x)
        elif self.covariance_type == 'diagonal':
            return self._dist.log_prob(x).sum()
        elif self.covariance_type == 'spherical':
            return self._dist.log_prob(x).sum()
        else:
            raise ValueError("Unsupported covariance matrix shape")

class InverseGammaPrior:
    def __init__(self, alpha, beta):
        self._dist = torch.distributions.InverseGamma(alpha, beta)
        
    def log_prob(self, x):
        if x.dim() == 1:  # Diagonal or spherical
            return self._dist.log_prob(x).sum()
        elif x.dim() == 2:  # Full or tied
            diag_elements = torch.diagonal(x)
            return self._dist.log_prob(diag_elements).sum()
        elif x.dim() == 3:  # Full
            diag_elements = torch.diagonal(x, dim1=-2, dim2=-1)
            return self._dist.log_prob(diag_elements).sum(dim=1).sum()
        else:
            raise ValueError("Unsupported covariance matrix shape")
        
class LogNormalPrior:
    def __init__(self, mean, stddev):
        self._dist = torch.distributions.LogNormal(mean, stddev)
        
    def log_prob(self, x):
        if x.dim() == 1:  # Diagonal or spherical
            return self._dist.log_prob(x).sum()
        elif x.dim() == 2:  # Tied
            diag_elements = torch.diagonal(x)
            return self._dist.log_prob(diag_elements).sum()
        elif x.dim() == 3:  # Full
            diag_elements = torch.diagonal(x, dim1=-2, dim2=-1)
            return self._dist.log_prob(diag_elements).sum(dim=1).sum()
        else:
            raise ValueError("Unsupported covariance matrix shape")