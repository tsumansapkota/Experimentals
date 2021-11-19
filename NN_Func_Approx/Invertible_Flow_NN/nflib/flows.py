"""
Implements various flows.
Each flow is invertible so it can be forward()ed and inverse()ed.
Notice that inverse() is not backward as in backprop but simply inversion.
Each flow also outputs its log det J "regularization"

Reference:

NICE: Non-linear Independent Components Estimation, Dinh et al. 2014
https://arxiv.org/abs/1410.8516

Variational Inference with Normalizing Flows, Rezende and Mohamed 2015
https://arxiv.org/abs/1505.05770

Density estimation using Real NVP, Dinh et al. May 2016
https://arxiv.org/abs/1605.08803
(Laurent's extension of NICE)

Improved Variational Inference with Inverse Autoregressive Flow, Kingma et al June 2016
https://arxiv.org/abs/1606.04934
(IAF)

Masked Autoregressive Flow for Density Estimation, Papamakarios et al. May 2017 
https://arxiv.org/abs/1705.07057
"The advantage of Real NVP compared to MAF and IAF is that it can both generate data and estimate densities with one forward pass only, whereas MAF would need D passes to generate data and IAF would need D passes to estimate densities."
(MAF)

Glow: Generative Flow with Invertible 1x1 Convolutions, Kingma and Dhariwal, Jul 2018
https://arxiv.org/abs/1807.03039

"Normalizing Flows for Probabilistic Modeling and Inference"
https://arxiv.org/abs/1912.02762
(review paper)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import abc
from nets import LeafParam, MLP, ARMLP


# ------------------------------------------------------------------------
class Flow(nn.Module):
    """
    Base flow model
    """
    def __init__(self):
        super().__init__()

    def _forward_yes_logDetJ(self, x):
        raise NotImplementedError

    def _forward_no_logDetJ(self, x):
        raise NotImplementedError

    def _inverse_yes_logDetJ(self, z):
        raise NotImplementedError

    def _inverse_no_logDetJ(self, z):
        raise NotImplementedError
    
    def forward(self, x, logDetJ:bool=False):
        return self._forward_yes_logDetJ(x) if logDetJ else self._forward_no_logDetJ(x)
    
    def inverse(self, z, logDetJ:bool=False):
        return self._inverse_yes_logDetJ(z) if logDetJ else self._inverse_no_logDetJ(z)

    # def get_logdetJ(self):
    #     raise NotImplementedError


class SequentialFlow(Flow):
    """ A sequence of Normalizing Flows is a SequentialFlow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def _forward_yes_logDetJ(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m)
        for flow in self.flows:
            x, ld = flow.forward(x, True)
            log_det += ld
        return x, log_det

    def _forward_no_logDetJ(self, x):
        for flow in self.flows:
            x = flow.forward(x, False)
        return x

    def forward_intermediate(self, x, logDetJ=False):
        return self._forward_intermediate_yes_logDetJ(x) if logDetJ else self._forward_intermediate_no_logDetJ(x)
        
    def _forward_intermediate_yes_logDetJ(self, x):
        m, _ = x.shape
        log_det = [torch.zeros(m)]
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x, True)
            log_det += [ld]
            zs += [x]
        return zs, log_det

    def _forward_intermediate_no_logDetJ(self, x):
        zs = [x]
        for flow in self.flows:
            x = flow.forward(x, False)
            zs.append(x)
        return zs

    def _inverse_yes_logDetJ(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z, True)
            log_det += ld
        return z, log_det

    def _inverse_no_logDetJ(self, z):
        for flow in self.flows[::-1]:
            z = flow.inverse(z, False)
        return z

    def _inverse_intermediate_yes_logDetJ(self, z):
        m, _ = z.shape
        log_det = [torch.zeros(m)]
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z, True)
            log_det += [ld]
            xs.append(z)
        return xs, log_det

    def _inverse_intermediate_no_logDetJ(self, z):
        xs = [z]
        for flow in self.flows[::-1]:
            z = flow.inverse(z, False)
            xs.append(z)
        return xs

    def inverse_intermediate(self, x, logDetJ=False):
        return self._inverse_intermediate_yes_logDetJ(x) if logDetJ else self._inverse_intermediate_no_logDetJ(x)
    

class NormalizingFlow(nn.Module):
    """ A Normalizing Flow Model is a (flow, prior) pair """
    
    def __init__(self, flow_list, prior):
        super().__init__()
        self.flow = SequentialFlow(flow_list)
        self.prior = prior
    
    def forward(self, x, logDetJ=False, intermediate=False):
        if logDetJ:
            if intermediate:
                zs, log_det = self.flow.forward_intermediate(x, True)
                z = zs[-1]
            else:
                zs, log_det = self.flow.forward(x, True)
                z = zs
            prior_logprob = self.prior.log_prob(z).view(x.size(0), -1).sum(1)
            return zs, log_det, prior_logprob
        else:
            if intermediate:
                zs = self.flow.forward_intermediate(x, False)
                z = zs[-1]
            else:
                zs = self.flow.forward(x, False)
                z = zs
            prior_logprob = self.prior.log_prob(z).view(x.size(0), -1).sum(1)
            return z, prior_logprob

    def inverse(self, z, logDetJ=False, intermediate=False):
        if logDetJ:
            if intermediate:
                xs, log_det = self.flow.inverse_intermediate(z, True)
            else:
                xs, log_det = self.flow.inverse(z, True)
            return xs, log_det
        else:
            if intermediate:
                xs = self.flow.inverse_intermediate(z, False)
            else:
                xs = self.flow.inverse(z, False)
            return z
    
    def sample(self, num_samples:int):
        z = self.prior.sample((num_samples,))
        xs = self.flow.inverse(z, False)
        return xs


# ------------------------------------------------------------------------


class AffineConstantFlow(Flow):
    """ 
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """
    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else torch.zeros(1, dim)
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else torch.zeros(1, dim)
        
    def _forward_yes_logDetJ(self, x):
        z = x * torch.exp(self.s) + self.t
        log_det = torch.sum(self.s, dim=1)
        return z, log_det.expand(len(x))

    def _forward_no_logDetJ(self, x):
        z = x * torch.exp(self.s) + self.t
        return z
    
    def _inverse_yes_logDetJ(self, z):
        x = (z - self.t) * torch.exp(-self.s)
        log_det = torch.sum(-self.s, dim=1)
        return x, log_det.expand(len(x))

    def _inverse_no_logDetJ(self, z):
        x = (z - self.t) * torch.exp(-self.s)
        return x

class ActNorm(AffineConstantFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False
    
    def _initialize_data(self, x):
        assert self.s is not None and self.t is not None # for now
        self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
        
        self.s.data[torch.isinf(self.s.data)] = 0
        self.s.data[torch.isnan(self.s.data)] = 0
        
        self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
        self.data_dep_init_done = True

    def forward(self, x, logDetJ=False):
        # first batch is used for init
        if not self.data_dep_init_done:
            self._initialize_data(x)    
        return super().forward(x, logDetJ)

class ActNorm2D(ActNorm):
    '''
    Per channel normalization rather than per neuron normalization
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x, logDetJ=False):
        ### (N, dim, h, w)
        x = x.transpose(1,3)
        xs1 = x.shape #### (N, w, h, dim)
        x = x.view(-1, xs1[3]) ## now in shape (N*h*w, dim)
        # the batch of data needs to be reshaped
        if logDetJ:
            z, _logdetJ = super().forward(x, True)
            return z.view(xs1).transpose(1,3), _logdetJ
        else:
            return super().forward(x, False).view(xs1).transpose(1,3)


class LinearFlow(Flow):
    def __init__(self, dim, bias=True, identity_init=True):
        super().__init__()
        self.dim = dim

        _l = nn.Linear(dim, dim, bias)
        if identity_init:
            self.weight = nn.Parameter(torch.eye(dim))#[torch.randperm(dim)])
        else:
            UDV = torch.svd(_l.weight.data.t())
            self.weight = nn.Parameter(UDV[0])
            del UDV
        if bias:
            self.bias = nn.Parameter(_l.bias.data)
        else:
            self.bias = None
        del _l

    def _forward_yes_logDetJ(self, x):
        if self.bias is not None:
            y = x + self.bias
        y = y @ self.weight
        return y, self._logdetJ().expand(x.shape[0])

    def _forward_no_logDetJ(self, x):
        if self.bias is not None:
            y = x + self.bias
        y = y @ self.weight
        return y

    def _inverse_yes_logDetJ(self, y):
        x = y @ self.weight.inverse()
        if self.bias is not None:
            x = x - self.bias
        return x, -self._logdetJ().expand(y.shape[0])

    def _inverse_no_logDetJ(self, y):
        x = y @ self.weight.inverse()
        if self.bias is not None:
            x = x - self.bias
        return x

    def _logdetJ(self):
        # return torch.log(torch.abs(torch.det(self.weight))) 
        ### torch.logdet gives nan if negative determinant
        # return self.weight.det().abs().log()
        return self.weight.det().log()


# ------------------------------------------------------------------------

# class SlowMAF(nn.Module):
#     """ 
#     Masked Autoregressive Flow, slow version with explicit networks per dim
#     """
#     def __init__(self, dim, parity, net_class=MLP, nh=24):
#         super().__init__()
#         self.dim = dim
#         self.layers = nn.ModuleDict()
#         self.layers[str(0)] = LeafParam(2)
#         for i in range(1, dim):
#             self.layers[str(i)] = net_class(i, 2, nh)
#         self.order = list(range(dim)) if parity else list(range(dim))[::-1]
        
#     def forward(self, x):
#         z = torch.zeros_like(x)
#         log_det = torch.zeros(x.size(0))
#         for i in range(self.dim):
#             st = self.layers[str(i)](x[:, :i])
#             s, t = st[:, 0], st[:, 1]
#             z[:, self.order[i]] = x[:, i] * torch.exp(s) + t
#             log_det += s
#         return z, log_det

#     def inverse(self, z):
#         x = torch.zeros_like(z)
#         log_det = torch.zeros(z.size(0))
#         for i in range(self.dim):
#             st = self.layers[str(i)](x[:, :i])
#             s, t = st[:, 0], st[:, 1]
#             x[:, i] = (z[:, self.order[i]] - t) * torch.exp(-s)
#             log_det += -s
#         return x, log_det

# class MAF(nn.Module):
#     """ Masked Autoregressive Flow that uses a MADE-style network for fast forward """
    
#     def __init__(self, dim, parity, net_class=ARMLP, nh=24):
#         super().__init__()
#         self.dim = dim
#         self.net = net_class(dim, dim*2, nh)
#         self.parity = parity

#     def forward(self, x):
#         # here we see that we are evaluating all of z in parallel, so density estimation will be fast
#         st = self.net(x)
#         s, t = st.split(self.dim, dim=1)
#         z = x * torch.exp(s) + t
#         # reverse order, so if we stack MAFs correct things happen
#         z = z.flip(dims=(1,)) if self.parity else z
#         log_det = torch.sum(s, dim=1)
#         return z, log_det
    
#     def inverse(self, z):
#         # we have to decode the x one at a time, sequentially
#         x = torch.zeros_like(z)
#         log_det = torch.zeros(z.size(0))
#         z = z.flip(dims=(1,)) if self.parity else z
#         for i in range(self.dim):
#             st = self.net(x.clone()) # clone to avoid in-place op errors if using IAF
#             s, t = st.split(self.dim, dim=1)
#             x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
#             log_det += -s[:, i]
#         return x, log_det

# class IAF(MAF):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         """
#         reverse the flow, giving an Inverse Autoregressive Flow (IAF) instead, 
#         where sampling will be fast but density estimation slow
#         """
#         self.forward, self.inverse = self.inverse, self.forward


# class Invertible1x1Conv(nn.Module):
#     """ 
#     As introduced in Glow paper.
#     """
    
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
#         Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
#         P, L, U = torch.lu_unpack(*Q.lu())
#         self.P = P # remains fixed during optimization
#         self.L = nn.Parameter(L) # lower triangular portion
#         self.S = nn.Parameter(U.diag()) # "crop out" the diagonal to its own parameter
#         self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S

#     def _assemble_W(self):
#         """ assemble W from its pieces (P, L, U, S) """
#         L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim))
#         U = torch.triu(self.U, diagonal=1)
#         W = self.P @ L @ (U + torch.diag(self.S))
#         return W

#     def forward(self, x):
#         W = self._assemble_W()
#         z = x @ W
#         log_det = torch.sum(torch.log(torch.abs(self.S)))
#         return z, log_det

#     def inverse(self, z):
#         W = self._assemble_W()
#         W_inv = torch.inverse(W)
#         x = z @ W_inv
#         log_det = -torch.sum(torch.log(torch.abs(self.S)))
#         return x, log_det

# ------------------------------------------------------------------------
