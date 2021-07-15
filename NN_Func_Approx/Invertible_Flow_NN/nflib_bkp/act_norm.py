import torch
import torch.nn as nn
from torch.nn import Parameter
from .basic_flows import Flow

__all__ = ['ActNorm1d', 'ActNorm2d']


class ActNormNd(Flow):

    def __init__(self, num_features, eps=1e-12):
        super(ActNormNd, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.register_buffer('initialized', torch.tensor(0))

    @property
    def shape(self):
        raise NotImplementedError        

    def initialize_with_first_batch(self, x):
        c = x.size(1)

        with torch.no_grad():
            # compute batch statistics
            x_t = x.transpose(0, 1).view(c, -1)
            batch_var = torch.var(x_t, dim=1)
            # for numerical issues
            batch_std = -0.5*torch.log(torch.max(batch_var, torch.tensor(0.2).to(batch_var)))
            batch_mean = -torch.mean(x_t*torch.exp(batch_std).reshape(-1,1), dim=1)

            self.bias.data = batch_mean.reshape(*self.shape)
            self.weight.data = batch_std.reshape(*self.shape)
            self.initialized.fill_(1)

    def forward(self, x, logdetJ=False):
        if not self.initialized:
            self.initialize_with_first_batch(x)

        y = (x + self.bias) * torch.exp(self.weight)
        if logdetJ:
            return y, self._logdetJ(x.shape)
        else:
            return y

    def inverse(self, y, logdetJ=False):
        assert self.initialized

        x = y * torch.exp(-self.weight) - self.bias
        if logdetJ:
            return x, self._logdetJ(y.shape)
        else:
            return x

    def _logdetJ(self, xs):
        return self.weight.expand(*xs).view(xs[0], -1).sum(dim=1)

    def __repr__(self):
        return ('{name}({num_features})'.format(name=self.__class__.__name__, **self.__dict__))


class ActNorm1d(ActNormNd):

    @property
    def shape(self):
        return [1, -1]


class ActNorm2d(ActNormNd):

    @property
    def shape(self):
        return [1, -1, 1, 1]