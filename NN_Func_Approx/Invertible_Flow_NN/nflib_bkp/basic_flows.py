import torch
from torch import nn
import torch.nn.functional as F


class Flow(nn.Module):
    """
    Base flow model
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, logdetJ:bool=False):
        raise NotImplementedError

    def inverse(self, y, logdetJ:bool=False):
        raise NotImplementedError

    # def _logdetJ(self):
    #     raise NotImplementedError
        

class NormalizingFlow(nn.Module):

    def __init__(self, flow_list, prior_dist):
        super().__init__()
        assert len(flow_list) > 0
        self.flow = SequentialFlow(flow_list)
        self.prior_dist = prior_dist

    def forward(self, x, logdetJ=False, logprob_prior=False):
        if logdetJ:
            yy, logdetJ = self.flow(x, True)
            if logprob_prior:
                return yy, logdetJ, self.prior_dist.log_prob(yy).view(yy.shape[0], -1).sum(dim=1)
            return yy
        else:
            yy = self.flow(x, False)
            if logprob_prior:
                return yy, self.prior_dist.log_prob(yy).view(yy.shape[0], -1).sum(dim=1)
            return yy
        

    def forward_intermediate(self, x, logdetJ=False, logprob_prior=False):
        if logdetJ:
            xs = [x]
            logdetJs = [torch.zeros(x.shape[0])]
            for f in self.flow.flow_list:
                x, logdetJ_ = f(x, True)
                logdetJs += [logdetJ_]
                xs.append(x)

            if logprob_prior:
                return xs, logdetJs, self.prior_dist.log_prob(x).view(x.shape[0], -1).sum(dim=1) 
            return xs, logdetJs
        else:
            xs = [x]
            for f in self.flow.flow_list:
                x = f(x, False)
                xs.append(x)
            
            if logprob_prior:
                return xs, self.prior_dist.log_prob(x).view(x.shape[0], -1).sum(dim=1)
            return xs

    def inverse(self, y, logdetJ=False):
        return self.flow.inverse(y, logdetJ)


    def inverse_intermediate(self, y, logdetJ=False):
        if logdetJ:
            ys = [y]
            logdetJs = [torch.zeros(y.shape[0])]
            for f in reversed(self.flow.flow_list):
                y, logdetJ_ = f.inverse(y, True)
                logdetJs += [logdetJ_]
                ys.append(y)
            return ys, logdetJs
        else:
            ys = [y]
            for f in reversed(self.flow.flow_list):
                y = f.inverse(y, False)
                ys.append(y)
            return ys

    def sample(self, num_samples):
        y = self.prior_dist.sample((num_samples,))
        x = self.flow.inverse(y, False)
        return x


class SequentialFlow(Flow):
    def __init__(self, flow_list:list):
        super().__init__()
        self.flow_list = nn.ModuleList(flow_list)
    
    def forward(self, x, logdetJ=False):
        if logdetJ:
            logdetJ = 0
            for f in self.flow_list:
                x, logdetJ_ = f(x, True)
                logdetJ += logdetJ_
            return x, logdetJ
        else:
            for f in self.flow_list:
                x = f(x, False)
            return x

    def inverse(self, y, logdetJ=False):
        if logdetJ:
            logdetJ = 0
            for f in reversed(self.flow_list):
                y, logdetJ_ = f.inverse(y, True)
                logdetJ += logdetJ_
            return y, logdetJ
        else:
            for f in reversed(self.flow_list):
                y = f.inverse(y, False)
            return y


class LinearFlow(Flow):
    def __init__(self, dim, bias=True, identity_init=True):
        super().__init__()
        self.dim = dim

        _l = nn.Linear(dim, dim, bias)
        if identity_init:
            self.weight = nn.Parameter(torch.eye(dim)[torch.randperm(dim)])
        else:
            UDV = torch.svd(_l.weight.data.t())
            self.weight = nn.Parameter(UDV[0])
            del UDV
        if bias:
            self.bias = nn.Parameter(_l.bias.data)
        else:
            self.bias = None
        del _l

    def forward(self, x, logdetJ:bool=False):
        if self.bias is not None:
            y = x + self.bias
        y = y @ self.weight

        #### returning
        if logdetJ:
            return y, self._logdetJ().expand(x.shape[0])
        return y


    def inverse(self, y, logdetJ:bool=False):
        x = y @ self.weight.inverse()
        if self.bias is not None:
            x = x - self.bias

        #### returning
        if logdetJ:
            return x, self._logdetJ().expand(y.shape[0])
        return x

    def _logdetJ(self):
        # return torch.log(torch.abs(torch.det(self.weight))) 
        ### torch.logdet gives nan if negative determinant
        return self.weight.det().abs().log()

    def extra_repr(self):
        return 'dim={}'.format(self.dim)



### From GLOW paper
class Conv2d_1x1(Flow):
    '''
    Invertible 1x1 convolution with identity initialization
    '''

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.eye(dim)[torch.randperm(dim)])

    def forward(self, x, logdetJ=False):
        y = F.conv2d(x, self.weight.view(self.dim, self.dim, 1, 1))
        if logdetJ:
            return y, self._logdetJ(x.shape)
        return y

    def inverse(self, y, logdetJ=False):
        x = F.conv2d(y, self.weight.inverse().view(self.dim, self.dim, 1, 1))
        if logdetJ:
            return x, self._logdetJ(x.shape)
        return x
    
    def _logdetJ(self, s):
        return torch.log(torch.abs(torch.det(self.weight))).expand(s[0]) * s[2] * s[3]

    def extra_repr(self):
        return 'dim={}'.format(self.dim)


class LeakyReluFLow(Flow):

    def __init__(self, alpha=0.1):
        super().__init__()
        assert alpha > 0 , "Alpha should be greater than 0"
        self.alpha = alpha

    def forward(self, x, logdetJ=False):
        mask = (x>0)
        y = torch.where(mask, x, x*self.alpha)
        if logdetJ:
            det_ = torch.where(mask, torch.Tensor([1]), torch.Tensor([self.alpha]))
            return y, det_.log().sum(dim=1)
        return y

    def inverse(self, y, logdetJ=False):
        mask = (y>0)
        y = torch.where(mask, y, y/self.alpha)
        if logdetJ:
            det_ = torch.where(mask, torch.Tensor([1]), torch.Tensor([self.alpha]))
            return y, det_.log().sum(dim=1)
        return y


class PReluFLow(Flow):

    def __init__(self, dim, init_alpha=0.5):
        super().__init__()
        assert init_alpha > 0 , "Alpha should be greater than 0"
        self.alpha = nn.Parameter(torch.ones(1, dim)*init_alpha)

    def forward(self, x, logdetJ=False):
        self.clip_alpha()
        mask = (x>0)
        y = torch.where(mask, x, x*self.alpha)
        if logdetJ:
            mask = (mask).type(x.dtype)
            det_ = mask+(1-mask)*self.alpha
            return y, det_.log().sum(dim=1)
        return y

    def inverse(self, y, logdetJ=False):
        self.clip_alpha()
        mask = (y>0)
        x = torch.where(mask, y, y/self.alpha)
        if logdetJ:
            mask = (mask).type(y.dtype)
            det_ = mask+(1-mask)*self.alpha
            return x, det_.log().sum(dim=1)
        return x

    def clip_alpha(self):
        self.alpha.data = torch.clamp(self.alpha.data, 0.1, 10)



