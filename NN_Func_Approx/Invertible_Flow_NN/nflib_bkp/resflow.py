import torch
import torch.nn as nn
from .basic_flows import Flow


class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        mask = (x > 0).type(x.dtype)
        return x*mask, mask

class LeakyReLU(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        assert alpha >= 0, "Alpha should be positive"
        self.alpha = alpha

    def forward(self, x):
        mask = (x > 0).type(x.dtype)
        mask += (1-mask)*self.alpha
        return x*mask, mask


class ResidualMLP(Flow):
    def __init__(self, dim, hidden_dims:list, activation=ReLU, scaler=0.97, n_iter=200):
        super().__init__()
        assert len(hidden_dims)>0, "Dims should include N x hidden units"
        assert activation is ReLU or activation is LeakyReLU, "Use ReLU or LeakyReLU"
        self.n_iter = n_iter

        dims = [dim, *hidden_dims, dim]
        resblock = []
        for i in range(0, len(dims)-1):
            linear = nn.utils.spectral_norm(nn.Linear(dims[i], dims[i+1]))
            resblock.append(linear)
            if i < len(dims)-2:
                actf = activation()
                resblock.append(actf)
        self.resblock = nn.ModuleList(resblock)
        self.scaler = scaler

        ### update spectral norm layer for some steps.
        with torch.no_grad():
            for _ in range(20):
                _a = torch.randn(1, dim)
                for i, b in enumerate(self.resblock):
                    if i%2==0: ### if linear layer
                        _a = b(_a)
                    else: ### if activation function
                        _a, _ = b(_a)

    def forward(self, x, logdetJ=False):
        res = x
        if logdetJ:
            J = torch.eye(x.shape[1]).expand(x.shape[0],x.shape[1],x.shape[1])
            for i, b in enumerate(self.resblock):
                if i%2==0: ### if linear layer
                    res = b(res)
                    J = J @ b.weight.t()
                else: ### if activation function
                    res, _j = b(res)
                    J = J* _j.unsqueeze(dim=1) ## unsqueeze in middle dimension
                    ### input J -> 5x2x2
                    ### transformation by 2x3 matrix: J -> 5x2x3
                    ### mask shape -> 5x3; so convert to -> 5x1x3
                    ### unsqueeze mask in dim 1 and multiply

            return x + res*self.scaler,\
                    torch.det(torch.eye(x.shape[1])+ J*self.scaler).abs().log()
        else:
            for i, b in enumerate(self.resblock):
                if i%2==0: ### if linear layer
                    res = b(res)
                else: ### if activation function
                    res, _j = b(res)
            return x + res*self.scaler

    def inverse(self, y, logdetJ:False):
        x = self._inverse_fixed_point(y)
        if logdetJ:
            _, _logdetJ = self.forward(x, True)
            return x, _logdetJ
        else:
            return x

    def _inverse_fixed_point(self, y, atol=1e-5, rtol=1e-5):
        x, x_prev = y - self.forward(y, False), y
        i = 0
        tol = atol + y.abs() * rtol
        while not torch.all((x - x_prev)**2 / tol < 1):
            x, x_prev = y - self.forward(x), x
            i += 1
            if i > self.n_iter:
                break
        return x