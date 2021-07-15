from .flows import Flow
import torch
import torch.nn as nn
from .utils.broyden import broyden
from .res_flow import *



def smooth_l1(x, beta=1):
    mask = torch.abs(x)<beta
    y = torch.empty_like(x)
    y[mask] = 0.5*(x[mask]**2)/beta
    y[~mask] = torch.abs(x[~mask])-0.5*beta
    return y

class iNN_Flow(Flow):
    def __init__(self, dim, hidden_dims:list, activation=Swish, jacob_iter=2, inverse_iter=200, reverse=False):
        super().__init__()
        assert len(hidden_dims)>0, "Dims should include N x hidden units"
        assert activation in [ReLU, LeakyReLU, Swish], "Use ReLU or LeakyReLU or Swish"
        self.n_iter = inverse_iter
        self.jacob_iter = jacob_iter
        self.reverse = reverse
        self.dim = dim
        dims = [dim, *hidden_dims, dim]
        resblock = []
        for i in range(0, len(dims)-1):
            linear = nn.utils.spectral_norm(nn.Linear(dims[i], dims[i+1]), n_power_iterations=3)
            resblock.append(linear)
            actf = activation()
            resblock.append(actf)
        resblock = resblock[:-1]
        self.resblock = nn.ModuleList(resblock)
        self._update_spectral_norm_and_remove()

    def forward(self, x, logDetJ:bool=False):
        if self.reverse:
            return self._inverse_yes_logDetJ(x) if logDetJ else self._inverse_no_logDetJ(x)
        return self._forward_yes_logDetJ(x) if logDetJ else self._forward_no_logDetJ(x)
    
    def inverse(self, z, logDetJ:bool=False):
        if self.reverse:
            return self._forward_yes_logDetJ(z) if logDetJ else self._forward_no_logDetJ(z)
        return self._inverse_yes_logDetJ(z) if logDetJ else self._inverse_no_logDetJ(z)

    def _constrain_detJ(self, x):
        self.optim = torch.optim.Adam(self.parameters(), lr=0.0001)
        for j_iter in range(1000):
            res = x
            J = torch.eye(x.shape[1]).expand(x.shape[0],x.shape[1],x.shape[1])
            #####  FORWARD  PROPAGATION  #######
            for i, b in enumerate(self.resblock):
                if i%2==0: ### if linear layer
                    res = b(res)
                    J = J @ b.weight.t()
                else: ### if activation function
                    res, _j = b(res)
                    J = J* _j.unsqueeze(dim=1)
            detJ = torch.det(torch.eye(x.shape[1])+ J)

            if torch.min(detJ) < 0.1:
                # loss = nn.functional.softplus(-40*detJ)/4
                # loss = nn.functional.softplus(-20*detJ)
                # loss = smooth_l1(torch.minimum(detJ-0.2, torch.Tensor([0])), beta=0.2)*4
                loss = 8*(torch.minimum(detJ-0.2, torch.Tensor([0]))**2).mean()

                
                self.optim.zero_grad()
                loss.mean().backward(retain_graph=True)
                self.optim.step()
            else:
                if j_iter>0 or j_iter == 1000-1:
                    print(f"Loss in iter {j_iter}: {float(loss)}, Min detJ: {float(torch.min(detJ))}")
                break


    def _forward_yes_logDetJ(self, x):
        # self._constrain_detJ(x)
        res = x
        J = torch.eye(x.shape[1]).expand(x.shape[0],x.shape[1],x.shape[1])
        #####  FORWARD  PROPAGATION  #######
        for i, b in enumerate(self.resblock):
            if i%2==0: ### if linear layer
                res = b(res)
                J = J @ b.weight.t()
            else: ### if activation function
                res, _j = b(res)
                J = J* _j.unsqueeze(dim=1)
        y = x + res
        detJ = torch.det(torch.eye(x.shape[1])+ J)
        self.detJ = detJ

        ######## find penalty and clip value
        # self.clipval = smooth_l1(torch.maximum(detJ-0.2, torch.Tensor([0])))*10
        # self.clipval = nn.functional.softplus(20*(detJ-0.5))
        # self.penalty = smooth_l1(torch.minimum(detJ-0.2, torch.Tensor([0])), beta=0.2)*4
        # self.penalty = nn.functional.softplus(-20*detJ)

        return y, detJ.abs().log()

    def _forward_no_logDetJ(self, x):
        res = x
        for i, b in enumerate(self.resblock):
            if i%2==0: ### if linear layer
                res = b(res)
            else: ### if activation function
                res, _j = b(res)
        return x + res#*self.scaler


    def _inverse_yes_logDetJ(self, y):
        g = lambda z: y - self._forward_no_logDetJ(z)
        x = broyden(g, torch.zeros_like(y), threshold=1000, eps=1e-5)["result"]
        _, _logdetJ = self.forward(x, True)
        return x, -_logdetJ

    def _inverse_no_logDetJ(self, y):
        g = lambda z: y - self._forward_no_logDetJ(z)
        x = broyden(g, torch.zeros_like(y), threshold=1000, eps=1e-5)["result"]
        return x
    

    def _update_spectral_norm_and_remove(self):
        ### update spectral norm layer for some steps.
        with torch.no_grad():
            for _ in range(10):
                _a = torch.randn(1, self.dim)
                for i, b in enumerate(self.resblock):
                    if i%2==0: ### if linear layer
                        _a = b(_a)
                    else: ### if activation function
                        _a, _ = b(_a)
            for i, b in enumerate(self.resblock):
                if i%2==0: ### if linear layer
                    nn.utils.remove_spectral_norm(b)                    