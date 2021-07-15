import torch
import torch.nn as nn
from .flows import Flow
from .utils.broyden import broyden

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

def jacobian(Y, X, create_graph=False):
    jac = torch.zeros(X.shape[0], X.shape[1], Y.shape[1])
    for i in range(Y.shape[1]):
        J_i = torch.autograd.grad(outputs=Y[:,i], inputs=X,
                                  grad_outputs=torch.ones(jac.shape[0]),
                                  only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        jac[:,:,i] = J_i
    if create_graph:
        jac.requires_grad_()
    return jac

class ResidualFlow(Flow):
    def __init__(self, dim, hidden_dims:list, activation=ReLU, scaler=0.97, n_iter=200, reverse=False):
        super().__init__()
        assert len(hidden_dims)>0, "Dims should include N x hidden units"
        assert activation is ReLU or activation is LeakyReLU, "Use ReLU or LeakyReLU"
        self.n_iter = n_iter

        self.reverse = reverse
        self.dim = dim
        dims = [dim, *hidden_dims, dim]
        resblock = []
        for i in range(0, len(dims)-1):
            linear = nn.utils.spectral_norm(nn.Linear(dims[i], dims[i+1]), n_power_iterations=3)
            resblock.append(linear)
            actf = activation()
            resblock.append(actf)
        self.resblock = nn.ModuleList(resblock[:-1])
        self.scaler = scaler
        self._update_spectral_norm_init()

    def forward(self, x, logDetJ:bool=False):
        if self.reverse:
            return self._inverse_yes_logDetJ(x) if logDetJ else self._inverse_no_logDetJ(x)
        return self._forward_yes_logDetJ(x) if logDetJ else self._forward_no_logDetJ(x)
    
    def inverse(self, z, logDetJ:bool=False):
        if self.reverse:
            return self._forward_yes_logDetJ(z) if logDetJ else self._forward_no_logDetJ(z)
        return self._inverse_yes_logDetJ(z) if logDetJ else self._inverse_no_logDetJ(z)

    def _forward_yes_logDetJ(self, x):
        res = x
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


    # def _forward_yes_logDetJ(self, x):
    #     if not x.requires_grad:
    #         x = torch.autograd.Variable(x, requires_grad=True)
    #     res = x
    #     for i, b in enumerate(self.resblock):
    #         if i%2==0: ### if linear layer
    #             res = b(res)
    #         else: ### if activation function
    #             res, _j = b(res)

    #     y = x + res*self.scaler
    #     J = jacobian(y, x, True)
    #     return y, torch.det(J).abs().log()

    def _forward_no_logDetJ(self, x):
        res = x
        for i, b in enumerate(self.resblock):
            if i%2==0: ### if linear layer
                res = b(res)
            else: ### if activation function
                res, _j = b(res)
        return x + res*self.scaler


    def _inverse_yes_logDetJ(self, y):
        g = lambda z: y - self._forward_no_logDetJ(z)
        x = broyden(g, torch.zeros_like(y), threshold=1000, eps=1e-5)["result"]
        _, _logdetJ = self.forward(x, True)
        return x, -_logdetJ

    def _inverse_no_logDetJ(self, y):
        g = lambda z: y - self._forward_no_logDetJ(z)
        x = broyden(g, torch.zeros_like(y), threshold=1000, eps=1e-5)["result"]
        return x


    # def _inverse_fixed_point(self, y, atol=1e-5, rtol=1e-5):
    #     x, x_prev = y - self._forward_no_logDetJ(y), y
    #     i = 0
    #     tol = atol + y.abs() * rtol
    #     while not torch.all((x - x_prev)**2 / tol < 1):
    #         x, x_prev = y - self._forward_no_logDetJ(x), x
    #         i += 1
    #         if i > 1000:
    #             print('Iterations exceeded 1000 for inverse.')
    #             break
    #     return x
    
    

    def _update_spectral_norm_init(self):
        ### update spectral norm layer for some steps.
        with torch.no_grad():
            for _ in range(200):
                _a = torch.randn(1, self.dim)
                for i, b in enumerate(self.resblock):
                    if i%2==0: ### if linear layer
                        _a = b(_a)
                    else: ### if activation function
                        _a, _ = b(_a)

            # for i, b in enumerate(self.resblock):
            #     if i%2==0: ### if linear layer
            #         nn.utils.remove_spectral_norm(b)

