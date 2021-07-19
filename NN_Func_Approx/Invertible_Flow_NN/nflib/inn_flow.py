from torch._C import dtype
from .flows import Flow
import torch
import torch.nn as nn
from .utils.broyden import broyden
from tqdm import tqdm
from typing import Union
from scipy import optimize
# ------------------------------------------------------------------------

class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def det_bound(self):
        return torch.Tensor([0,1])

    def forward(self, x):
        mask = (x > 0).type(x.dtype)
        return x*mask, mask

class LeakyReLU(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        assert alpha >= 0.01, "Alpha should be positive"
        self.alpha = alpha

    @property
    def det_bound(self):
        return torch.Tensor([self.alpha,1])

    def forward(self, x):
        mask = (x > 0).type(x.dtype)
        mask += (1-mask)*self.alpha
        return x*mask, mask
    

class TanhLU(nn.Module):

    @property
    def det_bound(self):
        return torch.Tensor([0.01,1])

    def forward(self, x):
        y = x
        dy = torch.ones_like(x)
        
        mask = x < 0
        y[mask] = torch.tanh(x[mask])
        dy[mask] =(1 - y[mask] ** 2)
        return y, dy


class Swish(nn.Module):

    def __init__(self, beta=0.8):
        super().__init__()
        self.beta = nn.Parameter(torch.Tensor([beta]))

    @property
    def det_bound(self):
        return torch.Tensor([-0.1,1.1])

    def forward(self, x):
        z = torch.sigmoid(self.beta*x)
        y = x * z
        
        by = self.beta*y
        j = by+z*(1-by)
        return y, j

# ------------------------------------------------------------------------
def smooth_l1(x, beta=1):
    mask = torch.abs(x)<beta
    y = torch.empty_like(x)
    y[mask] = 0.5*(x[mask]**2)/beta
    y[~mask] = torch.abs(x[~mask])-0.5*beta
    return y


class iMLP_Flow(Flow):
    def __init__(self, dim, hidden_dim, activation=TanhLU, jacob_iter=2, inverse_iter=200, reverse=False, scipy_inverse:Union[None, str]='broyden1'):
        super().__init__()
        assert activation in [ReLU, LeakyReLU, Swish, TanhLU], "Use only defined Activations"
        self.n_iter = inverse_iter
        self.jacob_iter = jacob_iter
        self.reverse = reverse
        self.dim = dim
        self.scipy_inverse = scipy_inverse
        self.hidden_dim = hidden_dim
        assert hidden_dim > dim, "Hidden dim should be >= input output dim to be invertible"

        self.layers = [
            nn.Linear(dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, dim)
        ]
        ## since with equal dim, the function with increasing function is invertible.
        if hidden_dim > dim:
            self.layers[2].weight.data[dim:] *= 0.
        self.layers = nn.ModuleList(self.layers)

        # find initial jacobian direction to keep the same direction (+ or -)
        self.sign = 1
        with torch.no_grad():
            x = torch.randn(1, self.dim)
            J = torch.eye(x.shape[1]).expand(x.shape[0],x.shape[1],x.shape[1])
            x = self.layers[0](x)
            J = J @ self.layers[0].weight.t()
            x, _j = self.layers[1](x)
            J = J* _j.unsqueeze(dim=1)
            x = self.layers[2](x)
            J = J @ self.layers[2].weight.t()
            detJ = torch.det(J)
            print(torch.sign(detJ))
            if detJ[0]<0:
                self.sign = -1

        self.det_penalty = None
        ## TODO: Optimize weights to have exact sign of determinant for all inputs.

        print("Initializing to have all same sign of determinant")
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for i in tqdm(range(100)):
            optimizer.zero_grad()
            x = torch.randn(1, self.dim)
            self._forward_yes_logDetJ(x)
            optimizer.step()

        print("Det Min Max", self._dets.min(), self._dets.max())


    def forward(self, x, logDetJ:bool=False):
        if self.reverse:
            return self._inverse_yes_logDetJ(x) if logDetJ else self._inverse_no_logDetJ(x)
        return self._forward_yes_logDetJ(x) if logDetJ else self._forward_no_logDetJ(x)
    
    def inverse(self, z, logDetJ:bool=False):
        if self.reverse:
            return self._forward_yes_logDetJ(z) if logDetJ else self._forward_no_logDetJ(z)
        return self._inverse_yes_logDetJ(z) if logDetJ else self._inverse_no_logDetJ(z)

    def _jacobian_determinant_loss(self, dets):
        minmax = self.layers[1].det_bound
        combo = torch.bernoulli(torch.ones(len(dets), 1, self.hidden_dim)*0.5).type(torch.long)
        jacobian = self.layers[0].weight.t().expand(len(dets), self.dim, self.hidden_dim)
        jacobian = jacobian*minmax[combo]
        jacobian = jacobian@self.layers[2].weight.t()
        rand_det = torch.det(jacobian)
        dets = torch.cat([dets, rand_det])
        self._dets = dets.data

        c = 0.1 ## 0.1, 0.2
        penalty = (1 - torch.minimum(dets*self.sign-0.2, torch.Tensor([c]))/c)**2
        # penalty = (1 - torch.minimum(dets*self.sign, torch.Tensor([1])))**25
        self.det_penalty =  penalty.mean() + penalty.max()/10
        self.det_penalty.backward(retain_graph=True)

        for p in self.parameters():
            if p.grad is None:
                p.grad_det = None
                continue
            p.grad_det = p.grad.clone()
        return

    def clip_output_gradients(self):
        '''
        Called after the loss of target is also backpropagated
        '''
        for p in self.parameters():
            if p.grad_det is None:
                continue
            ########### METHOD 1 ##############
            # print("gD", p.grad_det)
            # clipval = (1-torch.minimum(500*p.grad_det.data.abs(), torch.Tensor([1])))**100
            # clipval = torch.clamp(100*(p.grad_det.data.abs()-0.01), 0.0, 1.0)
            clipval = 10e-8/torch.clamp(p.grad_det.data.abs(), 0.0, 1.0)

            # print("CV",clipval)
            p.grad = torch.minimum(torch.maximum(p.grad, -clipval), clipval)
            p.grad += p.grad_det

            ########## METHOD 2 ##################
            # mask = p.grad_det.data.abs() > 1e-4
            # p.grad += p.grad_det
            # p.grad[mask] = p.grad_det[mask]

            ######### METHOD 3 #####################


    def _forward_yes_logDetJ(self, x):
        # self._constrain_detJ(x)
        J = torch.eye(x.shape[1]).expand(x.shape[0],x.shape[1],x.shape[1])
        #####  FORWARD  PROPAGATION  #######
        x = self.layers[0](x)
        J = J @ self.layers[0].weight.t()
        x, _j = self.layers[1](x)
        J = J* _j.unsqueeze(dim=1)
        x = self.layers[2](x)
        J = J @ self.layers[2].weight.t()

        self.detJ = torch.det(J)
        self._jacobian_determinant_loss(self.detJ)
        return x, self.detJ.abs().log()

    def _forward_no_logDetJ(self, x):
        x = self.layers[0](x)
        x, _j = self.layers[1](x)
        x = self.layers[2](x)
        return x

    def _inverse_yes_logDetJ(self, y):
        if self.scipy_inverse is None:
            g = lambda z: y - self._forward_no_logDetJ(z)
            x = broyden(g, torch.zeros_like(y), threshold=1000, eps=1e-5)["result"]
        else:
            x = torch.empty_like(y)
            def func_g(_z, _y):
                _z = torch.Tensor(_z)
                _x = _y - self._forward_no_logDetJ(_z)
                return _x.data.numpy()

            for i in tqdm(range(len(y))):
                out = optimize.root(func_g, torch.zeros_like(y[i]) ,args=(y[i]), method=self.scipy_inverse)['x']
                x[i] = torch.Tensor(out)

        _, _logdetJ = self.forward(x, True)
        return x, -_logdetJ

    def _inverse_no_logDetJ(self, y):
        if self.scipy_inverse is None:
            g = lambda z: y - self._forward_no_logDetJ(z)
            x = broyden(g, torch.zeros_like(y), threshold=1000, eps=1e-5)["result"]
        else:
            x = torch.empty_like(y)
            def func_g(_z, _y):
                _z = torch.Tensor(_z)
                _x = _y - self._forward_no_logDetJ(_z)
                return _x.data.numpy()

            for i in tqdm(range(len(y))):
                out = optimize.root(func_g, torch.zeros_like(y[i]) ,args=(y[i]), method=self.scipy_inverse)['x']
                x[i] = torch.Tensor(out)
        return x                

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------


class DeterminantScaler_Flow(Flow):
    def __init__(self, dim, scale=0.25):
        super().__init__()
        self.dim = dim
        self.scale = (torch.ones(1, dim)*scale)**(1/dim)

    def _forward_yes_logDetJ(self, x):
        x = x*self.scale
        logdet = torch.log(self.scale).sum()
        return x, logdet.expand(x.shape[0])

    def _forward_no_logDetJ(self, x):
        return x*self.scale

    def _inverse_yes_logDetJ(self, z):
        z = z/self.scale
        logdet = -torch.log(self.scale).sum()
        return z, logdet.expand(z.shape[0])

    def _inverse_no_logDetJ(self, z):
        return z/self.scale


# ------------------------------------------------------------------------


# class StaticScaleShift_Flow(Flow):
#     def __init__(self, scale=0.25, shift=0.0):
#         super().__init__()
#         self.scale = scale
#         self.shift = shift

#     def _forward_yes_logDetJ(self, x):
#         x = x*self.scale + self.shift
#         logdet = torch.log(torch.Tensor([self.scale]).to(x))*x.shape[1]
#         return x, logdet.expand(x.shape[0])

#     def _forward_no_logDetJ(self, x):
#         return x*self.scale + self.shift

#     def _inverse_yes_logDetJ(self, z):
#         z = (z-self.shift)/self.scale
#         logdet = -torch.log(torch.Tensor([self.scale]).to(z))*z.shape[1]
#         return z, logdet.expand(z.shape[0])

#     def _inverse_no_logDetJ(self, z):
#         return (z-self.shift)/self.scale



# class iMLP_Flow_xxx(Flow):
#     def __init__(self, dim, hidden_dim, activation=TanhLU, jacob_iter=2, inverse_iter=200, reverse=False):
#         super().__init__()
#         assert activation in [ReLU, LeakyReLU, Swish, TanhLU], "Use only defined Activations"
#         self.n_iter = inverse_iter
#         self.jacob_iter = jacob_iter
#         self.reverse = reverse
#         self.dim = dim
#         self.hidden_dim = hidden_dim
#         assert hidden_dim > dim, "Hidden dim should be >= input output dim to be invertible"

#         self.layers = [
#             nn.Linear(dim, hidden_dim),
#             activation(),
#             nn.Linear(hidden_dim, dim)
#         ]
#         ## since with equal dim, the function with increasing function is invertible.
#         if hidden_dim > dim:
#             self.layers[2].weight.data[dim:] *= 0.
#         self.layers = nn.ModuleList(self.layers)

#         ### since we limit our determinant on > 0.4, we can change that range to 0.1 by scaling it down by 4
#         self.scaler = DeterminantScaler_Flow(self.dim, scale=1/4)

#         # find initial jacobian direction to keep the same direction (+ or -)
#         self.sign = 1
#         with torch.no_grad():
#             x = torch.randn(1, self.dim)
#             J = torch.eye(x.shape[1]).expand(x.shape[0],x.shape[1],x.shape[1])
#             x = self.layers[0](x)
#             J = J @ self.layers[0].weight.t()
#             x, _j = self.layers[1](x)
#             J = J* _j.unsqueeze(dim=1)
#             x = self.layers[2](x)
#             J = J @ self.layers[2].weight.t()
#             detJ = torch.det(J)
#             print(torch.sign(detJ))
#             if detJ[0]<0:
#                 self.sign = -1

#         self.det_penalty = None
#         ## TODO: Optimize weights to have exact sign of determinant for all inputs.

#         print("Initializing to have all same sign of determinant")
#         optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
#         for i in tqdm(range(100)):
#             optimizer.zero_grad()
#             x = torch.randn(1, self.dim)
#             self._forward_yes_logDetJ(x)
#             optimizer.step()

#         print("Det Min Max", self._dets.min(), self._dets.max())


#     def forward(self, x, logDetJ:bool=False):
#         if self.reverse:
#             return self._inverse_yes_logDetJ(x) if logDetJ else self._inverse_no_logDetJ(x)
#         return self._forward_yes_logDetJ(x) if logDetJ else self._forward_no_logDetJ(x)
    
#     def inverse(self, z, logDetJ:bool=False):
#         if self.reverse:
#             return self._forward_yes_logDetJ(z) if logDetJ else self._forward_no_logDetJ(z)
#         return self._inverse_yes_logDetJ(z) if logDetJ else self._inverse_no_logDetJ(z)

#     def _jacobian_determinant_loss(self, dets):
#         minmax = self.layers[1].det_bound
#         combo = torch.bernoulli(torch.ones(len(dets), 1, self.hidden_dim)*0.5).type(torch.long)
#         jacobian = self.layers[0].weight.t().expand(len(dets), self.dim, self.hidden_dim)
#         jacobian = jacobian*minmax[combo]
#         jacobian = jacobian@self.layers[2].weight.t()
#         rand_det = torch.det(jacobian)
#         dets = torch.cat([dets, rand_det])
#         self._dets = dets.data
#         penalty = (1 - torch.minimum(dets*self.sign-0.2, torch.Tensor([0.2]))/0.2)**2 ## 0.4 as no loss point
#         # c = 0.2
#         # penalty = (1 - torch.minimum(dets*self.sign, torch.Tensor([1])))**25
#         self.det_penalty =  penalty.mean() + penalty.max()/10
#         self.det_penalty.backward(retain_graph=True)

#         for p in self.parameters():
#             if p.grad is None:
#                 p.grad_det = None
#                 continue
#             p.grad_det = p.grad.clone()
#         return

#     def clip_output_gradients(self):
#         '''
#         Called after the loss of target is also backpropagated
#         '''
#         for p in self.parameters():
#             if p.grad_det is None:
#                 continue
#             ########### METHOD 1 ##############
#             # print("gD", p.grad_det)
#             # clipval = (1-torch.minimum(500*p.grad_det.data.abs(), torch.Tensor([1])))**100
#             # clipval = torch.clamp(100*(p.grad_det.data.abs()-0.01), 0.0, 1.0)
#             clipval = 10e-8/torch.clamp(p.grad_det.data.abs(), 0.0, 1.0)


#             # print("CV",clipval)
#             p.grad = torch.minimum(torch.maximum(p.grad, -clipval), clipval)
#             p.grad += p.grad_det

#             ########## METHOD 2 ##################
#             # mask = p.grad_det.data.abs() > 1e-4
#             # p.grad += p.grad_det
#             # p.grad[mask] = p.grad_det[mask]

#             ######### METHOD 3 #####################


#     def _forward_yes_logDetJ(self, x):
#         # self._constrain_detJ(x)
#         J = torch.eye(x.shape[1]).expand(x.shape[0],x.shape[1],x.shape[1])
#         #####  FORWARD  PROPAGATION  #######
#         x = self.layers[0](x)
#         J = J @ self.layers[0].weight.t()
#         x, _j = self.layers[1](x)
#         J = J* _j.unsqueeze(dim=1)
#         x = self.layers[2](x)
#         J = J @ self.layers[2].weight.t()

#         self.detJ = torch.det(J)
#         self._jacobian_determinant_loss(self.detJ)

#         _x, _ldetJ = self.scaler._forward_yes_logDetJ(x)
#         return _x, self.detJ.abs().log()+_ldetJ

#     def _forward_no_logDetJ(self, x):
#         x = self.layers[0](x)
#         x, _j = self.layers[1](x)
#         x = self.layers[2](x)
#         return self.scaler._forward_no_logDetJ(x)

#     def _inverse_yes_logDetJ(self, y):
#         y, _ldetJ = self.scaler._inverse_yes_logDetJ(y)

#         if self.slow_inverse:
#             def func_g(z, y):
#                 z = torch.Tensor(z)
#                 _x = y - self._forward_no_logDetJ(z)
#                 return _x.data.numpy()

#         else:
#             g = lambda z: y - self._forward_no_logDetJ(z)
#             x = broyden(g, torch.zeros_like(y), threshold=1000, eps=1e-5)["result"]
#         _, _logdetJ = self.forward(x, True)
#         return x, (-_logdetJ) + _ldetJ

#     def _inverse_no_logDetJ(self, y):
#         y = self.scaler._inverse_no_logDetJ(y)
#         g = lambda z: y - self._forward_no_logDetJ(z)
#         x = broyden(g, torch.zeros_like(y), threshold=1000, eps=1e-5)["result"]
#         return x            


# ------------------------------------------------------------------------

# class iNN_Flow(Flow):
#     def __init__(self, dim, hidden_dims:list, activation=TanhLU, jacob_iter=2, inverse_iter=200, reverse=False):
#         super().__init__()
#         assert len(hidden_dims)>0, "Dims should include N x hidden units"
#         assert activation in [ReLU, LeakyReLU, Swish, TanhLU], "Use only defined Activations"
#         self.n_iter = inverse_iter
#         self.jacob_iter = jacob_iter
#         self.reverse = reverse
#         self.dim = dim
#         dims = [dim, *hidden_dims, dim]
#         resblock = []
#         for i in range(0, len(dims)-1):
#             linear = nn.utils.spectral_norm(nn.Linear(dims[i], dims[i+1]), n_power_iterations=3)
#             resblock.append(linear)
#             actf = activation()
#             resblock.append(actf)
#         resblock = resblock[:-1]
#         self.resblock = nn.ModuleList(resblock)
#         self._update_spectral_norm_and_remove()

#     def forward(self, x, logDetJ:bool=False):
#         if self.reverse:
#             return self._inverse_yes_logDetJ(x) if logDetJ else self._inverse_no_logDetJ(x)
#         return self._forward_yes_logDetJ(x) if logDetJ else self._forward_no_logDetJ(x)
    
#     def inverse(self, z, logDetJ:bool=False):
#         if self.reverse:
#             return self._forward_yes_logDetJ(z) if logDetJ else self._forward_no_logDetJ(z)
#         return self._inverse_yes_logDetJ(z) if logDetJ else self._inverse_no_logDetJ(z)

#     def _constrain_detJ(self, x):
#         self.optim = torch.optim.Adam(self.parameters(), lr=0.0001)
#         for j_iter in range(1000):
#             res = x
#             J = torch.eye(x.shape[1]).expand(x.shape[0],x.shape[1],x.shape[1])
#             #####  FORWARD  PROPAGATION  #######
#             for i, b in enumerate(self.resblock):
#                 if i%2==0: ### if linear layer
#                     res = b(res)
#                     J = J @ b.weight.t()
#                 else: ### if activation function
#                     res, _j = b(res)
#                     J = J* _j.unsqueeze(dim=1)
#             detJ = torch.det(torch.eye(x.shape[1])+ J)

#             if torch.min(detJ) < 0.1:
#                 # loss = nn.functional.softplus(-40*detJ)/4
#                 # loss = nn.functional.softplus(-20*detJ)
#                 # loss = smooth_l1(torch.minimum(detJ-0.2, torch.Tensor([0])), beta=0.2)*4
#                 loss = 8*(torch.minimum(detJ-0.2, torch.Tensor([0]))**2).mean()

                
#                 self.optim.zero_grad()
#                 loss.mean().backward(retain_graph=True)
#                 self.optim.step()
#             else:
#                 if j_iter>0 or j_iter == 1000-1:
#                     print(f"Loss in iter {j_iter}: {float(loss)}, Min detJ: {float(torch.min(detJ))}")
#                 break


#     def _forward_yes_logDetJ(self, x):
#         # self._constrain_detJ(x)
#         res = x
#         J = torch.eye(x.shape[1]).expand(x.shape[0],x.shape[1],x.shape[1])
#         #####  FORWARD  PROPAGATION  #######
#         for i, b in enumerate(self.resblock):
#             if i%2==0: ### if linear layer
#                 res = b(res)
#                 J = J @ b.weight.t()
#             else: ### if activation function
#                 res, _j = b(res)
#                 J = J* _j.unsqueeze(dim=1)
#         y = x + res
#         detJ = torch.det(torch.eye(x.shape[1])+ J)
#         self.detJ = detJ

#         ######## find penalty and clip value
#         # self.clipval = smooth_l1(torch.maximum(detJ-0.2, torch.Tensor([0])))*10
#         # self.clipval = nn.functional.softplus(20*(detJ-0.5))
#         # self.penalty = smooth_l1(torch.minimum(detJ-0.2, torch.Tensor([0])), beta=0.2)*4
#         # self.penalty = nn.functional.softplus(-20*detJ)

#         return y, detJ.abs().log()

#     def _forward_no_logDetJ(self, x):
#         res = x
#         for i, b in enumerate(self.resblock):
#             if i%2==0: ### if linear layer
#                 res = b(res)
#             else: ### if activation function
#                 res, _j = b(res)
#         return x + res#*self.scaler


#     def _inverse_yes_logDetJ(self, y):
#         g = lambda z: y - self._forward_no_logDetJ(z)
#         x = broyden(g, torch.zeros_like(y), threshold=1000, eps=1e-5)["result"]
#         _, _logdetJ = self.forward(x, True)
#         return x, -_logdetJ

#     def _inverse_no_logDetJ(self, y):
#         g = lambda z: y - self._forward_no_logDetJ(z)
#         x = broyden(g, torch.zeros_like(y), threshold=1000, eps=1e-5)["result"]
#         return x
    

#     def _update_spectral_norm_and_remove(self):
#         ### update spectral norm layer for some steps.
#         with torch.no_grad():
#             for _ in range(10):
#                 _a = torch.randn(1, self.dim)
#                 for i, b in enumerate(self.resblock):
#                     if i%2==0: ### if linear layer
#                         _a = b(_a)
#                     else: ### if activation function
#                         _a, _ = b(_a)
#             for i, b in enumerate(self.resblock):
#                 if i%2==0: ### if linear layer
#                     nn.utils.remove_spectral_norm(b)                    