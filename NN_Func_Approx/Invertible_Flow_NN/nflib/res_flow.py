import torch
from torch._C import Value
import torch.nn as nn
from .flows import Flow
from .utils.broyden import broyden
from .utils.mixed_lipschitz import InducedNormLinear, update_lipschitz

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
    

class TanhLU(nn.Module):

    def forward(self, x):
        y = x
        dy = torch.ones_like(x)
        
        mask = x < 0
        y[mask] = torch.tanh(x[mask])
        dy[mask] =(1 - y[mask] ** 2)
        return y, dy


class Swish(nn.Module):

    def __init__(self, beta=2.5): #beta=0.8
        super().__init__()
        self.beta = nn.Parameter(torch.Tensor([beta]))

    def forward(self, x):
        z = torch.sigmoid(self.beta*x)
        y = x * z
        
        by = self.beta*y
        j = by+z*(1-by)
        return y/1.1, j/1.1


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

class ResidualFlow_x(Flow):
    def __init__(self, dim, hidden_dims:list, activation=Swish, scaler=0.97, lipschitz_iter=5, n_iter=200, reverse=False):
        super().__init__()
        assert activation in [ReLU, LeakyReLU, Swish], "Use ReLU or LeakyReLU or Swish"
        self.n_iter = n_iter
        self.lipschitz_iter = lipschitz_iter
        self.reverse = reverse
        self.dim = dim
        
        if isinstance(hidden_dims, list) or isinstance(hidden_dims, tuple):
            assert len(hidden_dims)>0, "Dims should include N x hidden units"
            dims = [dim, *hidden_dims, dim]
            resblock = []
            for i in range(0, len(dims)-1):
                linear = InducedNormLinear(dims[i], dims[i+1],
                                            coeff=scaler, n_iterations=lipschitz_iter)
                resblock.append(linear)
                actf = activation()
                resblock.append(actf)
            self.resblock = nn.ModuleList(resblock[:-1])
        elif isinstance(hidden_dims, nn.Module):
            self.resblock = hidden_dims
        else:
            raise ValueError("hidden_dims must either be network or hidden dims of network")
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
        return x + res,\
                torch.det(torch.eye(x.shape[1])+ J).abs().log()


    # def _forward_yes_logDetJ(self, x):
    #     if not x.requires_grad:
    #         x = torch.autograd.Variable(x, requires_grad=True)
    #     res = x
    #     for i, b in enumerate(self.resblock):
    #         if i%2==0: ### if linear layer
    #             res = b(res)
    #         else: ### if activation function
    #             res, _j = b(res)

    #     y = x + res
    #     J = jacobian(y, x, True)
    #     return y, torch.det(J).abs().log()

    def _forward_no_logDetJ(self, x):
        res = x
        for i, b in enumerate(self.resblock):
            if i%2==0: ### if linear layer
                res = b(res)
            else: ### if activation function
                res, _j = b(res)
        return x + res


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
                    # if i%2==0: ### if Induced Norm Linear layer
                    if isinstance(b, InducedNormLinear):
                        b.compute_weight(update=True, n_iterations=self.lipschitz_iter*2)


# class ResidualFlow2(Flow):
#     def __init__(self, dim, resblock, activation=ReLU, scaler=0.97, lipschitz_iter=5, n_iter=200, reverse=False):
#         super().__init__()
#         assert activation is ReLU or activation is LeakyReLU, "Use ReLU or LeakyReLU"
#         self.n_iter = n_iter
#         self.lipschitz_iter = lipschitz_iter
#         self.reverse = reverse
#         self.dim = dim
        
#         self.resblock = resblock
#         self.scaler = scaler
#         self._update_spectral_norm_init()

#     def forward(self, x, logDetJ:bool=False):
#         if self.reverse:
#             return self._inverse_yes_logDetJ(x) if logDetJ else self._inverse_no_logDetJ(x)
#         return self._forward_yes_logDetJ(x) if logDetJ else self._forward_no_logDetJ(x)
    
#     def inverse(self, z, logDetJ:bool=False):
#         if self.reverse:
#             return self._forward_yes_logDetJ(z) if logDetJ else self._forward_no_logDetJ(z)
#         return self._inverse_yes_logDetJ(z) if logDetJ else self._inverse_no_logDetJ(z)


#     def _forward_yes_logDetJ(self, x):
#         if not x.requires_grad:
#             x = torch.autograd.Variable(x, requires_grad=True)

#         res = self.resblock(x)
#         y = x + res
#         J = jacobian(y, x, True)
#         return y, torch.det(J).abs().log()

#     def _forward_no_logDetJ(self, x):
#         res = self.resblock(x)
#         return x + res


#     def _inverse_yes_logDetJ(self, y):
#         g = lambda z: y - self._forward_no_logDetJ(z)
#         x = broyden(g, torch.zeros_like(y), threshold=1000, eps=1e-5)["result"]
#         _, _logdetJ = self.forward(x, True)
#         return x, -_logdetJ

#     def _inverse_no_logDetJ(self, y):
#         g = lambda z: y - self._forward_no_logDetJ(z)
#         x = broyden(g, torch.zeros_like(y), threshold=1000, eps=1e-5)["result"]
#         return x

#     def _update_spectral_norm_init(self):
#         ### update spectral norm layer for some steps.
#         update_lipschitz(self.resblock, 5)


class ResidualFlow(Flow):
    def __init__(self, dim, hidden_dims:list, activation=Swish, scaler=0.97, lipschitz_iter=2, inverse_iter=200, reverse=False):
        super().__init__()
        assert len(hidden_dims)>0, "Dims should include N x hidden units"
        # assert activation in [ReLU, LeakyReLU, Swish], "Use ReLU or LeakyReLU or Swish"
        self.n_iter = inverse_iter

        self.reverse = reverse
        self.dim = dim
        dims = [dim, *hidden_dims, dim]
        resblock = []
        for i in range(0, len(dims)-1):
            linear = nn.utils.spectral_norm(nn.Linear(dims[i], dims[i+1]), n_power_iterations=lipschitz_iter)
            resblock.append(linear)
            actf = activation()
            resblock.append(actf)
        self.resblock = nn.ModuleList(resblock[:-1])
        self.scaler = scaler
        # self._update_spectral_norm_init()

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
        # return x + res*self.scaler,\
        #         torch.det(torch.eye(x.shape[1])+ J*self.scaler).abs().log()
        return x + res*self.scaler,\
                torch.det(torch.eye(x.shape[1])+ J*self.scaler).log()

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
    

    def _update_spectral_norm_init(self):
        ### update spectral norm layer for some steps.
        with torch.no_grad():
            for _ in range(10):
                _a = torch.randn(1, self.dim)
                for i, b in enumerate(self.resblock):
                    if i%2==0: ### if linear layer
                        _a = b(_a)
                    else: ### if activation function
                        _a, _ = b(_a)



class ConvResidualFlow(Flow):
    def __init__(self, in_channel, channels:list, kernels=3, activation=Swish, scaler=0.97, lipschitz_iter=2, inverse_iter=500, reverse=False):
        super().__init__()
        assert len(channels)>0, "Dims should include N x hidden units"
        # assert activation in [ReLU, LeakyReLU, Swish], "Use ReLU or LeakyReLU or Swish"
        self.n_iter = inverse_iter

        self.reverse = reverse

        self.in_channel = in_channel
        self.channels = channels
        self.activation = activation

        if isinstance(kernels, int):
            self.kernels = [kernels for _ in range(len(channels)+1)]
        else:
            assert len(channels)+1 == len(kernels), "The length of kernels must be one greater than number of channels"
            self.kernels = kernels

        self.paddings = [(kernel-1)//2 for kernel in self.kernels]

        layers = []
        channels = [int(self.in_channel)]+\
                    self.channels + [int(self.in_channel)]
        for i in range(len(channels)-1):
            conv = nn.Conv2d(channels[i], channels[i+1], self.kernels[i], 
                                    padding=self.paddings[i])
            conv = nn.utils.spectral_norm(conv, n_power_iterations=lipschitz_iter)
            layers.append(conv)
            layers.append(self.activation())
        
        layers = layers[:-1]
        self.resblock = nn.ModuleList(layers)
        self.scaler = scaler
        self._update_spectral_norm_init(10)

    def forward(self, x, logDetJ:bool=False):
        if self.reverse:
            return self._inverse_yes_logDetJ(x) if logDetJ else self._inverse_no_logDetJ(x)
        return self._forward_yes_logDetJ(x) if logDetJ else self._forward_no_logDetJ(x)
    
    def inverse(self, z, logDetJ:bool=False):
        if self.reverse:
            return self._forward_yes_logDetJ(z) if logDetJ else self._forward_no_logDetJ(z)
        return self._inverse_yes_logDetJ(z) if logDetJ else self._inverse_no_logDetJ(z)

    def _forward_yes_logDetJ(self, x):
        if not x.requires_grad:
            x = torch.autograd.Variable(x, requires_grad=True)
        res = x
        for i, b in enumerate(self.resblock):
            if i%2==0: ### if conv layer
                res = b(res)
            else: ### if activation function
                res, _j = b(res)
        y = x + res*self.scaler
        J = jacobian(y, x, True)
        print("debug !! Computing jacobian, computationally expensive")
        return y, torch.det(J).abs().log()

    def _forward_no_logDetJ(self, x):
        res = x
        for i, b in enumerate(self.resblock):
            if i%2==0: ### if conv layer
                res = b(res)
            else: ### if activation function
                res, _j = b(res)
        return x + res*self.scaler

    def _fixed_point_(self, y):
        # inversion of ResNet-block (fixed-point iteration) -- copied from
        # https://github.com/jhjacobsen/invertible-resnet/blob/master/models/conv_iResNet.py
        x = y.data.clone()
        for iter_index in range(self.n_iter):
            ### forward propagation
            summand = x
            for i, b in enumerate(self.resblock):
                if i%2==0: ### if conv layer
                    summand = b(summand)
                else: ### if activation function
                    summand, _ = b(summand)
            ## fixed point iteration
            x = y - summand*self.scaler
        del y, summand
        return x

    def _inverse_yes_logDetJ(self, y):
        # g = lambda z: y - self._forward_no_logDetJ(z)
        # x = broyden(g, torch.zeros_like(y), threshold=self.n_iter, eps=1e-7)["result"]
        # x = broyden(g, y.data.clone(), threshold=self.n_iter, eps=1e-7)["result"]

        x = self._fixed_point_(y)
        _, _logdetJ = self.forward(x, True)
        return x, -_logdetJ

    def _inverse_no_logDetJ(self, y):
        g = lambda z: y - self._forward_no_logDetJ(z)
        # x = broyden(g, torch.zeros_like(y), threshold=self.n_iter, eps=1e-7)["result"]
        # x = broyden(g, y.data.clone(), threshold=self.n_iter, eps=1e-7)["result"]

        return self._fixed_point_(y)
    
    def _update_spectral_norm_init(self, N=10):
        ### update spectral norm layer for some steps.
        with torch.no_grad():
            for _ in range(N):
                _a = torch.randn(1, self.in_channel, 256, 256)
                for i, b in enumerate(self.resblock):
                    if i%2==0: ### if conv layer
                        _a = b(_a)
                    else: ### if activation function
                        _a, _ = b(_a)



class InvertiblePooling(Flow):
    def __init__(self, block_size=2):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def _forward_yes_logDetJ(self, x):
        ### x -> [N, C, H, W]
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, d, new_h, new_w = x.shape[0], x.shape[1], x.shape[2] // bl, x.shape[3] // bl
        y = x.reshape(bs, d, new_h, bl, new_w, bl).permute(0, 3, 5, 1, 2, 4).reshape(bs, d * bl_sq, new_h, new_w)
        return y, 0

    def _forward_no_logDetJ(self, x):
        return self._forward_yes_logDetJ(x)[0]

    def _inverse_yes_logDetJ(self, y):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, new_d, h, w = y.shape[0], y.shape[1] // bl_sq, y.shape[2], y.shape[3]
        x = y.reshape(bs, bl, bl, new_d, h, w).permute(0, 3, 4, 1, 5, 2).reshape(bs, new_d, h * bl, w * bl)
        return x, 0

    def _inverse_no_logDetJ(self, y):
        return self._inverse_yes_logDetJ(y)[0]


class Flatten(Flow):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size

    def _forward_yes_logDetJ(self, x):
        y = x.reshape(-1, self.img_size[0]*self.img_size[1]*self.img_size[2])
        return y, 0

    def _forward_no_logDetJ(self, x):
        return self._forward_yes_logDetJ(x)[0]

    def _inverse_yes_logDetJ(self, y):
        x = y.reshape(-1, *self.img_size)
        return x, 0

    def _inverse_no_logDetJ(self, y):
        return self._inverse_yes_logDetJ(y)[0]







####### Using diffent mechanism for inverting convolution original paper method
from .utils.spectral_norm_conv_inplace import spectral_norm_conv, SpectralNormConv
from .utils.spectral_norm_fc import spectral_norm_fc, SpectralNorm

def remove_spectral_norm(module, name='weight'):
    r"""Removes the spectral normalization reparameterization from a module.
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
    Example:
        >>> m = spectral_norm_conv(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNormConv, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}".format(
        name, module))


class ConvResidualFlow_v2(Flow):
    '''
    image_dim: in_channel, h, w
    '''
    def __init__(self, image_dim, channels:list, kernels=3, activation=Swish, scaler=0.97, lipschitz_iter=2, inverse_iter=100, reverse=False):
        super().__init__()
        assert len(channels)>0, "Dims should include N x hidden units"
        # assert activation in [ReLU, LeakyReLU, Swish], "Use ReLU or LeakyReLU or Swish"
        self.n_iter = inverse_iter

        self.reverse = reverse

        self.in_channel = image_dim[0]
        self.channels = channels
        self.activation = activation

        if isinstance(kernels, int):
            self.kernels = [kernels for _ in range(len(channels)+1)]
        else:
            assert len(channels)+1 == len(kernels), "The length of kernels must be one greater than number of channels"
            self.kernels = kernels

        self.paddings = [(kernel-1)//2 for kernel in self.kernels]

        layers = []
        channels = [int(self.in_channel)]+\
                    self.channels + [int(self.in_channel)]
        for i in range(len(channels)-1):
            conv = nn.Conv2d(channels[i], channels[i+1], self.kernels[i], 
                                    padding=self.paddings[i])
            # conv = nn.utils.spectral_norm(conv, n_power_iterations=lipschitz_iter)
            if self.kernels[i] == 1:
                conv = spectral_norm_fc(conv, scaler, n_power_iterations=lipschitz_iter)
            else:
                idim = (channels[i], image_dim[1], image_dim[2])
                conv = spectral_norm_conv(conv, scaler, idim, n_power_iterations=lipschitz_iter)
            layers.append(conv)
            layers.append(self.activation())
        
        layers = layers[:-1]
        self.resblock = nn.ModuleList(layers)
        # self.scaler = scaler
        # self._update_spectral_norm_init(10)

    def forward(self, x, logDetJ:bool=False):
        if self.reverse:
            return self._inverse_yes_logDetJ(x) if logDetJ else self._inverse_no_logDetJ(x)
        return self._forward_yes_logDetJ(x) if logDetJ else self._forward_no_logDetJ(x)
    
    def inverse(self, z, logDetJ:bool=False):
        if self.reverse:
            return self._forward_yes_logDetJ(z) if logDetJ else self._forward_no_logDetJ(z)
        return self._inverse_yes_logDetJ(z) if logDetJ else self._inverse_no_logDetJ(z)

    def _forward_yes_logDetJ(self, x):
        if not x.requires_grad:
            x = torch.autograd.Variable(x, requires_grad=True)
        res = x
        for i, b in enumerate(self.resblock):
            if i%2==0: ### if conv layer
                res = b(res)
            else: ### if activation function
                res, _j = b(res)
        y = x + res
        J = jacobian(y, x, True)
        print("debug !! Computing jacobian, computationally expensive")
        return y, torch.det(J).abs().log()

    def _forward_no_logDetJ(self, x):
        res = x
        for i, b in enumerate(self.resblock):
            if i%2==0: ### if conv layer
                res = b(res)
            else: ### if activation function
                res, _j = b(res)
        return x + res

    def _fixed_point_(self, y):
        # inversion of ResNet-block (fixed-point iteration) -- copied from
        # https://github.com/jhjacobsen/invertible-resnet/blob/master/models/conv_iResNet.py
        x = y.data.clone()
        for iter_index in range(self.n_iter):
            ### forward propagation
            summand = x
            for i, b in enumerate(self.resblock):
                if i%2==0: ### if conv layer
                    summand = b(summand)
                else: ### if activation function
                    summand, _ = b(summand)
            ## fixed point iteration
            x = y - summand
        del y, summand
        return x

    def _inverse_yes_logDetJ(self, y):
        # g = lambda z: y - self._forward_no_logDetJ(z)
        # x = broyden(g, torch.zeros_like(y), threshold=self.n_iter, eps=1e-7)["result"]
        # x = broyden(g, y.data.clone(), threshold=self.n_iter, eps=1e-7)["result"]

        x = self._fixed_point_(y)
        _, _logdetJ = self.forward(x, True)
        return x, -_logdetJ

    def _inverse_no_logDetJ(self, y):
        # g = lambda z: y - self._forward_no_logDetJ(z)
        # x = broyden(g, torch.zeros_like(y), threshold=self.n_iter, eps=1e-7)["result"]
        # x = broyden(g, y.data.clone(), threshold=self.n_iter, eps=1e-7)["result"]

        return self._fixed_point_(y)
