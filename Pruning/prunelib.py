import torch
import torch.nn as nn
import numpy as np
import abc
from tqdm import tqdm

taylorfo_mode_config = {
    "taylorfo":{"imp_norm":"none", "grad_rescale":False, "allow_linear":True},
    "taylorfo_abs":{"imp_norm":"abs", "grad_rescale":False, "allow_linear":True},
    "taylorfo_sq":{"imp_norm":"sq", "grad_rescale":False, "allow_linear":True},

    "taylorfo_norm":{"imp_norm":"none", "grad_rescale":True, "allow_linear":True},
    "taylorfo_abs_norm":{"imp_norm":"abs", "grad_rescale":True, "allow_linear":True},
    "taylorfo_sq_norm":{"imp_norm":"sq", "grad_rescale":True, "allow_linear":True},

    "taylorfo_nolin":{"imp_norm":"none", "grad_rescale":False, "allow_linear":False},
    "taylorfo_abs_nolin":{"imp_norm":"abs", "grad_rescale":False, "allow_linear":False},
    "taylorfo_sq_nolin":{"imp_norm":"sq", "grad_rescale":False, "allow_linear":False},

    "taylorfo_norm_nolin":{"imp_norm":"none", "grad_rescale":True, "allow_linear":False},
    "taylorfo_abs_norm_nolin":{"imp_norm":"abs", "grad_rescale":True, "allow_linear":False},
    "taylorfo_sq_norm_nolin":{"imp_norm":"sq", "grad_rescale":True, "allow_linear":False}
}
taylorfo_mode_list = list(taylorfo_mode_config.keys())

class Importance(abc.ABC):

    @abc.abstractmethod
    def compute_significance(self, x, t, **kwargs):
        pass

class Importance_TaylorFO_Modified(Importance):
    
    def __init__(self, net, criterion, config=None):
        self.net = net
        self.config = config
        self.criterion = criterion
        
        self.activations = {}
        self.gradients = {}
        self.forward_hook = {}
        self.backward_hook = {}
        self.keys = []
        
        pass

    def add_hook(self):
        self.activations = {}
        self.gradients = {}
        self.forward_hook = {}
        self.backward_hook = {}
        self.keys = []
        
        for name, module in list(self.net._modules.items()):
            if isinstance(module, torch.nn.Linear):
                hook = module.register_backward_hook(self.capture_gradients)
                self.backward_hook[module] = hook
                hook = module.register_forward_hook(self.capture_inputs)
                self.forward_hook[module] = hook
                
                self.activations[module] = None
                self.gradients[module] = None
                self.keys.append(module)
        
    def remove_hook(self):
        for module in self.keys:
            hook = self.forward_hook[module]
            hook.remove()
            hook = self.backward_hook[module]
            hook.remove()
    
    def capture_inputs(self, module, inp, out):
        self.activations[module] = out.data
        
    def capture_gradients(self, module, gradi, grado):
        self.gradients[module] = grado[0]
        
    def gather_inputs_gradients(self, x, t, use_unit_grad=None):
        self.add_hook()

        self.net.zero_grad()
        y = self.net(x)
        
        if use_unit_grad:
            # grad = torch.ones_like(y)
            torch.manual_seed(use_unit_grad)
            grad = torch.randn_like(y)
            grad = grad/torch.norm(grad, dim=1, keepdim=True)
            y.backward(gradient=grad)
        else:
            error = self.criterion(y,t)
            error.backward()
            
        
        
        self.remove_hook()
        return
    
    def compute_significance(self, x, t, config=None, normalize=True, layerwise_norm=False, use_unit_grad=None):
        self.gather_inputs_gradients(x, t, use_unit_grad)
        
        if config is None:
            if self.config is None:
                raise ValueError("config is not known. Please specify the config.") 
            else:
                config = self.config
        
        ## compute importance score
        importance = []
        if config["grad_rescale"]:
            scaler = torch.norm(self.gradients[self.keys[-1]], p=2, dim=1, keepdim=True) + 1e-5

        for module in self.keys:
            z = self.activations[module] * self.gradients[module]
            if config["grad_rescale"]:
                z = z / scaler
            if config["imp_norm"] == "abs":
                z = z.abs()
            elif config["imp_norm"] == "sq":
                z = z.pow(2)

            z = z.sum(dim=0).abs()
            if not config["allow_linear"]:
                apnz = torch.sum(self.activations[module] > 0., dim=0, dtype=torch.float)
                z = z*(1-apnz) * 4 ## tried on desmos.

            # if config["layerwise_norm"]:
            if layerwise_norm:
                z = z / torch.norm(z, p=2)

            importance.append(z)

        importance = importance[:-1]
        if normalize:
            sums = 0
            count = 0
            for imp in importance:
                sums += imp.sum()
                count += len(imp)
            divider = sums/count ## total importance is number of neurons
            for i in range(len(importance)):
                importance[i] = importance[i]/divider
            
        
        self.activations = {}
        self.gradients = {}
        self.forward_hook = {}
        self.backward_hook = {}
        
        return importance

class Importance_TaylorFO_Normalized(Importance):
    
    def __init__(self, net, criterion, config=None):
        self.net = net
        self.config = config
        self.criterion = criterion
        
        self.activations = {}
        self.gradients = {}
        self.forward_hook = {}
        self.backward_hook = {}
        self.keys = []
        
        pass

    def add_hook(self):
        self.activations = {}
        self.gradients = {}
        self.forward_hook = {}
        self.backward_hook = {}
        self.keys = []
        
        for name, module in list(self.net._modules.items()):
            if isinstance(module, torch.nn.Linear):
                hook = module.register_backward_hook(self.capture_gradients)
                self.backward_hook[module] = hook
                hook = module.register_forward_hook(self.capture_inputs)
                self.forward_hook[module] = hook
                
                self.activations[module] = None
                self.gradients[module] = None
                self.keys.append(module)
        
    def remove_hook(self):
        for module in self.keys:
            hook = self.forward_hook[module]
            hook.remove()
            hook = self.backward_hook[module]
            hook.remove()
    
    def capture_inputs(self, module, inp, out):
        self.activations[module] = out.data
        
    def capture_gradients(self, module, gradi, grado):
        self.gradients[module] = grado[0]
        
    def gather_inputs_gradients(self, x, t):
        self.add_hook()

        self.net.zero_grad()
        y = self.net(x)
        
        error = self.criterion(y,t)
        error.backward()
        
        self.remove_hook()
        return
    
    def compute_significance(self, x, t, config=None, normalize=True, layerwise_norm=False):
        self.gather_inputs_gradients(x, t)
        
        if config is None:
            if self.config is None:
                raise ValueError("config is not known. Please specify the config.") 
            else:
                config = self.config
        
        ## compute importance score
        importance = [0]*len(self.keys[:-1])
        if config["grad_rescale"]:
            scaler = torch.norm(self.gradients[self.keys[-1]], p=2, dim=1, keepdim=True) + 1e-5


        for i, module in enumerate(self.keys[:-1]):
            z = self.activations[module] * self.gradients[module]
            if config["grad_rescale"]:
                z = z / scaler
            if config["imp_norm"] == "abs":
                z = z.abs()
            elif config["imp_norm"] == "sq":
                z = z.pow(2)

            # z = z.sum(dim=0).abs()
            # if not config["allow_linear"]:
            #     apnz = torch.sum(self.activations[module] > 0., dim=0, dtype=torch.float)
            #     z = z*(1-apnz) * 4 ## tried on desmos.
            
            # # if config["layerwise_norm"]:
            # if layerwise_norm:
            #     z = z / torch.norm(z, p=2)

            importance[i] = z

        shapes = [imp.shape[1] for imp in importance]
        allimp = torch.cat(importance, dim=1)
        # print(allimp.shape)

        normalizer = torch.norm(allimp, p=2, dim=1, keepdim=True)
        # print(normalizer.shape)
        allimp = allimp/normalizer
        allimp = allimp.sum(dim=0).abs()
        # print(allimp.shape)
        # return allimp
        importance = [split for split in torch.split(allimp, [256, 128, 64], dim=0)]

        # for i, imp in enumerate(importance):

            

        # importance = importance[:-1]
        if normalize:
            sums = 0
            count = 0
            for imp in importance:
                sums += imp.sum()
                count += len(imp)
            divider = sums/count ## total importance is number of neurons
            for i in range(len(importance)):
                importance[i] = importance[i]/divider
            
        self.activations = {}
        self.gradients = {}
        self.forward_hook = {}
        self.backward_hook = {}
        
        return importance

class Importance_Molchanov_2019(Importance):

    def __init__(self, net, criterion):
        self.net = net
        self.criterion = criterion
        self.keys = []
        for name, module in list(self.net._modules.items()):
            if isinstance(module, torch.nn.Linear):
                self.keys.append(module)
        
    def compute_significance(self, x, t, normalize=True, batch_size=32):

        importance = [0]*len(self.keys)
        bstrt = list(range(0, len(x), batch_size))
        bstop = bstrt[1:]+[len(x)]
        for i in tqdm(range(len(bstrt))):
            self.net.zero_grad()
            y = self.net(x[bstrt[i]:bstop[i]])
            error = self.criterion(y,t[bstrt[i]:bstop[i]])
            error.backward()
        
            ## compute importance for each input
            for j, module in enumerate(self.keys):
                z = (module.weight.data*module.weight.grad).pow(2).sum(dim=1) + \
                    (module.bias*module.bias.grad).pow(2)
                importance[j] += z
                
        ## compute mean
        for i, module in enumerate(self.keys):
            importance[i] = importance[i]/len(bstrt) 


        importance = importance[:-1]
        if normalize:
            sums = 0
            count = 0
            for imp in importance:
                sums += imp.sum()
                count += len(imp)
            divider = sums/count ## total importance is number of neurons
            for i in range(len(importance)):
                importance[i] = importance[i]/divider
            
        return importance

class Importance_APoZ(Importance):

    def __init__(self, net, criterion):
        self.net = net
        self.criterion = criterion
        
        self.activations = {}
        self.forward_hook = {}
        self.keys = []
        pass

    def add_hook(self):
        self.activations = {}
        self.forward_hook = {}
        self.keys = []
        
        for name, module in list(self.net._modules.items()):
            if isinstance(module, torch.nn.ReLU):
                hook = module.register_forward_hook(self.capture_inputs)
                self.forward_hook[module] = hook
                
                self.activations[module] = None
                self.keys.append(module)
        
    def remove_hook(self):
        for module in self.keys:
            hook = self.forward_hook[module]
            hook.remove()
    
    def capture_inputs(self, module, inp, out):
        self.activations[module] = out.data
        
    def gather_activations(self, x, t):
        self.add_hook()

        self.net.zero_grad()
        y = self.net(x)
        
        self.remove_hook()
        return
    
    def compute_significance(self, x, t, normalize=True):
        self.gather_activations(x, t)

        importance = []
        for module in self.keys:
            apnz = torch.sum(self.activations[module] > 0., dim=0, dtype=torch.float)
            importance.append(apnz)

        if normalize:
            sums = 0
            count = 0
            for imp in importance:
                sums += imp.sum()
                count += len(imp)
            divider = sums/count ## total importance is number of neurons
            for i in range(len(importance)):
                importance[i] = importance[i]/divider
            
        self.activations = {}
        self.forward_hook = {}
        return importance


class Importance_Magnitude(Importance):

    def __init__(self, net, criterion=None):
        self.net = net
        self.keys = []
        for name, module in list(self.net._modules.items()):
            if isinstance(module, torch.nn.Linear):
                self.keys.append(module)
        
    def compute_significance(self, x=None, t=None, normalize=True):

        importance = []
        for module in self.keys:
            z = torch.norm(module.weight.data, p=2, dim=1)
            importance.append(z)

        importance = importance[:-1]
        if normalize:
            sums = 0
            count = 0
            for imp in importance:
                sums += imp.sum()
                count += len(imp)
            divider = sums/count ## total importance is number of neurons
            for i in range(len(importance)):
                importance[i] = importance[i]/divider
            
        return importance


###############################################################

def get_pruning_mask(importance, num_prune):
    layer_dims = []
    for imp in importance:
        layer_dims.append(len(imp))
    
    imps = torch.ones(len(imp), max(layer_dims))*sum(layer_dims)*10
    imps_shape = imps.shape
    for i, imp in enumerate(importance):
        imps[i, :len(imp)] = imp
        
    imps = imps.reshape(-1)
    indices = torch.argsort(imps)
    imps[indices[:num_prune]] = -1.
    imps = imps.reshape(imps_shape)
    
    mask = (imps>=0).type(torch.float)
    masks = []
    for i, imp in enumerate(importance):
        masks.append(mask[i, :len(imp)])
    return masks


