###### MODIFIED from 
## https://github.com/chenyaofo/CIFAR-pretrained-models

import torch
import torch.nn as nn
from typing import Union, Tuple

__all__ = ['CifarResNet', 'cifar_resnet20', 'cifar_resnet32', 'cifar_resnet44', 'cifar_resnet56']


### shift normalized dists towards 0 for sparse activation with exponential
# class DistanceTransform(nn.Module):
    
#     def __init__(self, input_dim, num_centers, p=2, bias=True):
#         super().__init__()
#         self.input_dim = input_dim
#         self.num_centers = num_centers
#         self.p = p
        
# #         self.centers = torch.randn(num_centers, input_dim)/2.
#         self.centers = torch.rand(num_centers, input_dim)
#         self.centers = nn.Parameter(self.centers)
        
#         self.scaler = nn.Parameter(torch.ones(1, num_centers)*2/3)
#         self.bias = nn.Parameter(torch.ones(1, num_centers)*-0.1) if bias else None
        
#     def forward(self, x):
# #         x = x[:, :self.input_dim]
#         dists = torch.cdist(x, self.centers)
        
#         ### normalize similar to UMAP
# #         dists = dists-dists.min(dim=1, keepdim=True)[0]
#         dists = dists-dists.mean(dim=1, keepdim=True)
#         dists = dists/dists.std(dim=1, keepdim=True)

#         dists = torch.exp((-dists-3)*self.scaler)
#         if self.bias is not None: dists = dists+self.bias
#         return dists


class DistanceTransformBase(nn.Module):
    
    def __init__(self, input_dim, num_centers, p=2):
        super().__init__()
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.p = p
        
        self.centers = torch.randn(num_centers, input_dim)/3.
#         self.centers = torch.rand(num_centers, input_dim)
        self.centers = nn.Parameter(self.centers)
    
    def forward(self, x):
        dists = torch.cdist(x, self.centers, p=self.p)
        
        ### normalize similar to UMAP
#         dists = dists-dists.min(dim=1, keepdim=True)[0]
#         dists = dists-dists.mean(dim=1, keepdim=True)
#         dists = dists/dists.std(dim=1, keepdim=True)

        return dists
    
    def set_centroid_to_data_randomly(self, data_loader):
        indices = np.random.permutation(len(data_loader.dataset.data))[:self.centers.shape[0]]
        self.centers.data = data_loader.dataset.data[indices].to(self.centers.device)
        pass
    
    def set_centroid_to_data_maxdist(self, data_loader):
        ## sample N points
        N = self.centers.shape[0]
        new_center = torch.empty_like(self.centers)
        min_dists = torch.empty(N)
        count = 0
        for i, (xx, _) in enumerate(tqdm(data_loader)):
            if count < N:
                if N-count < batch_size:
                    #### final fillup
                    new_center[count:count+N-count] = xx[:N-count]
                    xx = xx[N-count:]
                    dists = torch.cdist(new_center, new_center)+torch.eye(N)*1e5
                    min_dists = dists.min(dim=0)[0]
                    count = N

                else:#### fill the center
                    new_center[count:count+len(xx)] = xx
                    count += len(xx)
                    continue

            ammd = min_dists.argmin()
            for i, x in enumerate(xx):
                dists = torch.norm(new_center-x, dim=1)
                md = dists.min()
                if md > min_dists[ammd]:
                    min_dists[ammd] = md
                    new_center[ammd] = x
                    ammd = min_dists.argmin()
        self.centers.data = new_center.to(self.centers.device)
        pass
        
    
    def set_centroid_to_data(self, data_loader):
        new_center = self.centers.data.clone()
        min_dists = torch.ones(self.centers.shape[0])*1e9

        for xx, _ in data_loader:

            dists = torch.cdist(xx, self.centers.data)
            ### min dist of each center to the data points
            min_d, arg_md = dists.min(dim=0)

            ### dont allow same point to be assigned as closest to multiple centroid
            occupied = []
            for i in np.random.permutation(len(arg_md)):
        #     for i, ind in enumerate(arg_md):
                ind = arg_md[i]
                if ind in occupied:
                    min_d[i] = min_dists[i]
                    arg_md[i] = -1
                else:
                    occupied.append(ind)

            ### the index of centroids that have new min_dist
            idx = torch.nonzero(min_d<min_dists).reshape(-1)

            ### assign new_center to the nearest data point
            new_center[idx] = xx[arg_md[idx]]
            min_dists[idx] = min_d[idx]
            
        self.centers.data = new_center.to(self.centers.device)
        pass

### shift normalized dists towards 0 for sparse activation with exponential
class DistanceTransform_Exp(DistanceTransformBase):
    
    def __init__(self, input_dim, num_centers, p=2, bias=False, eps=1e-5):
        super().__init__(input_dim, num_centers, p=2)
        
        self.scaler = nn.Parameter(torch.ones(1, num_centers)*3/3)
#         self.bias = nn.Parameter(torch.ones(1, num_centers)*-0.1) if bias else None
        self.bias = nn.Parameter(torch.ones(1, num_centers)*0) if bias else None
        self.eps = eps
        
    def forward(self, x):
        dists = super().forward(x)
        
        ### normalize similar to UMAP
#         dists = dists-dists.min(dim=1, keepdim=True)[0]
        dists = dists-dists.mean(dim=1, keepdim=True)
        dists = dists/torch.sqrt(dists.var(dim=1, keepdim=True)+self.eps)
#         a = ((-dists-2)*self.scaler).data.cpu().numpy()
#         print(a.mean(), a.std(), a.min(), a.max())
        dists = torch.exp((-dists-2)*self.scaler)
#         dists = torch.softmax((-dists-3)*self.scaler, dim=1)
        if self.bias is not None: dists = dists+self.bias
        return dists


### shift normalized dists towards 0 for sparse activation with exponential
class DistanceTransform_MinExp(DistanceTransformBase):
    
    def __init__(self, input_dim, num_centers, p=2, bias=False, eps=1e-5):
        super().__init__(input_dim, num_centers, p=2)
        
        self.scaler = nn.Parameter(torch.ones(1, num_centers)*6/3)
        self.scaler.requires_grad = False
#         self.bias = nn.Parameter(torch.ones(1, num_centers)*-0.1) if bias else None
        self.bias = nn.Parameter(torch.ones(1, num_centers)*0) if bias else None
        self.eps = eps
        
    def forward(self, x):
        dists = super().forward(x)
        
        ### normalize similar to UMAP
        dists = dists-dists.min(dim=1, keepdim=True)[0]
#         dists = dists-dists.mean(dim=1, keepdim=True)
        dists = dists/torch.sqrt(dists.var(dim=1, keepdim=True)+self.eps)

        dists = torch.exp(-dists*self.scaler)
#         dists = torch.softmax(-dists*self.scaler, dim=1)
        if self.bias is not None: dists = dists+self.bias
        return dists
    

## bias to basic dist
class DistanceTransform(nn.Module):
    
    def __init__(self, input_dim, num_centers, p=2, bias=True):
        super().__init__()
        bias=False
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.p = p
        self.bias = nn.Parameter(torch.zeros(1, num_centers)) if bias else None
        
        self.centers = torch.rand(num_centers, input_dim)
        self.centers = nn.Parameter(self.centers)
        
    def forward(self, x):
#         x = x[:, :self.input_dim]
        dists = torch.cdist(x, self.centers, p=self.p)
        
        ### normalize similar to UMAP
#         dists = dists-dists.min(dim=1, keepdim=True)[0]
#         dists = dists-dists.max(dim=1, keepdim=True)[0]
        dists = dists-dists.mean(dim=1, keepdim=True)
        dists = dists/dists.std(dim=1, keepdim=True)

        if self.bias is not None: dists = dists+self.bias
        return dists
    
    
class Conv2D_DT(nn.Module):
    
    def __init__(self, in_channels, out_channels,
                 kernel_size: Union[int, Tuple[int, ...]],
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 bias = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding  # format for padding -> l, r, t, b
        self.stride = stride
        self._preprocess_()
        
        self.unfold = nn.Unfold(self.kernel_size, self.dilation, self.padding, self.stride)
        self.dt = DistanceTransform(self.kernel_size[0]*self.kernel_size[1]*in_channels, out_channels)
    
    def _preprocess_(self):
        if not isinstance(self.kernel_size, (tuple, list)):
            self.kernel_size = (self.kernel_size, self.kernel_size)
        if not isinstance(self.dilation, (tuple, list)):
            self.dilation = (self.dilation, self.dilation)
        if not isinstance(self.stride, (tuple, list)):
            self.stride = (self.stride, self.stride)
#         if not isinstance(self.padding, (tuple, list)):
#             self.padding = (self.padding, self.padding, self.padding, self.padding)
#         assert len(self.padding) == 4, 'padding must be specified for all sides of image'
        if not isinstance(self.padding, (tuple, list)):
            self.padding = (self.padding, self.padding)
        assert len(self.padding) == 2, 'padding must be specified for TB, LR'
        return
        
    def _get_output_size_(self, inputH, inputW):
        ### input change due to padding
        inputH, inputW = (inputH+2*self.padding[0], inputW+2*self.padding[1])
        
        ### kernel change due to dilation
        kH = (self.kernel_size[0]-1)*self.dilation[0]+1
        kW = (self.kernel_size[1]-1)*self.dilation[1]+1
        
        oH = (inputH-kH)/self.stride[0]+1
        oW = (inputW-kW)/self.stride[1]+1
        return (int(oH), int(oW))
        
    def forward(self, x):
        c = x.shape
        x = self.unfold(x).transpose(1,2).reshape(-1, self.dt.input_dim)
        x = self.dt(x).view(c[0], -1, self.dt.num_centers).transpose(1,2)\
                    .view(c[0], -1, *self._get_output_size_(c[2], c[3]))
        return x
    
class ScaleShift(nn.Module):
    
    def __init__(self, input_dim):
        super().__init__()
        self.scaler = nn.Parameter(torch.ones(1, input_dim))
        self.shifter = nn.Parameter(torch.zeros(1, input_dim))
        
    def forward(self, x):
        return x*self.scaler+self.shifter

    

class ChannelNorm2D(nn.Module):
    
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        x = x-x.mean(dim=1, keepdim=True)
        x = x/torch.sqrt(x.var(dim=1, keepdim=True)+self.eps)
        return x
    

def conv3x3_linear(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1_linear(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3_linear2(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        ChannelNorm2D()
    )

def conv1x1_linear2(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
        ChannelNorm2D()
    )

def conv3x3_distance(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2D_DT(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1_distance(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2D_DT(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

conv3x3 = conv3x3_linear
conv1x1 = conv1x1_linear

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, transform="distance"):
        '''
        transform: Either "linear" or "distance"
        '''
        super(CifarResNet, self).__init__()
        global conv3x3, conv1x1
        
        if transform == "distance":
            conv3x3 = conv3x3_distance
            conv1x1 = conv1x1_distance
            
        elif transform == "linear":
            conv3x3 = conv3x3_linear
            conv1x1 = conv1x1_linear
        elif transform == "linear2":
            conv3x3 = conv3x3_linear2
            conv1x1 = conv1x1_linear2
        else:
            raise ValueError("transform could not be identified")
        
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if transform == "linear":
            self.fc = nn.Linear(64 * block.expansion, num_classes)
        elif transform == "linear2":
            self.fc = nn.Sequential(
                nn.Linear(64 * block.expansion, num_classes),
                nn.LayerNorm(num_classes)
            )
        else:
            self.fc = nn.Sequential(
                DistanceTransform(64 * block.expansion, num_classes),
                ScaleShift(num_classes)
            )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def cifar_resnet20(**kwargs):
    model = CifarResNet(BasicBlock, [3, 3, 3], **kwargs)
    return model


def cifar_resnet32(**kwargs):
    model = CifarResNet(BasicBlock, [5, 5, 5], **kwargs)
    return model


def cifar_resnet44(**kwargs):
    model = CifarResNet(BasicBlock, [7, 7, 7], **kwargs)
    return model


def cifar_resnet56(**kwargs):
    model = CifarResNet(BasicBlock, [9, 9, 9], **kwargs)
    return model