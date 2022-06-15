import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

def reshape_1d(matrix, m):
    # If not a nice multiple of m, fill with zeroes.
    if matrix.shape[1] % m > 0:
        mat = torch.cuda.FloatTensor(matrix.shape[0], matrix.shape[1] + (m-matrix.shape[1]%m)).fill_(0)
        mat[:, :matrix.shape[1]] = matrix
        shape = mat.shape
        return mat.view(-1,m),shape
    else:
        return matrix.view(-1,m), matrix.shape

def default_conv(in_channels, out_channels, kernel_size, bias=True, M=None, args=None):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class SparseConv(nn.Conv2d):
    """" implement N:M sparse convolution layer """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', N=2, M=8, args=None, **kwargs):
        self.N = N
        self.M = M
        
        if padding!=0 and kernel_size//2!=padding:
            pass
        else:
            padding = kernel_size//2

        super(SparseConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **kwargs)
        
        self.priority_scores = nn.Parameter(torch.ones(self.M-1), requires_grad=True)
        
        self.op_per_position = (kernel_size ** 2) * in_channels * out_channels

        # Pruning units
        # Since pruning units are updated at the begining of an epoch, they should be saved to reproduce final performance after training
        self.pruning_units = nn.Parameter(torch.ones(self.M, self.weight.shape[0], self.weight.shape[1], self.weight.shape[2], self.weight.shape[3]), requires_grad=False)

        self.final_binary_scores = None
        self.final_priority_scores = None

    def compute_costs(self):
        scores = torch.clamp(self.priority_scores, 0, 1.03)
        # Priority-Ordered Pruning
        scores = torch.cumprod(scores, dim=0)

        binary_scores = (scores > 0.5).int()
        # Straight-Through Estimator
        binary_scores = binary_scores - scores.detach() + scores

        current_N = 1 + binary_scores.sum()
        
        spatial_size = self.h*self.w

        pruned_costs = (current_N / self.M) * self.op_per_position * spatial_size
        original_costs = self.op_per_position * spatial_size
        
        return pruned_costs, original_costs

    # Determine pruning units using magnitude of weights
    # Reference: https://github.com/NM-sparsity/NM-sparsity
    def update_pruning_units(self):
        weight = self.weight.clone()
        shape = weight.shape
        tensor_type = weight.type()
        weight = weight.float().contiguous()

        weight = weight.permute(2,3,0,1).contiguous().view(shape[2]*shape[3]*shape[0], shape[1])
        mat, mat_shape = reshape_1d(weight, self.M)

        _,idxes = abs(mat).sort()
        ws = []
        mask = torch.cuda.IntTensor(weight.shape).fill_(0).view(-1, self.M)
        for n in range(1,self.M+1):
            mask.scatter_(1,idxes[:,-n].unsqueeze(-1),n)

        mask.view(weight.shape)
        mask = mask.view(shape[2], shape[3], shape[0], shape[1]).permute(2,3,0,1).contiguous()
        mask = mask.view(shape).type(tensor_type)
        
        pruning_units = []
        for i in range(self.M):
            # preserve only (i+1)-th largest weight (w.r.t magnitude) in each group of M weights
            preserved_weights = (mask == i+1).int()
            pruning_units.append(preserved_weights)
        
        self.pruning_units.data = torch.stack(pruning_units, 0)

    # Fix scores when the target budget is reached
    def fix_scores(self):
        scores = torch.clamp(self.priority_scores, 0, 1.03)
        prob = torch.cumprod(scores, 0)
        binary = (prob > 0.5).int()
        self.final_binary_scores = binary
        self.final_priority_scores = copy.deepcopy(self.priority_scores)
        
        self.priority_scores.requires_grad = False
    
    # To eliminate the effect of moving average of optimizer
    def reassign_scores(self):
        if self.final_priority_scores is not None:
            self.priority_scores.data = self.final_priority_scores.data

    # Fix pruning masks for inference
    def get_fixed_mask(self):
        scores = torch.clamp(self.priority_scores, 0, 1.03)
        prob = torch.cumprod(scores, 0)
        binary = (prob > 0.5).int()

        mask = self.pruning_units[0]
        mask = mask + torch.sum(self.pruning_units[1:] * binary.view(-1,1,1,1,1), dim=0)
        self.mask= mask

    def forward(self, x):
        if self.training:
            if self.final_binary_scores is None:
                scores = torch.clamp(self.priority_scores, 0, 1.03)
                prob = torch.cumprod(scores, 0)
                binary = (prob > 0.5).int()
                mn_mask = binary - prob.detach() + prob

                mask = self.pruning_units[0]
                mask = mask + torch.sum(self.pruning_units[1:] * mn_mask.view(-1,1,1,1,1), dim=0)
            else:
                mask = self.pruning_units[0]
                mask = mask + torch.sum(self.pruning_units[1:] * self.final_binary_scores.view(-1,1,1,1,1), dim=0)            
            
            out = F.conv2d(x, self.weight * mask, self.bias, self.stride, self.padding, self.dilation, self.groups) 
        else:
            out = F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups) 

        self.h = out.shape[-2]
        self.w = out.shape[-1]

        return out

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1, args=None):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias, M=args.M))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True, args=None):
        
        if conv == 'default':
            conv = default_conv

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias=bias, M=args.M))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias=bias, M=args.M))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


