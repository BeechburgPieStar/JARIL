import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.init import normal_

# hyper-parameters
OUT_NUM = 3

def ifnone(a, b):
    # From fastai.fastcore
    "`b` if `a` is None else `a`"
    return b if a is None else a

def noop(x=None, *args, **kwargs):
    "Do nothing"
    return x

def Norm(nf, ndim=1, norm='Batch', zero_norm=False, init=True, **kwargs):
    "Norm layer with `nf` features and `ndim` with auto init."
    assert 1 <= ndim <= 3
    nl = getattr(nn, f"{snake2camel(norm)}Norm{ndim}d")(nf, **kwargs)
    if nl.affine and init:
        nl.bias.data.fill_(1e-3)
        nl.weight.data.fill_(0. if zero_norm else 1.)
    return nl


class Add(nn.Module):
    def forward(self, x, y):
        return x.add(y)
    def __repr__(self):
        return f'{self.__class__.__name__}'

class Concat(nn.Module):
    def __init__(self, dim=1): 
        super(Concat, self).__init__()
        self.dim = dim
    def forward(self, *x): 
        return torch.cat(*x, dim=self.dim)
    def __repr__(self): 
        return f'{self.__class__.__name__}(dim={self.dim})'

class Squeeze(nn.Module):
    def __init__(self, dim=-1): 
        super(Squeeze, self).__init__()
        self.dim = dim
    def forward(self, x): 
        return x.squeeze(dim=self.dim)
    def __repr__(self): 
        return f'{self.__class__.__name__}(dim={self.dim})'

class Reshape(nn.Module):
    def __init__(self, *shape): 
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.reshape(x.shape[0], -1) if not self.shape else x.reshape(-1) if self.shape == (-1,) else x.reshape(x.shape[0], *self.shape)
    def __repr__(self): 
        return f"{self.__class__.__name__}({', '.join(['bs'] + [str(s) for s in self.shape])})"

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class GAP1d(nn.Module):
    "Global Adaptive Pooling + Flatten"
    def __init__(self, output_size=1):
        super(GAP1d, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = Flatten()
    def forward(self, x):
        return self.flatten(self.gap(x))

def same_padding1d(seq_len, ks, stride=1, dilation=1):
    "Same padding formula as used in Tensorflow"
    p = (seq_len - 1) * stride + (ks - 1) * dilation + 1 - seq_len
    return p // 2, p - p // 2

class Pad1d(nn.ConstantPad1d):
    def __init__(self, padding, value=0.):
        super().__init__(padding, value)

class SameConv1d(nn.Module):
    "Conv1d with padding='same'"
    def __init__(self, ni, nf, ks=3, stride=1, dilation=1, **kwargs):
        super(SameConv1d, self).__init__()
        self.ks, self.stride, self.dilation = ks, stride, dilation
        self.conv1d_same = nn.Conv1d(ni, nf, ks, stride=stride, dilation=dilation, **kwargs)
        self.weight = self.conv1d_same.weight
        self.bias = self.conv1d_same.bias
        self.pad = Pad1d

    def forward(self, x):
        self.padding = same_padding1d(x.shape[-1], self.ks, dilation=self.dilation)
        return self.conv1d_same(self.pad(self.padding)(x))


def Conv1d(ni, nf, kernel_size=None, ks=None, stride=1, padding='same', dilation=1, **kwargs):
    "conv1d layer with padding='same', 'causal', 'valid', or any integer (defaults to 'same')"
    assert not (kernel_size and ks), 'use kernel_size or ks but not both simultaneously'
    assert kernel_size is not None or ks is not None, 'you need to pass a ks'
    kernel_size = kernel_size or ks
    if padding == 'same':
        if kernel_size%2==1:
            conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=kernel_size//2 * dilation, dilation=dilation, **kwargs)
        else:
            conv = SameConv1d(ni, nf, kernel_size, stride=stride, dilation=dilation, **kwargs)
    else: conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=padding, dilation=dilation, **kwargs)
    return conv

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class Conv1d_new_padding(nn.Module):
    def __init__(self, ni, nf, ks=None, stride=1, bias=False, pad_zero=True):
        super(Conv1d_new_padding, self).__init__()

        self.ks = ks
        # self.padding = nn.ConstantPad1d((0, int(self.ks-1)), 0)
        if pad_zero:
            self.padding = nn.ConstantPad1d((int((self.ks - 1) / 2), int(self.ks / 2)), 0)
        else:
            self.padding = nn.ReplicationPad1d((0, int(self.ks-1)))
        self.conv1d = nn.Conv1d(ni, nf, self.ks, stride=stride, bias=bias)

    def forward(self, x):
        out = self.padding(x)
        out = self.conv1d(out)
        return out

class ConvBlock(nn.Module):
    "Create a sequence of conv1d (`ni` to `nf`), activation (if `act_cls`) and `norm_type` layers."
    def __init__(self, ni, nf, kernel_size=None, stride=1, act=None, pad_zero=True):
        super(ConvBlock, self).__init__()
        kernel_size = kernel_size
        self.layer_list = []

        self.conv = Conv1d_new_padding(ni, nf, ks=kernel_size, stride=stride, pad_zero=pad_zero)
        self.bn = nn.BatchNorm1d(num_features=nf)
        self.layer_list += [self.conv, self.bn]
        if act is not None: self.layer_list.append(act)

        self.net = nn.Sequential(*self.layer_list)

    def forward(self, x):
        x = self.net(x)
        return x

class SeparableConv1d(nn.Module):
    def __init__(self, ni, nf, ks, stride=1, padding='same', dilation=1, bias=True, bias_std=0.01):
        super(SeparableConv1d, self).__init__()
        self.depthwise_conv = nn.Conv1d(ni, ni, ks, stride=stride, padding=padding, dilation=dilation, groups=ni, bias=bias)
        self.pointwise_conv = nn.Conv1d(ni, nf, 1, stride=1, padding=0, dilation=1, groups=1, bias=bias)
        if bias:
            if bias_std != 0: 
                normal_(self.depthwise_conv.bias, 0, bias_std)
                normal_(self.pointwise_conv.bias, 0, bias_std)
            else: 
                self.depthwise_conv.bias.data.zero_()
                self.pointwise_conv.bias.data.zero_()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

def weights_init(m):
    # print('=> weights init')
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(m.weight, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.constant_(m.weight, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        # Note that BN's running_var/mean are
        # already initialized to 1 and 0 respectively.
        if m.weight is not None:
            m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.zero_()