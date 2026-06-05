# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, mobilenet_v2, efficientnet_b0
import math
import numpy as np
import itertools
from typing import Tuple

# YOLOv26系列组件
class YOLOv26Head(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super(YOLOv26Head, self).__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_channels, 3 * (num_classes + 5), 1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# FasterNet基础组件
class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=4, forward='split_cat', kernel_size=3):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, kernel_size, 1, kernel_size//2, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        x1, x2 = x.split([self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x

    def forward_split_cat(self, x):
        x = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x[0])
        x = torch.cat((x1, x[1]), 1)
        return x

class FasterBlock(nn.Module):
    def __init__(self, dim, n_div=4, mlp_ratio=2.0, drop_path=0., act_layer=nn.ReLU, inplace=True):
        super().__init__()
        self.n_div = n_div
        self.mlp_ratio = mlp_ratio

        c_hidden = int(dim * mlp_ratio)
        self.layer1 = nn.Conv2d(dim, c_hidden, 1, bias=False)
        self.pconv = Partial_conv3(dim, n_div=n_div)
        self.layer2 = nn.Conv2d(c_hidden, dim, 1, bias=False)
        self.act = act_layer(inplace=inplace)

    def forward(self, x):
        shortcut = x
        x = self.layer1(x)
        x = self.pconv(x)
        x = self.act(x)
        x = self.layer2(x)
        x = x + shortcut
        return x

# ConvNeXt基础组件
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

# MobileViT组件
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def rearrange(tensor, pattern, **axes_lengths):
    """简化版rearrange函数，用于模拟einops.rearrange的功能"""
    if 'b p n (h d)' in pattern and 'b (ph pw) (h w) d' in pattern:
        # 从 'b (ph pw) (h w) d' 到 'b p n (h d)' 的转换
        b, ph_pw, hw, d = tensor.shape
        p = axes_lengths['p']
        n = axes_lengths['n']
        h = axes_lengths['h']
        ph, pw = axes_lengths['ph'], axes_lengths['pw']
        
        # 假设 ph_pw = ph * pw, hw = h * w
        w = hw // axes_lengths['h']  # 计算w
        
        # reshape tensor
        tensor = tensor.view(b, ph, pw, h, w, d)
        # 转换为所需的输出格式
        tensor = tensor.contiguous().view(b, p, n, h * d)
    elif 'b (ph pw) (h w) d' in pattern and 'b p n (h d)' in pattern:
        # 从 'b p n (h d)' 到 'b (ph pw) (h w) d' 的转换
        b, p, n, hd = tensor.shape
        h = axes_lengths['h']
        d = hd // h
        ph, pw = axes_lengths['ph'], axes_lengths['pw']
        w = axes_lengths['w']
        
        # 假设 p = ph * pw, n = h * w
        tensor = tensor.view(b, ph, pw, h, w, d)
        tensor = tensor.permute(0, 1, 3, 2, 4, 5).contiguous()
        tensor = tensor.view(b, ph * pw, h * w, d)
    elif 'b p h n d' in pattern and 'b p n (h d)' in pattern:
        # 从 'b p h n d' 到 'b p n (h d)' 的转换
        b, p, h, n, d = tensor.shape
        tensor = tensor.view(b, p, n, h * d)
    elif 'b p n (h d)' in pattern and 'b p h n d' in pattern:
        # 从 'b p n (h d)' 到 'b p h n d' 的转换
        b, p, n, hd = tensor.shape
        h = axes_lengths['h']
        d = hd // h
        tensor = tensor.view(b, p, h, n, d)
    return tensor

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MobileViTBlock(nn.Module):
    def __init__(self, dim, kernel_size, patch_size, depth, mlp_dim, dim_head=64, channels=16):
        super().__init__()
        self.ph, self.pw = pair(patch_size)

        self.conv1 = ConvBlock(channels, channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = ConvBlock(channels, dim, kernel_size=1)

        self.transformer = Transformer(dim, depth, 4, dim_head, mlp_dim)

        self.conv3 = ConvBlock(dim, channels, kernel_size=1)
        self.conv4 = ConvBlock(2 * channels, channels, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x):
        y = x.clone()

        # Local representation
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global representation
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw, h=h//self.ph, w=w//self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat([x, y], 1)
        x = self.conv4(x)
        return x

# EdgeNeXt组件
class PositionalEncodingFourier(nn.Module):
    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        device = self.token_projection.weight.device
        y_embed = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, dtype=torch.float32, device=device).repeat(H, 1)
        y_embed = y_embed.unsqueeze(0).repeat(B, 1, 1)
        x_embed = x_embed.unsqueeze(0).repeat(B, 1, 1)

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :] / dim_t
        pos_y = y_embed[:, :, :] / dim_t
        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=64):
        super(ConvTokenizer, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.proj(x)
        return x

# LeViT组件
def stem(in_chs, out_chs, activation):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        activation(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        activation(),
    )

class Residual(nn.Module):
    def __init__(self, m, add=True):
        super().__init__()
        self.m = m
        self.add = add

    def forward(self, x):
        if self.add:
            return x + self.m(x)
        else:
            return torch.cat([x, self.m(x)], 1)

def group(w, g, b, w_min):
    # group function implementation
    w = w.reshape(b, g, -1, w.shape[-2], w.shape[-1])
    return w

def attention2d(q, k, v):
    # attention function implementation
    B, _, N, _ = q.shape
    scale_factor = q.size(-1) ** -0.5
    att = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
    att = F.softmax(att, dim=-1)
    x = torch.matmul(att, v)
    return x

class Attention2d(nn.Module):
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, act_layer=None, resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)
        
        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N))

    
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x

class LevitMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Hardswish, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LevitLayer(nn.Module):
    def __init__(self, dim, key_dim, num_heads=8, mlp_ratio=2, attn_ratio=2, act_layer=nn.Hardswish, resolution=14):
        super().__init__()
        self.attn = Attention2d(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, act_layer=act_layer, resolution=resolution)
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = LevitMlp(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=0.)
        self.drop_path = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x).flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x).flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W))
        return x

# Swin Transformer组件
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 初始化相对位置偏置表
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def to_2tuple(x):
    """将输入转换为二元组"""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    else:
        return (x, x)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

# VMamba相关组件 - 这里我们用一个简化的版本来代替SS2D
class SS2D(nn.Module):
    def __init__(self, d_model, d_state=64, ssm_ratio=1.0, dt_rank="auto", act_layer=nn.SiLU,
                 conv_kernel=3, bias=False, forward_type="v0"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.ssm_ratio = ssm_ratio
        self.dt_rank = dt_rank
        self.act_layer = act_layer
        self.conv_kernel = conv_kernel
        self.bias = bias
        self.forward_type = forward_type
        
        # 简化版本的SS2D，使用卷积层代替复杂的SSM
        self.conv = nn.Conv2d(d_model, d_model, kernel_size=conv_kernel, padding=conv_kernel//2, groups=d_model, bias=bias)
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.act = act_layer()

    def forward(self, x):
        # 假设x的形状为 (B, H, W, C)
        if len(x.shape) == 4:
            B, H, W, C = x.shape
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        else:
            # 假设x的形状为 (B, L, C)，其中L=H*W
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # 卷积操作
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = self.linear(x)
        x = self.act(x)
        
        if len(x.shape) == 4:
            x = x.view(B, H * W, C)  # (B, L, C)
        
        return x

class VSSBlock(nn.Module):
    def __init__(self, hidden_dim: int = 0, drop_path: float = 0, norm_layer: nn.Module = nn.LayerNorm, 
                 ssm_d_state: int = 64, ssm_ratio: float = 1.0, ssm_dt_rank: any = "auto", ssm_act_layer=nn.SiLU,
                 ssm_conv: int = 3, ssm_conv_bias=True, forward_type="v0"):
        super().__init__()
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim, 
            d_state=ssm_d_state, 
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            conv_kernel=ssm_conv,
            bias=ssm_conv_bias,
            forward_type=forward_type,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.op(self.norm(x)))

# 主要模型定义
def get_yolov26_nano(num_classes=10, pretrained=False):
    """YOLOv26-Nano (End-to-End)"""
    class YOLOv26Nano(nn.Module):
        def __init__(self, num_classes):
            super(YOLOv26Nano, self).__init__()
            self.backbone = nn.Sequential(
                ConvBlock(3, 16, kernel_size=3, stride=2, padding=1),
                ConvBlock(16, 32, kernel_size=3, stride=2, padding=1),
                ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
                ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
                ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.backbone(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    return YOLOv26Nano(num_classes)

def get_faster_net_p2(num_classes=10, pretrained=False):
    """FasterNet + P2 Detect Head"""
    class FasterNetP2(nn.Module):
        def __init__(self, num_classes):
            super(FasterNetP2, self).__init__()
            self.stem = ConvBlock(3, 32, kernel_size=3, stride=2, padding=1)
            
            self.stage1 = nn.Sequential(
                FasterBlock(32, n_div=4),
                FasterBlock(32, n_div=4),
            )
            
            self.downsample1 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
            self.stage2 = nn.Sequential(
                FasterBlock(64, n_div=4),
                FasterBlock(64, n_div=4),
                FasterBlock(64, n_div=4),
            )
            
            self.downsample2 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
            self.stage3 = nn.Sequential(
                FasterBlock(128, n_div=4),
                FasterBlock(128, n_div=4),
                FasterBlock(128, n_div=4),
                FasterBlock(128, n_div=4),
            )
            
            self.p2_head = YOLOv26Head(128, num_classes)
            
            # For classification task
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.stage1(x)
            x = self.downsample1(x)
            x = self.stage2(x)
            x = self.downsample2(x)
            x = self.stage3(x)
            
            # For classification
            x_class = self.avgpool(x)
            x_class = torch.flatten(x_class, 1)
            x_class = self.classifier(x_class)
            
            return x_class

    return FasterNetP2(num_classes)

def get_convnext_tiny_yolo26(num_classes=10, pretrained=False):
    """ConvNeXt-Tiny + YOLOv26 Head"""
    class ConvNeXtTinyYOLO26(nn.Module):
        def __init__(self, num_classes):
            super(ConvNeXtTinyYOLO26, self).__init__()
            # ConvNeXt Tiny configuration
            depths = [3, 3, 9, 3]
            dims = [96, 192, 384, 768]
            
            self.downsample_layers = nn.ModuleList()
            stem = nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
            self.downsample_layers.append(stem)
            
            for i in range(3):
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            self.stages = nn.ModuleList()
            for i in range(4):
                stage = nn.Sequential(
                    *[Block(dim=dims[i]) for j in range(depths[i])]
                )
                self.stages.append(stage)

            self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
            self.head = nn.Linear(dims[-1], num_classes)
            
            # YOLO head
            self.yolo_head = YOLOv26Head(dims[-1], num_classes)

        def forward(self, x):
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)

            x = self.norm(x)
            
            # Global average pooling for classification
            x = F.adaptive_avg_pool2d(x, 1)
            x = torch.flatten(x, 1)
            x = self.head(x)
            
            return x

    return ConvNeXtTinyYOLO26(num_classes)

def get_pp_lcnet_picodet(num_classes=10, pretrained=False):
    """PP-LCNet + PicoDet"""
    class PPLCNetPicoDet(nn.Module):
        def __init__(self, num_classes):
            super(PPLCNetPicoDet, self).__init__()
            # Simplified PP-LCNet structure
            self.features = nn.Sequential(
                ConvBlock(3, 16, kernel_size=3, stride=2, padding=1),
                ConvBlock(16, 32, kernel_size=3, stride=2, padding=1),
                ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
                ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
                ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),
            )
            
            # PicoDet style detection head
            self.pico_head = YOLOv26Head(256, num_classes)
            
            # Classification head
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.features(x)
            
            # For classification
            x_class = self.avgpool(x)
            x_class = torch.flatten(x_class, 1)
            x_class = self.classifier(x_class)
            
            return x_class

    return PPLCNetPicoDet(num_classes)

def get_repvgg_yolov6s(num_classes=10, pretrained=False):
    """RepVGG + YOLOv6s"""
    class RepVGGBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(RepVGGBlock, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.stride = stride
            
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
            self.conv2 = nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()

        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)) + self.bn2(self.conv2(x)))
            return out

    class RepVGGYOLOv6s(nn.Module):
        def __init__(self, num_classes):
            super(RepVGGYOLOv6s, self).__init__()
            self.stem = RepVGGBlock(3, 64, stride=2)
            
            self.stage1 = nn.Sequential(
                RepVGGBlock(64, 128, stride=2),
                RepVGGBlock(128, 128),
                RepVGGBlock(128, 128),
            )
            
            self.stage2 = nn.Sequential(
                RepVGGBlock(128, 256, stride=2),
                RepVGGBlock(256, 256),
                RepVGGBlock(256, 256),
                RepVGGBlock(256, 256),
            )
            
            self.stage3 = nn.Sequential(
                RepVGGBlock(256, 512, stride=2),
                RepVGGBlock(512, 512),
                RepVGGBlock(512, 512),
                RepVGGBlock(512, 512),
                RepVGGBlock(512, 512),
                RepVGGBlock(512, 512),
            )
            
            self.yolo_head = YOLOv26Head(512, num_classes)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            
            # For classification
            x_class = self.avgpool(x)
            x_class = torch.flatten(x_class, 1)
            x_class = self.classifier(x_class)
            
            return x_class

    return RepVGGYOLOv6s(num_classes)

def get_vmamba_detect(num_classes=10, pretrained=False):
    """VMamba (Visual Mamba) + Detect Head"""
    class VMambaDetect(nn.Module):
        def __init__(self, num_classes):
            super(VMambaDetect, self).__init__()
            # Simplified VMamba structure
            self.stem = ConvBlock(3, 96, kernel_size=7, stride=4, padding=2)
            
            self.layers = nn.ModuleList([
                VSSBlock(hidden_dim=96) for _ in range(3)
            ])
            
            self.downsample1 = nn.Conv2d(96, 192, kernel_size=2, stride=2)
            self.layers2 = nn.ModuleList([
                VSSBlock(hidden_dim=192) for _ in range(3)
            ])
            
            self.downsample2 = nn.Conv2d(192, 384, kernel_size=2, stride=2)
            self.layers3 = nn.ModuleList([
                VSSBlock(hidden_dim=384) for _ in range(9)
            ])
            
            self.detect_head = YOLOv26Head(384, num_classes)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(384, num_classes)

        def forward(self, x):
            x = self.stem(x)
            
            for layer in self.layers:
                x = layer(x)
            
            x = self.downsample1(x)
            for layer in self.layers2:
                x = layer(x)
                
            x = self.downsample2(x)
            for layer in self.layers3:
                x = layer(x)
            
            # For classification
            x_class = self.avgpool(x)
            x_class = torch.flatten(x_class, 1)
            x_class = self.classifier(x_class)
            
            return x_class

    return VMambaDetect(num_classes)

def get_mobilevit_s_yolo26(num_classes=10, pretrained=False):
    """MobileViT-S + YOLOv26"""
    class MobileViTSYOLO26(nn.Module):
        def __init__(self, num_classes):
            super(MobileViTSYOLO26, self).__init__()
            # MobileViT-S configuration
            self.conv1 = ConvBlock(3, 32, kernel_size=3, stride=2, padding=1)
            self.conv2 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)
            
            self.mobilevit_block1 = MobileViTBlock(dim=96, kernel_size=3, patch_size=2, depth=2, mlp_dim=144, channels=64)
            self.conv3 = ConvBlock(64, 96, kernel_size=3, stride=2, padding=1)
            
            self.mobilevit_block2 = MobileViTBlock(dim=120, kernel_size=3, patch_size=2, depth=4, mlp_dim=180, channels=96)
            self.conv4 = ConvBlock(96, 120, kernel_size=3, stride=2, padding=1)
            
            self.mobilevit_block3 = MobileViTBlock(dim=144, kernel_size=3, patch_size=2, depth=3, mlp_dim=216, channels=120)
            
            self.yolo_head = YOLOv26Head(120, num_classes)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(120, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.mobilevit_block1(x)
            x = self.conv3(x)
            x = self.mobilevit_block2(x)
            x = self.conv4(x)
            x = self.mobilevit_block3(x)
            
            # For classification
            x_class = self.avgpool(x)
            x_class = torch.flatten(x_class, 1)
            x_class = self.classifier(x_class)
            
            return x_class

    return MobileViTSYOLO26(num_classes)

def get_edgenext_yolo8(num_classes=10, pretrained=False):
    """EdgeNeXt + YOLOv8"""
    class EdgeNeXtYOLO8(nn.Module):
        def __init__(self, num_classes):
            super(EdgeNeXtYOLO8, self).__init__()
            # EdgeNeXt structure
            self.stem = ConvTokenizer(3, 64)
            
            self.stage1 = nn.Sequential(
                ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
                ConvBlock(128, 128, kernel_size=3, padding=1),
            )
            
            self.stage2 = nn.Sequential(
                ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),
                ConvBlock(256, 256, kernel_size=3, padding=1),
                ConvBlock(256, 256, kernel_size=3, padding=1),
            )
            
            self.stage3 = nn.Sequential(
                ConvBlock(256, 512, kernel_size=3, stride=2, padding=1),
                ConvBlock(512, 512, kernel_size=3, padding=1),
                ConvBlock(512, 512, kernel_size=3, padding=1),
                ConvBlock(512, 512, kernel_size=3, padding=1),
            )
            
            self.yolo_head = YOLOv26Head(512, num_classes)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            
            # For classification
            x_class = self.avgpool(x)
            x_class = torch.flatten(x_class, 1)
            x_class = self.classifier(x_class)
            
            return x_class

    return EdgeNeXtYOLO8(num_classes)

def get_levit(num_classes=10, pretrained=False):
    """LeViT (Vision Transformer in Vision)"""
    class LeViT(nn.Module):
        def __init__(self, num_classes, image_size=224):
            super(LeViT, self).__init__()
            self.stem = stem(3, 32, nn.Hardswish)
            self.trunk = nn.Sequential(
                LevitLayer(32, key_dim=16, num_heads=4, resolution=image_size//4),
                LevitLayer(32, key_dim=16, num_heads=4, resolution=image_size//4),
                LevitLayer(32, key_dim=16, num_heads=4, resolution=image_size//4),
            )
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(32, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.trunk(x)
            
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            
            return x

    return LeViT(num_classes)

def get_swin_tiny_maskrcnn(num_classes=10, pretrained=False):
    """Swin-Tiny + Mask R-CNN (Distilled)"""
    class SwinTinyMaskRCNN(nn.Module):
        def __init__(self, num_classes):
            super(SwinTinyMaskRCNN, self).__init__()
            # Swin Tiny configuration
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            embed_dim = 96
            window_size = 7
            
            self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
            self.pos_drop = nn.Dropout(p=0.0)
            
            self.layers = nn.ModuleList()
            for i in range(4):
                layer = SwinTransformerBlock(
                    dim=int(embed_dim * 2 ** i),
                    input_resolution=(224 // (2 ** (i+2)), 224 // (2 ** (i+2))),
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.1,
                    norm_layer=nn.LayerNorm
                )
                self.layers.append(layer)
            
            self.norm = nn.LayerNorm(int(embed_dim * 2 ** 3))
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(int(embed_dim * 2 ** 3), num_classes)

        def forward(self, x):
            x = self.patch_embed(x)
            x = self.pos_drop(x)
            
            for layer in self.layers:
                x = layer(x, (x.shape[2], x.shape[3]))
            
            x = self.norm(x)
            x = x.mean(dim=1)
            x = self.head(x)
            
            return x

    return SwinTinyMaskRCNN(num_classes)

def get_resnet18_cbam(num_classes=10, pretrained=False):
    """ResNet-18 + CBAM (Convolutional Block Attention Module)"""
    from torchvision.models import resnet18
    
    class ChannelAttention(nn.Module):
        def __init__(self, in_planes, ratio=16):
            super(ChannelAttention, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)

            self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
            self.relu1 = nn.ReLU()
            self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
            out = avg_out + max_out
            return self.sigmoid(out)

    class SpatialAttention(nn.Module):
        def __init__(self, kernel_size=7):
            super(SpatialAttention, self).__init__()

            assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
            padding = 3 if kernel_size == 7 else 1

            self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x = torch.cat([avg_out, max_out], dim=1)
            x = self.conv1(x)
            return self.sigmoid(x)

    class ResNet18CBAM(nn.Module):
        def __init__(self, num_classes):
            super(ResNet18CBAM, self).__init__()
            base_model = resnet18(weights="DEFAULT" if pretrained else None)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            self.ca = ChannelAttention(512)
            self.sa = SpatialAttention()
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = self.ca(x) * x
            x = self.sa(x) * x
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    return ResNet18CBAM(num_classes)

def get_resnet18_simam(num_classes=10, pretrained=False):
    """ResNet-18 + SimAM (Simple Attention Module)"""
    from torchvision.models import resnet18
    
    class SimAM(nn.Module):
        def __init__(self, channels=None, e_lambda=1e-4):
            super(SimAM, self).__init__()

            self.activaton = nn.Sigmoid()
            self.e_lambda = e_lambda

        def forward(self, x):
            b, c, h, w = x.size()
            
            n = w * h - 1
            x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
            y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

            return x * self.activaton(y)

    class ResNet18SimAM(nn.Module):
        def __init__(self, num_classes):
            super(ResNet18SimAM, self).__init__()
            base_model = resnet18(weights="DEFAULT" if pretrained else None)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            self.simam = SimAM()
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = self.simam(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    return ResNet18SimAM(num_classes)

