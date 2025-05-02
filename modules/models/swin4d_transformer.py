import itertools
from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp

from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from .patchembedding import PatchEmbed

rearrange = _ = optional_import("eionops", name='rearrange')

__all__  = [
    "window_partition",
    "window_reverse",
    "WindowAttention4D",
    "SwinTransformerBlock4D",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayer",
    "SwinTransformer4D",
]


def window_partion(x, window_size):
    """window partition function based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Partition tokens into their respective windows

    Args:
        x: input tensor (B, D, H, W, T, C)

        window_size: local window size(looking at the window that captures the fMRI).


    Returns:
        windows: (B*num_windows, window_size*window_size*window_size*window_size, C)
    """
    x_shape = x.size()
    b,d,h,w,t,c = x_shape
    x = x.view(
        b,
        d // window_size[0],
        window_size[0],
        h // window_size[1],
        window_size[1],
        w // window_size[2],
        window_size[2],
        t // window_size[3],
        window_size[3],
        c
    )
    windows = (
        x.permute(0,1,3,5,6,2,4,6,8,9)
        .contiguous().view(-1, window_size[0]*window_size[1]*window_size[2]*window_size[3])
    )
    return windows

def window_reverse(windows, window_size, dims):
    b,d,h,w,t = dims
    x = windows.view(
        b,
        torch.div(d, window_size[0], rounding_mode='floor'),
        torch.div(h, window_size[0], rounding_mode='floor'),
        torch.div(w, window_size[0], rounding_mode='floor'),
        torch.div(t, window_size[0], rounding_mode='floor'),
        window_size[0],
        window_size[1],
        window_size[2],
        window_size[3],
        -1,
    )
    x = x.permute(0,1,5,2,6,3,7,4,8,9)
    
    
    return x

def get_window_size(x_size, window_size, shift_size = None):
    
    
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
                
    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size) , tuple(use_shift_size)
    
class WindowAttention4D(nn.Module):
    
    def __init__(
        self,
        dim:int,
        num_heads:int,
        window_size: Sequence[int],
        qkv_bias : bool = False,
        attn_drop : float = 0.0,
        proj_drop : float = 0.0,
    ) -> None:
        """
        Args:
            dim (int): number of channels(feature)
            num_heads (int): number of attention heads
            window_size (Sequence[int]): local window size
            qkv_bias (bool, optional): add a bias to the query, key and value tensor. Defaults to False.
            attn_drop (float, optional): attention dropout rate. Defaults to 0.0.
            proj_drop (float, optional): dropout rate of output. Defaults to 0.0.
        """
        
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        mesh_args = torch.meshgrid.__kwdefaults__
        
        self.qkv = nn.Linear (dim , dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim , dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim= -1)
        
        
    def forward(self, x, mask):
        
        b_, n,c = x.shape
        qkv = self.qkv(x).reshape(b_, n,3,self.num_heads, c // self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2,-1)
        
        
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.to(attn.dtype).unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(b_,n,c)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
class SwinTransformerBlock4D(nn.Module):
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size:Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop:float = 0.0,
        attn_drop : float = 0.0,
        drop_path:float  = 0.0,
        act_layer: str = "GELU",
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool= False,
    ) -> None:


        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention4D(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")
