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
    None
