from unittest.mock import patch

from torch import nn as nn
import torch


class PatchEmbed(nn.Module):
    """
    Converts the 4D image to Patch Embedding
    """
    def __init__(self, 
                img_size = (96,96,96,20),
                patch_size = (4,4,4,1),
                in_chans = 2,
                embed_dim = 24,
                norm_layer = None,
                flatten = True,
                spatial_dim = 3,
                ):
                assert len(patch_size) == 4, "Need four numbers pertaining to the dimensions, Height, Width, Depth and Temporal space"
                assert patch_size[3] == 1, "temporal axis merging is not implemented yet"
                super().__init__()
                self.img_size = img_size
                self.patc_size = patch_size
                self.grid_size= (
                    img_size[0] // patch_size[0],
                    img_size[1] // patch_size[1],
                    img_size[2] // patch_size[2],
                )
                self.embed_dim = embed_dim
                self.num_patches = self.grid_size[0] * self.grid_size[1]
                self.flatten = flatten
                
                self.fc=nn.Linear(in_features=in_chans * patch_size[0] * patch_size[1] * patch_size[2] * patch_size[3], out_features=embed_dim)
                self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        torch.cuda.nvtx.range_push("PatchEmbed")
        B,C,H,W,D,T = x.shape
        assert H == self.img_size[0], f"Input image height({H} does not match model ({self.img_size[0]}))"
        assert W == self.img_size[1], f"Input image height({W} does not match model ({self.img_size[1]}))"
        assert D == self.img_size[2], f"Input image height({D} does not match model ({self.img_size[2]}))"
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1,2) #Batch, Channels, Height, Width --> Batch,Number of pathces, Channels
        x = self.norm(x)
        torch.cuda.nvtx.range_pop()
        return x
    
    def proj(self,x):
        B,C,H,W,D,T = x.shape
        pH, pW, pD = self.grid_size
        sH,sW,sD,sT = self.patc_size
        
        x = x.view(B,C,pH,sH,pW,sW,pD,sD,-1, sT)
        x = x.permute(0,2,4,6,8, 3, 5,7,9,1).contiguous().view(-1,sH * sW * sD * sT * C)
        x = self.fc(x)
        x = x.view(B, pH, pW, pD, -1, self.embed_dim).contiguous()
        x = x.permute(0,5,1,2,3,4)
        return x