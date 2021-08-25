import math
from math import log, pi, sqrt
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, device=None):
        """
        Normalize the input before the function `fn` is applied. GroupNorm with 
        num_groups = 4 is used if `dim` is divisible by 4, else InstanceNorm is
        applied.

        Args:
            dim (int): number of channels in the input
            fn (class with `__call__()` or `forward()`): func to apply after norm
            device (str, torch.device): device with which to normalize
        """
        super().__init__()
        self.fn = fn
        num_groups = dim//4 if dim % 4 == 0 else dim
        self.norm = nn.GroupNorm(num_groups, dim, device=device)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

    
class FeedForward(nn.Module):
    def __init__(self, dim, in_encoder, mult = 4):
        """
        Two-layered MLP for transformer blocks.

        Args:
            dim (int): size of each token in the input sequence `x`
            in_encoder (bool): whether the FeedForward block is in the encoder
            mult (int): factor by which to scale dim to compute hidden dimension
        """
        super().__init__()
        hidden_dim = dim * mult

        if in_encoder: 
            self.project_in = nn.Conv3d(dim, hidden_dim, 1)
            self.project_out = nn.Sequential(
                nn.Conv3d(hidden_dim, hidden_dim, (1, 3, 3), padding = (0, 1 1)),
                nn.GELU(),
                nn.Conv3d(hidden_dim, dim, 1)
            )

        else: 
            self.project_in = nn.Conv2d(dim, hidden_dim, 1)
            self.project_out = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding = 1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, dim, 1)
            )

    def forward(self, x):
        x = self.project_in(x)
        return self.project_out(x)
      
      
################################################# ELEPHANTFORMER #####################################
# preliminary model

class Elephantformer(nn.Module):
    def __init__(
        self,
        num_frames = 2,
        dim = 64,
        channels = 3,
        stages = 4,
        num_blocks = 2,
        dim_head = 64,
        window_size = 16,
        heads = 8,
        ff_mult = 4,
        input_channels = None,
        output_channels = None,
    ):
        """
        Uformer that consumes multiple images and outputs one synthesized image.

        Args:
            dim (int): output channel dimension of the input projection
            num_frames (int): number of frames in input clip `x`
            channels (int): size of channel dimension of the input image
            stages (int): number of downsampling and upsampling blocks for encoder
                          and decoder respectively
            num_blocks (int): number of transformer blocks before each down/upsample
            dim_head (int): size of each head for the multi-head spacetime attention
            window_size (int): size of window for the multi-head spacetime attention.
                               Same idea as splitting input sequence into patches, so
                               this can be thought of as patch size.
            heads (int): number of heads used in the multi-head spacetime attention
            ff_mult (int): feed forward layers contain the following linear projections:
                           - nn.Linear(dim, dim * ff_mult)
                           - nn.Linear(dim * ff_mult, dim)
            input_channels (int): in case `channels` is different for input and output
                                  images, specifies input images' channel dimension.
            output_channels (int): in case `channels` is different for input and output
                                   images, specifies output image's channel dimension.
        """
        super().__init__()
        input_channels = channels if input_channels is None else input_channels
        output_channels = channels if output_channels is None else output_channels

        self.project_in = nn.Sequential(
            nn.Conv3d(input_channels, dim, kernel_size=(1, 3, 3)),
            nn.GELU()
        )

        self.project_out = nn.Sequential(
            nn.Conv2d(dim, output_channels, 3, padding = 1),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        # render each hyperparam a tuple so that we can iterate thru them
        heads, window_size, dim_head, num_blocks = map(partial(cast_tuple, depth = stages), (heads, window_size, dim_head, num_blocks))

        for ind, heads, window_size, dim_head, num_blocks in zip(range(stages), heads, window_size, dim_head, num_blocks):
            is_last = ind == (stages - 1)

            self.downs.append(nn.ModuleList([ # need 3d version of this
                EncoderBlock(dim, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size),
                nn.Conv3d(dim, dim * 2, kernel_size = (1, 4, 4), stride = 2, padding = 1)
            ]))

            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(dim * 2, dim, 2, stride = 2),
                Block(dim, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size)
            ]))

            dim *= 2

            if is_last:
                # need to concatenate the feature maps in between each block
                self.mid = Block(dim = dim, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size)
                # 1. 3DBlock(dim -> dim), 2. rearrange(x, 'b f c -> b (f, c)'), 3. SelfAttn2dBlock(f*dim -> f*dim), 4. SelfAttn2dBlock(f*dim -> dim)

    def forward(
        self,
        x,
    ):
        x = rearrange(x, 'b f c h w -> b c f h w')
        x = self.project_in(x)

        set_trace()
        skips = []
        for block, downsample in self.downs:
            x = block(x)
            skips.append(x)
            x = downsample(x)

        x = self.mid(x)

        for (upsample, block), skip in zip(reversed(self.ups), reversed(skips)):
            x = upsample(x)
            x = block(x, skip = skip)

        x = self.project_out(x)
        return x
