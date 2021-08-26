import math
from math import log, pi, sqrt
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from elephantformer.rotary import AxialRotaryEmbedding, apply_rotary_emb
from elephantformer.utils import cast_tuple

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
    def __init__(self, dim, mult = 4):
        """
        Two-layered MLP for transformer blocks.

        Args:
            dim (int): size of each token in the input sequence `x`
            in_encoder (bool): whether the FeedForward block is in the encoder
            mult (int): factor by which to scale dim to compute hidden dimension
        
        NOTE: Input tensor `x` is of shape (B, C, F, H, W). If `x` is a single 
            image of shape (B, C, H, W), reshape it to (B, C, 1, H, W).
        """
        super().__init__()
        hidden_dim = dim * mult

        self.project_in = nn.Conv3d(dim, hidden_dim, 1)
        self.project_out = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, (1, 3, 3), padding = (0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        x = self.project_in(x)
        return self.project_out(x)


class Attention3d(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8, window_size = 16, skip_type = 'ConcatCross'):
        """
        Window-based multi-head self-attention generalized to a 3d input.
        Each pixel attends to every other pixel in its patch, within its frame
        and in all other frames.

        Args: 
            dim (int): size of channel dimension of input
            dim_head (int): size of each head for the multi-head attention
            heads (int): number of heads to use for the multi-head attention
            window_size (int): size of window for the multi-head spacetime attention.
                        It's the same idea as splitting input sequence into patches, 
                        so this can be thought of as equivalent to patch size.
            skip_type (str): type of skip if in decoder. Will only use if `skip` arg
                        in `forward()` is provided. The options are 'ConcatCross', 
                        'Cross', or 'ConcatSelfSum'.

        NOTE: shape of input `x` is (B, C, F, H, W)
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.window_size = window_size
        self.skip_type = skip_type
        inner_dim = dim_head * heads

        self.to_q = nn.Conv3d(dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv3d(dim, inner_dim * 2, 1, bias = False)
        self.to_out = nn.Conv3d(inner_dim, dim, 1)

    def forward(self, x, skip=None, pos_emb = None):
        h, w, b, f = self.heads, self.window_size, x.shape[0], x.shape[2]
    
        q = self.to_q(x)

        kv_input = x
        
        if skip is not None:
            if self.skip_type == 'ConcatCross':
                kv_input = torch.cat((kv_input, skip), dim = 2)
            else:
                assert False, "Invalid skip type. The options are 'ConcatSkip', 'Cross' or 'ConcatCrossSum'"

        k, v = self.to_kv(kv_input).chunk(2, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) f x y -> (b h) f x y c', h = h), (q, k, v))

        if pos_emb is not None:
            q, k = apply_rotary_emb(q, k, pos_emb)

        q, k, v = map(lambda t: rearrange(t, 'b f (x w1) (y w2) c -> (b x y) (f w1 w2) c', w1 = w, w2 = w), (q, k, v))
        
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h x y) (f w1 w2) c -> b (h c) f (x w1) (y w2)', b = b, h = h, y = x.shape[-1] // w, w1 = w, w2 = w, f = f)
        return self.to_out(out)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        window_size = 16,
        rotary_emb = True,
        device = None,
        skip_type = None,
    ):
        """
        Transfomer block of Elephantformer. Constitutes the following layers:
            - GroupNorm, Attention, GroupNorm, FeedForward.

        Args:
            dim (int): output channel dimension of the input projection
            depth (int): number of blocks to repeat
            dim_head (int): size of each head for the multi-head spacetime attention
            heads (int): number of heads used in the multi-head spacetime attention
            ff_mult (int): feed forward layers contain the following linear projections:
                        - nn.Linear(dim, dim * ff_mult)
                        - nn.Linear(dim * ff_mult, dim)
            window_size (int): size of window for the multi-head spacetime attention.
                        Same idea as splitting input sequence into patches, so this
                        can be thought of as equivalent to patch size.
            rotary_emb (bool): whether to apply the relative positional encoding from
                        RoPE.
            device (torch.device, str): device to use, 'cuda' or 'cpu'            
            skip_type (str): type of skip for the decoder. The options are 
                        'ConcatCross', 'Cross', or 'ConcatSelfSum'.
        """
        super().__init__()
        self.pos_emb = AxialRotaryEmbedding(dim_head) if rotary_emb else None 

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention3d(dim, dim_head = dim_head, heads = heads, window_size = window_size, skip_type=skip_type), device),
                PreNorm(dim, FeedForward(dim, mult = ff_mult), device)
            ]))

    def forward(self, x, skip = None):
        pos_emb = None

        if self.pos_emb is not None:
            # NAIVE SOLUTION: apply 2d AxialRotaryEmbs independently on each frame; to be updated
            # frames = x.chunk(x.shape[2], dim=2)
            # embd_frames = [self.pos_emb(f) for f in frames]
            # pos_emb = torch.stack(embd_frames, dim = 2)

            pos_emb = self.pos_emb(x) # NEEDS UPDATING

        for attn, ff in self.layers:
            x = attn(x, skip = skip, pos_emb = pos_emb) + x
            x = ff(x) + x
        return x
    
    
################################################# ELEPHANTFORMER #####################################
# Generator

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
        device = None,
        skip_type = 'ConcatCross'
    ):
        """
        Uformer that consumes a 3d image or multiple images and outputs one synthesized image.

        Args:
            dim (int): output channel dimension of the input projection
            num_frames (int): number of frames in input clip `x`
            channels (int): size of channel dimension of the input image
            stages (int): number of downsampling and upsampling blocks for encoder
                        and decoder respectively
            num_blocks (int): number of transformer blocks before each down/upsample
            dim_head (int): size of each head for the multi-head spacetime attention
            window_size (int): size of window for the multi-head spacetime attention.
                        Same idea as splitting input sequence into patches, so this
                        can be thought of as equivalent to patch size.
            heads (int): number of heads used in the multi-head spacetime attention
            ff_mult (int): feed forward layers contain the following linear projections:
                        - nn.Linear(dim, dim * ff_mult)
                        - nn.Linear(dim * ff_mult, dim)
            input_channels (int): in case `channels` is different for input and output
                        images, specifies input images' channel dimension.
            output_channels (int): in case `channels` is different for input and output
                        images, specifies output image's channel dimension.
            device (torch.device, str): device to use, 'cuda' or 'cpu'
            skip_type (str): type of skip for the decoder. The options are 
                        'ConcatCross', 'Cross', or 'ConcatSelfSum'.
        
        NOTE: `forward()` consumes a tensor of shape (B, C, F, H, W) and spits out
            a tensor of shape (B, C, H, W)
        """
        super().__init__()
        input_channels = channels if input_channels is None else input_channels
        output_channels = channels if output_channels is None else output_channels

        self.project_in = nn.Sequential(
            nn.Conv3d(input_channels, dim, kernel_size = (1, 3, 3), padding = (0, 1, 1)),
            nn.GELU()
        )

        self.project_out = nn.Sequential(
            nn.Conv3d(dim, output_channels, kernel_size = (1, 3, 3), padding = (0, 1, 1))
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        # render each hyperparam a tuple so that we can iterate thru them
        heads, window_size, dim_head, num_blocks = map(partial(cast_tuple, depth = stages), (heads, window_size, dim_head, num_blocks))

        for ind, heads, window_size, dim_head, num_blocks in zip(range(stages), heads, window_size, dim_head, num_blocks):
            is_last = ind == (stages - 1)

            self.downs.append(nn.ModuleList([ # need 3d version of this
                Block(dim, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size, device = device),
                nn.Conv3d(dim, dim * 2, kernel_size = (1, 4, 4), stride = (1, 2, 2), padding = (0, 1, 1))
            ]))

            self.ups.append(nn.ModuleList([
                nn.ConvTranspose3d(dim * 2, dim, (1, 2, 2), stride = (1, 2, 2)),
                Block(dim, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size, device = device, skip_type = skip_type)
            ]))

            dim *= 2
            if is_last:
                self.mid = nn.Sequential(
                    Rearrange('b c f h w -> b (c f) 1 h w'),
                    Block(dim = dim*num_frames, depth = 1, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size),
                    nn.Conv3d(dim*num_frames, dim, kernel_size = (1, 3, 3), padding = (0, 1, 1)),
                    Block(dim = dim, depth = 2, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size),
                )

    def forward(self, x,):
        x = self.project_in(x)

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
        return x.squeeze(2)
