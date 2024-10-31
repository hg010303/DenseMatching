import os
import sys
from operator import add
from functools import reduce, partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from timm.models.layers import DropPath, trunc_normal_
import torchvision.models as models

# from models.feature_backbones import resnet
from models.croco.mod import FeatureL2Norm, unnormalise_and_convert_mapping_to_flow
from models.croco.SCOT.rhm_map import rhm
from models.croco.SCOT.geometry import gaussian2d, center, receptive_fields
from einops import rearrange, repeat
from timm.layers import to_2tuple

## uncertainty


r'''
Modified timm library Vision Transformer implementation
https://github.com/rwightman/pytorch-image-models
'''

def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of channels per head (dim // num_heads if not set)
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, appearance_guidance_dim, num_heads, head_dim=None, window_size=7, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)  # Wh, Ww
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim + appearance_guidance_dim, attn_dim, bias=qkv_bias)
        self.k = nn.Linear(dim + appearance_guidance_dim, attn_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, attn_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim + appearance_guidance_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        
        q = self.q(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x[:, :, :self.dim]).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

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
        
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        window_size (int): Window size.
        num_heads (int): Number of attention heads.
        head_dim (int): Enforce the number of channels per head
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self, dim, appearance_guidance_dim, input_resolution, num_heads=4, head_dim=None, window_size=7, shift_size=0,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., qk_scale=None,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_hyperpixel=13):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim+appearance_guidance_dim)
        self.attn = WindowAttention(
            dim, appearance_guidance_dim=appearance_guidance_dim, num_heads=num_heads, head_dim=head_dim, window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
    

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim+appearance_guidance_dim)
        self.norm3 = norm_layer(dim)
        
        self.mlp = Mlp(in_features=dim+appearance_guidance_dim, hidden_features=int((dim+appearance_guidance_dim) * mlp_ratio), act_layer=act_layer, drop=drop)
        
        self.num_hyperpixel = num_hyperpixel

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)):
                for w in (
                        slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # num_win, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # if appearance_guidance is not None:
        #     D = appearance_guidance.shape[-1]
        #     appearance_guidance = appearance_guidance.view(B, H, W, -1)
        #     x = torch.cat([x, appearance_guidance], dim=-1)
        #     shortcut=x
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, x_windows.shape[-1])  # num_win*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # num_win*B, window_size*window_size, C

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


class MultiscaleBlock(nn.Module):

    def __init__(self, dim, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, img_size=14, num_hyperpixel=13):
        super().__init__()
        # self.attn = Attention(
        #     dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.block_1 = SwinTransformerBlock(dim, embed_dim, (img_size,img_size), num_heads=num_heads, head_dim=None, window_size=7, shift_size=0, num_hyperpixel=num_hyperpixel)
        self.block_2 = SwinTransformerBlock(dim, embed_dim, (img_size,img_size), num_heads=num_heads, head_dim=None, window_size=7, shift_size=7 // 2, num_hyperpixel=num_hyperpixel)
        
        self.attn_multiscale = Attention(
            dim+embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_multiscale2 = Attention(
            dim+embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim+embed_dim)
        self.norm2 = norm_layer(dim+embed_dim)
        self.norm3 = norm_layer(dim+embed_dim)
        self.norm4 = norm_layer(dim+embed_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim+embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim+embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        

    def forward(self, x):
        '''
        Multi-level aggregation
        '''
        B, N, H, W = x.shape
        if N == 1:
            x = x.flatten(0, 1)
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x.view(B, N, H, W)
        
        x = x.flatten(0, 1)
        x = self.block_1(x)
        
        x = x.view(B, N, H, -1).transpose(1, 2).flatten(0, 1) 
        x = x + self.drop_path(self.attn_multiscale(self.norm1(x)))
        x = x.view(B, H, N, -1).transpose(1, 2).flatten(0, 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(B, N, H, -1)
        
        
        
        x = x.flatten(0, 1)        
        
        x = self.block_2(x)
        x = x.view(B, N, H, -1).transpose(1, 2).flatten(0, 1) 
        x = x + self.drop_path(self.attn_multiscale2(self.norm3(x)))
        x = x.view(B, H, N, -1).transpose(1, 2).flatten(0, 1)
        x = x + self.drop_path(self.mlp2(self.norm4(x)))
        x = x.view(B, N, H, -1)
        

        return x


class TransformerAggregator(nn.Module):
    def __init__(self, num_hyperpixel, dim, img_size=224, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.pos_embed_x = nn.Parameter(torch.zeros(1, num_hyperpixel, 1, img_size, (embed_dim+dim) // 2))
        self.pos_embed_y = nn.Parameter(torch.zeros(1, num_hyperpixel, img_size, 1, (embed_dim+dim) // 2))
        self.pos_drop = nn.Dropout(p=drop_rate)

        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            MultiscaleBlock(
                dim=dim, embed_dim = embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, img_size=img_size, num_hyperpixel=num_hyperpixel)
            for i in range(2)])
        self.proj = nn.Linear(embed_dim+dim, img_size ** 2)
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed_x, std=.02)
        trunc_normal_(self.pos_embed_y, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, attn_maps, source_feat):
        B = attn_maps.shape[0]
        x = attn_maps.clone()
        
        pos_embed = torch.cat((self.pos_embed_x.repeat(1, 1, self.img_size, 1, 1), self.pos_embed_y.repeat(1, 1, 1, self.img_size, 1)), dim=4)
        pos_embed = pos_embed.flatten(2, 3)
        x = torch.cat((x, source_feat), dim=3) + pos_embed
        x = self.blocks(x)
        
        x = self.proj(x) + attn_maps 

        return x.mean(1)


class CATs_SWIN(nn.Module):
    def __init__(self,
    feature_size=16,
    feature_proj_dim=128,
    depth=4,
    num_heads=6,
    mlp_ratio=4,
    hyperpixel_ids=[0,8,20,21,26,28,29,30],
    output_interp=False,
    cost_transformer=True,
    args=None,):
        super().__init__()
        self.feature_size = feature_size
        self.feature_proj_dim = feature_proj_dim
        self.decoder_embed_dim = self.feature_size + self.feature_proj_dim
        self.cost_transformer=cost_transformer
        self.args=args
        self.correlation = getattr(args,'correlation',False)
        self.reciprocity = getattr(args,'reciprocity',False)
        self.occlusion_mask = getattr(args,'occlusion_mask',False)

        channels = [768]*12
        if self.correlation:
            channels = [1024]+channels
            hyperpixel_ids = hyperpixel_ids + [12]
            



        # self.feature_extraction = FeatureExtractionHyperPixel(hyperpixel_ids, feature_size, freeze)
        self.ln = nn.ModuleList([nn.LayerNorm(channels[i]) for i in hyperpixel_ids])
        if self.reciprocity:
            self.ln_src = nn.ModuleList([nn.LayerNorm(channels[i]) for i in hyperpixel_ids])
        
        self.proj = nn.ModuleList([ 
            nn.Linear(channels[i], self.feature_proj_dim) for i in hyperpixel_ids
        ])

        self.decoder = TransformerAggregator(
            img_size=self.feature_size, dim = self.feature_size**2, embed_dim=self.feature_proj_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_hyperpixel=len(hyperpixel_ids))
            
        self.l2norm = FeatureL2Norm()
    
        self.x_normal = np.linspace(-1,1,self.feature_size)
        self.x_normal = nn.Parameter(torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False))
        self.y_normal = np.linspace(-1,1,self.feature_size)
        self.y_normal = nn.Parameter(torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False))
        
        self.output_interp = output_interp
        
    def softmax_with_temperature(self, x, beta, d = 1):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M # subtract maximum value for stability
        exp_x = torch.exp(x/beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    def soft_argmax(self, corr, beta=0.02):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        b,_,h,w = corr.size()
        corr = self.softmax_with_temperature(corr, beta=beta, d=1)
        corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
        x_normal = self.x_normal.expand(b,w)
        x_normal = x_normal.view(b,w,1,1)
        grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        
        grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
        y_normal = self.y_normal.expand(b,h)
        y_normal = y_normal.view(b,h,1,1)
        grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        return grid_x, grid_y
    
    def mutual_nn_filter(self, correlation_matrix):
        r"""Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18)"""
        corr_src_max = torch.max(correlation_matrix, dim=3, keepdim=True)[0]
        corr_trg_max = torch.max(correlation_matrix, dim=2, keepdim=True)[0]
        corr_src_max[corr_src_max == 0] += 1e-30
        corr_trg_max[corr_trg_max == 0] += 1e-30

        corr_src = correlation_matrix / corr_src_max
        corr_trg = correlation_matrix / corr_trg_max

        return correlation_matrix * (corr_src * corr_trg)
    
    def corr(self, src, trg):
        return src.flatten(2).transpose(-1, -2) @ trg.flatten(2)
    
    @staticmethod
    def constrain_large_log_var_map(var_min, var_max, large_log_var_map):
        """
        Constrains variance parameter between var_min and var_max, returns log of the variance. Here large_log_var_map
        is the unconstrained variance, outputted by the network
        Args:
            var_min: min variance, corresponds to parameter beta_minus in paper
            var_max: max variance, corresponds to parameter beta_plus in paper
            large_log_var_map: value to constrain

        Returns:
            larger_log_var_map: log of variance parameter
        """
        if var_min > 0 and var_max > 0:
            large_log_var_map = torch.log(var_min +
                                          (var_max - var_min) * torch.sigmoid(large_log_var_map - torch.log(var_max)))
        elif var_max > 0:
            large_log_var_map = torch.log((var_max - var_min) * torch.sigmoid(large_log_var_map - torch.log(var_max)))
        elif var_min > 0:
            # large_log_var_map = torch.log(var_min + torch.exp(large_log_var_map))
            max_exp = large_log_var_map.detach().max() - 10.0
            large_log_var_map = torch.log(var_min / max_exp.exp() + torch.exp(large_log_var_map - max_exp)) + max_exp
        return large_log_var_map

    def forward(self, attn_maps, tgt_feats,output_shape, feat_source, feat_target, attn_maps_source=None, src_feats=None, tgt_img = None, src_img = None):
        B, _,_ = tgt_feats[0].size()

        tgt_feats_proj, src_feats_proj = [],[]
        
        if self.correlation:
            corr = self.corr(self.l2norm(feat_target.permute(0,2,1)),self.l2norm(feat_source.permute(0,2,1)))
            attn_maps = [corr] + attn_maps
            tgt_feats = [feat_target] + tgt_feats
            
            if self.reciprocity:
                corr = corr.transpose(-1,-2)
                attn_maps_source = [corr] + attn_maps_source
                src_feats = [feat_source] + src_feats
        
        for i in range(len(self.proj)):
            B,L,C = tgt_feats[i].shape
            
            tgt_feats[i] = self.ln[i](tgt_feats[i])
            tgt_feats_proj.append(self.proj[i](tgt_feats[i]))
            
            if self.reciprocity:
                src_feats[i] = self.ln_src[i](src_feats[i])
                src_feats_proj.append(self.proj[i](src_feats[i]))
        
        
        tgt_feats = torch.stack(tgt_feats_proj, dim=1)
        attn_maps = torch.stack(attn_maps, dim=1)
        # attn_maps = self.mutual_nn_filter(attn_maps)
        refined_corr = self.decoder(attn_maps, tgt_feats)
        
        if self.args.reverse:
            src_feats = torch.stack(src_feats_proj, dim=1)
            attn_maps_source = torch.stack(attn_maps_source, dim=1)
            refined_corr_source = self.decoder(attn_maps_source, src_feats)
            refined_corr_target = refined_corr
            refined_corr = (refined_corr_source.transpose(-1,-2))
        elif self.reciprocity:
            src_feats = torch.stack(src_feats_proj, dim=1)
            attn_maps_source = torch.stack(attn_maps_source, dim=1)
            refined_corr_source = self.decoder(attn_maps_source, src_feats)
            refined_corr_target = refined_corr
            refined_corr = (refined_corr + refined_corr_source.transpose(-1,-2)) / 2
            

        
        if not self.cost_transformer:
            refined_corr = attn_maps.mean(dim=1) ## target source
            # refined_corr = (attn_maps.mean(dim=1) + attn_maps_source.mean(dim=1).transpose(-1,-2))/2.
            # refined_corr = self.corr(self.l2norm(feat_target.permute(0,2,1)),self.l2norm(feat_source.permute(0,2,1)))
            



        grid_x, grid_y = self.soft_argmax(refined_corr.transpose(-1,-2).view(B, -1, self.feature_size, self.feature_size),beta=2e-2)
        self.grid_x = grid_x
        self.grid_y = grid_y

        flow = torch.cat((grid_x, grid_y), dim=1)
        flow = unnormalise_and_convert_mapping_to_flow(flow)
        h, w = flow.shape[-2:]
        if self.output_interp:
            flow = F.interpolate(flow, size=output_shape, mode='bilinear', align_corners=False)
            flow[:, 0] *= float(output_shape[1]) / float(w)
            flow[:, 1] *= float(output_shape[0]) / float(h)

        return flow
    
    def get_FCN_map(self, img, feat_map, fc, sz):
        #scales = [1.0,1.5,2.0]
        scales = [1.0]
        map_list = []
        for scale in scales:
            if scale*scale*sz[0]*sz[1] > 1200*800:
                scale = 1.5
            img = F.interpolate(img, (int(scale*sz[0]),int(scale*sz[1])), None, 'bilinear', True) # 1x3xHxW
            #feat_map, fc = self.extract_intermediate_feat(img,return_hp=False,backbone='fcn101')
            # feat_map = self.backbone1.evaluate(img)
            
            predict = torch.max(feat_map, 1)[1]
            mask = predict-torch.min(predict)
            mask_map = mask / torch.max(mask)
            mask_map = F.interpolate(mask_map.unsqueeze(0).double(), (sz[0],sz[1]), None, 'bilinear', True)[0,0] # HxW
    
        return mask_map