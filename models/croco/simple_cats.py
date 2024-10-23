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

## uncertainty
from models.PDCNet.mod_uncertainty import MixtureDensityEstimatorFromCorr, MixtureDensityEstimatorFromUncertaintiesAndFlow
from models.croco.conv4d_coponerf import Conv4d


r'''
Modified timm library Vision Transformer implementation
https://github.com/rwightman/pytorch-image-models
'''
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


class MultiscaleBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_multiscale = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp2(self.norm4(x)))
        x = x.view(B, N, H, W).transpose(1, 2).flatten(0, 1) 
        x = x + self.drop_path(self.attn_multiscale(self.norm3(x)))
        x = x.view(B, H, N, W).transpose(1, 2).flatten(0, 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(B, N, H, W)
        return x


class TransformerAggregator(nn.Module):
    def __init__(self, num_hyperpixel, img_size=224, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.pos_embed_x = nn.Parameter(torch.zeros(1, num_hyperpixel, 1, img_size, embed_dim // 2))
        self.pos_embed_y = nn.Parameter(torch.zeros(1, num_hyperpixel, img_size, 1, embed_dim // 2))
        self.pos_drop = nn.Dropout(p=drop_rate)

        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            MultiscaleBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.proj = nn.Linear(embed_dim, img_size ** 2)
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

    def forward(self, attn_maps, source):
        B = attn_maps.shape[0]
        x = attn_maps.clone()
        
        pos_embed = torch.cat((self.pos_embed_x.repeat(1, 1, self.img_size, 1, 1), self.pos_embed_y.repeat(1, 1, 1, self.img_size, 1)), dim=4)
        pos_embed = pos_embed.flatten(2, 3)
        # x = torch.cat((x.transpose(-1, -2), target), dim=3) + pos_embed
        # x = self.proj(self.blocks(x)).transpose(-1, -2) + corr  # swapping the axis for swapping self-attention.
        x = torch.cat((x, source), dim=3) + pos_embed
        x = self.proj(self.blocks(x)) + attn_maps 

        return x.mean(1)


class FeatureExtractionHyperPixel(nn.Module):
    def __init__(self, hyperpixel_ids, feature_size, freeze=True):
        super().__init__()
        # self.backbone = resnet.resnet101(pretrained=True)
        self.feature_size = feature_size
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        nbottlenecks = [3, 4, 23, 3]
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.hyperpixel_ids = hyperpixel_ids
    
    
    def forward(self, img):
        r"""Extract desired a list of intermediate features"""

        feats = []

        # Layer 0
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)
        if 0 in self.hyperpixel_ids:
            feats.append(feat.clone())

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.layer_ids)):
            res = feat
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

            if bid == 0:
                res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

            feat += res

            if hid + 1 in self.hyperpixel_ids:
                feats.append(feat.clone())
                #if hid + 1 == max(self.hyperpixel_ids):
                #    break
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        # Up-sample & concatenate features to construct a hyperimage
        for idx, feat in enumerate(feats):
            feats[idx] = F.interpolate(feat, self.feature_size, None, 'bilinear', True)

        return feats


class CATs(nn.Module):
    def __init__(self,
    feature_size=16,
    feature_proj_dim=128,
    depth=4,
    num_heads=6,
    mlp_ratio=4,
    hyperpixel_ids=[0,8,20,21,26,28,29,30],
    output_interp=False,
    cost_transformer=True,
    args=None,
    conv4d=False,):
        super().__init__()
        self.feature_size = feature_size
        self.feature_proj_dim = feature_proj_dim
        self.decoder_embed_dim = self.feature_size ** 2 + self.feature_proj_dim
        self.cost_transformer=cost_transformer
        self.args=args
        self.correlation = getattr(args,'correlation',False)
        self.reciprocity = getattr(args,'reciprocity',False)
        self.occlusion_mask = getattr(args,'occlusion_mask',False)
        self.uncertainty = getattr(args,'uncertainty',False)
        self.give_layer_before_flow_to_uncertainty_decoder = True
        self.conv4d = conv4d

        if self.args.cost_agg == 'hierarchical_cats' or self.args.cost_agg == 'hierarchical_residual_cats' or self.args.cost_agg == 'hierarchical_conv4d_cats' or self.args.cost_agg == 'hierarchical_conv4d_cats_level' or self.args.cost_agg == 'hierarchical_conv4d_cats_level_4stage':
            channels = [1024]*12
        else:
            channels = [768]*12
        if self.correlation:
            channels = [1024]+channels
            hyperpixel_ids = hyperpixel_ids + [12]
            
        if self.uncertainty:
            self.corr_uncertainty_decoder4 = MixtureDensityEstimatorFromCorr(in_channels=1,
                                                                         batch_norm=True,
                                                                         search_size=14, output_channels=6,
                                                                         output_all_channels_together=True)

            if self.give_layer_before_flow_to_uncertainty_decoder:
                uncertainty_input_channels = 6 + 196
            else:
                uncertainty_input_channels = 6 + 2
            self.uncertainty_decoder4 = MixtureDensityEstimatorFromUncertaintiesAndFlow(in_channels=uncertainty_input_channels,
                                                                                        batch_norm=True,
                                                                                        output_channels=3)
            
        if self.conv4d:
            self.conv4d_seq = nn.Sequential(
                Conv4d(in_channels=1, out_channels=16, kernel_size=[3,3,3,3], stride=[1,1,2,2], padding=[1,1,1,1]),
                nn.GroupNorm(4,16),
                nn.ReLU(),
                Conv4d(in_channels=16, out_channels=64, kernel_size=[3,3,3,3], stride=[1,1,2,2], padding=[1,1,1,1]),
                nn.GroupNorm(4,64),
                nn.ReLU(),
                Conv4d(in_channels=64, out_channels=128, kernel_size=[3,3,3,3], stride=[1,1,2,2], padding=[1,1,1,1]),
                nn.GroupNorm(4,128),
                nn.ReLU(),
                )



        # self.feature_extraction = FeatureExtractionHyperPixel(hyperpixel_ids, feature_size, freeze)
        self.ln = nn.ModuleList([nn.LayerNorm(channels[i]) for i in hyperpixel_ids])
        if self.reciprocity:
            self.ln_src = nn.ModuleList([nn.LayerNorm(channels[i]) for i in hyperpixel_ids])
        
        self.proj = nn.ModuleList([ 
            nn.Linear(channels[i], self.feature_proj_dim) for i in hyperpixel_ids
        ])

        self.decoder = TransformerAggregator(
            img_size=self.feature_size, embed_dim=self.decoder_embed_dim, depth=depth, num_heads=num_heads,
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
    
    def estimate_uncertainty_components(self, corr_uncertainty_module, uncertainty_predictor,
                                        corr_type, corr, c_target, c_source, flow, up_previous_flow=None,
                                        up_previous_uncertainty=None, global_local='global_corr'):
        # corr uncertainty decoder
        x_second_corr = None
        if 'gocor' in corr_type.lower():
            if self.corr_for_corr_uncertainty_decoder == 'gocor':
                input_corr_uncertainty_dec = corr
            elif self.corr_for_corr_uncertainty_decoder == 'corr':
                input_corr_uncertainty_dec = getattr(self, global_local)(c_target, c_source)

            elif self.corr_for_corr_uncertainty_decoder == 'corr_and_gocor':
                input_corr_uncertainty_dec = getattr(self, global_local)(c_target, c_source)
                x_second_corr = corr
            else:
                raise NotImplementedError
        else:
            input_corr_uncertainty_dec = corr

        input_corr_uncertainty_dec = input_corr_uncertainty_dec.transpose(-1, -2).reshape(input_corr_uncertainty_dec.size(0), -1, self.feature_size, self.feature_size)
        corr_uncertainty = corr_uncertainty_module(input_corr_uncertainty_dec, x_second_corr=x_second_corr)

        flow = flow.transpose(-1, -2).reshape(flow.size(0), -1, self.feature_size, self.feature_size)
        # final uncertainty decoder
        if up_previous_flow is not None and up_previous_uncertainty is not None:
            input_uncertainty = torch.cat((corr_uncertainty, flow,
                                           up_previous_uncertainty, up_previous_flow), 1)
        else:
            input_uncertainty = torch.cat((corr_uncertainty, flow), 1)

        large_log_var_map, weight_map = uncertainty_predictor(input_uncertainty)
        return large_log_var_map, weight_map
    
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
            
        if self.uncertainty:
            uncertainty4 = self.estimate_uncertainty_components(self.corr_uncertainty_decoder4,
                                                                        self.uncertainty_decoder4,
                                                                        'corr',
                                                                        attn_maps[:,0], None, None, refined_corr,
                                                                        global_local='use_global_corr_layer')
            # large_log_var_map4 = self.constrain_large_log_var_map(torch.tensor(2.0), torch.tensor(0.0), large_log_var_map4)
            # small_log_var_map4 = torch.ones_like(large_log_var_map4, requires_grad=False) * torch.log(torch.tensor(1.0))
            # log_var_map4 = torch.cat((small_log_var_map4, large_log_var_map4), 1)
            
            # log_var_map4 = F.interpolate(log_var_map4, size=output_shape, mode='bilinear', align_corners=False)
            uncertainty4 = F.interpolate(uncertainty4, size=output_shape, mode='bilinear', align_corners=False)
            
        
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

        if self.occlusion_mask:
            grid_x_target, grid_y_target = self.soft_argmax(refined_corr_target.transpose(-1,-2).view(B, -1, self.feature_size, self.feature_size),beta=2e-2)
            flow_target = torch.cat((grid_x_target, grid_y_target), dim=1)
            flow_target = unnormalise_and_convert_mapping_to_flow(flow_target)
            if self.output_interp:
                flow_target = F.interpolate(flow_target, size=output_shape, mode='bilinear', align_corners=False)
                flow_target[:, 0] *= float(output_shape[1]) / float(w)
                flow_target[:, 1] *= float(output_shape[0]) / float(h)
            
            # grid_x_source, grid_y_source = self.soft_argmax(refined_corr_source.transpose(-1,-2).view(B, -1, self.feature_size, self.feature_size),beta=2e-2)
            grid_x_source, grid_y_source = self.soft_argmax(refined_corr_source.view(B, -1, self.feature_size, self.feature_size),beta=2e-2)
            
            flow_source = torch.cat((grid_x_source, grid_y_source), dim=1)
            flow_source = unnormalise_and_convert_mapping_to_flow(flow_source)
            if self.output_interp:
                flow_source = F.interpolate(flow_source, size=output_shape, mode='bilinear', align_corners=False)
                flow_source[:, 0] *= float(output_shape[1]) / float(w)
                flow_source[:, 1] *= float(output_shape[0]) / float(h)
            
            return flow, flow_target,flow_source
        
        if self.conv4d:
            PH, PW = output_shape[0]//16, output_shape[1]//16
            refined_corr = refined_corr.view(B,PH, PW, PH, PW).unsqueeze(dim=1)
            refined_corr = self.conv4d_seq(refined_corr)
            bsz, ch, ha, wa, hb, wb = refined_corr.size()
            refined_corr = refined_corr.view(bsz, ch, ha, wa, -1).mean(dim=-1)
            
            return flow, refined_corr

        if self.uncertainty:
            
            output = {1:{'dense_flow': flow,
                    'dense_certainty': uncertainty4}}
            return output
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