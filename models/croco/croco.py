# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# CroCo model during pretraining
# --------------------------------------------------------



import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True # for gpu >= Ampere and pytorch >= 1.12
from functools import partial

from models.croco.blocks import Block, DecoderBlock, PatchEmbed
from models.croco.pos_embed import get_2d_sincos_pos_embed, RoPE2D 
from models.croco.masking import RandomMask
from models.croco.simple_cats import CATs
from models.croco.cats_swin import CATs_SWIN
from models.croco.cats_swin_decoder import CATs_SWIN_Decoder

from .craft import CRAFT
from .mod import FeatureL2Norm, unnormalise_and_convert_mapping_to_flow
from einops import rearrange, repeat
import numpy as np
from .hierarchical_cats import HierarchicalCATs
from models.croco.head_downstream import PixelwiseTaskWithDPT
import torch.nn.functional as F



class CroCoNet(nn.Module):

    def __init__(self,
                 img_size=224,           # input image size
                 patch_size=16,          # patch_size 
                 mask_ratio=0.9,         # ratios of masked tokens 
                 enc_embed_dim=768,      # encoder feature dimension
                 enc_depth=12,           # encoder depth 
                 enc_num_heads=12,       # encoder number of heads in the transformer block 
                 dec_embed_dim=512,      # decoder feature dimension 
                 dec_depth=8,            # decoder depth 
                 dec_num_heads=16,       # decoder number of heads in the transformer block 
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_im2_in_dec=True,   # whether to apply normalization of the 'memory' = (second image) in the decoder 
                 pos_embed='cosine',     # positional embedding (either cosine or RoPE100)
                 attn_map_output=False,  # whether to output the attention maps
                 output_interp=True,
                 args=None,
                 cats_depth = 4,
                ):
                
        super(CroCoNet, self).__init__()
         
        self.cost_agg = args.cost_agg
        self.attn_map_output = attn_map_output or self.cost_agg
        self.cost_transformer = args.cost_transformer
        self.kwargs = args
        self.hierarchical = getattr(args,'hierarchical',False)
        self.occlusion_mask = getattr(args,'occlusion_mask',False)
        self.output_interp = output_interp
        
        self.reciprocity = getattr(args, 'reciprocity', False)
        if self.cost_agg=='cats':
            self.cats = CATs(feature_size=(img_size[0]//16), hyperpixel_ids = [i for i in range(0, 12)], output_interp=output_interp, cost_transformer=self.cost_transformer, args=args,depth=cats_depth)
        elif self.cost_agg=='CRAFT':
            self.craft = CRAFT(args=args, dim_tokens_enc=768)#dim_tokens_enc=)
            self.craft.init()
            self.x_normal = np.linspace(-1,1,14)
            self.x_normal = nn.Parameter(torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False))
            self.y_normal = np.linspace(-1,1,14)
            self.y_normal = nn.Parameter(torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False))
            
            self.x_normal_rev = np.linspace(-1,1,56)
            self.x_normal_rev = nn.Parameter(torch.tensor(self.x_normal_rev, dtype=torch.float, requires_grad=False))
            self.y_normal_rev = np.linspace(-1,1,56)
            self.y_normal_rev = nn.Parameter(torch.tensor(self.y_normal_rev, dtype=torch.float, requires_grad=False))
        elif self.cost_agg=='hierarchical_cats' or self.cost_agg=='hierarchical_residual_cats':
            # self.cats1 = CATs(feature_size=(img_size[0]//16), hyperpixel_ids = [i for i in range(0, 4)], output_interp=False, cost_transformer=self.cost_transformer, args=args,depth=1)
            self.cats2 = CATs(feature_size=(img_size[0]//16), hyperpixel_ids = [i for i in range(0, 4)], output_interp=False, cost_transformer=self.cost_transformer, args=args,depth=1)
            self.cats3 = CATs(feature_size=(img_size[0]//16), hyperpixel_ids = [i for i in range(0, 4)], output_interp=False, cost_transformer=self.cost_transformer, args=args,depth=1)
            self.cats4 = CATs(feature_size=(img_size[0]//16), hyperpixel_ids = [i for i in range(0, 4)], output_interp=False, cost_transformer=self.cost_transformer, args=args,depth=1)
            
            self.hierarchical_cats = HierarchicalCATs(dim_tokens_enc = 768+2, hooks = [0,1,2,], args=args)
            
        elif self.cost_agg=='hierarchical_conv4d_cats' or self.cost_agg=='hierarchical_conv4d_cats_level':
            # self.cats1 = CATs(feature_size=(img_size[0]//16), hyperpixel_ids = [i for i in range(0, 4)], output_interp=False, cost_transformer=self.cost_transformer, args=args,depth=1)
            self.cats2 = CATs(feature_size=(img_size[0]//16), hyperpixel_ids = [i for i in range(0, 4)], output_interp=False, cost_transformer=self.cost_transformer, args=args,depth=1, conv4d=True)
            self.cats3 = CATs(feature_size=(img_size[0]//16), hyperpixel_ids = [i for i in range(0, 4)], output_interp=False, cost_transformer=self.cost_transformer, args=args,depth=1, conv4d=True)
            self.cats4 = CATs(feature_size=(img_size[0]//16), hyperpixel_ids = [i for i in range(0, 4)], output_interp=False, cost_transformer=self.cost_transformer, args=args,depth=1, conv4d=True)
            
            dim_tokens_enc = 768 if self.cost_agg=='hierarchical_conv4d_cats' else 768 + 128
            self.hierarchical_cats = HierarchicalCATs(dim_tokens_enc = dim_tokens_enc, hooks = [0,1,2,], args=args, conv4d_feature = 128*3)
            
        elif self.cost_agg=='hierarchical_conv4d_cats_level_4stage':
            self.cats = CATs(feature_size=(img_size[0]//16), hyperpixel_ids = [i for i in range(0, 12)], output_interp=False, cost_transformer=self.cost_transformer, args=args,depth=4, conv4d=True)
            
            dim_tokens_enc = 768
            self.hierarchical_cats = HierarchicalCATs(dim_tokens_enc = dim_tokens_enc, hooks = [0,1,2,3], args=args, conv4d_feature = 128, depth=args.cats_depth)
            
        elif self.cost_agg=='cats_swin':
            self.cats_swin = CATs_SWIN(feature_size=(img_size[0]//16), hyperpixel_ids = [i for i in range(0, 12)], output_interp=output_interp, cost_transformer=self.cost_transformer, args=args,depth=cats_depth)
            
        elif self.cost_agg=='cats_swin_decoder':
            self.cats_swin_decoder = CATs_SWIN_Decoder(feature_size=(img_size[0]//16), hyperpixel_ids = [i for i in range(0, 12)], output_interp=output_interp, cost_transformer=self.cost_transformer, args=args,depth=cats_depth)
            
            
        elif self.cost_agg == 'croco_flow':
            self.head = PixelwiseTaskWithDPT()
            self.head.num_channels = 2
            
        
        # patch embeddings  (with initialization done as in MAE)
        self._set_patch_embed(img_size, patch_size, enc_embed_dim)

        # mask generations
        self._set_mask_generator(self.patch_embed.num_patches, mask_ratio)

        self.pos_embed = pos_embed
        if pos_embed=='cosine':
            # positional embedding of the encoder 
            enc_pos_embed = get_2d_sincos_pos_embed(enc_embed_dim, int(self.patch_embed.num_patches**.5), n_cls_token=0)
            self.register_buffer('enc_pos_embed', torch.from_numpy(enc_pos_embed).float())
            # positional embedding of the decoder  
            dec_pos_embed = get_2d_sincos_pos_embed(dec_embed_dim, int(self.patch_embed.num_patches**.5), n_cls_token=0)
            self.register_buffer('dec_pos_embed', torch.from_numpy(dec_pos_embed).float())
            # pos embedding in each block
            self.rope = None # nothing for cosine 
        elif pos_embed.startswith('RoPE'): # eg RoPE100 
            self.enc_pos_embed = None # nothing to add in the encoder with RoPE
            self.dec_pos_embed = None # nothing to add in the decoder with RoPE
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(pos_embed[len('RoPE'):])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError('Unknown pos_embed '+pos_embed)

        # transformer for the encoder 
        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)
        
        # masked tokens 
        self._set_mask_token(dec_embed_dim)

        # decoder 
        self._set_decoder(enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec,softmaxattn=args.softmaxattn)
        
        # prediction head 
        self._set_prediction_head(dec_embed_dim, patch_size)
        
        # initializer weights
        self.initialize_weights()   
        
        if self.cost_agg == 'croco_flow':
            self.head.setup(self)        

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_embed_dim)

    def _set_mask_generator(self, num_patches, mask_ratio):
        self.mask_generator = RandomMask(num_patches, mask_ratio)
        
    def _set_mask_token(self, dec_embed_dim):
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        
    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec, softmaxattn=False):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder 
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder 
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope, softmaxattn=softmaxattn)
            for i in range(dec_depth)])
        # final norm layer 
        self.dec_norm = norm_layer(dec_embed_dim)
        
    def _set_prediction_head(self, dec_embed_dim, patch_size):
         self.prediction_head = nn.Linear(dec_embed_dim, patch_size**2 * 3, bias=True)
        
        
    def initialize_weights(self):
        # patch embed 
        self.patch_embed._init_weights()
        # mask tokens
        if self.mask_token is not None: torch.nn.init.normal_(self.mask_token, std=.02)
        # linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # if isinstance(m, nn.Linear):
        #     # we use xavier_uniform following official JAX ViT:
        #     torch.nn.init.xavier_uniform_(m.weight)
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _encode_image(self, image, do_mask=False, return_all_blocks=False):
        """
        image has B x 3 x img_size x img_size 
        do_mask: whether to perform masking or not
        return_all_blocks: if True, return the features at the end of every block 
                           instead of just the features from the last block (eg for some prediction heads)
        """
        # embed the image into patches  (x has size B x Npatches x C) 
        # and get position if each return patch (pos has size B x Npatches x 2)
        x, pos = self.patch_embed(image)              
        # add positional embedding without cls token  
        if self.enc_pos_embed is not None: 
            x = x + self.enc_pos_embed[None,...]
        # apply masking 
        B,N,C = x.size()
        if do_mask:
            masks = self.mask_generator(x)
            x = x[~masks].view(B, -1, C)
            posvis = pos[~masks].view(B, -1, 2)
        else:
            B,N,C = x.size()
            masks = torch.zeros((B,N), dtype=bool)
            posvis = pos
        # now apply the transformer encoder and normalization        
        if return_all_blocks:
            out = []
            for blk in self.enc_blocks:
                x = blk(x, posvis)
                out.append(x)
            out[-1] = self.enc_norm(out[-1])
            return out, pos, masks
        else:
            for blk in self.enc_blocks:
                x = blk(x, posvis)
            x = self.enc_norm(x)
            return x, pos, masks
 
    def _decoder(self, feat1, pos1, masks1, feat2, pos2, return_all_blocks=False):
        """
        return_all_blocks: if True, return the features at the end of every block 
                           instead of just the features from the last block (eg for some prediction heads)
                           
        masks1 can be None => assume image1 fully visible 
        """
        # encoder to decoder layer 
        visf1 = self.decoder_embed(feat1)
        f2 = self.decoder_embed(feat2)
        # append masked tokens to the sequence
        B,Nenc,C = visf1.size()
        if masks1 is None: # downstreams
            f1_ = visf1
        else: # pretraining 
            Ntotal = masks1.size(1)
            f1_ = self.mask_token.repeat(B, Ntotal, 1).to(dtype=visf1.dtype)
            f1_[~masks1] = visf1.view(B * Nenc, C)
        # add positional embedding
        if self.dec_pos_embed is not None:
            f1_ = f1_ + self.dec_pos_embed
            f2 = f2 + self.dec_pos_embed
        # apply Transformer blocks
        out = f1_
        out2 = f2 
        attn_maps = []
        if return_all_blocks:
            _out, out = out, []
            for blk in self.dec_blocks:
                _out, out2, attn_map = blk(_out, out2, pos1, pos2)
                out.append(_out)
                attn_maps.append(attn_map)
            out[-1] = self.dec_norm(out[-1])
        else:
            for blk in self.dec_blocks:
                out, out2, attn_map = blk(out, out2, pos1, pos2)
                attn_maps.append(attn_map)
            out = self.dec_norm(out)
            
        if self.attn_map_output:
            return out, attn_maps
        return out, None

    def patchify(self, imgs):
        """
        imgs: (B, 3, H, W)
        x: (B, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        
        return x

    def unpatchify(self, x, channels=3):
        """
        x: (N, L, patch_size**2 *channels)
        imgs: (N, 3, H, W)
        """
        patch_size = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], channels, h * patch_size, h * patch_size))
        return imgs
    
    def softmax_with_temperature(self, x, beta, d = 1):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M # subtract maximum value for stability
        exp_x = torch.exp(x/beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum
    
    def soft_argmax(self, corr, beta=0.02):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        b,th,tw,h,w = corr.size()
        corr = corr.view(b,th*tw,h,w)
        
        corr = self.softmax_with_temperature(corr, beta=beta, d=1)
        corr = corr.view(-1,th,tw,h,w) # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
        x_normal = self.x_normal.expand(b,tw)
        x_normal = x_normal.view(b,tw,1,1)
        grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        
        grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
        y_normal = self.y_normal.expand(b,th)
        y_normal = y_normal.view(b,th,1,1)
        grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        return grid_x, grid_y

    def soft_argmax_rev(self, corr, beta=0.02):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        b,th,tw,h,w = corr.size()
        corr = corr.view(b,th*tw,h,w)
        
        corr = self.softmax_with_temperature(corr, beta=beta, d=1)
        corr = corr.view(-1,th,tw,h,w) # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
        x_normal_rev = self.x_normal_rev.expand(b,tw)
        x_normal_rev = x_normal_rev.view(b,tw,1,1)
        grid_x = (grid_x*x_normal_rev).sum(dim=1, keepdim=True) # b x 1 x h x w
        
        grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
        y_normal_rev = self.y_normal_rev.expand(b,th)
        y_normal_rev = y_normal_rev.view(b,th,1,1)
        grid_y = (grid_y*y_normal_rev).sum(dim=1, keepdim=True) # b x 1 x h x w
        return grid_x, grid_y

    def forward(self, img_target, img_source,mode=None):
        """
        img1: tensor of size B x 3 x img_size x img_size
        img2: tensor of size B x 3 x img_size x img_size
        
        out will be    B x N x (3*patch_size*patch_size)
        masks are also returned as B x N just in case 
        """
        B,_,H,W = img_target.size()
        feat_targets, pos_target, mask_target = self._encode_image(img_target, do_mask=False, return_all_blocks=True)
        feat_sources, pos_source, mask_source = self._encode_image(img_source, do_mask=False, return_all_blocks=True)

        feat_target = feat_targets[-1]
        feat_source = feat_sources[-1]

        # decoder
        decfeat, attn_map = self._decoder(feat_target, pos_target, mask_target, feat_source, pos_source, return_all_blocks=True)
        if self.reciprocity:
            decfeat_source, attn_map_source = self._decoder(feat_source, pos_source, mask_source, feat_target, pos_target, return_all_blocks=True)
        
        
        ## heuristic attention refine
        attn_map = [attn.mean(dim=1).detach() for attn in attn_map]
        for i in range(len(attn_map)):
            attn_map[i][:,:,0]=attn_map[i].min()
        self.attn_map = attn_map
        if self.reciprocity:
            attn_map_source = [attn.mean(dim=1).detach() for attn in attn_map_source]
            for i in range(len(attn_map_source)):
                attn_map_source[i][:,:,0]=attn_map_source[i].min()    
            
        
        if self.cost_agg == 'cats':
            decfeat = [feat.detach() for feat in decfeat]
            if self.reciprocity:
                decfeat_source = [feat.detach() for feat in decfeat_source]
                
                if self.occlusion_mask:
                    out, out_target, out_source = self.cats(attn_map, decfeat, (H,W), feat_source, feat_target, attn_map_source, decfeat_source, img_target, img_source)
                    return out,out_target,out_source
                
                out = self.cats(attn_map, decfeat, (H,W), feat_source, feat_target, attn_map_source, decfeat_source, img_target, img_source)
                
            else:
                out = self.cats(attn_map, decfeat, (H,W), feat_source, feat_target)
            
            return out
        
        elif self.cost_agg == 'cats_swin':
            decfeat = [feat.detach() for feat in decfeat]
            if self.reciprocity:
                decfeat_source = [feat.detach() for feat in decfeat_source]
                
                out = self.cats_swin(attn_map, decfeat, (H,W), feat_source, feat_target, attn_map_source, decfeat_source, img_target, img_source)
                
            else:
                out = self.cats_swin(attn_map, decfeat, (H,W), feat_source, feat_target)
            
            return out
        
        elif self.cost_agg == 'cats_swin_decoder':
            decfeat = [feat.detach() for feat in decfeat]
            if self.reciprocity:
                decfeat_source = [feat.detach() for feat in decfeat_source]
                
                out = self.cats_swin_decoder(attn_map, decfeat, (H,W), feat_source, feat_target, attn_map_source, decfeat_source, img_target, img_source, appearance_feature = [feat_targets[8],feat_targets[16]])
                
            else:
                out = self.cats_swin_decoder(attn_map, decfeat, (H,W), feat_source, feat_target, appearance_feature = [feat_targets[8],feat_targets[16]])
            
            return out
        
        elif self.cost_agg == 'hierarchical_cats' or self.cost_agg == 'hierarchical_residual_cats':
            assert self.reciprocity, "reciprocity must be True for hierarchical_cats"
            encfeat_targets = [feat for feat in feat_targets]
            encfeat_sources = [feat for feat in feat_sources]
                        
            aggregates_flow1 = self.cats4(attn_map[0:4], encfeat_targets[0:8:2], (H,W), feat_source, feat_target, attn_map_source[0:4], encfeat_sources[0:4])
            aggregates_flow2 = self.cats3(attn_map[4:8], encfeat_targets[8:16:2], (H,W), feat_source, feat_target, attn_map_source[4:8], encfeat_sources[4:8])
            aggregates_flow3 = self.cats2(attn_map[8:12], encfeat_targets[16:24:2], (H,W), feat_source, feat_target, attn_map_source[8:12], encfeat_sources[8:12])            
            
            aggregates_flows = [aggregates_flow1, aggregates_flow2, aggregates_flow3]
            enc_feats = [decfeat[3], decfeat[7], decfeat[11]]
            
            outputs = self.hierarchical_cats(enc_feats,aggregates_flows, (H,W))
            outputs = outputs + [aggregates_flow1, aggregates_flow2, aggregates_flow3]
            return outputs

        elif self.cost_agg == 'hierarchical_conv4d_cats' or self.cost_agg == 'hierarchical_conv4d_cats_level':
            assert self.reciprocity, "reciprocity must be True for hierarchical_cats"
            encfeat_targets = [feat.detach() for feat in feat_targets]
            encfeat_sources = [feat.detach() for feat in feat_sources]
                        
            aggregates_flow1, conv4d_feature1 = self.cats4(attn_map[0:4], encfeat_targets[0:8:2], (H,W), feat_source, feat_target, attn_map_source[0:4], encfeat_sources[0:4])
            aggregates_flow2, conv4d_feature2 = self.cats3(attn_map[4:8], encfeat_targets[8:16:2], (H,W), feat_source, feat_target, attn_map_source[4:8], encfeat_sources[4:8])
            aggregates_flow3, conv4d_feature3 = self.cats2(attn_map[8:12], encfeat_targets[16:24:2], (H,W), feat_source, feat_target, attn_map_source[8:12], encfeat_sources[8:12])            
            
            aggregates_flows = [conv4d_feature1, conv4d_feature2, conv4d_feature3]
            enc_feats = [decfeat[3], decfeat[7], decfeat[11]]
            
            outputs = self.hierarchical_cats(enc_feats,aggregates_flows, (H,W))
            outputs = outputs + [aggregates_flow1, aggregates_flow2, aggregates_flow3]
            return outputs
        
        elif self.cost_agg == 'hierarchical_conv4d_cats_level_4stage':
            assert self.reciprocity, "reciprocity must be True for hierarchical_cats"
            encfeat_targets = [feat.detach() for feat in feat_targets]
            encfeat_sources = [feat.detach() for feat in feat_sources]
                        
            # aggregates_flow1, conv4d_feature1 = self.cats4(attn_map[0:3], encfeat_targets[0:6:2], (H,W), feat_source, feat_target, attn_map_source[0:3], encfeat_sources[0:6:2])
            # aggregates_flow2, conv4d_feature2 = self.cats3(attn_map[3:6], encfeat_targets[6:12:2], (H,W), feat_source, feat_target, attn_map_source[3:6], encfeat_sources[6:12:2])
            # aggregates_flow3, conv4d_feature3 = self.cats2(attn_map[6:9], encfeat_targets[12:18:2], (H,W), feat_source, feat_target, attn_map_source[6:9], encfeat_sources[12:18:2])            
            # aggregates_flow4, conv4d_feature4 = self.cats1(attn_map[9:12], encfeat_targets[18:24:2], (H,W), feat_source, feat_target, attn_map_source[9:12], encfeat_sources[18:24:2])            
            
            aggregates_flow, conv4d_feature = self.cats(attn_map, decfeat, (H,W), feat_source, feat_target, attn_map_source, decfeat_source)
            
            
            aggregates_flows = [conv4d_feature]
            enc_feats = [decfeat[2], decfeat[5], decfeat[8], decfeat[11]]
            
            outputs = self.hierarchical_cats(enc_feats,aggregates_flows, (H,W))
            outputs = outputs + [aggregates_flow]
            
            if self.output_interp:
                for i in range(len(outputs)):
                    H_output, W_output = outputs[i].shape[-2:]
                    outputs[i] = F.interpolate(outputs[i], size=(H,W), mode='bilinear', align_corners=False)
                    outputs[i][:,0] = outputs[i][:,0] * (W/W_output)
                    outputs[i][:,1] = outputs[i][:,1] * (H/H_output)    
                                
            return outputs
        
        elif self.cost_agg == 'CRAFT':
            out = self.craft(decfeat, (H,W), attn_map)
            # out[-1] = rearrange(out[-1].transpose(-2,-1)+self.craft.tmp, 'b (sh sw) (th tw) -> b sh sw th tw', sh=14, sw=14, th=14, tw=14) 
            if self.hierarchical:
                predicted_flow = []
                
                for i in range(len(out)):
                    out[i] = rearrange(out[i], 'b (sh sw) th tw -> b sh sw th tw', sh=14, sw=14) 
                    
                    grid_x, grid_y = self.soft_argmax(out[i], beta=0.02)
                    flow = torch.cat((grid_x, grid_y), dim=1)
                    flow = unnormalise_and_convert_mapping_to_flow(flow)
                    predicted_flow.append(flow)
                    
                return predicted_flow
            
            
            out[-1] = rearrange(out[-1], 'b (sh sw) th tw -> b sh sw th tw', sh=14, sw=14) 
            
            grid_x, grid_y = self.soft_argmax(out[-1], beta=0.02)
            flow = torch.cat((grid_x, grid_y), dim=1)
            flow = unnormalise_and_convert_mapping_to_flow(flow)
            

            if self.reciprocity:
                out = self.craft(decfeat_source, (H,W), attn_map_source)
                # out[-1] = rearrange(out[-1].transpose(-2,-1)+self.craft.tmp, 'b (sh sw) (th tw) -> b sh sw th tw', sh=14, sw=14, th=14, tw=14) 
                out[-1] = rearrange(out[-1], 'b (sh sw) th tw -> b th tw sh sw', sh=14, sw=14) 
                
                grid_x, grid_y = self.soft_argmax_rev(out[-1], beta=0.02)
                flow_reci = torch.cat((grid_x, grid_y), dim=1)
                flow_reci = unnormalise_and_convert_mapping_to_flow(flow_reci)
                return flow, flow_reci
            
            return flow
        
        elif self.cost_agg == 'croco_flow':
            decfeat = feat_targets + decfeat
            img_info = {'height': H, 'width': W}
            return self.head(decfeat, img_info)
        
        return out, mask_target, None
