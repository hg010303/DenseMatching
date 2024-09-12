import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Union, Tuple, Iterable, List, Optional, Dict
from .cats import MultiscaleBlock
from timm.models.layers import DropPath, trunc_normal_


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    scratch.layer_rn = nn.ModuleList([
        scratch.layer1_rn,
        scratch.layer2_rn,
        scratch.layer3_rn,
        scratch.layer4_rn,
    ])

    return scratch

class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        width_ratio=1,
    ):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()
        self.width_ratio = width_ratio

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            if self.width_ratio != 1:
                res = F.interpolate(res, size=(output.shape[2], output.shape[3]), mode='bilinear')

            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        if self.width_ratio != 1:
            # and output.shape[3] < self.width_ratio * output.shape[2]
            #size=(image.shape[])
            if (output.shape[3] / output.shape[2]) < (2 / 3) * self.width_ratio:
                shape = 3 * output.shape[3]
            else:
                shape = int(self.width_ratio * 2 * output.shape[2])
            output  = F.interpolate(output, size=(2* output.shape[2], shape), mode='bilinear')
        else:
            output = nn.functional.interpolate(output, scale_factor=2,
                    mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output

def make_fusion_block(features, use_bn, width_ratio=1):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        width_ratio=width_ratio,
    )

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


class CRAFT(nn.Module):
    """DPT output adapter.

    :param num_cahnnels: Number of output channels
    :param stride_level: tride level compared to the full-sized image.
        E.g. 4 for 1/4th the size of the image.
    :param patch_size_full: Int or tuple of the patch size over the full image size.
        Patch size for smaller inputs will be computed accordingly.
    :param hooks: Index of intermediate layers
    :param layer_dims: Dimension of intermediate layers
    :param feature_dim: Feature dimension
    :param last_dim: out_channels/in_channels for the last two Conv2d when head_type == regression
    :param use_bn: If set to True, activates batch norm
    :param dim_tokens_enc:  Dimension of tokens coming from encoder
    """

    def __init__(self,
                 num_channels: int = 1,
                 stride_level: int = 1,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 main_tasks: Iterable[str] = ('rgb',),
                 hooks: List[int] = [2, 5, 8, 11],
                 layer_dims: List[int] = [96, 192, 384, 768],
                 feature_dim: int = 256,
                 last_dim: int = 32,
                 use_bn: bool = False,
                 dim_tokens_enc: Optional[int] = None,
                 head_type: str = 'regression',
                 output_width_ratio=1,
                 max_depth = 80.,
                 residual = False,
                 args = None,
                 **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size = pair(patch_size)
        self.main_tasks = main_tasks
        self.hooks = hooks
        self.layer_dims = layer_dims
        self.feature_dim = feature_dim
        self.dim_tokens_enc = dim_tokens_enc * len(self.main_tasks) if dim_tokens_enc is not None else None
        self.head_type = head_type
        self.max_depth=max_depth
        self.residual = residual
        self.args = args

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size[0] // stride_level)
        self.P_W = max(1, self.patch_size[1] // stride_level)

        self.scratch = make_scratch(layer_dims, feature_dim, groups=1, expand=False)

        self.attention_aggregator0 = nn.Sequential(MultiscaleBlock(196, 7, 196), MultiscaleBlock(196, 7, 196),nn.Linear(196, 196))
        self.attention_aggregator1 = nn.Sequential(MultiscaleBlock(196, 7, 196), MultiscaleBlock(196, 7, 196),nn.Linear(196, 196))
        self.attention_aggregator2 = nn.Sequential(MultiscaleBlock(196, 7, 196), MultiscaleBlock(196, 7, 196),nn.Linear(196, 196))
        self.attention_aggregator3 = nn.Sequential(MultiscaleBlock(196, 7, 196), MultiscaleBlock(196, 7, 196),nn.Linear(196, 196))
            
        feature = 196
        
        self.aggregator0 = nn.Sequential( nn.GELU(),
                                          nn.Conv2d(256+feature, 256, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(256, 256+feature, kernel_size=3, stride=1, padding=1))
        self.aggregator1 = nn.Sequential(nn.GELU(),
                                        nn.Conv2d(256+feature, 256, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(256, 256+feature, kernel_size=3, stride=1, padding=1))
        self.aggregator2 = nn.Sequential( nn.GELU(),
                                          nn.Conv2d(256+feature, 256, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(256, 256+feature, kernel_size=3, stride=1, padding=1))
        self.aggregator3 = nn.Sequential( nn.GELU(),
                                          nn.Conv2d(256+feature, 256, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(256, 256+feature, kernel_size=3, stride=1, padding=1))
        
        self.proj = nn.ModuleList([nn.Conv2d(256+feature, feature,  kernel_size=1, stride=1, padding=0) for i in range(4)])
        self.tmp = nn.Parameter(torch.zeros(1,196,56,56))
        
        self.apply(self._init_weights)
        
        
    def init(self, dim_tokens_enc=768):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        """
        #print(dim_tokens_enc)

        # Set up activation postprocessing layers
        if isinstance(dim_tokens_enc, int):
            dim_tokens_enc = 4 * [dim_tokens_enc]

        self.dim_tokens_enc = [dt * len(self.main_tasks) for dt in dim_tokens_enc]

        self.act_1_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[0],
                out_channels=self.layer_dims[0],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=self.layer_dims[0],
                out_channels=self.layer_dims[0],
                kernel_size=4, stride=4, padding=0,
                bias=True, dilation=1, groups=1,
            )
        )

        self.act_2_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[1],
                out_channels=self.layer_dims[1],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=self.layer_dims[1],
                out_channels=self.layer_dims[1],
                kernel_size=2, stride=2, padding=0,
                bias=True, dilation=1, groups=1,
            )
        )

        self.act_3_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[2],
                out_channels=self.layer_dims[2],
                kernel_size=1, stride=1, padding=0,
            )
        )

        self.act_4_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[3],
                out_channels=self.layer_dims[3],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.Conv2d(
                in_channels=self.layer_dims[3],
                out_channels=self.layer_dims[3],
                kernel_size=3, stride=2, padding=1,
            )
        )

        self.act_postprocess = nn.ModuleList([
            self.act_1_postprocess,
            self.act_2_postprocess,
            self.act_3_postprocess,
            self.act_4_postprocess
        ])
        
    

    def adapt_tokens(self, encoder_tokens):
        # Adapt tokens
        x = []
        x.append(encoder_tokens[:, :])
        x = torch.cat(x, dim=-1)
        return x
    
    def r6d2mat(self,d6: torch.Tensor) -> torch.Tensor:
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalisation per Section B of [1].
        Args:
            d6: 6D rotation representation, of size (*, 6). Here corresponds to the two
                first two rows of the rotation matrix. 
        Returns:
            batch of rotation matrices of size (*, 3, 3)
        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """

        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)  # corresponds to row

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            # trunc_normal_(m.weight, std=.02)
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)     
        
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, encoder_tokens: List[torch.Tensor], image_size, attn_map,intrinsics=None):
            #input_info: Dict:
        outputs={}
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        H, W = image_size
        
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]
        if self.residual:
            layers = [layers[0]+layers[4], layers[1]+layers[5], layers[2]+layers[6], layers[3]+layers[7]]
        attn_maps = [(attn_map[hook] + attn_map[hook-1] + attn_map[hook-2])/3. for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(l) for l in layers]
        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # attn_maps = [l.mean(dim=1) for l in attn_maps]
        attn_maps[3] = self.attention_aggregator3(attn_maps[3].unsqueeze(dim=1)).squeeze(dim=1)
        attn_maps[2] = self.attention_aggregator2(attn_maps[2].unsqueeze(dim=1)).squeeze(dim=1)
        attn_maps[1] = self.attention_aggregator1(attn_maps[1].unsqueeze(dim=1)).squeeze(dim=1)
        attn_maps[0] = self.attention_aggregator0(attn_maps[0].unsqueeze(dim=1)).squeeze(dim=1)
        
            
        
        attn_maps = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in attn_maps]
        attn_sizes = [(7,7),(14,14),(28,28),(56,56)]
        
        attn_map = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in attn_map]
        
        attn3_croco = F.interpolate(attn_maps[3], size=attn_sizes[0], mode='bilinear')
        attn_input3 = torch.cat([attn3_croco, layers[3]], dim=1)
        attn3_out = self.aggregator3(attn_input3) + attn_input3
        attn3_out = self.proj[3](attn3_out) + F.interpolate(attn_map[3], size=attn_sizes[0], mode='bilinear')
        
        attn2_croco = F.interpolate(attn3_out, size=attn_sizes[1], mode='bilinear')
        attn_maps[2] = F.interpolate(attn_maps[2], size=attn_sizes[1], mode='bilinear')
        attn2 = attn2_croco + attn_maps[2] 
        
        attn_input2 = torch.cat([attn2, layers[2]], dim=1)
        attn2_out = self.aggregator2(attn_input2) + attn_input2
        attn2_out = self.proj[2](attn2_out) + F.interpolate(attn_map[2], size=attn_sizes[1], mode='bilinear')

        attn1_croco = F.interpolate(attn2_out, size=attn_sizes[2], mode='bilinear')
        attn_maps[1] = F.interpolate(attn_maps[1], size=attn_sizes[2], mode='bilinear')
        attn1 = attn1_croco + attn_maps[1]
        
        attn_input1 = torch.cat([attn1, layers[1]], dim=1)
        attn1_out = self.aggregator1(attn_input1) + attn_input1
        attn1_out = self.proj[1](attn1_out) + F.interpolate(attn_map[1], size=attn_sizes[2], mode='bilinear')
        
        
        attn0_croco = F.interpolate(attn1_out, size=attn_sizes[3], mode='bilinear')
        attn_maps[0] = F.interpolate(attn_maps[0], size=attn_sizes[3], mode='bilinear')
        attn0 = attn0_croco + attn_maps[0]
        
        attn_input0 = torch.cat([attn0, layers[0]], dim=1)
        attn0_out = self.aggregator0(attn_input0) + attn_input0
        attn0_out = self.proj[0](attn0_out) + F.interpolate(attn_map[0], size=attn_sizes[3], mode='bilinear')
        # attn0_out =  F.interpolate(attn_map[0], size=attn_sizes[3], mode='bilinear') + self.tmp
        

        return [attn0_out, attn1_out, attn2_out, attn3_out]