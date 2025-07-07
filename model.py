import math
import random
import time
import os
import numpy as np
from collections import OrderedDict
from functools import partial
from typing import Optional, Union
from pos_embed import get_2d_sincos_pos_embed
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from torchcomplex.complex_layers import ComplexConv1d, ComplexBatchNorm2d, ComplexLinear
import torchcomplex.complex_functions as Complex_F
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
from transformers import GPT2Tokenizer, GPT2Model
from CommUtils import *
from ops.bra_legacy import BiLevelRoutingAttention
import time
from _common import Attention, AttentionLePE, DWConv


def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    # if name == 'sum':
    #     return Summer(PositionalEncodingPermute2D(emb_dim))
    # elif name == 'npe.sin':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='sin')
    # elif name == 'npe.coord':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='coord')
    # elif name == 'hpe.conv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='conv', res_shortcut=True)
    # elif name == 'hpe.dsconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='dsconv', res_shortcut=True)
    # elif name == 'hpe.pointconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='pointconv', res_shortcut=True)
    else:
        raise ValueError(f'PE name {name} is not surpported!')


class BiBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                 num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 mlp_ratio=4, mlp_dwconv=False,
                 side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                                qk_scale=qk_scale, kv_per_win=kv_per_win,
                                                kv_downsample_ratio=kv_downsample_ratio,
                                                kv_downsample_kernel=kv_downsample_kernel,
                                                kv_downsample_mode=kv_downsample_mode,
                                                topk=topk, param_attention=param_attention, param_routing=param_routing,
                                                diff_routing=diff_routing, soft_routing=soft_routing,
                                                side_dwconv=side_dwconv,
                                                auto_pad=auto_pad)
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w c -> n c h w'),  # compatiability
                                      nn.Conv2d(dim, dim, 1),  # pseudo qkv linear
                                      nn.Conv2d(dim, dim, 5, padding=2, groups=dim),  # pseudo attention
                                      nn.Conv2d(dim, dim, 1),  # pseudo out linear
                                      Rearrange('n c h w -> n h w c')
                                      )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
                                 DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio * dim), dim)
                                 )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))  # (N, H, W, C)
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, H, W, C)
        else:  # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x)))  # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x)))  # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class BiFormer(nn.Module):
    def __init__(self, depth=[3, 4, 8, 3], in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 head_dim=64, qk_scale=None, representation_size=None,
                 drop_path_rate=0., drop_rate=0.,
                 use_checkpoint_stages=[],
                 ########
                 n_win=7,
                 kv_downsample_mode='ada_avgpool',
                 kv_per_wins=[2, 2, -1, -1],
                 topks=[8, 8, -1, -1],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[None, None, None, None],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=False,
                 # -----------------------
                 kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1],  # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios=[4, 4, 4, 4],
                 param_attention='qkvo',
                 mlp_dwconv=False):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()
        # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        )
        if (pe is not None) and 0 in pe_stages:
            stem.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
        if use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i + 1])
            )
            if (pe is not None) and i + 1 in pe_stages:
                downsample_layer.append(get_pe_layer(emb_dim=embed_dim[i + 1], name=pe))
            if use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)
        ##########################################################################

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[BiBlock(dim=embed_dim[i], drop_path=dp_rates[cur + j],
                          layer_scale_init_value=layer_scale_init_value,
                          topk=topks[i],
                          num_heads=nheads[i],
                          n_win=n_win,
                          qk_dim=qk_dims[i],
                          qk_scale=qk_scale,
                          kv_per_win=kv_per_wins[i],
                          kv_downsample_ratio=kv_downsample_ratios[i],
                          kv_downsample_kernel=kv_downsample_kernels[i],
                          kv_downsample_mode=kv_downsample_mode,
                          param_attention=param_attention,
                          param_routing=param_routing,
                          diff_routing=diff_routing,
                          soft_routing=soft_routing,
                          mlp_ratio=mlp_ratios[i],
                          mlp_dwconv=mlp_dwconv,
                          side_dwconv=side_dwconv,
                          before_attn_dwconv=before_attn_dwconv,
                          pre_norm=pre_norm,
                          auto_pad=auto_pad) for j in range(depth[i])],
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]

        ##########################################################################
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)  # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
            x = self.stages[i](x)
        # x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        return x

#################### model variants #######################


model_urls = {
    "biformer_tiny_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHEOoGkgwgQzEDlM/root/content",
    "biformer_small_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHDyM-x9KWRBZ832/root/content",
    "biformer_base_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHI_XPhoadjaNxtO/root/content",
}


# https://github.com/huggingface/pytorch-image-models/blob/4b8cfa6c0a355a9b3cb2a77298b240213fb3b921/timm/models/_factory.py#L93

@register_model
def biformer_tiny(pretrained=False, pretrained_cfg=None,
                  pretrained_cfg_overlay=None, **kwargs):
    model = BiFormer(
        depth=[2, 2, 8, 2],
        embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
        # ------------------------------
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,

        # -------------------------------
        **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        # model_key = 'biformer_tiny_in1k'
        # url = model_urls[model_key]
        # checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True,
        #                                                 file_name=f"{model_key}.pth")
        # model.load_state_dict(checkpoint["model"])
        checkpoint = torch.load("checkpoints/SE.pth", map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint, strict=True)
    return model

class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.real_bn = nn.BatchNorm1d(num_features)
        self.imag_bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        real = self.real_bn(x.real)
        imag = self.imag_bn(x.imag)
        return torch.complex(real, imag)

class ComplexResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = ComplexConv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = ComplexBatchNorm1d(out_channels)  # 规范化通道维度 (C)
        self.residual = ComplexConv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv(x)
        x = self.norm(x)  # 只对通道维度进行正则化
        x = Complex_F.complex_relu(x)
        x = x + residual
        return x


class SignalSemanticFeatureExtractor(nn.Module):
    def __init__(self, input_length=6000, output_length=49, pretrain=False):
        super().__init__()
        self.input_length = input_length
        self.pretrain = pretrain
        # 定义三个复数残差模块
        self.block1 = ComplexResidualBlock(in_channels=10, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.block2 = ComplexResidualBlock(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
        self.block3 = ComplexResidualBlock(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2)
        # 计算最终的序列长度
        self.feature_length = math.ceil(input_length / 8)  # 假设经过 3 层 stride=2 的卷积

        # 全连接层，将特征映射到指定输出长度
        self.fc = nn.Linear(1500,output_length)


        self.head_1 = nn.Linear(49*512, 1)
        # 输出感知目标距离
        self.head_2 = nn.Linear(49*512, 1)
        # 输出感知目标的速度
        self.head_3 = nn.Linear(49*512, 1)

    def forward(self, x):
        # 通过每个残差模块
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # 实部和虚部展开并通过全连接层
        x = torch.cat([x.real, x.imag], dim=-1)
        x = self.fc(x)
        if self.pretrain:
            x = x.view(x.shape[0],-1)
            angle = F.sigmoid(self.head_1(x))
            distance = F.sigmoid(self.head_2(x))
            rate = F.sigmoid(self.head_3(x))
            return angle,distance,rate
        else:
            x = x.permute(0, 2, 1)  # 调整输出形状
            return x


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.2):
        """
        初始化交叉注意力模块。
        :param dim: 输入特征的维度。
        :param num_heads: 多头注意力中的头数。
        :param dropout: dropout 概率。
        """
        super(CrossAttentionFusion, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        """
        前向传播。
        :param x1: 输入特征 1，形状为 (batch_size, dim)。
        :param x2: 输入特征 2，形状为 (batch_size, dim)。
        :return: 融合后的特征，形状为 (batch_size, dim)。
        """
        # 将输入特征调整为 (seq_len, batch_size, dim) 形式
        x1 = x1.permute(1, 0, 2)
        x2 = x2.permute(1, 0, 2)

        # 计算交叉注意力
        q1 = self.query(x1)  # 对 x1 计算 Query
        k2 = self.key(x2)    # 对 x2 计算 Key
        v2 = self.value(x2)  # 对 x2 计算 Value
        fusion1, _ = self.attention(q1, k2, v2)  # x1 查询 x2 信息

        q2 = self.query(x2)  # 对 x2 计算 Query
        k1 = self.key(x1)    # 对 x1 计算 Key
        v1 = self.value(x1)  # 对 x1 计算 Value
        fusion2, _ = self.attention(q2, k1, v1)  # x2 查询 x1 信息

        # 将两个融合特征相加并归一化
        fusion = self.norm(fusion1 + fusion2)
        # fusion = self.norm(fusion)

        # 添加 Dropout 和残差连接
        fusion = self.dropout(fusion) + x1 + x2

        # 转回原始格式 (batch_size, seq_len, dim)
        fusion = fusion.permute(1, 0, 2)
        return fusion

# MSF
class MultimodalSemanticFusion(nn.Module):
    def __init__(self, input_length=6000, output_length=49, dims = 512):
        super().__init__()
        self.SigSE = SignalSemanticFeatureExtractor(input_length, output_length)
        # checkpoint = torch.load("checkpoints/signal1.pth", map_location='cpu', weights_only=True)
        # self.SigSE.load_state_dict(checkpoint)
        # self.SigSE.requires_grad_(False)
        self.ImgSE = biformer_tiny(pretrained=True)
        # self.ImgSE.requires_grad_(False)
        self.CrossAttn = CrossAttentionFusion(dim=dims,num_heads=8)
        self.linear = nn.Linear(512,768)
        self.norm = nn.LayerNorm(dims)

    def forward(self,signal, image):
        sig_sf = self.SigSE(signal)
        img_sf = self.ImgSE(image)
        img_sf = img_sf.permute(0, 2, 3, 1)
        img_sf = img_sf.view(img_sf.shape[0],img_sf.shape[1]*img_sf.shape[2],-1)
        sig_sf = self.norm(sig_sf)
        img_sf = self.norm(img_sf)
        mul_sf = F.gelu(self.CrossAttn(img_sf, sig_sf))
        mul_sf = self.linear(mul_sf)
        return mul_sf

# LSE
class LLMSemanticEncoder(nn.Module):
    def __init__(self):
        super(LLMSemanticEncoder, self).__init__()
        # 将特征维度映射到 token 空间
        self.llm = GPT2Model.from_pretrained("checkpoints/GPT-2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("checkpoints/GPT-2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 使用 eos_token 作为填充符号
        # self.pool = nn.MaxPool1d(2,2)


    def forward(self, MulSF, channel_info, device="cuda"):
        tokenized_texts = self.tokenizer(channel_info, return_tensors="pt", padding="max_length", truncation=True, max_length=49)
        input_ids = tokenized_texts["input_ids"].to(device)  # (2, seq_len)
        attention_mask = tokenized_texts["attention_mask"].to(device)  # (2, seq_len)

        # 将文本输入映射到 GPT-2 嵌入空间
        text_embeddings = self.llm.wte(input_ids)  # (2, seq_len, 768)
        # 拼接特征和文本嵌入
        fused_input = torch.cat([MulSF, text_embeddings], dim=1)  # (2, 98, 768)

        # 构造 Attention Mask
        feature_mask = torch.ones((MulSF.size(0), MulSF.size(1))).to(device)  # (2, 98, 768)
        fused_attention_mask = torch.cat([feature_mask, attention_mask], dim=1)  # (2, 98, 768)

        # 使用 GPT-2 提取融合后的特征
        outputs = self.llm(inputs_embeds=fused_input, attention_mask=fused_attention_mask)
        SemanticEncoding = outputs.last_hidden_state  # (2, 98, 768)
        SE = F.tanh(SemanticEncoding)
        return SE

def WirelessCommunication(SE, modulator, Eb_N0_dB):
    SE_Bit = (SE > 0).float()
    mdtor.m_type = modulator
    # 调制
    modulated_symbols = mdtor.modulate(SE_Bit)

    # 信道传输
    received_symbols = chmodel(modulated_symbols, Eb_N0_dB)
    # 解调
    demodulated_bits = mdtor.demodulate(received_symbols,SE_Bit.shape)
    # if not os.path.exists(f"logs/{Eb_N0_dB}_Constellation.pdf"):
    #     mdtor.Constellation_draw(modulated_symbols, received_symbols, f"logs/{Eb_N0_dB}_Constellation.pdf")
    demodulated_bits = F.tanh(demodulated_bits)
    return demodulated_bits

class VitDecoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # self.blocks = nn.ModuleList([
        #     Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward(self, x):
        x = self.Upsample(x)
        bs, fs, w, h = x.shape
        x = x.view(bs, fs, -1)
        x = x.permute(0, 2, 1)
        # x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        x = self.unpatchify(x)
        # x = F.tanh(x)
        return x

# SSD
class SensingSemanticDecoder(nn.Module):
    def __init__(self):
        super(SensingSemanticDecoder, self).__init__()
        # self.linear = nn.Linear(384,256)
        self.backbone = nn.Sequential(
            nn.Conv2d(1536, 512, 3, 1, 1),
            nn.ReLU(),
        )
        self.head_1 = VitDecoder()
        checkpoint = torch.load("checkpoints/SD.pth", map_location='cpu',weights_only=True)
        self.head_1.load_state_dict(checkpoint,strict=True)
        self.conv = nn.Conv2d(512,128,3,1,1)
        self.head_2 = nn.Linear(128 * 7 * 7, 1)
        self.head_3 = nn.Linear(128 * 7 * 7, 1)
        self.head_4 = nn.Linear(128 * 7 * 7, 1)

    def forward(self, se):
        se = se.view(se.shape[0], -1, 7, 7)
        se = self.backbone(se)
        rec_img = self.head_1(se)


        x = F.relu(self.conv(se))

        x = x.view(x.shape[0],-1)

        angle = F.sigmoid(self.head_2(x))

        distance = F.sigmoid(self.head_3(x))

        rate = F.sigmoid(self.head_4(x))

        return rec_img, angle, distance, rate

mdtor = ModulatorDemodulator()
chmodel = PhysicalChannel("AWGN")


class SIMAC(nn.Module):
    def __init__(self,input_length=6000, output_length=49, dims=512):
        super().__init__()
        self.MSF = MultimodalSemanticFusion(input_length=input_length,output_length=output_length,dims=dims)
        self.LSE = LLMSemanticEncoder()
        self.SSD = SensingSemanticDecoder()


    def forward(self, signal, img, modulator, snr, ChannelInfo, device="cuda"):
        SF = self.MSF(signal, img)
        SE = self.LSE(SF, ChannelInfo, device=device)
        se = WirelessCommunication(SE, modulator, snr)
        p_sensing_img, p_angle, p_distance, p_rate = self.SSD(se)
        return p_sensing_img, p_angle, p_distance, p_rate


if __name__ == '__main__':
    batch_size = 32
    input_dim = 6000
    signal_tensor = torch.randn(batch_size, 10, input_dim, dtype=torch.complex64).cuda()  # Create a complex input tensor
    image = torch.randn((batch_size, 3, 224, 224)).cuda()


    model = SIMAC().cuda()
    params = list(model.LSE.parameters())
    num_params = sum([p.numel() for p in params])
    print("LSE parameters: {}".format(num_params))

    params = list(model.MSF.parameters())
    num_params = sum([p.numel() for p in params])
    print("MSF parameters: {}".format(num_params))

    params = list(model.SSD.parameters())
    num_params = sum([p.numel() for p in params])
    print("SSD parameters: {}".format(num_params))

    params = list(model.parameters())
    num_params = sum([p.numel() for p in params])
    print("SIMAC parameters: {}".format(num_params))

    snr = 25
    modulator = "8PSK"
    ChannelInfo = [f"the SNR is {snr} dB, the signal modulation is {modulator}"]*batch_size
    t = time.time()
    rec_img,rate,distance,cls = model(signal_tensor, image, modulator, snr, ChannelInfo, device="cuda")
    print("SIMAC run time: ", time.time() - t)
    from ptflops import get_model_complexity_info

    dummy_input1 = (10, 6000)  # signal shape
    dummy_input2 = (3, 224, 224)
    MSF = MultimodalSemanticFusion(input_length=6000, output_length=49, dims=512).cuda()
    LSE = LLMSemanticEncoder().cuda()
    SSD = SensingSemanticDecoder().cuda()

    with torch.no_grad():
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(MSF, (signal_tensor, image))
        print("MSF FLOPs: ", flops.total())
    t = time.time()
    SF = MSF(signal_tensor, image)
    print("MSF run time: ", time.time() - t)
    with torch.no_grad():
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(LSE, (SF, ChannelInfo))
        print("LSE FLOPs: ", flops.total())

    t = time.time()
    SE = LSE(SF, ChannelInfo, device="cuda")
    print("LSE run time: ", time.time() - t)

    with torch.no_grad():
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(SSD, (SE))
        print("SSD FLOPs: ", flops.total())
    t = time.time()
    res = SSD(SE)
    print("SSD run time: ", time.time() - t)
    with torch.no_grad():
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, (signal_tensor, image, modulator, snr, ChannelInfo))
        print("SIMAC FLOPs: ", flops.total())
