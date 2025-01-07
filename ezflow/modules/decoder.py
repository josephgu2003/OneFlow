
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from torch.jit import Final
from timm.layers import use_fused_attn

import math 
from torch.nn.init import trunc_normal_ 
from timm.layers import DropPath


class DWConv(nn.Module):
    def __init__(self, in_features, out_features=None, act_layer=nn.GELU, kernel_size=3):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(
            in_features, in_features, kernel_size=kernel_size, padding=padding, groups=in_features)
        self.act = act_layer()
        self.conv2 = torch.nn.Conv2d(
            in_features, out_features, kernel_size=kernel_size, padding=padding, groups=out_features)

    def forward(self, x, H=32, W=32):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)


        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)
        return x


class WierdMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.gamma = nn.Parameter(torch.ones(hidden_features), requires_grad=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H=32, W=32):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.drop(self.gamma * self.dwconv(x, H, W)) + x
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ChannelProcessing(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., linear=False, drop_path=0., 
                 mlp_hidden_dim=None, act_layer=nn.GELU, drop=None, norm_layer=nn.LayerNorm, cha_sr_ratio=1, c_head_num=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        num_heads = c_head_num or num_heads
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.cha_sr_ratio = cha_sr_ratio if num_heads > 1 else 1

        # config of mlp for v processing
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_v = WierdMlp(in_features=dim//self.cha_sr_ratio, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)
        self.norm_v = norm_layer(dim//self.cha_sr_ratio)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def _gen_attn(self, q, k):
        q = q.softmax(-2).transpose(-1,-2)
        _, _, N, _  = k.shape
        k = torch.nn.functional.adaptive_avg_pool2d(k.softmax(-2), (N, 1))
        
        attn = torch.nn.functional.sigmoid(q @ k)
        return attn  * self.temperature
    def forward(self, x, H=32, W=32, atten=None):
        B, N, C = x.shape
        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.num_heads,  C // self.num_heads).permute(0, 2, 1, 3)

        attn = self._gen_attn(q, k)
        attn = self.attn_drop(attn)

        Bv, Hd, Nv, Cv = v.shape
        v = self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd*Cv))).reshape(Bv, Nv, Hd, Cv).transpose(1, 2)

        repeat_time = N // attn.shape[-1]
        attn = attn.repeat_interleave(repeat_time, dim=-1) if attn.shape[-1] > 1 else attn
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        return x,  (attn * v.transpose(-1, -2)).transpose(-1, -2) #attn
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}



class CrossAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = True, # TODO: !!
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q_mat = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_mat = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.qk_norm = qk_norm
     #   self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
      #  self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, N1, C = q.shape
        B, N2, C = x.shape
        q = self.q_mat(q).reshape(B, N1, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        kv = self.kv_mat(x).reshape(B, N2, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q = q.squeeze(0)
        k, v = kv.unbind(0)
       
        if self.qk_norm:
            q = torch.nn.functional.normalize(q, p=2, dim=-1) # TODO: this is wrong
            k = torch.nn.functional.normalize(k, p=2, dim=-1) 
        #q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class DecoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=False, qk_norm=False, proj_drop=0.3, **block_kwargs)
        self.attn2 = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=False, qk_norm=False, proj_drop=0.3, **block_kwargs)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = ChannelProcessing(dim=hidden_size, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, act_layer=approx_gelu, drop=0.3)
        self.norm1 = torch.nn.LayerNorm(hidden_size)
        self.norm2 = torch.nn.LayerNorm(hidden_size)
        self.norm3 = torch.nn.LayerNorm(hidden_size)

    def forward(self, q, x1, x2, c):
    #    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
    
        # reinject
        q = q + x1
        
        # self attn
        x1 = self.attn(q)
        q = q + x1
        
        q = self.norm1(q)
        
        # cross attn
        x2 = self.attn2(q, x2)
        q = q + x2
        
        q = self.norm2(q)
        
        # mlp
        q = q + self.mlp(q)[0]
        q = self.norm3(q)
        # add gelu here?
        return q

class ConcatenateDecoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=False, qk_norm=False, proj_drop=0.3, **block_kwargs)
        self.attn2 = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=False, qk_norm=False, proj_drop=0.3, **block_kwargs)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = ChannelProcessing(dim=hidden_size, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, act_layer=approx_gelu, drop=0.3)
        self.norm1 = torch.nn.LayerNorm(hidden_size)
        self.norm2 = torch.nn.LayerNorm(hidden_size)
        self.norm3 = torch.nn.LayerNorm(hidden_size)

    def forward(self, q, x1, x2, c):
        # self attn
        out = self.attn(q)
        q = q + out
        
        q = self.norm1(q)
        
        # cross attn
        feats = torch.concat((x1, x2), dim=1)
        out = self.attn2(q, feats)
        q = q + out
        
        q = self.norm2(q)
        
        # mlp
        q = q + self.mlp(q)[0]
        q = self.norm3(q)
        # add gelu here?
        return q


