
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from torch.jit import Final
from timm.layers import use_fused_attn




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
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.3, bias=True)
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
        q = q + self.mlp(q)
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
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.3, bias=True)
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
        q = q + self.mlp(q)
        q = self.norm3(q)
        # add gelu here?
        return q


