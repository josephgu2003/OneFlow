from functools import partial
import torch 
import torch.nn as nn

from ezflow.encoder.dinov2.backbones import dinov2_vits14_reg
from ezflow.encoder.dinov2.block import Block
from ezflow.encoder.dinov2.native_attention import NativeAttention
from ezflow.encoder.dinov2.vision_transformer import DinoVisionTransformer, vit_small
from ezflow.models.build import MODEL_REGISTRY
from ezflow.models.dit import get_2d_sincos_pos_embed
from ezflow.modules.decoder import DecoderBlock
from ezflow.modules.base_module import BaseModule 
from ezflow.utils.invert_flow import reparameterize

def simple_interpolate(x, size):
    return torch.nn.functional.interpolate(x, size=size, mode='bilinear')

@MODEL_REGISTRY.register()
class Dino(BaseModule):
    def __init__(self, cfg):
        super().__init__()
        self.scaler = cfg.SCALER
        self.hidden_size = cfg.HIDDEN_SIZE
        self.proj = nn.Linear(384, cfg.HIDDEN_SIZE)
        self.z_dim = -1
        self.final_layer = nn.Linear(cfg.HIDDEN_SIZE, self.z_dim)
        num_patches = 16 * 16 * self.scaler * self.scaler
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, cfg.HIDDEN_SIZE), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(cfg.HIDDEN_SIZE, cfg.NUM_HEADS) for _ in range(cfg.DECODER_BLOCKS)
        ])
        self.init_weights()
        self.vits16 = dinov2_vits14_reg(block_fn=partial(Block, attn_class=NativeAttention))
        
        self.vqgan = None
        for param in self.vqgan.parameters():
            param.requires_grad_(False)
        self.vqgan.eval()
      
    def init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.orthogonal_(module.weight, 0.1)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
       
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, grid_size=16 * self.scaler) 
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.z_dim
        p = 1
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
 
    def encode_flow(self, flow_gt): 
        self.vqgan.eval()
        h, w = flow_gt.size(2), flow_gt.size(3)
        flow = torch.nn.functional.interpolate(flow_gt, size=(256, 256), mode='bilinear', align_corners=True)
        
        return self.vqgan.encode(torch.concat((flow, torch.zeros_like(flow[:, 0:1, :, :])), dim=1))[0]
        
    
    def forward(self, both_img):        
        self.vqgan.eval()
        hw = both_img.shape[2:]
        both_img = simple_interpolate(both_img, size=(224, 224)) # TODO: fix

        x = self.vits16.prepare_tokens_with_masks(both_img, None)
        
        x1x2 = torch.chunk(x, 2)
        img1 = x1x2[0] # TODO: img encoding??
        img2 = x1x2[1]
        
        x = torch.concat((img1, img2), dim=1)
        
        for blk in self.vits16.blocks:
            x = blk(x)
        x = self.vits16.norm(x)
        
        x = self.proj(x)
        
        x1x2 = torch.chunk(x, 2, dim=1)
        x1 = x1x2[0][:, 1+4:] + self.pos_embed
        x2 = x1x2[1][:, 1+4:] + self.pos_embed
        
        q = x1
        
        for block in self.decoder_blocks:
            q = block(q, x1, x2, None)
            
        q = self.final_layer(q) 
        q = self.unpatchify(q)

        latents = q

        if self.training:
            return {'latents': latents}
        else:
            with torch.no_grad():
                q = self.vqgan.decode(q)
                flow = torch.nn.functional.interpolate(q[:, :2, :, :], size=hw, mode='bilinear', align_corners=True)
                
            return {'latents': latents, 'flow_preds': flow, "flow_upsampled": flow}
