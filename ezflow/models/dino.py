import torch 
import torch.nn as nn

from ezflow.models.build import MODEL_REGISTRY
from ezflow.models.dit import DecoderBlock, get_2d_sincos_pos_embed
from ezflow.modules.base_module import BaseModule 
from ezflow.utils.invert_flow import reparameterize

def simple_interpolate(x, size):
    return torch.nn.functional.interpolate(x, size=size, mode='bilinear')

@MODEL_REGISTRY.register()
class Dino(BaseModule):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg.HIDDEN_SIZE
        self.img1_embed = nn.Parameter(torch.randn(1, 1, 384), requires_grad=True)
        self.img2_embed = nn.Parameter(torch.randn(1, 1, 384), requires_grad=True)
        self.proj = nn.Linear(384, cfg.HIDDEN_SIZE)
        self.final_layer = nn.Linear(cfg.HIDDEN_SIZE, 3)
        num_patches = 32 * 32
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, cfg.HIDDEN_SIZE), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(cfg.HIDDEN_SIZE, cfg.NUM_HEADS) for _ in range(cfg.DECODER_BLOCKS)
        ])
        self.init_weights()
        self.vits16 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    
    def init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.orthogonal_(module.weight, 0.1)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
       
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, grid_size=32) 
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = 3
        p = 1
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def forward(self, both_img):        
        hw = both_img.shape[2:]
        both_img = simple_interpolate(both_img, size=(448, 448))

        x = self.vits16.prepare_tokens_with_masks(both_img, None)
        
        x1x2 = torch.chunk(x, 2)
        img1 = x1x2[0] + self.img1_embed
        img2 = x1x2[1] + self.img2_embed
        
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

        flow = reparameterize(q[:, :2, :, :], hw)
        #var = simple_interpolate(flow[:, -1:, :, :], size=hw)
        var = torch.nn.functional.interpolate(q[:, -1:, :, :], size=hw, mode='bilinear', align_corners=True)

        return {'flow_preds': flow, "flow_upsampled": flow, 'var': var}