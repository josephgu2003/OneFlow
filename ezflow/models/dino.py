import torch 
import torch.nn as nn

from ezflow.models.build import MODEL_REGISTRY
from ezflow.models.dit import DecoderBlock, get_2d_sincos_pos_embed
from ezflow.modules.base_module import BaseModule 

def reparameterize(x, hw):
    h, w = hw[0], hw[1]
    x = torch.nn.functional.interpolate(x, size=hw, mode='bilinear')
    hs = torch.arange(h, device=x.device)
    ws = torch.arange(w, device=x.device)
    
    wh = torch.tensor(hw[::-1], device=x.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    flow = (x[:, :2, :, :] + 0.5) * wh - torch.stack(torch.meshgrid(hs, ws, indexing='ij'))
    return flow 

def simple_interpolate(x, size):
    return torch.nn.functional.interpolate(x, size=size, mode='bilinear')

@MODEL_REGISTRY.register()
class Dino(BaseModule):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg.HIDDEN_SIZE
        self.vits16 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        self.proj = nn.Linear(384, cfg.HIDDEN_SIZE)
        self.final_layer = nn.Linear(cfg.HIDDEN_SIZE, 2 * 14 * 14)
        num_patches = 32 * 32
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, cfg.HIDDEN_SIZE), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(cfg.HIDDEN_SIZE, cfg.NUM_HEADS) for _ in range(cfg.DECODER_BLOCKS)
        ])
        self.init_weights()
    
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
        c = 2
        p = 14
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
        for blk in self.vits16.blocks:
            x = blk(x)
        x = self.vits16.norm(x)
        x = x[:, 1 + 4:]
        
        x = self.proj(x)
        
        x1x2 = torch.chunk(x, 2)
        x1 = x1x2[0] + self.pos_embed 
        x2 = x1x2[1] + self.pos_embed 
        
        q = x1 
        
        for block in self.decoder_blocks:
            q = block(q, x1, x2, None)
            
        q = self.final_layer(q) 
        q = self.unpatchify(q)
        q = reparameterize(q, hw)
        return {'flow_preds': q, "flow_upsampled": q}