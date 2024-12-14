import torch 
import torch.nn as nn

from ezflow.models.build import MODEL_REGISTRY
from ezflow.models.dit import DecoderBlock
from ezflow.modules.base_module import BaseModule 

def reparameterize(x, hw):
    x = torch.nn.functional.interpolate(x, size=hw, mode='bilinear')
    hs = torch.arange(hw[0], device=x.device)
    ws = torch.arange(hw[1], device=x.device)
    
    wh = torch.tensor(hw[::-1], device=x.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    flow = (x[:, :2, :, :] + 0.5) * wh - torch.stack(torch.meshgrid(hs, ws, indexing='ij'))
    x = flow 
    return x 

def simple_interpolate(x, size):
    return torch.nn.functional.interpolate(x, size=size, mode='bilinear')

@MODEL_REGISTRY.register()
class DiT(BaseModule):
    def __init__(self, cfg):
        super().__init__()
        self.vits16 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        self.proj = nn.Linear(384, cfg.HIDDEN_SIZE)
        num_patches = 32 * 32
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, cfg.HIDDEN_SIZE), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(cfg.HIDDEN_SIZE, cfg.NUM_HEADS) for _ in range(cfg.DECODER_BLOCKS)
        ])
        
    def forward(self, both_img):
        hw = both_img.shape[2:]
        both_img = simple_interpolate(both_img)
        
        x = self.vits16.prepare_tokens_with_masks(both_img, None)
        for blk in self.vits16.blocks:
            x = blk(x)
        x = self.vits16.norm(x)
        x = x[:, 1 + 4:]
        
        x1x2 = torch.chunk(x, 2)
        x1 = x1x2[0] + self.pos_embed 
        x2 = x1x2[1] + self.pos_embed 
        
        q = x1 
        
        for block in self.decoder_blocks:
            q = block(q, x1, x2, None)
        
        x = reparameterize(q, hw)
        return {'flow_preds': x, "flow_upsampled": x}