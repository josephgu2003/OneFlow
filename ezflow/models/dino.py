from functools import partial
import torch 
import torch.nn as nn

from ezflow.encoder.dinov2.backbones import dinov2_vits14_reg
from ezflow.encoder.dinov2.block import Block
from ezflow.encoder.dinov2.native_attention import NativeAttention
from ezflow.encoder.dinov2.vision_transformer import DinoVisionTransformer, vit_small
from ezflow.models.build import MODEL_REGISTRY
from ezflow.models.dit import get_2d_sincos_pos_embed
from ezflow.modules.decoder import DecoderBlock, ConcatenateDecoderBlock
from ezflow.modules.base_module import BaseModule 
from ezflow.utils.invert_flow import reparameterize
from ezflow.decoder.dpt import DPTHead
from ezflow.encoder.croco.croco_downstream import CroCoDownstreamBinocular
from ezflow.encoder.croco.croco import CroCoNet
from ezflow.encoder.croco.head_downstream import PixelwiseTaskWithDPT

def simple_interpolate(x, size):
    return torch.nn.functional.interpolate(x, size=size, mode='bilinear')

@MODEL_REGISTRY.register()
class Dino(BaseModule):
    def __init__(self, cfg):
        super().__init__()
        self.scaler = cfg.SCALER
        self.hidden_size = cfg.HIDDEN_SIZE
        self.proj = nn.Linear(768, 384)
        # self.final_layer = nn.Linear(cfg.HIDDEN_SIZE, 3)
        self.dpt = DPTHead(384)
        num_patches = 32 * 32 * self.scaler * self.scaler
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, cfg.HIDDEN_SIZE), requires_grad=False)
        
        block_classes = {'DecoderBlock': DecoderBlock, 'ConcatenateDecoderBlock': ConcatenateDecoderBlock}
        block_class = block_classes[cfg.BLOCK_CLASS]
        assert cfg.DECODER_BLOCKS % 4 == 0
        self.decoder_blocks = nn.ModuleList([
            block_class(cfg.HIDDEN_SIZE, cfg.NUM_HEADS) for _ in range(cfg.DECODER_BLOCKS)
        ])
        self.out_indices = list([i for i in range(0, cfg.DECODER_BLOCKS, cfg.DECODER_BLOCKS // 4)])
        self.init_weights()
        ckpt = torch.load('./pretrained_models/CroCo_V2_ViTBase_BaseDecoder.pth', 'cpu')
        
        model = CroCoNet( **ckpt.get('croco_kwargs',{}))
    
        msg = model.load_state_dict(ckpt['model'], strict=True)
        print(msg)
    
        self.croco = model
        
        self.encoder_concat = cfg.ENCODER_CONCAT 
        self.reparam = cfg.REPARAM
      
    def init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.orthogonal_(module.weight, 0.1)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
       
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, grid_size=32 * self.scaler) 
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
    
    def forward(self, img1, img2):
        both_img = torch.concat((img1, img2))        
        hw = both_img.shape[2:]
        both_img = simple_interpolate(both_img, size=(448 * self.scaler, 448 * self.scaler)) # TODO: fix

        x, pos = self.croco.patch_embed(both_img)
       
        if self.encoder_concat: 
            x1x2 = torch.chunk(x, 2)
            img1 = x1x2[0] # TODO: img encoding??
            img2 = x1x2[1]
        
            x = torch.concat((img1, img2), dim=1)
            pos = torch.concat((torch.chunk(pos, 2)),dim=1)
        
        feats = []
        for i, blk in enumerate(self.croco.enc_blocks[:12]):
            x = blk(x, pos)
            if i in [2, 5, 8, 11]:
                feats.append([self.proj(x[:,:x.size(1)//2])])

        q = self.dpt(feats, 28, 28, 448, 448)

        if self.reparam:
            flow = reparameterize(q[:, :2, :, :], hw)
        else:
            flow = torch.nn.functional.interpolate(q[:, :2, :, :], size=hw, mode='bilinear', align_corners=True)
            wh = torch.tensor(hw[::-1], device=x.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
            flow = (flow + 0.5) * wh
            
        #var = simple_interpolate(flow[:, -1:, :, :], size=hw)
        var = torch.nn.functional.interpolate(q[:, -1:, :, :], size=hw, mode='bilinear', align_corners=True)

        return {'flow_preds': flow, "flow_upsampled": flow, 'var': var}
