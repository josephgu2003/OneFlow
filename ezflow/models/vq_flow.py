from functools import partial
import torch 
import torch.nn as nn

from ezflow.decoder.vq.utils import instantiate_from_config
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

def simple_interpolate(x, size):
    return torch.nn.functional.interpolate(x, size=size, mode='bilinear')
flow_scale = 0.1
base = 1
iters = 4

@MODEL_REGISTRY.register()
class VQFlow(BaseModule):
    def __init__(self, cfg):
        super().__init__()
        self.scaler = cfg.SCALER
        self.hidden_size = cfg.HIDDEN_SIZE
        # self.final_layer = nn.Linear(cfg.HIDDEN_SIZE, 3)
        self.dpt = DPTHead(cfg.HIDDEN_SIZE)
        num_patches = 16 * 16 * self.scaler * self.scaler
        self.n_embed = cfg.model.params.n_embed
        self.embed_dim = cfg.model.params.embed_dim
        self.final_layer = nn.Linear(384, self.n_embed * 2) # two way reparam

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, cfg.HIDDEN_SIZE), requires_grad=False)
        
        block_classes = {'DecoderBlock': DecoderBlock, 'ConcatenateDecoderBlock': ConcatenateDecoderBlock}
        block_class = block_classes[cfg.BLOCK_CLASS]
        assert cfg.DECODER_BLOCKS % 4 == 0

        self.use_dec = cfg.DECODER_BLOCKS > 0

        if self.use_dec:
            self.proj = nn.Linear(384, cfg.HIDDEN_SIZE)
            self.decoder_blocks = nn.ModuleList([
                block_class(cfg.HIDDEN_SIZE, cfg.NUM_HEADS) for _ in range(cfg.DECODER_BLOCKS)
            ])
            self.out_indices = list([i for i in range(0, cfg.DECODER_BLOCKS, cfg.DECODER_BLOCKS // 4)])
        self.init_weights()
        self.vits16 = dinov2_vits14_reg(block_fn=partial(Block, attn_class=NativeAttention))
        
        self.encoder_concat = cfg.ENCODER_CONCAT 
        self.reparam = cfg.REPARAM
        
        self.vqgan = instantiate_from_config(cfg.model)
        for param in self.vqgan.parameters():
            param.requires_grad_(False)
            
        self.vqgan.eval()
        self.vq_dim = 256 * self.scaler
      
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
        c = self.n_embed * 2 # two way reparam
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
        flow_gt = torch.concat((flow_gt, 0 * flow_gt[:, :1, :, :]), dim=1)
        flow = torch.nn.functional.interpolate(flow_gt, size=(self.vq_dim, self.vq_dim), mode='bilinear', align_corners=True)
        flow[:, 0, :, :] = flow[:, 0, :, :] / w / flow_scale
        flow[:, 1, :, :] = flow[:, 1, :, :] / h / flow_scale
       
        logit_stack = [] 
        for i in range(iters):
            # in normalized space
            logits = self.vqgan.encode(flow)[-1][-1]
            logits = logits.reshape(flow_gt.shape[0], 16, 16)
            logit_stack.append(logits)
            bhwc = [logits.shape[0], logits.shape[1], logits.shape[2], self.embed_dim]
            z_q = self.vqgan.quantize.get_codebook_entry(logits, shape=bhwc)
            new_flow = self.vqgan.decode(z_q)
            flow = flow - new_flow
            flow *= (base ** i)
            flow[:, -1, :, :] *= 0
        return logit_stack

    def _encode(self, x):
        feats = []
        for i, blk in enumerate(self.vits16.blocks):
            x = blk(x)

            if i in [2, 5, 8, 11]:
                x1x2 = torch.chunk(x, 2, dim=1 if self.encoder_concat else 0)
                x1 = x1x2[0][:, 1+4:]
                feats.append([x1])
        return x, feats

    def _decode(self, x):
        x = self.vits16.norm(x)
        
        x = self.proj(x)
        
        x1x2 = torch.chunk(x, 2, dim=1 if self.encoder_concat else 0)
        x1 = x1x2[0][:, 1+4:]
        x2 = x1x2[1][:, 1+4:]
        
        q = x1
        
        feats = []
        for i, block in enumerate(self.decoder_blocks):
            q = block(q, x1, x2, None)
            
            if i in self.out_indices:
                feats.append([q])
        return feats

    def decode_flow(self, logit_stack, bhwc, hw):
        flow = None
        
        for i in range(iters):
            logits = logit_stack[i]
            z_q = self.vqgan.quantize.get_codebook_entry(logits, shape=bhwc)
            new_flow = self.vqgan.decode(z_q) / (base ** i)
            if flow is None:
                flow = new_flow
            else:
                flow = flow + new_flow
                
        flow = torch.nn.functional.interpolate(flow[:, :2, :, :], size=hw, mode='bilinear', align_corners=True) * flow_scale
        flow[:, 0, :, :] = flow[:, 0, :, :] * hw[1]
        flow[:, 1, :, :] = flow[:, 1, :, :] * hw[0]
        return flow

    def forward(self, img1, img2):
        self.vqgan.eval()
        both_img = torch.concat((img1, img2))        
        hw = both_img.shape[2:]
        both_img = simple_interpolate(both_img, size=(224 * self.scaler, 224 * self.scaler)) # TODO: fix

        x = self.vits16.prepare_tokens_with_masks(both_img, None)
       
        if self.encoder_concat: 
            x1x2 = torch.chunk(x, 2)
            img1 = x1x2[0] # TODO: img encoding??
            img2 = x1x2[1]
        
            x = torch.concat((img1, img2), dim=1)

        x, feats = self._encode(x)

        if self.use_dec:
            feats = self._decode(x)
          
        q = self.final_layer(feats[-1][0])
        q = self.unpatchify(q) 
        q_dir, q_mag = q[:, :self.n_embed, :, :], q[:, self.n_embed:, :, :]
        latents = (q_dir, q_mag) 
        if self.training:
            return {'latents': (q_dir, q_mag)}
        else:
            with torch.no_grad():
                bhwc = [q.shape[0], q.shape[2], q.shape[3], self.embed_dim]
                q_dir = torch.argmax(q_dir, dim=1)
                q_mag = torch.argmax(q_mag, dim=1)
                flow = self.decode_flow(q_dir, q_mag, bhwc, hw)
                
            return {'latents': latents, 'flow_preds': flow, "flow_upsampled": flow}
