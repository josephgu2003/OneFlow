
import torch

def segment_mag(mag, segments=2, interval=50, divisor=2):
    mags = []
    for i in range(segments):
        threshold = i * interval
        m = torch.relu(mag - threshold) / interval
       
        if i != segments - 1:
            m = torch.clamp(m, 0, 1)
        m = m / divisor
        mags.append(m)
    return mags

def unsegment_mag(mags, interval=50, divisor=2):
    return torch.sum(torch.stack(mags), dim=0) * interval * divisor 

def encode_mag_segmentwise(mag):
    segments = segment_mag(mag[:, :1])
    segments.append(torch.zeros_like(mag[:, :1]))
    return torch.concat(segments, dim=1)

def decode_mag_segmentwise(mags):
    return unsegment_mag([mags[:, :1], mags[:, 1:2]])
    
def colorize_mag(channel):
    c1 = torch.zeros_like(channel)
    c1[:, 0] += 0.5
    c1[:, 2] += 0.0
    c2 = torch.zeros_like(channel)
    c2[:, 1] += 0.0
    c2[:, 2] -= 0.0
    mag = channel[:, 0:1] / 50
    c3 = c1 * mag + c2 * (1 - mag)
    return c3 

def decolorize_mag(channel):
    mag = channel[:, 0:1] / 0.5
    return mag * 50
    
def pad_flow(flow):
    return torch.concat((flow, torch.zeros_like(flow[:, :1])), dim=1)

def unpad_flow(flow):
    return flow[:, :2]

def decompose_flow(flow):
    dir = torch.nn.functional.normalize(flow, p=2, dim=1)
    mag = torch.norm(flow, p=2, dim=1, keepdim=True)
    return pad_flow(dir), mag.repeat(1, 3, 1, 1)
    
def reparameterize(x, hw):
    h, w = hw[0], hw[1]
    x = torch.nn.functional.interpolate(x, size=hw, mode='bilinear', align_corners=True)
    hs = torch.arange(h, device=x.device)
    ws = torch.arange(w, device=x.device)
    
    wh = torch.tensor(hw[::-1], device=x.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    # flow is xy, hmm is this correct?
    flow = (x[:, :2, :, :] + 0.5) * wh - torch.stack(torch.meshgrid(hs, ws, indexing='ij')[::-1])
    return flow 

def invert_flow(flow):
    b, c, h, w = flow.shape
    
    hs = torch.arange(h, device=flow.device)
    ws = torch.arange(w, device=flow.device)
    
    coords = torch.stack(torch.meshgrid(hs, ws, indexing='ij')) + torch.flip(flow, dims=[1]) # b c h w
    coords = coords.long() 

    coords[:, 0, :, :] = torch.clamp(coords[:, 0, :, :], 0, h-1)
    coords[:, 1, :, :] = torch.clamp(coords[:, 1, :, :], 0, w-1) 
    
    flow = flow.flatten(2)
    coords = coords.flatten(2)
    coords = coords[:, 0:1, :] * w + coords[:, 1:, :]
    coords = torch.concat((coords, coords), dim=1)
    inv_flow = torch.zeros_like(flow)
    valid = torch.zeros_like(flow)
    
    inv_flow.scatter_(dim=2, index=coords, src=-flow)
    inv_flow = inv_flow.view(b, 2, h, w)
    valid.scatter_(dim=2, index=coords, src=torch.ones_like(flow))
    valid = torch.mean(valid.view(b, 2, h, w), dim=1)

    return inv_flow, valid
