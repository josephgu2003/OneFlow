
import torch

def encode_digits(mag):
    mag = mag.to(torch.int32)[:, :1]

    hunds = torch.floor(mag / 100)
    mag = mag - hunds * 100

    tens = torch.floor(mag / 10)
    mag = mag - tens * 10

    ones = torch.floor(mag)

    hunds = (hunds - 5) / 10
    tens = (tens - 5) / 10
    ones = (ones - 5) / 10
    return torch.concat((hunds, tens, ones), dim=1)

def decode_digits(digits):
    digits = digits * 10 + 5
    digits = torch.round(digits)
    return digits[:, :1] * 100 + digits[:, 1:2] * 10 + digits[:, 2:3] * 1

def encode_polar(mag):
    mag = torch.pow(mag[:, :1], 0.7)
    mag = mag / 20 * 2 * torch.pi
    sin = torch.sin(mag)
    cos = torch.cos(mag)
    return torch.concat((sin, cos, torch.zeros_like(cos)), dim=1)

def decode_polar(polar):
    mag = torch.pow(torch.atan2(polar[:, 0:1], polar[:, 1:2]), 1 / 0.7) / (2 * torch.pi) * 20
    return mag

divisors = [50, 50, 100]
def segment_mag(mag, segments=3):
    mags = []
    intervals = [0, 50, 100]
    for i in range(segments):
        threshold = intervals[i]
        m = torch.relu(mag - threshold)
        
        if i != segments - 1:
            m = torch.clamp(m, 0, intervals[i+1]-intervals[i])

        mags.append(m / divisors[i])
    return mags

def unsegment_mag(mags):
    mags = [mag for mag in mags]
    mags[1][mags[1] < 0.05] = 0
    mags[2][mags[2] < 0.05] = 0
    return (mags[0] * divisors[0] + mags[1] * divisors[1] * 1 + mags[2] * divisors[2] * 1)

def encode_mag_segmentwise(mag):
    segments = segment_mag(mag[:, :1])
    
    if len(segments) == 2:
        segments.append(torch.zeros_like(mag[:, :1]))
    return torch.concat(segments, dim=1)

def decode_mag_segmentwise(mags):
    return unsegment_mag([mags[:, :1], mags[:, 1:2], mags[:, 2:3]])
 
means = [0, 50, 100]
sds = [100, 100, 100]

def gaussians_mag(mag):
    mags = []

    for i in range(3):
        mu = means[i]
        sd = sds[i]
        m = (mag - mu) / sd
        mags.append(m)

    return mags
    return torch.concat(mags, dim=1)

def ungaussians_mag(mags):
 
    es1 = mags[0] * sds[0] + means[0]
    es2 = mags[1] * sds[1] + means[1]
    es3 = mags[2] * sds[2] + means[2]


    mags = torch.concat([mags[0][:, :1], mags[1][:, :1], mags[2][:, :1]], dim=1)
    mags = torch.abs(mags)
    one_hot = torch.zeros_like(es1)
    one_hot.scatter_(1, mags.argmin(dim=1, keepdim=True), 1) 
    return es1 * one_hot[:, :1] + es2 * one_hot[:, 1:2] + es3 * one_hot[:, 2:3]
   
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
