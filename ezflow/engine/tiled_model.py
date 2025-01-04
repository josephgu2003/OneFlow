import torch 
import torch.nn as nn
import numpy as np
import torchshow
import random

@torch.no_grad()
def tiled_pred(model, img1, img2, overlap=0.5, crop=512):     
    # for each image, we are going to run inference on many overlapping patches
    # then, all predictions will be weighted-averaged
    B, _, H, W = img1.shape
    win_height, win_width = crop, crop

    def crop_generator():
        for sy in _overlapping(H, win_height, overlap):
          for sx in _overlapping(W, win_width, overlap):
            yield sy, sx, sy, sx, True

    # keep track of weighted sum of prediction*weights and weights
    accu_pred = torch.zeros((B, 2, H, W), device=img1.device) # accumulate the weighted sum of predictions 
    accu_conf = img1.new_zeros((B, H, W)) # accumulate the weights 


    for sy1, sx1, sy2, sx2, aligned in crop_generator():
        # compute optical flow there
        pred =  model(_crop(img1,sy1,sx1), _crop(img2,sy2,sx2))['flow_upsampled']
                        
        accu_pred[...,sy1,sx1] += pred
        accu_conf[...,sy1,sx1] += 1
        
    pred = accu_pred / accu_conf[:, None,:,:]
    assert not torch.any(torch.isnan(pred))
    return pred

def _overlapping(total, window, overlap=0.5):
    assert total >= window and 0 <= overlap < 1, (total, window, overlap)
    num_windows = 1 + int(np.ceil( (total - window) / ((1-overlap) * window) ))
    offsets = np.linspace(0, total-window, num_windows).round().astype(int)
    yield from (slice(x, x+window) for x in offsets)

def _crop(img, sy, sx):
    B, THREE, H, W = img.shape
    if 0 <= sy.start and sy.stop <= H and 0 <= sx.start and sx.stop <= W:
        return img[:,:,sy,sx]
    assert False # let's not pad for now
    l, r = max(0,-sx.start), max(0,sx.stop-W)
    t, b = max(0,-sy.start), max(0,sy.stop-H)
    img = torch.nn.functional.pad(img, (l,r,t,b), mode='constant')
    return img[:, :, slice(sy.start+t,sy.stop+t), slice(sx.start+l,sx.stop+l)]

class TiledModel(nn.Module):
    def __init__(self, model, crop=320, overlap=0.5):
        super().__init__()
        self.model = model
        self.overlap = overlap 
        self.crop = crop
    
    def forward(self, img1, img2):
        # run tiled inference
        out = {'flow_upsampled': tiled_pred(self.model, img1, img2, overlap=self.overlap, crop=self.crop)}
        return out
