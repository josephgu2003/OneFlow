import torch 

def model_stats(model: torch.nn.Module):
    norm = 0.0
    n = 0
    
    for p in model.parameters():
        if p.requires_grad:
            n += 1
            norm += p.norm().item()
    
    return norm / n
    
def layer_stats(model: torch.nn.Module, tgt: str):
    for n, p in model.named_parameters():
        if tgt in n:
            return p.norm().item(), p.grad.norm().item()
        
    raise ValueError("Did not find tgt!")