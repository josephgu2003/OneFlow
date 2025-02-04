
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import torch.optim as optim

def get_ortho_like(dim, heads, alpha, beta, sign=1, dist='uniform'):
    if dist == 'normal':
        A = alpha * np.random.normal(size=(dim,dim)) / (dim**0.5) + sign * beta * np.eye(dim)
    if dist == 'uniform':
        A = alpha * np.random.uniform(size=(dim,dim), low=-3**0.5 / (dim**0.5), high = 3**0.5 / (dim**0.5))\
    + sign * beta * np.eye(dim)

    U, S, V = np.linalg.svd(A)
    L = U @ np.diag(np.sqrt(S))
    R = np.diag(np.sqrt(S)) @ V
    return L, R

def impulse_init(heads,img_size,att_rank,ff,scale=1.0,spatial_pe=None,norm=1):
    weight = torch.zeros((heads,img_size[0]*img_size[1],img_size[0]*img_size[1]))
    k = torch.randint(0,ff**2,(heads,))
    for i in range(heads):
        m = (k[i]//ff)-(ff//2)
        n = (k[i]%ff)-(ff//2)
        tmp_weight = torch.zeros((img_size[1],img_size[1]))
        for j in range(0-min(0,n),img_size[1]-max(0,n)):
            tmp_weight[j,j+n] = 1
        for j in range(0-min(0,m),img_size[0]-max(0,m)):
            weight[i,j*img_size[1]:(j+1)*img_size[1],(j+m)*img_size[1]:(j+m+1)*img_size[1]] = tmp_weight
    # weight = np.sqrt(1/3)*weight
    class PermuteM(nn.Module):
        def __init__(self, heads, att_size, att_rank,scale=1.0,spatial_pe=None):
            super().__init__()
            self.scale = scale
            if spatial_pe is None:
                self.spatial_pe = False
                weights_Q = np.sqrt(1/att_rank/heads)*(2*torch.rand(heads,att_size,att_rank)-1)
                weights_K = np.sqrt(1/att_rank/heads)*(2*torch.rand(heads,att_rank,att_size)-1)
            else:
                self.spatial_pe = True
                self.pe = torch.nn.functional.layer_norm(spatial_pe.cuda(),[spatial_pe.shape[1]])
                weights_Q = np.sqrt(1/att_rank/heads)*(2*torch.rand(heads,spatial_pe.shape[1],att_rank)-1)
                weights_K = np.sqrt(1/att_rank/heads)*(2*torch.rand(heads, att_rank, spatial_pe.shape[1])-1)

            self.weights_K = nn.Parameter(weights_K)
            self.weights_Q = nn.Parameter(weights_Q)
        def forward(self):
            if self.spatial_pe:
                M = self.pe@self.weights_Q@self.weights_K@(self.pe.T)
            else:
                M = torch.bmm(self.weights_Q,self.weights_K)
            return torch.softmax(M*self.scale,-1)
    
    net = PermuteM(heads,img_size[0]*img_size[1],att_rank,scale,spatial_pe)
    net.cuda()

    nq = net.weights_Q.detach().cpu().norm(dim=(1)).mean()
    weight = weight.cuda()
    num_epoch = 10000
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)#,weight_decay=1e-6)
    for i in range(num_epoch):
        if i%norm==0:
            with torch.no_grad():
                net.weights_Q.div_(net.weights_Q.detach().norm(dim=(1),keepdim=True)/nq)
                net.weights_K.div_(net.weights_K.detach().norm(dim=(1),keepdim=True)/nq)
        optimizer.zero_grad()
        outputs = net()
        loss = criterion(outputs, weight)
        loss.backward()
        optimizer.step()
    print(loss.data)

    return net.weights_Q.detach().cpu(),net.weights_K.detach().cpu()


def advanced_init_weights(self, mode) -> None:
    # this fn left here for compat with downstream users
    # init_weights_vit_timm(m)
    # modified this funcition as our initialization
    if 'mimetic' in mode:
        alpha, beta = mode[7:].split('_')
        alpha = float(alpha)
        beta = float(beta)
        print(f'using mimetic init with alpha={alpha}, beta={beta}')
        head_dim = self.embed_dim // self.num_heads
        for i in range(self.depth):
            d = i / float(self.depth - 1)

            for h in range(self.num_heads):
                Q, K = get_ortho_like(self.embed_dim, -float('inf'), alpha, beta, 1)
                Q = Q[:,:head_dim]
                K = K.T[:,:head_dim]

                self.blocks[i].attn.qkv.weight.data[(h*head_dim):((h+1)*head_dim)] = torch.tensor(Q.T).float()
                self.blocks[i].attn.qkv.weight.data[self.embed_dim+(h*head_dim):self.embed_dim+((h+1)*head_dim)] = torch.tensor(K.T).float()

        for block in self.blocks:
            V, Proj = get_ortho_like(self.embed_dim, self.num_heads, 0.4, 0.4, -1)
            block.attn.qkv.weight.data[2*self.embed_dim:] = torch.tensor(V).float()
            block.attn.proj.weight.data = torch.tensor(Proj).float()
    elif 'impulse' in mode:
        a = mode[7:]
        # scale = float(a)
        scale = (self.embed_dim // self.num_heads) ** -0.5
        norm_epoch = 100 # int(b)
        ff = int(a)
        print(f'using impulse init with scale={scale}, norm_epoch={norm_epoch}, kernel size={ff}')
        head_dim = self.embed_dim // self.num_heads

        def get_pseudo_input(code,self):
            if code == 'P':
                return self.pos_embed
            elif code == 'U':
                return 2*torch.rand(self.pos_embed.shape,device=self.pos_embed.device)-1
            elif code == 'G':
                return trunc_normal_(torch.zeros_like(self.pos_embed), std=torch.sqrt(torch.tensor(1/2)))
            elif code == 'A':
                return self.pos_embed+2*torch.rand(self.pos_embed.shape,device=self.pos_embed.device)-1
            elif code == 'B':
                return self.pos_embed+trunc_normal_(torch.zeros_like(self.pos_embed), std=torch.sqrt(torch.tensor(1/2)))
            else:
                return None
        
        for i in range(self.depth):
            d = i / float(self.depth - 1)
            if i == 0:
                Q, K = impulse_init(self.num_heads,self.attn_size,head_dim,ff,scale=scale,spatial_pe=get_pseudo_input(self.pseudo_input1,self),norm=norm_epoch)
            elif i == 1:
                if not self.pseudo_input1==self.pseudo_input2:
                    Q, K = impulse_init(self.num_heads,self.attn_size,head_dim,ff,scale=scale,spatial_pe=get_pseudo_input(self.pseudo_input2,self),norm=norm_epoch)
                elif not self.init_once:
                    Q, K = impulse_init(self.num_heads,self.attn_size,head_dim,ff,scale=scale,spatial_pe=get_pseudo_input(self.pseudo_input2,self),norm=norm_epoch)
            else:
                if not self.init_once:
                    Q, K = impulse_init(self.num_heads,self.attn_size,head_dim,ff,scale=scale,spatial_pe=get_pseudo_input(self.pseudo_input2,self),norm=norm_epoch)
            for h in range(self.num_heads):
                self.blocks[i].attn.qkv.weight.data[(h*head_dim):((h+1)*head_dim)] = torch.tensor(Q[h].T).float()
                self.blocks[i].attn.qkv.weight.data[self.embed_dim+(h*head_dim):self.embed_dim+((h+1)*head_dim)] = torch.tensor(K[h]).float()
