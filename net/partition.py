import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from net.utils import *
from net.lsconv import LSConv
from torchvision import transforms
import os

def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B*N, C)

    x[idx1.reshape(-1)] = x1.reshape(B*N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B*N2, C)

    x = x.reshape(B, N, C)
    return x

class PredictorLG(nn.Module):
    """ Importance Score Predictor
    """
    def __init__(self, dim, window_size=8, k=4,ratio=0.5,device=""):
        super().__init__()
        self.ratio = ratio
        self.dim = dim
        self.window_size = window_size
        cdim = dim + k
        embed_dim = window_size**2
        
        self.in_conv = nn.Sequential(
            nn.Conv2d(cdim, cdim//4, 1),
            LayerNorm(cdim//4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.out_mask = nn.Sequential(
            nn.Linear(embed_dim, window_size),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(window_size, 2),
            nn.Softmax(dim=-1)
        )

        self.out_CA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cdim//4, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_x, mask=None, ratio=0.5, train_mode=False):

        x = self.in_conv(input_x)

        ca = self.out_CA(x)
        
        x = torch.mean(x, keepdim=True, dim=1) 
        x = rearrange(x,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        B, N, C = x.size()
        pred_score = self.out_mask(x)
        mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]

        if self.training or train_mode:
            return mask, ca
        else:
            score = pred_score[:, : , 0]
            B, N = score.shape
            r = torch.mean(mask,dim=(0,1))*1.0
            if self.ratio == 1:
                num_keep_node = N 
            else:
                num_keep_node = min(int(N * r * 2 * self.ratio), N)
            idx = torch.argsort(score, dim=1, descending=True)
            idx1 = idx[:, :num_keep_node]
            idx2 = idx[:, num_keep_node:]
            return [idx1, idx2], ca

class CAMixer(nn.Module):
    def __init__(self, dim, window_size=8, bias=True, is_deformable=True, ratio=0.5,device=""):
        super().__init__()    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim = dim
        self.window_size = window_size
        self.ratio = ratio
        k = 3
        d = 2
        self.project_v = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)
        self.project_q = nn.Linear(dim, dim, bias = bias)
        self.project_k = nn.Linear(dim, dim, bias = bias)
        # Conv
        self.conv_sptial = nn.Sequential(
            nn.Conv2d(dim, dim, k, padding=k//2, groups=dim),
            nn.Conv2d(dim, dim, k, stride=1, padding=((k//2)*d), groups=dim, dilation=d))        
        self.project_out = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)
        self.act = nn.GELU()
        # Predictor
        self.route = PredictorLG(dim,window_size,ratio=ratio,device=device)
        self.lsconv = LSConv(dim=dim,groups=6).to(self.device)

    def forward(self,x,condition_global=None, mask=None, train_mode=False):
        x = x.to(self.device)
        condition_global = condition_global.to(self.device)
        N,C,H,W = x.shape
        v = self.project_v(x)
        condition_wind = torch.stack(torch.meshgrid(torch.linspace(-1,1,self.window_size),torch.linspace(-1,1,self.window_size)))\
                .type_as(x).unsqueeze(0).repeat(N, 1, H//self.window_size, W//self.window_size)
        _condition = torch.cat([v, condition_global, condition_wind], dim=1)
        mask, ca = self.route(_condition,ratio=self.ratio,train_mode=train_mode)
        qk = torch.cat([x,x],dim=1)
        vs = self.lsconv(v)
        v  = rearrange(v,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        vs = rearrange(vs,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        qk = rearrange(qk,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        if self.training or train_mode:
            N_ = v.shape[1]
            v1,v2 = v*mask, vs*(1-mask)   
            qk1 = qk*mask 
        else:
            idx1, idx2 = mask
            _, N_ = idx1.shape
            v1,v2 = batch_index_select(v,idx1),batch_index_select(vs,idx2)
            qk1 = batch_index_select(qk,idx1)

        v1 = rearrange(v1,'b n (dh dw c) -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        qk1 = rearrange(qk1,'b n (dh dw c) -> b (n dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)

        q1,k1 = torch.chunk(qk1,2,dim=2)
        q1 = self.project_q(q1)
        k1 = self.project_k(k1)
        q1 = rearrange(q1,'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        k1 = rearrange(k1,'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
  
        attn = q1 @ k1.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        f_attn = attn@v1

        f_attn = rearrange(f_attn,'(b n) (dh dw) c -> b n (dh dw c)', 
            b=N, n=N_, dh=self.window_size, dw=self.window_size)

        if not (self.training or train_mode):
            attn_out = batch_index_fill(v.clone(), f_attn, v2.clone(), idx1, idx2)
        else:
            attn_out = f_attn + v2

        attn_out = rearrange(
            attn_out, 'b (h w) (dh dw c) -> b (c) (h dh) (w dw)', 
            h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size
        )
        out = attn_out
        out = self.act(self.conv_sptial(out))*ca + out
        out = self.project_out(out)
        if self.training:
            return out, torch.mean(mask,dim=1)
        return out,[mask]

class GatedFeedForward(nn.Module):
    def __init__(self, dim, mult = 1, bias=False, dropout = 0.,device=""):
        super().__init__()
        self.dim = dim

        self.project_in = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Block(nn.Module):
    def __init__(self, n_feats, window_size=8, ratio=0.5,device=""):
        super(Block,self).__init__()
        
        self.n_feats = n_feats
        self.norm = LayerNorm(n_feats)
        self.mixer = CAMixer(n_feats,window_size=window_size,ratio=ratio,device=device)
        self.ffn = GatedFeedForward(n_feats,device=device)
        
    def forward(self,x,condition_global=None):
        if self.training:
            res, decision = self.mixer(x,condition_global)
            x = self.norm(x+res)
            res = self.ffn(x)
            x = self.norm(x+res)
            return x, decision
        else:
            res,decision = self.mixer(x,condition_global)
            x = self.norm(x+res)
            res = self.ffn(x)
            x = self.norm(x+res)
            return x,decision

class CAMixerSR(nn.Module):
    def __init__(self, n_block=[1], n_group=4, in_channle=3, n_feats=60, ratio=0.5, window_sizes=16, device=""):
        super().__init__()
        self.ratio =ratio
        self.n_feats = n_feats
        self.window_sizes = window_sizes
        self.global_predictor = nn.Sequential(nn.Conv2d(n_feats, 8, 1, 1, 0, bias=True),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                        nn.Conv2d(8, 2, 3, 1, 1, bias=True),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.head = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.body = nn.ModuleList([Block(n_feats, window_size=self.window_sizes, ratio=ratio) \
                                   for i in range(n_group)])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.tail = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

    def forward(self, x):
        _,_,H,W = x.shape
        decision = []
        x = self.check_image_size(x)
        x = self.head(x)
        condition_global = self.global_predictor(x) 
        shortcut = x.clone()
        if self.training:
            for _, blk in enumerate(self.body):
                x, mask = blk(x,condition_global)
                decision.extend(mask)
        else:
            for _, blk in enumerate(self.body):
                x, mask = blk(x,condition_global)
                decision.extend(mask)  
        x = self.body_tail(x) + shortcut
        if self.training:
            return x[:,:,:H,:W],2*self.ratio
        else:
            return x[:,:,:H,:W],decision
    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(1,1,256,256).to(device)
    net = CAMixerSR(in_channle=1, n_feats=36, ratio=0.5,device=device).to(device)
    net.eval()
    output = net(x)
    print(output.shape)