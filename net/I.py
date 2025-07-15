import torch
import torch.nn as nn
from net.utils import *
from net.partition import CAMixerSR
from einops import rearrange
 
class I_CAB_encoder(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(I_CAB_encoder, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, image_features, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q1(image_features)*self.q2(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn,dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class I_CAB_decoder(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(I_CAB_decoder, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn,dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out   

class I_FFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(I_FFN, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
       
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x

class I_encoder0(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(I_encoder0, self).__init__()
        self.norm = LayerNorm(dim)
        self.ffn = I_FFN(dim) 
        self.cab = I_CAB_encoder(dim, num_heads, bias)
        self.clip_proj = nn.Sequential(
            nn.Linear(512, dim*16*16),
            nn.Unflatten(1, (dim, 16, 16)),
        )
    # x->q;y->kv
    def forward(self, image_features, x, y):
        H,W = x.shape[2],x.shape[3]
        image_features = self.clip_proj(image_features)
        image_features = F.interpolate(image_features, size=(H, W), mode='bilinear', align_corners=False)
        x = x + self.cab(self.norm(image_features),self.norm(x),self.norm(y))
        x = self.ffn(self.norm(x))
        return x
    
class I_decoder_after(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(I_decoder_after, self).__init__()
        self.norm = LayerNorm(dim)
        self.ffn = I_FFN(dim) 
        self.cab = I_CAB_decoder(dim, num_heads, bias)
    # x->q;y->kv
    def forward(self, x, y):
        H,W = x.shape[2],x.shape[3]
        x = x + self.cab(self.norm(x),self.norm(y))
        x = self.ffn(self.norm(x))
        return x

class I_decoder_before(nn.Module):
    def __init__(self, in_channle=1, n_feats=72, ratio=0.5,n_group=1,n_block=[1],device=""):
        super().__init__()
        self.CAMixerSR = CAMixerSR(in_channle=in_channle, n_feats=n_feats,n_block=n_block,ratio=ratio,n_group=n_group,device=device)
    def forward(self, inputI):
        return self.CAMixerSR(inputI)[0]  # [B,1,H,W]

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(1,1,128,128).to(device)
    net = I_decoder_before().to(device)
    net.eval()
    with torch.no_grad():
        output = net(x)
    print(output.shape)