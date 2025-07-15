import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import open_clip
from net.utils import *

class DACLIP(nn.Module):
    def __init__(self, checkpoint = '../universal-image-restoration/pretrained/daclip_ViT-B-32.pt'):
        super().__init__()
        self.model, self.preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=checkpoint)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        for p in self.model.parameters(): 
            p.requires_grad=False
    def forward(self, x):
        image = self.preprocess(x)
        with torch.no_grad():
            image_features, degra_features = self.model.encode_image(image, control=True)
            image_features = F.normalize(image_features, dim=-1) 
            degra_features = F.normalize(degra_features, dim=-1)
        return image_features,degra_features
 
class HV_CAB_encoder(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(HV_CAB_encoder, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, degra_features, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q1(degra_features)*self.q2(x))
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
    
class HV_CAB_decoder(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(HV_CAB_decoder, self).__init__()
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

class HV_FFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(HV_FFN, self).__init__()

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

class HV_encoder(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(HV_encoder, self).__init__()
        self.norm = LayerNorm(dim)
        self.ffn = HV_FFN(dim) 
        self.cab = HV_CAB_encoder(dim, num_heads, bias)
        self.clip_proj = nn.Sequential(
            nn.Linear(512, dim*16*16),
            nn.Unflatten(1, (dim, 16, 16)),
        )
    # x->q;y->kv
    def forward(self, degra_features, x, y):
        H,W = x.shape[2],x.shape[3]
        degra_features = self.clip_proj(degra_features)
        degra_features = F.interpolate(degra_features, size=(H, W), mode='bilinear', align_corners=False)
        x = x + self.cab(self.norm(degra_features),self.norm(x),self.norm(y))
        x = self.ffn(self.norm(x))
        return x
    
class HV_decoder(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(HV_decoder, self).__init__()
        self.norm = LayerNorm(dim)
        self.ffn = HV_FFN(dim) 
        self.cab = HV_CAB_decoder(dim, num_heads, bias)
    # x->q;y->kv
    def forward(self, x, y):
        H,W = x.shape[2],x.shape[3]
        x = x + self.cab(self.norm(x),self.norm(y))
        x = self.ffn(self.norm(x))
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x1 = torch.rand(1,3,256,256).to(device)
    x2 = torch.rand(1,72,256,256).to(device)
    path = '../TPAMI/pretrained/daclip_ViT-B-32.pt'
    daclip = DACLIP(checkpoint = path).to(device)
    clip_feature = daclip(x1)
    net = HV_CAB_encoder(dim=72,num_heads=4).to(device)
    net.eval()
    with torch.no_grad():
        output = net(clip_feature,x2,x2)