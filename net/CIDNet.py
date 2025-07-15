import torch
from data.data import *
from torchvision import transforms
import torch.nn as nn
from net.utils import *
from net.I import I_encoder0,I_decoder_before,I_decoder_after
from net.HV import HV_encoder,HV_decoder,DACLIP
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

class CIDNet(nn.Module):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(CIDNet, self).__init__()
        
        path = '../pretrained/daclip_ViT-B-32.pt'
        self.daclip = DACLIP(checkpoint = path)        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0,bias=False)
        )
        
        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
            )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
            )

        self.HV_LCA1 = HV_encoder(ch2, head2)
        self.HV_LCA2 = HV_encoder(ch3, head3)
        self.HV_LCA3 = HV_encoder(ch4, head4)
        self.HV_LCA4 = HV_decoder(ch4, head4)
        self.HV_LCA5 = HV_decoder(ch3, head3)
        self.HV_LCA6 = HV_decoder(ch2, head2)

        self.I_LCA1 = I_encoder0(ch2, head2)
        self.I_LCA2 = I_encoder0(ch3, head3)
        self.I_LCA3 = I_encoder0(ch4, head4)
        self.I_LCA4_0 = I_decoder_before(n_feats=ch4)
        self.I_LCA5_0 = I_decoder_before(n_feats=ch3)
        self.I_LCA6_0 = I_decoder_before(n_feats=ch2)

        self.I_LCA4 = I_decoder_after(ch4, head4)
        self.I_LCA5 = I_decoder_after(ch3, head3)
        self.I_LCA6 = I_decoder_after(ch2, head2)

        self.trans = RGB_HVI()
        
    def forward(self, x):
        dtypes = x.dtype
        image_features, degra_features = self.daclip(x)
        hvi = self.trans.HVIT(x)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)
        # low
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        i_enc2 = self.I_LCA1(image_features, i_enc1, hv_1)
        hv_2 = self.HV_LCA1(degra_features, hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)
        
        i_enc3 = self.I_LCA2(image_features, i_enc2, hv_2)
        hv_3 = self.HV_LCA2(degra_features, hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3

        i_enc3 = self.IE_block3(i_enc3)
        hv_3 = self.HVE_block3(hv_3)

        i_enc4 = self.I_LCA3(image_features, i_enc3, hv_3)
        hv_4 = self.HV_LCA3(degra_features, hv_3, i_enc3)
        
        mid4 = self.I_LCA4_0(i_enc4)
        i_dec4 = self.I_LCA4(mid4,hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)
        
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        mid3 = self.I_LCA5_0(i_dec3)
        i_dec2 = self.I_LCA5(mid3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)
        
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec2, v_jump1)
        mid2 = self.I_LCA6_0(i_dec2)
        i_dec1 = self.I_LCA6(mid2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)
        
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb
    
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi
    
