from torch import nn, einsum
from collections.abc import Iterable
import os
import copy
import torch
from einops import rearrange
import torch.nn.functional as F
from torch.nn import MultiheadAttention

norm_func = nn.InstanceNorm3d
norm_func2d = nn.InstanceNorm2d
act_func = nn.CELU

class SSA(nn.Module):
    def __init__(self, dim, n_segment):
        super(SSA, self).__init__()
        self.scale = dim ** -0.5    
        self.n_segment = n_segment 

        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size = 1)
        self.attend = nn.Softmax(dim = -1)
        self.to_temporal_qk = nn.Conv3d(dim, dim * 2, 
                                  kernel_size=(3, 1, 1), 
                                  padding=(1, 0, 0))

    def forward(self, x):
        bt, c, h, w = x.shape
        t = self.n_segment
        b = bt / t

        # Spatial Attention:
        qkv = self.to_qkv(x) # bt, 3*c, h, w
        q, k, v = qkv.chunk(3, dim = 1) # bt, c, h, w
        q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), (q, k, v)) # bt, hw, c

        # -pixel attention
        pixel_dots = einsum('b i c, b j c -> b i j', q, k) * self.scale # bt, hw, hw
        pixel_attn = torch.softmax(pixel_dots, dim=-1) # bt, hw, hw
        pixel_out = einsum('b i j, b j d -> b i d', pixel_attn, v) # bt, hw, c
        
        # -channel attention
        chan_dots = einsum('b i c, b i k -> b c k', q, k) * self.scale # bt, c, c
        chan_attn = torch.softmax(chan_dots, dim=-1)
        chan_out = einsum('b i j, b d j -> b d i', chan_attn, v) # bt, hw, c
        
        # aggregation
        x_hat = pixel_out + chan_out # bt, hw, c
        x_hat = rearrange(x_hat, '(b t) (h w) c -> b c t h w', t=t, h=h, w=w) # b c t h w  
        
        # Temporal attention
        t_qk = self.to_temporal_qk(x_hat) # b, 2*c, t, h, w
        tq, tk = t_qk.chunk(2, dim=1) # b, c, t, h, w
        tq, tk = map(lambda t: rearrange(t, 'b c t h w -> b t (c h w )'), (tq, tk)) # b, t, chw
        tv = rearrange(v, '(b t) (h w) c -> b t (c h w)', t=t, h=h, w=w) # shared value embedding ; b, t, chw
        dots = einsum('b i d, b j d -> b i j', tq, tk) # b, t, t
        attn = torch.softmax(dots, dim=-1)
        out = einsum('b k t, b t d -> b k d', attn, tv) # b, t, chw
        out = rearrange(out, 'b t (c h w) -> (b t) c h w', h=h,w=w,c=c) # bt, c, h, w

        return out


class SADA_Attention(nn.Module):
    def __init__(self, inchannel, n_segment):
        super(SADA_Attention, self).__init__()

        self.LF0 = SSA(inchannel, n_segment)
        self.LF1 = SSA(inchannel, n_segment)
        self.LF2 = SSA(inchannel, n_segment)  

        self.fusion = nn.Sequential(
            nn.Conv3d(inchannel, 1, kernel_size=1, padding=0, bias=False),          
            norm_func(1, affine=True),
            nn.Sigmoid(),
            )

    def forward(self, x):

        n, c, d, w, h = x.size()
 
        localx = copy.copy(x).transpose(1,2).contiguous().view(n*d, c, w, h)  # n*d, c, w, h
        localx = self.LF0(localx).transpose(1,2).contiguous() # n*d, w, c, h
        x0 = localx.view(n, c, d, w, h)  # n, c, d, w, h 

        localx = copy.copy(x).permute(0, 3, 1, 2, 4).contiguous().view(n*w, c, d, h)  # n*w, c, d, h
        localx = self.LF1(localx) # n*w, c, d, h
        x1 = localx.view(n, w, c, d, h).permute(0, 2, 3, 1, 4).contiguous()  # n, c, d, w, h 

        localx = copy.copy(x).permute(0, 4, 1, 2, 3).contiguous().view(n*h, c, d, w)  # n*h, c, d, w
        localx = self.LF2(localx) # n*h, c, d, w
        x2 = localx.view(n, h, c, d, w).permute(0, 2, 3, 4, 1).contiguous()  # n, c, d, w, h 

        # n, c, d, w, h
        return x0+x1+x2


class  MVCSBlock(nn.Module):
    def __init__(self, inchannel, outchannel, num_heads, atten):
        super(MVCSBlock, self).__init__()

        self.atten = atten

        self.conv_0 = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=1, padding=0, bias=False),          
            norm_func(outchannel, affine=True),
            nn.GELU(),
            )

        self.Atten = SADA_Attention(outchannel, num_heads)

        self.conv_1 = nn.Sequential(
            nn.Conv3d(outchannel, outchannel, kernel_size=3, padding=1, bias=False),
            norm_func(outchannel, affine=True),
            nn.GELU(),             
            )

        self.conv_2 = nn.Sequential(
            nn.Conv3d(outchannel, outchannel, kernel_size=3, padding=1, bias=False),          
            norm_func(outchannel, affine=True),
            nn.GELU(),
            )

    def forward(self, x):
        x = self.conv_0(x)
        # residual = x
        if self.atten:
            x = self.Atten(x)
        out = self.conv_1(x)
        return self.conv_2(out) # + residual


class Blocks(nn.Module):
    def __init__(self, inchannel, outchannel, num_heads, atten= [False,False]):
        super(Blocks, self).__init__()
        self.block0 = MVCSBlock(inchannel, outchannel, num_heads, atten[0])
        self.block1 = MVCSBlock(outchannel, outchannel, num_heads, atten[1])        
        self.conv_0 = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=1, padding=0, bias=False),          
            norm_func(outchannel, affine=True),
            nn.GELU(),
            )
        self.DropLayer = nn.Dropout(0.2)

        self.gamma = nn.Parameter(torch.zeros(1))          

    def forward(self, x):
        residual = x
        x = self.block0(x)
        x = self.DropLayer(x)
        x = self.block1(x)
        return x + self.conv_0(residual)

class Conv3dUnit(nn.Module):
    def __init__(self, inchannel, outchannel=1, kernel_size=1):
        super(Conv3dUnit, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            norm_func(outchannel, affine=True),
            nn.GELU()
        )
    def forward(self, x):
        return self.conv(x)

class InputUnit(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size=3):
        super(InputUnit, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            norm_func(outchannel, affine=True),
            nn.GELU()
        )
    def forward(self, x):
        return self.conv(x)

class MMMNet(nn.Module):

    def __init__(self,
                 inchannel=1,
                 num_classes=1,
                 num_head=[16,8,4,2],
                 drop_rate = 0.2,
                 **kwargs
                 ):       
        super(MMMNet, self).__init__()

        base_channel = 64
        num_heads = num_head

        self.A_input = InputUnit(inchannel, base_channel) 
        self.Pooling = nn.AvgPool3d(2, 2) 

        self.A_conv0 = Blocks(base_channel, base_channel*2,num_heads[0], [False, False])

        self.A_conv1 = Blocks(base_channel*2, base_channel*4, num_heads[1], [True, True])
        
        self.A_conv2 = Blocks(base_channel*4, base_channel*8, num_heads[2], [True, True])
          
        self.A_conv3 = Blocks(base_channel*8, base_channel*8, num_heads[3],[True, True])
        
        self.ClassHead = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')

    def forward(self, x):
       
        n, c, d, h, w = x.size()
     
        x0 = self.A_input(x) # n, 64, 32, h, w
        x0 = self.Pooling(x0) # n, 64, 16, h/2, w/2
        x1 = self.A_conv0(x0) # n, 128, 16, h/2, w/2

        x1 = self.Pooling(x1) # n, 128, 8, h/4, w/4
        x2 = self.A_conv1(x1) # n, 256, 8, h/4, w/4

        x3 = self.Pooling(x2) # n, 256, 4, h/8, w/8
        x3 = self.A_conv2(x3) # n, 512, 4, h/8, w/8

        '''x4 = self.Pooling(x3) # n, 512, 2, h/16, w/16
        x4 = self.A_conv3(x4) # n, 512, 2, h/16, w/16'''

        # n, 512, 1, 1, 1 => n, 512 
        out = self.ClassHead(nn.AdaptiveMaxPool3d((1,1,1))(x3).view(x3.shape[0],-1)) 

        return x3,out

    def args_dict(self):
        model_args = {}

        return model_args

###CMFA
class CMFA(nn.Module):
    def __init__(self,img_dim,tab_dim,hid_dim,heads=4,dropout=0.2):
        super().__init__()

        self.fi1=nn.Linear(img_dim,hid_dim)
        self.fi2=nn.Linear(hid_dim,hid_dim)
        self.ft1=nn.Linear(tab_dim,hid_dim)
        self.ft2=nn.Linear(hid_dim,hid_dim)

        self.conv_i1 = nn.Linear(hid_dim, hid_dim)
        self.conv_i2 = nn.Linear(hid_dim, hid_dim)
        self.conv_i3 = nn.Linear(hid_dim, hid_dim)
        self.conv_t1 = nn.Linear(hid_dim, hid_dim)
        self.conv_t2 = nn.Linear(hid_dim, hid_dim)
        self.conv_t3 = nn.Linear(hid_dim, hid_dim)

        self.self_attn_V = MultiheadAttention(hid_dim, heads,dropout=dropout)
        self.self_attn_T = MultiheadAttention(hid_dim, heads,dropout=dropout)
        
    def forward(self,i,t):
        #residual_i = i

        i_ = self.fi1(i)
        i_=F.relu(i_)
        t_ = self.ft1(t)
        t_=F.relu(t_)
        residual_i_ = i_
        residual_t_ = t_

        v1 = F.relu(self.conv_i1(i_))
        k1 = F.relu(self.conv_i2(i_))
        q1 = F.relu(self.conv_i3(i_))
        v2 = F.relu(self.conv_t1(t_))
        k2 = F.relu(self.conv_t2(t_))
        q2 = F.relu(self.conv_t3(t_))

        V_ = self.self_attn_V(q2, k1, v1)[0]
        T_ = self.self_attn_T(q1, k2, v2)[0]
        V_ = V_ + residual_i_
        T_ = T_ + residual_t_

        V_ = self.fi2(V_)
        T_ = self.ft2(T_)

        #V_ = V_ + residual_i    

        return torch.cat((V_,T_),1) 

###
class PEMVCNet(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.mv = MMMNet()
        self.gp = nn.AdaptiveAvgPool3d(1)
        self.li1 = nn.Linear(512,128)
        self.ai1 = nn.ReLU()
        self.li2 = nn.Linear(128,32)

        nn.init.kaiming_normal_(self.li1.weight,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_normal_(self.li2.weight,mode='fan_in',nonlinearity='relu')

        self.lt1 = nn.Linear(7,32)
        self.at1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)
        self.lt2 = nn.Linear(32,32)

        nn.init.kaiming_normal_(self.lt1.weight,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_normal_(self.lt2.weight,mode='fan_in',nonlinearity='relu')

        self.Fusion = CMFA(img_dim=32,tab_dim=32,hid_dim=32)
        self.c1 = nn.Linear(64,32)
        self.a1 = nn.ReLU()
        self.c2 = nn.Linear(32,1)

        nn.init.kaiming_normal_(self.c1.weight,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_normal_(self.c2.weight,mode='fan_in',nonlinearity='relu')

    def forward(self,i,t):
        i,_ = self.mv(i)
        i = self.gp(i)
        i = i.view(i.size(0),-1) # 8*512
        i = self.li1(i)
        i = self.ai1(i)
        i = self.li2(i) # 8*128

        t = self.lt1(t)
        t = self.at1(t)
        t = self.dropout1(t)
        t = self.lt2(t) # 8*128

        fusion = self.Fusion(i,t)
        out = self.c1(fusion)
        out = self.a1(out)
        out = self.c2(out)

        return out

    def args_dict(self):
        model_args = {}

        return model_args  
