import numpy as np
import torch
import torch.nn as nn

def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=False, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def conv2x2(in_channels, out_channels, stride=1,
            padding=1, bias=False, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=2,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def conv5x5(in_channels, out_channels, stride=1,
            padding=2, bias=False, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=5,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose', groups=1):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            groups=groups,
            stride=2,
            bias=False)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels, groups) if in_channels != out_channels else nn.Identity())

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1,
        bias=False)
    
class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, final_layer=False, mode='transpose', curr_dim=None):
        super(UpConv, self).__init__()

        #self.relu = nn.GELU()
        self.relu = nn.LeakyReLU(0.02)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_layer = final_layer

        #self.curr_dim = (self.out_channels,curr_dim,curr_dim)
        
        #self.norm0 = nn.InstanceNorm2d(self.in_channels,affine=True)
        #self.norm0 = nn.LayerNorm([self.in_channels,curr_dim,curr_dim],elementwise_affine=True)        
        #self.conv1 = torch.nn.utils.spectral_norm(upconv2x2(self.in_channels, self.out_channels, mode))
        self.conv1 = upconv2x2(self.in_channels, self.in_channels, mode)
        self.norm1 = nn.InstanceNorm2d(self.in_channels,affine=True)
        #self.norm1 = nn.LayerNorm([self.in_channels,curr_dim,curr_dim],elementwise_affine=True)
        #self.norm1 = torch.nn.Identity()
        #curr_dim *= 2
        self.norm2 = nn.InstanceNorm2d(self.in_channels,affine=True)        
        self.conv2 = conv3x3(self.in_channels, self.in_channels)
        #self.norm2 = nn.Identity()
        #self.norm2 = nn.InstanceNorm2d(self.out_channels,affine=True)
        #self.norm2 = nn.LayerNorm([self.out_channels,curr_dim,curr_dim]) #,elementwise_affine=True)        
        #self.conv3 = torch.nn.utils.spectral_norm(conv5x5(self.out_channels, self.out_channels))
        self.norm3 = nn.InstanceNorm2d(self.in_channels,affine=True)        
        self.conv3 = conv3x3(self.in_channels, self.out_channels)
        #self.norm3 = nn.Identity()
        #self.norm3 = nn.InstanceNorm2d(self.out_channels,affine=True)
        #self.norm3 = nn.LayerNorm([self.out_channels,curr_dim,curr_dim]) #,elementwise_affine=True)
        #self.drop = nn.Dropout(0.3)

        self.skippy = nn.Sequential(nn.InstanceNorm2d(self.in_channels),upconv2x2(self.in_channels,self.out_channels,mode='upsample'))
        if final_layer:
            self.conv_final = nn.Sequential(#nn.InstanceNorm2d(self.out_channels),
                conv3x3(self.out_channels, self.out_channels // 2),
                #nn.InstanceNorm2d(self.out_channels // 2),
                nn.LeakyReLU(0.02),
                #nn.InstanceNorm2d(self.out_channels // 2),
                conv3x3(self.out_channels // 2, self.out_channels // 4),
                nn.LeakyReLU(0.02),
                conv1x1(self.out_channels // 4, 3),
                #nn.InstanceNorm2d(3),
                #nn.Tanh())
                nn.Sigmoid())

    def forward(self, x):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """

        skip_x = self.skippy(x)
        x = self.relu(self.conv1(self.norm1(x)))
        x = self.relu(self.conv2(self.norm2(x)))
        x = self.relu(self.conv3(self.norm3(x)))

        #x = self.relu(self.norm1(self.conv1(x)))
        #x = self.relu(self.norm2(self.conv2(x)))
        #x = self.relu(self.norm3(self.conv3(x)))
                
        x = x + skip_x
        if self.final_layer:
            x = self.conv_final(x)
            
        return x
            
class ImageDecoder(nn.Module):
    def __init__(self, input_dim=(45,80), final_dim=(360,640), feature_dim=256, mode='transpose'):
        super().__init__()
        self.layers = []
        self.input_dim = input_dim
        self.final_dim = final_dim
        self.feature_dim = feature_dim

        #self.norm0 = nn.InstanceNorm2d(feature_dim)
        
        while (input_dim[0] != final_dim[0]):
            input_dim = (input_dim[0]*2,input_dim[1]*2)
            #self.layers.append(UpConv(self.feature_dim,self.feature_dim // 2 if input_dim != final_dim else 3,input_dim == final_dim,mode,input_dim//2))
            self.layers.append(UpConv(self.feature_dim,self.feature_dim // 2,input_dim[0] == final_dim[0],mode,input_dim[0]))            
            self.feature_dim = self.feature_dim // 2

        self.layers = nn.ModuleList(self.layers)

    def forward(self,x):

        #x = self.norm0(x)
        
        for l in self.layers:
            x = l(x)

        return x
    
