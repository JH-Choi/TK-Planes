import numpy as np
import torch
import torch.nn as nn

from .LimitGradLayer import LimitGradLayer

def conv3x3(in_channels, out_channels, stride=1,
            padding=0, bias=False, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        padding_mode='replicate',
        #padding_mode='zeros',        
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
            padding=0, bias=False, groups=1):
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
    def __init__(self, in_channels, out_channels, final_layer=False, mode='transpose', curr_dim=None, first_layer=False):
        super(UpConv, self).__init__()

        #self.relu = nn.GELU()
        self.relu = nn.LeakyReLU(0.02)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_layer = first_layer
        self.final_layer = final_layer

        self.limit_layer = LimitGradLayer.apply
        #self.curr_dim = (self.out_channels,curr_dim,curr_dim)

        curr_channels = self.in_channels #// 2

        self.conv1 = upconv2x2(self.out_channels, self.out_channels, mode)

        self.process = nn.Sequential(#nn.InstanceNorm2d(curr_channels_s,affine=True),
            conv3x3(curr_channels, 2*curr_channels),
            self.relu,
            #nn.InstanceNorm2d(2*curr_channels_s,affine=True),
            #conv3x3(2*curr_channels_s, 2*curr_channels_s),
            #self.relu,
            #nn.InstanceNorm2d(2*curr_channels_s,affine=True),
            #conv3x3(2*curr_channels_s, 2*curr_channels_s),
            #self.relu,                                            
            #nn.InstanceNorm2d(2*curr_channels_s,affine=True),
            conv3x3(2*curr_channels, self.out_channels),
            self.relu)

        #self.skippy = nn.Sequential(nn.InstanceNorm2d(curr_channels),upconv2x2(curr_channels,self.out_channels,mode='upsample'))
        #self.skippy_s = nn.Sequential(conv1x1(curr_channels_s,self.out_channels[0])) # // 2))
        #self.skippy_d = nn.Sequential(conv1x1(curr_channels_d,self.out_channels[1])) # // 2))        
        #self.skippy1 = nn.Sequential(upconv2x2(curr_channels,2*curr_channels,mode='upsample'))        
        #self.skippy2 = nn.Sequential(conv1x1(2*curr_channels,self.out_channels))        
        if final_layer:
            curr_out_channels = self.out_channels
            self.conv_final = nn.Sequential(#nn.InstanceNorm2d(self.out_channels),
                                            #conv5x5(curr_out_channels, curr_out_channels // 2, padding=0),
                                            conv3x3(curr_out_channels, curr_out_channels * 4),                
                                            #nn.InstanceNorm2d(self.out_channels // 2),
                                            nn.LeakyReLU(0.02),
                                            #nn.InstanceNorm2d(self.out_channels // 2),
                                            #conv5x5(curr_out_channels // 2, curr_out_channels // 4, padding=0),
                                            conv3x3(curr_out_channels * 4, curr_out_channels * 4),                
                                            nn.LeakyReLU(0.02),
                                            conv3x3(curr_out_channels * 4, curr_out_channels * 4),                
                                            nn.LeakyReLU(0.02),                
                                            #nn.InstanceNorm2d(self.out_channels // 4),                                            
                                            #conv3x3(curr_out_channels // 4, curr_out_channels // 8, padding=0),
                                            #nn.LeakyReLU(0.02),
                                            #nn.InstanceNorm2d(self.out_channels // 8),                                            
                                            conv3x3(curr_out_channels * 4, curr_out_channels * 2),
                                            nn.LeakyReLU(0.02))
            dir_encoding_dims = 0
            self.dir_render = nn.Sequential(
                                            conv1x1((curr_out_channels * 2) + dir_encoding_dims, curr_out_channels * 4),
                                            nn.LeakyReLU(0.02),                                
                                            conv1x1(curr_out_channels * 4, curr_out_channels * 4),
                                            nn.LeakyReLU(0.02),                                
                                            conv1x1(curr_out_channels * 4, curr_out_channels),
                                            nn.LeakyReLU(0.02),                                
                                            conv1x1(curr_out_channels, 3),                 
                                            #nn.InstanceNorm2d(3),
                                            #nn.Tanh())
                                            nn.Sigmoid()
            )
    def forward(self, x, dynamo, dir_encoding):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        x = self.process(x)

        x = self.relu(self.conv1(x))        
        
        if self.final_layer:
            x = self.conv_final(x)
            #x = torch.concat([x,dir_encoding],dim=1)
            x = self.dir_render(x)

        return x
            
class ImageDecoder(nn.Module):
    def __init__(self, input_dim=8, final_dim=64, feature_dim=512, mode='transpose'):
        super().__init__()
        self.layers = []
        self.input_dim = input_dim
        self.final_dim = final_dim
        self.feature_dim = feature_dim
        #self.norm0 = nn.InstanceNorm2d(feature_dim)

        done_first_loop = False
        while (input_dim != final_dim):
            input_dim = input_dim*2
            if done_first_loop:
                in_feature_dim = 2*self.feature_dim
            else:
                in_feature_dim = self.feature_dim
            out_feature_dim = self.feature_dim // 2

            self.layers.append(UpConv(in_feature_dim,out_feature_dim,input_dim == final_dim,mode,input_dim,not done_first_loop))
            self.feature_dim = out_feature_dim 
            done_first_loop = True

        self.num_layers = len(self.layers)
        self.layers = nn.ModuleList(self.layers)

    def forward(self,x,dynamo,dir_encoding):
        #x = self.norm0(x)
        x_lst = x
        x = x_lst[-1]

        reverse_counter = -2
        for idx,l in enumerate(self.layers):
            #print(x.shape)
            #print(x.shape)
            #exit(-1)

            if idx < self.num_layers - 1:
                x = l(x,dynamo,None)
                x = torch.cat([x,x_lst[reverse_counter]],dim=1)
                reverse_counter -= 1
            else:
                x = l(x,dynamo,dir_encoding)
                
        return x
    
