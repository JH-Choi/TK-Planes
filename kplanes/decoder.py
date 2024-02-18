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

        if not first_layer or True:
            if not first_layer:
                preprocess_channels_s = self.in_channels[0] // 2
                preprocess_channels_d = self.in_channels[1] // 2
            else:
                preprocess_channels_s = self.in_channels[0]
                preprocess_channels_d = self.in_channels[1]
            '''    
            self.preprocess_static = nn.Sequential(conv3x3(preprocess_channels_s, 2*preprocess_channels_s, padding=1),
                                                   self.relu,
                                                   nn.InstanceNorm2d(2*preprocess_channels_s, affine=True),
                                                   conv3x3(2*preprocess_channels_s, preprocess_channels_s, padding=1),
                                                   self.relu)
            self.preprocess_dynamic = nn.Sequential(conv3x3(preprocess_channels_d, 2*preprocess_channels_d, padding=1),
                                                    self.relu,
                                                    nn.InstanceNorm2d(2*preprocess_channels_d,affine=True),
                                                    conv3x3(2*preprocess_channels_d, preprocess_channels_d, padding=1),
                                                    self.relu)           
            '''
            #self.in_channels = self.in_channels // 2
            #self.norm1 = nn.InstanceNorm2d(self.in_channels,affine=True)
            self.preprocess_static = nn.Identity()
            self.preprocess_dynamic = nn.Identity()
            self.norm1_s = nn.Identity()
            self.norm1_d = nn.Identity()                        
        else:
            preprocess_channels_s = self.in_channels[0] // 2
            preprocess_channels_d = self.in_channels[0] // 2       
            self.preprocess_static = nn.Identity()
            self.preprocess_dynamic = nn.Identity()            
            self.norm1_s = nn.Identity()
            self.norm1_d = nn.Identity()            
            
        #self.norm0 = nn.InstanceNorm2d(self.in_channels,affine=True)
        #self.norm0 = nn.LayerNorm([self.in_channels,curr_dim,curr_dim],elementwise_affine=True)        
        #self.conv1 = torch.nn.utils.spectral_norm(upconv2x2(self.in_channels, self.out_channels, mode))

        curr_channels_s = self.in_channels[0] #// 2
        curr_channels_d = self.in_channels[1] #// 2
        #self.conv1_s = upconv2x2(curr_channels_s, curr_channels_s, mode)
        #self.conv1_d = upconv2x2(curr_channels_d, curr_channels_d, mode)
        self.conv1_s = upconv2x2(self.out_channels[0], self.out_channels[0], mode)
        self.conv1_d = upconv2x2(self.out_channels[1], self.out_channels[1], mode)                
        #self.norm1 = nn.LayerNorm([self.in_channels,curr_dim,curr_dim],elementwise_affine=True)
        #self.norm1 = torch.nn.Identity()
        #curr_dim *= 2
        self.process_static = nn.Sequential(#nn.InstanceNorm2d(curr_channels_s,affine=True),
                                            conv3x3(curr_channels_s, 2*curr_channels_s),
                                            self.relu,
                                            #nn.InstanceNorm2d(2*curr_channels_s,affine=True),
                                            #conv3x3(2*curr_channels_s, 2*curr_channels_s),
                                            #self.relu,
                                            #nn.InstanceNorm2d(2*curr_channels_s,affine=True),
                                            #conv3x3(2*curr_channels_s, 2*curr_channels_s),
                                            #self.relu,                                            
                                            #nn.InstanceNorm2d(2*curr_channels_s,affine=True),
                                            conv3x3(2*curr_channels_s, self.out_channels[0]),
                                            self.relu)

        self.process_dynamic = nn.Sequential(#nn.InstanceNorm2d(curr_channels_d,affine=True),
                                            conv3x3(curr_channels_d, 2*curr_channels_d),
                                            self.relu,
                                            #nn.InstanceNorm2d(2*curr_channels_d,affine=True),
                                            #conv3x3(2*curr_channels_d, 2*curr_channels_d),
                                            #self.relu,
                                            #nn.InstanceNorm2d(2*curr_channels_d,affine=True),
                                            #conv3x3(2*curr_channels_d, 2*curr_channels_d),
                                            #self.relu,                                             
                                            #nn.InstanceNorm2d(2*curr_channels_d,affine=True),
                                            conv3x3(2*curr_channels_d, self.out_channels[1]),
                                            self.relu)
        
        ##self.norm2 = nn.InstanceNorm2d(curr_channels,affine=True)
        #self.norm2 = nn.Identity()
        ##self.conv2 = conv3x3(curr_channels, 2*curr_channels)
        #self.norm2 = nn.Identity()
        #self.norm2 = nn.InstanceNorm2d(self.out_channels,affine=True)
        #self.norm2 = nn.LayerNorm([self.out_channels,curr_dim,curr_dim]) #,elementwise_affine=True)        
        #self.conv3 = torch.nn.utils.spectral_norm(conv5x5(self.out_channels, self.out_channels))
        ##self.norm3 = nn.InstanceNorm2d(2*curr_channels,affine=True)        
        ##self.conv3 = conv3x3(2*curr_channels, 2*curr_channels)
        ##self.norm4 = nn.InstanceNorm2d(2*curr_channels,affine=True)
        #self.norm4 = nn.Identity()        
        ##self.conv4 = conv3x3(2*curr_channels, self.out_channels)        
        #self.norm3 = nn.Identity()
        #self.norm3 = nn.InstanceNorm2d(self.out_channels,affine=True)
        #self.norm3 = nn.LayerNorm([self.out_channels,curr_dim,curr_dim]) #,elementwise_affine=True)
        #self.drop = nn.Dropout(0.3)

        #self.skippy = nn.Sequential(nn.InstanceNorm2d(curr_channels),upconv2x2(curr_channels,self.out_channels,mode='upsample'))
        #self.skippy_s = nn.Sequential(conv1x1(curr_channels_s,self.out_channels[0])) # // 2))
        #self.skippy_d = nn.Sequential(conv1x1(curr_channels_d,self.out_channels[1])) # // 2))        
        #self.skippy1 = nn.Sequential(upconv2x2(curr_channels,2*curr_channels,mode='upsample'))        
        #self.skippy2 = nn.Sequential(conv1x1(2*curr_channels,self.out_channels))        
        if final_layer:
            curr_out_channels = self.out_channels[0] + self.out_channels[1]
            self.conv_final = nn.Sequential(#nn.InstanceNorm2d(self.out_channels),
                                            #conv5x5(curr_out_channels, curr_out_channels // 2, padding=0),
                                            conv3x3(curr_out_channels, curr_out_channels * 4),                
                                            #nn.InstanceNorm2d(self.out_channels // 2),
                                            nn.LeakyReLU(0.02),
                                            #nn.InstanceNorm2d(self.out_channels // 2),
                                            #conv5x5(curr_out_channels // 2, curr_out_channels // 4, padding=0),
                                            conv3x3(curr_out_channels * 4, curr_out_channels),                
                                            nn.LeakyReLU(0.02),
                                            conv3x3(curr_out_channels, curr_out_channels // 4),                
                                            nn.LeakyReLU(0.02),                
                                            #nn.InstanceNorm2d(self.out_channels // 4),                                            
                                            #conv3x3(curr_out_channels // 4, curr_out_channels // 8, padding=0),
                                            #nn.LeakyReLU(0.02),
                                            #nn.InstanceNorm2d(self.out_channels // 8),                                            
                                            conv3x3(curr_out_channels // 4, 3),
                                            #nn.InstanceNorm2d(3),
                                            #nn.Tanh())
                                            nn.Sigmoid()
            )
    def forward(self, x, dynamo):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """

        #skip_x1 = self.skippy1(x)
        if not self.first_layer:    
            #x_s = x[:,:2*self.in_channel[0]]
            #x_s = x[:,2*self.in_channel[0]:]            
            x_prev = x[:,:x.shape[1] // 2]
            x_new = x[:,x.shape[1] // 2:]            
            x_prev_s = x_prev[:,:self.in_channels[0] // 2]
            x_prev_d = x_prev[:,self.in_channels[0] // 2:]
            x_new_s = x_new[:,:self.in_channels[0] // 2]
            x_new_d = x_new[:,self.in_channels[0] // 2:]            
            x_new_s = self.preprocess_static(x_new_s)
            x_new_d = self.preprocess_dynamic(x_new_d)

            x_s = torch.concat([x_prev_s,x_new_s],dim=1)
            x_d = torch.concat([x_prev_d,x_new_d],dim=1)            
        else:
            x_s = x[:,:self.in_channels[0]]
            x_d = x[:,self.in_channels[0]:]
            x_s = self.preprocess_static(x_s)
            x_d = self.preprocess_dynamic(x_d)

        #print('START')
        #print(x_s.shape)
        #x = torch.concat([x1,x2_s,x2_d],dim=1)

        #skip_x_s = self.skippy_s(x_s)
        #skip_x_d = self.skippy_d(x_d)
        #print(x_s.shape)
        x_s = self.process_static(x_s)
        x_d = self.process_dynamic(x_d)        
        #print(x_s.shape)
        x_s = self.relu(self.conv1_s(self.norm1_s(x_s)))        
        x_d = self.relu(self.conv1_d(self.norm1_d(x_d)))
        #print(x_s.shape)
        #exit(-1)
        #x = self.norm2(x)
        #x = self.relu(self.conv2(x))
        #x = x + skip_x1

        #skip_x2 = self.skippy2(x)
        #x = self.relu(self.conv3(self.norm3(x)))
        #x = self.relu(self.conv4(self.norm4(x)))        

        #x = self.relu(self.norm1(self.conv1(x)))
        #x = self.relu(self.norm2(self.conv2(x)))
        #x = self.relu(self.norm3(self.conv3(x)))

        #x = x + skip_x2

        #x_s = x_s + skip_x_s
        #x_d = x_d + skip_x_d
        #if self.final_layer:
            #x_d = 0*x_d
            #x_s = 0*x_s
        x_d = self.limit_layer(x_d,dynamo)
        x = torch.concat([x_s,x_d],dim=1)

        #print(x.shape)
        #print('STOP')
        
        if self.final_layer:
            
            x = self.conv_final(x)
            
        return x
            
class ImageDecoder(nn.Module):
    def __init__(self, input_dim=(8,8), final_dim=(64,64), feature_dim=(512,512), mode='transpose'):
        super().__init__()
        self.layers = []
        self.input_dim = input_dim
        self.final_dim = final_dim
        self.feature_dim = feature_dim
        #self.norm0 = nn.InstanceNorm2d(feature_dim)

        done_first_loop = False
        while (input_dim[0] != final_dim[0]):
            input_dim = (input_dim[0]*2,input_dim[1]*2)
            if done_first_loop:
                in_feature_dim = (2*self.feature_dim[0],2*self.feature_dim[1])
                #out_feature_dim = (self.feature_dim[0] // 2, self.feature_dim[1] // 2)
            else:
                in_feature_dim = self.feature_dim
            out_feature_dim = (self.feature_dim[0] // 2, self.feature_dim[1] // 2)

            self.layers.append(UpConv(in_feature_dim,out_feature_dim,input_dim[0] == final_dim[0],mode,input_dim[0],not done_first_loop))
            self.feature_dim = out_feature_dim #(self.feature_dim[0] // 2, self.feature_dim[1] // 2)
            done_first_loop = True

        self.num_layers = len(self.layers)
        self.layers = nn.ModuleList(self.layers)

    def forward(self,x,dynamo):
        #x = self.norm0(x)
        x_lst = x
        x = x_lst[-1]

        reverse_counter = -2
        for idx,l in enumerate(self.layers):
            #print(x.shape)
            x = l(x,dynamo)
            #print(x.shape)
            #exit(-1)

            if idx < self.num_layers - 1:
                x = torch.cat([x,x_lst[reverse_counter]],dim=1)
                reverse_counter -= 1
                
        return x
    
