import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import mean
import numpy as np



def Conv2D(in_dims, out_dims, k=3, p=1, s=1, g=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_dims, out_channels=out_dims, kernel_size=k,
                                        stride=s, padding=p, groups=g, bias=False))
    return result


class RepBlock(nn.Module):
    def __init__(self, in_dims, out_dims, k=3, g=1, s=1, deploy=False, CheapConv2d=True, expand_size=2):
        super().__init__()
        self.input_channel = in_dims
        self.output_channel = out_dims
        self.deploy = deploy
        self.CheapConv2d = CheapConv2d
        self.expand_size = expand_size
        self.kernel_size = k
        self.padding = k // 2
        self.groups = g
        self.strides = s
        self.activation = nn.LeakyReLU(inplace=True)
        self.exp_dims = in_dims * self.expand_size
        
        # make sure kernel_size=3 padding=1
        assert self.kernel_size == 3
        assert self.padding == 1

        #before rep
        if not self.deploy: 
            if not self.CheapConv2d:
                #basicconv2d
                self.conv_3x3 = Conv2D(in_dims, out_dims, k, self.padding, s, g)
                #branch
                self.conv_1x1 = Conv2D(in_dims, out_dims, k=1, p=0, s=1, g=g)
            else:
                #liner bottle layer
                self.conv_pw1 = Conv2D(in_dims, self.exp_dims, k=1, p=0, s=1, g=g)
                self.conv_3x3 = Conv2D(self.exp_dims, self.exp_dims, k=3, p=1, s=1, g=self.exp_dims)
                self.conv_pw2 = Conv2D(self.exp_dims, out_dims, k=1, p=0, s=1, g=g)
                #branch
                self.conv_1x1 = Conv2D(self.exp_dims, self.exp_dims, k=1, p=0, s=1, g=self.exp_dims)
                self.res = nn.Identity()
        
        #rep
        else: 
            if not self.CheapConv2d:
                self.rep_3x3 = nn.Conv2d(in_channels=in_dims, out_channels=out_dims, kernel_size=3, padding=1,
                                         padding_mode='zeros', stride=1, groups=1, bias=False)
            else:
                self.rep_3x3 = nn.Conv2d(in_channels=self.exp_dims, out_channels=self.exp_dims, kernel_size=3,
                                         padding=1, padding_mode='zeros', stride=1, groups=self.exp_dims, bias=False)

    def forward(self, inputs):
    
        #before rep
        if not self.deploy: 
            if not self.CheapConv2d:
                return self.activation(self.conv_1x1(inputs) + self.conv_3x3(inputs))
            else:
                z = self.activation(self.conv_pw1(inputs))
                return self.conv_pw2(self.activation(self.conv_3x3(z) + self.conv_1x1(z) + self.res(z)))
                
        
        #rep 
        else:
            if not self.CheapConv2d:
                return self.activation(self.rep_3x3(inputs))
            else:
                return self.conv_pw2(self.activation(self.rep_3x3(self.activation(self.conv_pw1(inputs)))))
                


    def _switch_to_deploy(self):
        self.deploy = True
        kernel = self._get_equivalent_kernel_bias()
        self.rep_3x3 = nn.Conv2d(in_channels = self.conv_3x3.conv.in_channels,
                                 out_channels = self.conv_3x3.conv.out_channels,
                                 kernel_size = self.conv_3x3.conv.kernel_size, padding=self.conv_3x3.conv.padding,
                                 padding_mode = self.conv_3x3.conv.padding_mode, stride=self.conv_3x3.conv.stride,
                                 groups = self.conv_3x3.conv.groups, bias=False)
        self.rep_3x3.weight.data = kernel
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv_3x3')
        self.__delattr__('conv_1x1')
        if self.CheapConv2d:
            self.__delattr__('res')

  
    def _pad_1x1_kernel(self, kernel):
        if kernel is None:
            return 0
        else:
            return F.pad(kernel, [1] * 4)

   
    def _get_equivalent_kernel_bias(self):
        conv_3x3_weight = self._fuse_conv_bn(self.conv_3x3)
        conv_1x1_weight = self._fuse_conv_bn(self.conv_1x1)
        if self.CheapConv2d:
            res_weight = self._fuse_conv_bn(self.res)
            return conv_3x3_weight + self._pad_1x1_kernel(conv_1x1_weight) + res_weight
        else:
            return conv_3x3_weight + self._pad_1x1_kernel(conv_1x1_weight)


    def _fuse_conv_bn(self, branch):
        if branch is None:
            return 0, 0
        elif isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
        else:
            if not hasattr(self, 'res_tensor'):
                input_dim = self.exp_dims // self.exp_dims
                kernel_value = np.zeros((self.exp_dims, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.exp_dims):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).cuda()

            kernel = self.id_tensor

        return kernel

