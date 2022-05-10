import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchstat import stat
from thop import profile


class GaitSet(nn.Module):
    def __init__(self, hidden_dim=256):
        super(GaitSet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None

        init_channels = 1
        channels = [32,64,128]
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=init_channels, out_channels=channels[0], kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=channels[0], out_channels=channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )
            
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=channels[1], out_channels=channels[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )   
         
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=channels[2], out_channels=channels[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True)
        )    
        
        
        self.gl1 = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=channels[1], out_channels=channels[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )     
               
        self.gl2 = nn.Sequential(
            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=channels[2], out_channels=channels[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
        )
        
        
        self.bin_num = [1,2,4,8,16]
        self.fc_bin = nn.ModuleList([nn.Linear(channels[2], self.hidden_dim, bias=False) for i in range(sum(self.bin_num)*2)])

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal(m.weight.data, 1.0, 0.02)
                nn.init.constant(m.bias.data, 0.0)


    def frame_max(self, x):
        x = torch.max(x,0)[0]
        return x.unsqueeze(0)
            
    def forward(self,x):
        x = self.block1(x)
        gl = self.gl1(self.frame_max(x))
        
        x = self.block2(x)
        gl = self.gl2(gl + self.frame_max(x))
       
        x  = self.block3(x)
        x = self.frame_max(x)[0]
       
        gl = gl + x

        feature = list()
        c, h, w = x.size()
        for num_bin in self.bin_num:
            z = x.view( c, num_bin, -1)
            z = z.mean(2) + z.max(2)[0]
            feature.append(z)
            z = gl.view(c, num_bin, -1)
            z = z.mean(2) + z.max(2)[0]
            feature.append(z)
            
        feature = torch.cat(feature, 1).unsqueeze(0).permute(0, 2, 1).contiguous()
        fc_feature = list()
        for i in range(sum(self.bin_num)*2):
            z = self.fc_bin[i](feature[:, i, :])
            fc_feature.append(z.unsqueeze(1))
        feature = torch.cat(fc_feature, 1)
      

if __name__ =='__main__':
   
    model = GaitSet()
    input = torch.randn(80, 1, 64, 44)
    flops, params = profile(model, inputs=(input, ))
    print(flops, params)
    print("Number of FLOPs: %.2fG" % (flops/1e9))
    print("Number of parameter: %.2fM" % (params/1e6))