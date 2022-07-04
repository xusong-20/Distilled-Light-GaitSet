import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import mean
import numpy as np


class HPM(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(HPM, self).__init__()
        self.bin_num = [16]
        self.fc_bin = nn.ModuleList([nn.Linear(in_dims, out_dims, bias=False) for i in range(sum(self.bin_num))])

    def forward(self, x):
    
        #stage-HP: Horizontal Pooling
        feature = list()
        n, c, h, w = x.size()
        for num_bin in self.bin_num:
            z = x.view(n, c, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)
        feature = torch.cat(feature, 2).permute(0, 2, 1).contiguous()
        
        #stage-FM: Feature Mapping
        fc_feature = list()
        for i in range(sum(self.bin_num)):
            z = self.fc_bin[i](feature[:, i, :])
            fc_feature.append(z.unsqueeze(1))
        feature = torch.cat(fc_feature, 1)

        return feature
