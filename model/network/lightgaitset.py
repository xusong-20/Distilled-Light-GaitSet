import torch
import torch.nn as nn
import numpy as np
from .basic_blocks import SetBlock
from .rep_block import RepBlock
from .hpm import HPM

class LightSetNet(nn.Module):
    def __init__(self, hidden_dim):
        super(LightSetNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None

        _set_in_channels = 1
        _set_channels = [32,64,128]
      
        self.set_layer1 = SetBlock(RepBlock(_set_in_channels, _set_channels[0], CheapConv2d=False, expand_size=2), True)
        self.set_layer2 = SetBlock(RepBlock(_set_channels[0], _set_channels[1], CheapConv2d=True,  expand_size=2), True)
        self.set_layer3 = SetBlock(RepBlock(_set_channels[1], _set_channels[2], CheapConv2d=True,  expand_size=2))
        self.hpm = HPM(_set_channels[-1],  self.hidden_dim)

      
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    
    def frame_max(self, x):
        if self.batch_frame is None:
            return torch.max(x, 1)
        else:
            _tmp = [
                torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
                ]
            max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return max_list, arg_max_list



    def forward(self, silho, batch_frame=None):
        if batch_frame is not None:
            batch_frame = batch_frame[0].data.cpu().numpy().tolist()
            _ = len(batch_frame)
            for i in range(len(batch_frame)):
                if batch_frame[-(i + 1)] != 0:
                    break
                else:
                    _ -= 1
            batch_frame = batch_frame[:_]
            frame_sum = np.sum(batch_frame)
            if frame_sum < silho.size(1):
                silho = silho[:, :frame_sum, :, :]
            self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
        n = silho.size(0)
        x = silho.unsqueeze(2)
        del silho
        
        x = self.set_layer1(x)
        x = self.set_layer2(x)
        x = self.set_layer3(x)
        fl_feature = x
        #sl : set-level fl: frame-level
        sl_feat = self.frame_max(x)[0]
        sl_feat = self.hpm(sl_feat)  
        sl_feature = sl_feat
        
        label_prob = None
        
        return sl_feature, label_prob, fl_feature
