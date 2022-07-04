import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CELoss(nn.Module):
    def __init__(self, num_classes, eps=0.1):
        super(CELoss,self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.confidence = 1.-eps

    def forward(self, output, target):
        logprobs = F.log_softmax(output, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(-1)
        loss = self.confidence * nll_loss + self.eps * smooth_loss
        return loss
