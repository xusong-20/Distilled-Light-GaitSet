import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class LocalDistillLoss(nn.Module):
    def __init__(self):
        super(LocalDistillLoss, self).__init__()
        
    
    def forward(self, target, output):
    
        n, s, c, h, w = output.size()
        student = output.view(n,-1)
        norm_student = F.normalize(student)
        
        with torch.no_grad():
            teacher = target.view(n, -1)
            norm_teacher = torch.nn.functional.normalize(teacher) 
            
        ts_diff =  norm_teacher - norm_student
        loss = (ts_diff * ts_diff).view(-1,1).sum(0) / (n * n)
        
        return loss

class GlobalDistillLoss(nn.Module):
    def __init__(self, margin):
        super(GlobalDistillLoss, self).__init__()
        self.margin = margin
        
    def forward(self, target, output, label):
        n, b, c =  output.size()
   
        hp_mask = (label.unsqueeze(0) == label.unsqueeze(1)).byte().view(-1).bool()
        hn_mask = (label.unsqueeze(0) != label.unsqueeze(1)).byte().view(-1).bool()

        #studet full triplet metric
        student  =  output.view(n, -1)
        student_dist = self.batch_dist(student).view(-1)
        student_hp_dist = torch.masked_select(student_dist, hp_mask).view(n, -1, 1)
        student_hn_dist = torch.masked_select(student_dist, hn_mask).view(n, 1, -1)
        student_full_loss_metric = F.relu(self.margin + student_hp_dist - student_hn_dist).view(-1)
        student_full_loss_metric_sum = student_full_loss_metric.sum(0)
        student_full_loss_num = (student_full_loss_metric != 0).sum(0).float()
        student_full_loss_metric_mean = student_full_loss_metric_sum / student_full_loss_num 
        student_full_loss_metric_mean[student_full_loss_num  == 0] = 0
        
        #teacher full triplet metric
        with torch.no_grad():
            teacher  =  target.view(n, -1)
            teacher_dist = self.batch_dist(teacher).view(-1)
            teacher_hp_dist = torch.masked_select(teacher_dist, hp_mask).view(n, -1, 1)
            teacher_hn_dist = torch.masked_select(teacher_dist, hn_mask).view(n, 1, -1)
            teacher_full_loss_metric = F.relu(self.margin + teacher_hp_dist - teacher_hn_dist).view(-1)
            teacher_full_loss_metric_sum = teacher_full_loss_metric.sum(0)
            teacher_full_loss_num = (teacher_full_loss_metric != 0).sum(0).float()
            teacher_full_loss_metric_mean = teacher_full_loss_metric_sum / teacher_full_loss_num 
            teacher_full_loss_metric_mean[teacher_full_loss_num  == 0] = 0
            
        loss = F.smooth_l1_loss(student_full_loss_metric_mean, teacher_full_loss_metric_mean)
        
        return loss

    def batch_dist(self, x):
        x2 = torch.sum(x ** 2, 1)
        dist = x2.unsqueeze(1) + x2.unsqueeze(1).transpose(0, 1) - 2 * torch.matmul(x, x.transpose(0, 1))
        dist = torch.sqrt(F.relu(dist))
        return dist


class GlobalDistillLoss_Hard(nn.Module):
    def __init__(self, margin):
        super(GlobalDistillLoss_Hard, self).__init__()
        self.margin = margin
        
    def forward(self, target, output, label):
        n, b, c =  output.size()
   
        hp_mask = (label.unsqueeze(0) == label.unsqueeze(1)).byte().view(-1).bool()
        hn_mask = (label.unsqueeze(0) != label.unsqueeze(1)).byte().view(-1).bool()
        
        #studet hard triplet metric
        student  =  output.view(n, -1)
        student_dist = self.batch_dist(student).view(-1)
        student_hard_hp_dist, student_hard_hp_index = torch.max(torch.masked_select(student_dist, hp_mask).view(n, -1), 1)
        student_hard_hn_dist, student_hard_hn_index = torch.min(torch.masked_select(student_dist, hn_mask).view(n, -1), 1)
        student_hard_loss_metric = F.relu(self.margin + student_hard_hp_dist - student_hard_hn_dist).view(-1)
        student_hard_loss_metric_mean = torch.mean(student_hard_loss_metric,0)
        
        #teacher hard triplet metric
        with torch.no_grad():
            teacher  =  target.view(n, -1)
            teacher_dist = self.batch_dist(teacher).view(-1)
            teacher_hard_hp_dist = torch.index_select(torch.masked_select(teacher_dist, hp_mask).view(n, -1), 1, student_hard_hp_index)
            teacher_hard_hn_dist = torch.index_select(torch.masked_select(teacher_dist, hn_mask).view(n, -1), 1, student_hard_hn_index)
            teacher_hard_loss_metric = F.relu(self.margin + teacher_hard_hp_dist - teacher_hard_hn_dist).view(-1)
            teacher_hard_loss_metric_mean = torch.mean(teacher_hard_loss_metric, 0)
            
        loss = F.smooth_l1_loss(student_hard_loss_metric_mean, teacher_hard_loss_metric_mean)
        
        return loss

    def batch_dist(self, x):
        x2 = torch.sum(x ** 2, 1)
        dist = x2.unsqueeze(1) + x2.unsqueeze(1).transpose(0, 1) - 2 * torch.matmul(x, x.transpose(0, 1))
        dist = torch.sqrt(F.relu(dist))
        return dist


