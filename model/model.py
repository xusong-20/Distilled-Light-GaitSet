import math
import os
import os.path as osp
import random
import sys
from datetime import datetime

import collections
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import joblib

from .network import TripletLoss, LocalDistillLoss, GlobalDistillLoss, GlobalDistillLoss_Hard
from .network import SetNet, LightSetNet
from .utils import TripletSampler
from tqdm import tqdm


class Model:
    def __init__(self,
                 distillation,
                 distillation_weight,
                 hidden_dim,
                 teacher_hidden_dim,
                 lr,
                 lr_decay_rate,
                 optimizer_type,
                 weight_decay,
                 momentum,
                 hard_or_full_trip,
                 margin,
                 num_workers,
                 batch_size,
                 restore_iter,
                 total_iter,
                 save_name,
                 teacher_save_name,
                 train_pid_num,
                 type_equalization,
                 frame_num,
                 model_name,
                 teacher_model_name,
                 train_source,
                 test_source,
                 img_size=64):

        self.distillation = distillation
        self.dw = distillation_weight

        self.save_name = save_name
        self.teacher_save_name = teacher_save_name
        

        self.train_pid_num = train_pid_num
        self.type_equalization = type_equalization
        self.train_source = train_source
        self.test_source = test_source
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.optimizer_type = optimizer_type
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.hard_or_full_trip = hard_or_full_trip
        self.margin = margin
        self.frame_num = frame_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.teacher_model_name = teacher_model_name
        self.P, self.M = batch_size
        self.restore_iter = restore_iter
        self.total_iter = total_iter
        self.img_size = img_size
        self.teacher_hidden_dim =  256

        
        # setting basic light model
        self.student_encoder = LightSetNet(self.hidden_dim).float()
        self.student_encoder = nn.DataParallel(self.student_encoder)
        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        self.triplet_loss = nn.DataParallel(self.triplet_loss)
        self.student_encoder.cuda()
        self.triplet_loss.cuda()

        if self.optimizer_type == 'sgd':
            self.student_optimizer = optim.SGD([{'params': self.student_encoder.parameters()}],  lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adam':
            self.student_optimizer = optim.Adam([{'params': self.student_encoder.parameters()}], lr=self.lr, weight_decay=self.weight_decay)

        # triple loss
        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []
        self.dist_list = []
        self.mean_dist = 0.01

        # setting teacher model and distillation loss
        if self.distillation: 
            self.teacher_encoder = SetNet(self.teacher_hidden_dim).float()
            self.lkdloss = LocalDistillLoss().float()
            self.gkdloss = GlobalDistillLoss(self.margin).float()
            self.gkdloss_hard = GlobalDistillLoss_Hard(self.margin).float()
            
            self.teacher_encoder = nn.DataParallel(self.teacher_encoder)
            self.lkdloss = nn.DataParallel(self.lkdloss)
            self.gkdloss = nn.DataParallel(self.gkdloss)
            self.gkdloss_hard = nn.DataParallel(self.gkdloss_hard)
            
            self.teacher_encoder.cuda()
            self.lkdloss.cuda()
            self.gkdloss.cuda()
            self.gkdloss_hard.cuda()
            
            self.teacher_optimizer = optim.Adam([{'params': self.teacher_encoder.parameters()}], lr=1e-4, weight_decay=0.0)

            self.lkd_loss_metric = []
            self.gkd_loss_metric = []

       
        self.sample_type = 'all'

    def collate_fn(self, batch):
        batch_size = len(batch)
        feature_num = len(batch[0][0])
        seqs = [batch[i][0] for i in range(batch_size)]
        frame_sets = [batch[i][1] for i in range(batch_size)]
        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]
        batch = [seqs, view, seq_type, label, None]

        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                frame_id_list = random.choices(frame_set, k=self.frame_num)
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        seqs = list(map(select_frame, range(len(seqs))))

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)
            batch_frames = [[
                len(frame_sets[i])
                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                if i < batch_size
            ] for _ in range(gpu_num)]
            if len(batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)
            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
            seqs = [[
                np.concatenate([
                    seqs[i][j]
                    for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                    if i < batch_size
                ], 0) for _ in range(gpu_num)]
                for j in range(feature_num)]
            seqs = [np.asarray([
                np.pad(seqs[j][_],
                       ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                       'constant',
                       constant_values=0)
                for _ in range(gpu_num)])
                for j in range(feature_num)]
            batch[4] = np.asarray(batch_frames)

        batch[0] = seqs
        return batch

    
    def load_teacher_model(self, restore_iter):
        self.teacher_encoder.load_state_dict(torch.load(osp.join(
            'teacher', self.teacher_model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.teacher_save_name, restore_iter))))
        self.teacher_optimizer.load_state_dict(torch.load(osp.join(
            'teacher', self.teacher_model_name,
            '{}-{:0>5}-optimizer.ptm'.format(self.teacher_save_name, restore_iter))))

    
    def fit(self):
        if self.distillation:
            self.teacher_encoder.eval()
            print("teacher_params:",sum(params.numel() for params in self.teacher_encoder.parameters()))
        if self.restore_iter != 0:
            self.load(self.restore_iter)
        self.student_encoder.train()
        print("student_params:",sum(params.numel() for params in self.student_encoder.parameters()))
        
        self.sample_type = 'random'
        for param_group in self.student_optimizer.param_groups:
            param_group['lr'] = self.lr

        triplet_sampler = TripletSampler(self.train_source, self.batch_size, self.type_equalization)
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        train_label_set = list(self.train_source.label_set)
        train_label_set.sort()

        _time1 = datetime.now()
        for seq, view, seq_type, label, batch_frame in train_loader:
            self.restore_iter += 1
            self.student_optimizer.zero_grad()

            for i in range(len(seq)):
                seq[i] = self.np2var(seq[i]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            # load feature of teacher model
            if self.distillation:
                with torch.no_grad():
                    sl_target, label_prob, fl_target = self.teacher_encoder(*seq, batch_frame)
            # load feature of student model
            sl_output, label_prob, fl_output = self.student_encoder(*seq, batch_frame)
            
            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.array(target_label)).long()
            triplet_sl_output = sl_output.permute(1, 0, 2).contiguous()
            triplet_label = target_label.unsqueeze(0).repeat(triplet_sl_output.size(0), 1)
            
            #basic loss
            (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num) = self.triplet_loss(triplet_sl_output, triplet_label)


            if self.distillation:
                lkd_loss_metric = self.lkdloss(fl_target, fl_output)
                gkd_loss_metric = self.gkdloss(sl_target, sl_output, target_label)
                #gkd_loss_metric = self.gkdloss_hard(sl_target, sl_output, target_label)
                loss = full_loss_metric.mean() + self.dw * (lkd_loss_metric.mean() + gkd_loss_metric.mean())
            else:
                loss = full_loss_metric.mean()

                
            self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
            self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
            self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
            self.dist_list.append(mean_dist.mean().data.cpu().numpy())

            
            if self.distillation:
                self.lkd_loss_metric.append(lkd_loss_metric.mean().data.cpu().numpy())
                self.gkd_loss_metric.append(gkd_loss_metric.mean().data.cpu().numpy())

            if loss > 1e-9:
                loss.backward()
                self.student_optimizer.step()

            if self.restore_iter % 1000 == 0:
                print(datetime.now() - _time1)
                _time1 = datetime.now()
                self.save()

            if self.restore_iter % 20000 == 0:
                self.lr = self.lr * self.lr_decay_rate
                if self.optimizer_type == 'sgd':
                    self.student_optimizer = optim.SGD([{'params': self.student_encoder.parameters()}], lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

          

            if self.restore_iter % 100 == 0:
                #self.save()
                print('iter {}:'.format(self.restore_iter), end='')
                print(', hard_loss_metric={0:.4f}'.format(np.mean(self.hard_loss_metric)), end='')
                print(', full_loss_metric={0:.4f}'.format(np.mean(self.full_loss_metric)), end='')
                print(', full_loss_num={0:.4f}'.format(np.mean(self.full_loss_num)), end='')
                self.mean_dist = np.mean(self.dist_list)
                print(', mean_dist={0:.4f}'.format(self.mean_dist), end='')
                
                if self.distillation:
                    print(',lkd_loss={0:4f}'.format(np.mean(self.lkd_loss_metric)), end='')
                    print(',gkd_loss={0:4f}'.format(np.mean(self.gkd_loss_metric)), end='')
                    
                print(', lr=%f' % self.student_optimizer.param_groups[0]['lr'], end='')
                print(', hard or full=%r' % self.hard_or_full_trip)
                sys.stdout.flush()

                self.hard_loss_metric = []
                self.full_loss_metric = []
                self.full_loss_num = []
                self.dist_list = []
                if self.distillation:
                    self.lkd_loss_metric = []
                    self.gkd_loss_metric = []


            if self.restore_iter == self.total_iter:
                break

    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def transform(self, flag, batch_size=1, deploy=False):
        self.student_encoder.eval()
        print("student_params:",sum(params.numel() for params in self.student_encoder.parameters()))
        if deploy:
            self.student_encoder.module.set_layer1.forward_block._switch_to_deploy()
            self.student_encoder.module.set_layer2.forward_block._switch_to_deploy()
            self.student_encoder.module.set_layer3.forward_block._switch_to_deploy()
            print("rep_student_params:",sum(params.numel() for params in self.student_encoder.parameters()))
        source = self.test_source if flag == 'test' else self.train_source
        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

 
        _time1 = datetime.now()
        for i, x in enumerate(tqdm(data_loader)):
            seq, view, seq_type, label, batch_frame = x
            for j in range(len(seq)):
                seq[j] = self.np2var(seq[j]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()


            feature, _, output = self.student_encoder(*seq, batch_frame)
            n, num_bin, _ = feature.size()
            feature_list.append(feature.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list += label
    
        
        return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list
        
      

    def save(self):
        os.makedirs(osp.join('checkpoint', self.model_name), exist_ok=True)
        torch.save(self.student_encoder.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-encoder.ptm'.format(
                                self.save_name, self.restore_iter)))
        torch.save(self.student_optimizer.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-optimizer.ptm'.format(
                                self.save_name, self.restore_iter)))



    def load(self, restore_iter):
        self.student_encoder.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter))))
        self.student_optimizer.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))))
