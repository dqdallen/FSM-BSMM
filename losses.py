import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import Delaunay
from sklearn.manifold import TSNE
import time
#import dgl
import scipy.sparse as spp
import networkx as nx
from sklearn.cluster import KMeans

# from pytorch_metric_learning import miners, losses


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

class MyLoss(nn.Module):
    def __init__(self, classes):
        super(MyLoss, self).__init__()
        self.classes = classes
        
        self.dim=64
        self.centers = torch.zeros((classes, self.dim)).cuda()
        self.maxR = torch.zeros((classes,)).cuda()
        self.maxf = torch.zeros((classes, classes)).cuda()
        self.cnt = 0
    
    def generatelabels(self, batchsize,real_labels =None):
        x = torch.Tensor(torch.zeros((batchsize,self.classes),requires_grad=True)).cuda()
        if real_labels is None: #生成随机标签
            y = [np.random.randint(0, 9) for i in range(batchsize)]
            x[np.arange(batchsize), y] = 1
        else:
            x[np.arange(batchsize),real_labels] = 1
    
        return x
        
    def forward(self, y_pred, target, weight, out, epoch, epochs,nums_cls_list):
        y_true = self.generatelabels(target.shape[0], target.long())
        num_batch = torch.sum(y_true, 0)
        x_means = []
        maxs = torch.zeros((self.classes,)).cuda()
        rs = torch.zeros((self.classes,)).cuda()

        inter_dis = 0
        inter_cnt = 0
        intra_cnt = 0
        intra_dis = 0
        if torch.any(torch.isnan(y_pred)):
            print('error')
            exit()
        for i in range(self.classes):
            if num_batch[i] == 0:
                continue
            ind = torch.where(target == i)[0]
            x = out[ind]
            #xx = torch.softmax(x.detach(), 1)
            #cind = torch.argmax(xx[:, i])
            xmean = torch.mean(x.detach(), 0)
            #xmean = x[cind].detach()
            self.centers[i] = self.centers[i] * 0.1 + 0.9 * xmean
            #eind = torch.argmin(xx[:, i])
            # maxs[i] = (xx[cind,i]-xx[eind,i]) / torch.sqrt(torch.sum((x[eind] - self.centers[i]) ** 2))
            
            #maxs[i] = torch.sqrt(torch.sum((x[eind].detach() - self.centers[i]) ** 2))
            xsum = torch.sqrt(torch.sum((x.detach() - self.centers[i]) ** 2, 1))
            rs[i] = max(xsum)
            # intra_dis += torch.sum((x - weight[i]) ** 2, 1).mean()
            # intra_cnt += 1
            xmax = max(xsum)
            maxs[i] = xmax
        
        i = 0
        while i < self.classes:
            if maxs[i] > 0:
                break
            i += 1
        maxx = maxs[i]
        mcs = torch.ones((self.classes,)).cuda()
        
        ncl = torch.Tensor(nums_cls_list).cuda()
        ncl = 10*ncl / ncl[0]
        #for i in range(self.classes//2, self.classes):
        #    maxs[i] = maxs[i] / max(maxs[0:self.classes//2])
        #    mcs[i] = 0.01
        #for i in range(0, self.classes//2):
        #    maxs[i] = 1
        maxs = maxs / maxx
        maxs[maxs > 1] = 1
        if self.classes > 10:
            if torch.all(self.maxR == 0):
                self.maxR = maxs
            else:
                for i in range(self.classes):
                    if maxs[i] != 0:
                        self.maxR[i] = self.maxR[i] * 0.1 + maxs[i] * 0.9
            maxs = self.maxR
        
        points = out
        p1 = points.repeat(points.shape[0], 1)
        p2 = points.repeat(1, points.shape[0]).reshape(p1.shape[0],-1)
        dis_mat = torch.sum((p1 - p2) ** 2, 1).reshape(points.shape[0], -1)
        dis_m = torch.zeros((dis_mat.shape[0], dis_mat.shape[0], dis_mat.shape[0])).cuda()
        target_mat = torch.zeros((dis_mat.shape[0], dis_mat.shape[0])).cuda()

        a = dis_mat.repeat(dis_mat.shape[0], 1)
        b = dis_mat.repeat(1, dis_mat.shape[0]).reshape(a.shape[0],-1)
        c = a + b
        c = c.reshape((dis_m.shape))
        dis_m = c
     
        target_d = target.unsqueeze(0)
        t1 = target_d.repeat(target_d.shape[1], 1)
        target_mat = t1 - t1.t()
        gamma = 2
        if self.classes == 100:
            gamma = 8
        elif self.classes == 200:
            gamma = 15
        elif self.classes == 365:
            gamma = 5
        alpha = (epoch/epochs) ** gamma
        #alpha = 1-0.5*np.cos(epoch/epochs*np.pi) - 0.5
        if self.classes == 10:
            sigma = -30
            lambd = 0.5
        elif self.classes == 100:
            sigma = -50
            lambd = 0.5
        elif self.classes == 200:
            sigma = -150
            lambd = 0.3
        elif self.classes == 365:
            sigma = -10
            lambd = 0.5
        dis_m = dis_m.permute(0,2,1)
        judge_mat = (dis_m >= dis_mat) + 0
        judge_mask = torch.sum(judge_mat, 1)
        judge_mask = (judge_mask == dis_m.shape[0]) + 0
        judge_mask = judge_mask.float()
        d = torch.abs((torch.eye(dis_m.shape[0])-1).cuda())
        
        judge_mask = torch.mul(d, judge_mask)
        target_mask = (target_mat != 0) + 0
        target_mask = target_mask.float()
        mask = torch.mul(target_mask, judge_mask)
        inter_dis = torch.exp(dis_mat/sigma)
        cnt = mask.sum()
        
        intra_dis = torch.sum((points - weight[target]) ** 2, 1)
        #intra_dis = torch.sum((points - self.centers[target]) ** 2, 1)
        intra_dis = torch.exp(intra_dis/sigma)
        intra_dis = intra_dis.unsqueeze(0)
        intra_dis = intra_dis.repeat(inter_dis.shape[0], 1)
        lossdis = inter_dis - intra_dis + lambd
        inter_dis_new = torch.mul(mask, lossdis)
        lossdis[lossdis < 0] = 0
        inter_dis_new[inter_dis_new < 0] = 0
        mcs = ncl
        scale = mcs[target]
        scale = scale.unsqueeze(0)
        scale = scale.repeat((inter_dis.shape[0], 1))
        inter_dis_new /= scale
        
        y_pred *= maxs**alpha
        x = torch.softmax(y_pred, 1)
        y = torch.log(x ** y_true)
        if torch.any(torch.isnan(y)):
            print(inter_dis)
            print(torch.any(torch.isnan(y_pred)))
            exit()
        beta = max((1-alpha), 0.7)
        #targ = torch.Tensor(target).cuda()
        loss = F.cross_entropy(y_pred, target)
        loss = loss + beta*inter_dis_new.sum() / cnt
        # loss = (-1 * y.sum() / y_true.shape[0]) #+ beta*inter_dis_new.sum() / cnt
        return loss
    
    def updateC(self):
        self.centers = torch.zeros((self.classes, self.dim)).cuda()
        self.maxR = torch.zeros((self.classes,)).cuda()
        self.maxf = torch.zeros((self.classes, self.classes)).cuda()

