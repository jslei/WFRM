#!/usr/bin/env python
# coding: gbk

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

import torch.autograd as autograd
from copy import deepcopy
import os
import argparse
from utils import set_seed, get_pretrain 
from dataset import PACS, VLCS, OfficeHome
from models import resnet18, AlexNet

parser = argparse.ArgumentParser(description="Script to launch training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--rho', type=float, default=6.5, help='perturbation intensity')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--seed', type=int, default=12345)

parser.add_argument('--dataset_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--dataset', type=str, default='vlcs')
parser.add_argument('--backbone', type=str, choice=['alexnet','resnet18'], default='alexnet')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == 'vlcs':
    num_class = 5
    dataset = VLCS(args.dataset_dir)
elif args.dataset == 'pacs':
    num_class = 7
    dataset = PACS(args.dataset_dir)
elif args.dataset == 'officehome':
    num_class = 65
    dataset = OfficeHome(args.dataset_dir)
 
def test(iter_):
    net.eval()
    n = 0
    acc = 0
    with torch.no_grad():
        for X,y in iter_:
            X = X.to(device)    
            y = y.to(device)
            y_hat,_ = net(X)
            acc += (y_hat.argmax(dim = 1) == y).float().sum().item()        
            n += len(X)
    net.train()
    return acc/n


def pert(grad,rho=args.rho):
    s = rho*torch.sign(grad)*torch.abs(grad)
    grad_norm = torch.norm(grad,dim=1).view(-1,1)
    return s/grad_norm


pretrain_param = get_pretrain(args.backbone)
if args.backbone=='alexnet':
    net = AlexNet(num_class)
    net.load_state_dict(pretrain_param,strict = False)
elif args.backbone=='resnet18':
    net = resnet18(num_class)
    net.load_state_dict(pretrain_param,strict = False)

for i in range(len(dataset.domains)):

    net.init_fc()
    dataset_t = deepcopy(dataset.trainset)
    dataset_v = deepcopy(dataset.valset)
    test_iter = torch.utils.data.DataLoader(dataset.testset[i], batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    del dataset_t[i]
    train_set = torch.utils.data.ConcatDataset(dataset_t)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True,drop_last=True)
    del dataset_v[i]
    val_set = torch.utils.data.ConcatDataset(dataset_v)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    # In[ ]:
    net.to(device)
    net.train()
    print(" ")
    print(" ")
    print('{} training!!'.format(dataset.domains[i]))
    optimizer = optim.SGD(net.parameters(),lr = args.lr,momentum=0.9,weight_decay = args.weight_decay)
    
    loss = nn.CrossEntropyLoss()
    best_acc_v = 0
    best_acc_t = 0
    param = 0
    
    for k in range(1,args.epochs+1):
            
        for X,y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat,f = net(X)
            l = loss(y_hat,y)
            grad = autograd.grad(l, f, retain_graph=True)[0]
            epsilon = pert(grad)
            f_adv = f+epsilon
            y_adv = net.predict(f_adv)
            l_adv = loss(y_adv,y)
            optimizer.zero_grad()
            ((1-args.alpha)*l+args.alpha*l_adv).backward()
            optimizer.step()

        
        if k%5==0:
            accu_t = test(test_iter)
            accu_v = test(val_iter)
            if accu_v>best_acc_v:
                best_acc_v = accu_v
                best_acc_t = accu_t
                param = net.cpu().state_dict()
                net.to(device)
            print('epoch {},loss:{},val_acc:{},best_val_acc:{},accuracy:{},best_accuracy:{}'.format(k,(1-args.alpha)*l+args.alpha*l_adv,accu_v,best_acc_v,accu_t,best_acc_t))
    
    torch.save(param, args.save_dir+dataset.domains[i]+'.pt')
    

