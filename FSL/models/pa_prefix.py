import torch
import numpy as np
import math
import torch.nn.init as init
import torch.nn as nn
import torch.optim as optim

from models.model_utils import sigmoid, cosine_sim
from models.losses import prototype_loss, cross_entropy_loss
from utils import device
import torch.nn.functional as F
from torchvision import transforms
import torch.autograd as autograd

import cv2
class PR(nn.Module):
    def __init__(self, init_prefix, in_dim=384, out_dim=64, warmup_teacher_temp=0.04, teacher_temp=0.04,
                 warmup_teacher_temp_epochs=5, nepochs=40, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

        self.projector = nn.Linear(in_dim, out_dim)
        self.init_prefix = init_prefix

    def forward(self, prefix, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        prefix = F.normalize(prefix, dim=-1, p=2)
        init_prefix = F.normalize(self.init_prefix, dim=-1, p=2)

        prefix = self.projector(prefix)
        init_prefix = self.projector(init_prefix)

        temp = self.teacher_temp_schedule[epoch]
        prefix = prefix / self.student_temp
        init_out = F.softmax((init_prefix - self.center) / temp, dim=-1).detach()
        #init_out = F.softmax(init_prefix, dim=-1).detach()

        loss = torch.sum(-init_out * F.log_softmax(prefix, dim=-1), dim=-1).mean()

        self.update_center(init_out)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

def apply_selection(features, vartheta):
    """
    Performs pre-classifier alignment of features (feature adaptation) via a linear transformation.
    """

    features = features
    features = F.linear(features, vartheta[0])

    return features

def pa(model, support_img, query_img, context_labels, target_labels, wd = 0,return_fea=False, wfrm=False, lower_fea = 0, init_prefix=None, max_iter=40, ad_opt='linear', lr=0.05, distance='cos', input_dim=512, T=1.0, num_gpu=2, episode_idx=0):
    """
    PA method: learning a linear transformation per task to adapt the features to a discriminative space 
    on the support set during meta-testing
    """
    rho = 10
    alpha = 0.5
    output_dim = input_dim
    linear_dim = input_dim*lower_fea
    n_layer = 12
    n_head = 6
    lam = 0.1
    #l1_w = 2
    n_hidden = input_dim // 2
    n_way = torch.unique(context_labels).shape[0]
    num_sample_per_cls = torch.zeros(n_way)

    for i in range(n_way):
        num_sample_per_cls[i] = (context_labels==i).float().sum()

    vartheta = []
    vartheta.append(torch.eye(output_dim, input_dim).to(device).requires_grad_(True))
    # if lower_fea == 0:
    #     vartheta.append(torch.eye(output_dim, input_dim).to(device).requires_grad_(True))
    # else:
    #     vartheta.append(torch.eye(linear_dim, linear_dim).to(device).requires_grad_(True))
    if init_prefix is not None:
        prefix_weight = init_prefix.clone().to("cuda:0").requires_grad_(True)
    else:
        prefix_weight = torch.randn(n_way, output_dim).to("cuda:0").requires_grad_(True)
    control_trans = nn.Sequential(
                        nn.Linear(input_dim, n_hidden), #1024 * 512
                        nn.Tanh(),
                        nn.Linear(n_hidden, n_layer * 2 * input_dim)).requires_grad_(True).to("cuda:0")

    optim_list = [{'params':vartheta[0], 'lr':lr, 'weight_decay':wd}, 
            {'params':model.parameters(), 'lr':lr/2}, 
            {'params':control_trans.parameters(), 'lr':lr/2}, 
            {'params':prefix_weight, 'lr':lr/2}]

    if init_prefix is not None:
        loss_fn = PR(init_prefix, nepochs=max_iter).to("cuda:0")
        optim_list.append({'params':loss_fn.parameters(), 'lr':lr/2})
    optimizer = optim.AdamW(optim_list, eps=1e-4)

    #model.train()
    for i in range(max_iter):
        optimizer.zero_grad()

        prefix = control_trans(prefix_weight).view(n_way, n_layer, 2, n_head, input_dim//n_head)
        prefix = prefix.permute(1, 2, 3, 0, 4)
        prefix = prefix.unsqueeze(0).expand(2, -1, -1, -1, -1, -1)
        context_features = model(support_img, prefix=prefix, return_feat=True)
        selected_features = apply_selection(context_features, vartheta)
        loss, stat, _ = prototype_loss(selected_features, context_labels,
                                       selected_features, context_labels, distance=distance, T=T)
        
        if wfrm:
            grad = torch.autograd.grad(loss,context_features, retain_graph=True)[0]
            perturbation = norm(rho, grad)
            context_features_pert = context_features+perturbation
            #context_features = F.normalize(context_features, p=2, dim=1)
            #context_features_pert = F.normalize(context_features_pert, p=2, dim=1)
            selected_features_pert = apply_selection(context_features_pert, vartheta)
            loss_, stat, _ = prototype_loss(selected_features, context_labels,
                                       selected_features_pert, context_labels, distance=distance, T=T)
            loss = (1-alpha)*loss+alpha*loss_
        
        #loss = loss + l1_w*model.module.get_ensemble()

        if init_prefix is not None:
            dt_loss = loss_fn(prefix_weight, i)
            loss = loss + lam * dt_loss
        
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        prefix = control_trans(prefix_weight).view(n_way, n_layer, 2, n_head, input_dim//n_head)
        prefix = prefix.permute(1, 2, 3, 0, 4)
        prefix = prefix.unsqueeze(0).expand(2, -1, -1, -1, -1, -1)
        context_features = model(support_img, prefix=prefix, return_feat=True)
        target_features = model(query_img, prefix=prefix, return_feat=True)
        selected_context = apply_selection(context_features, vartheta)
        selected_target = apply_selection(target_features, vartheta)
    
    if return_fea:
        return context_features, target_features, selected_context, selected_target
    return selected_context, selected_target

def norm(rho,grad):
    g_norm = torch.norm(grad,dim=1).view(-1,1)
    return rho*grad/(g_norm+1e-12)

def pa_wfrm(model, support_img, query_img, context_labels, target_labels, num_class, init_prefix=None, max_iter=40, ad_opt='linear', lr=0.05, distance='cos', input_dim=512, T=1.0, num_gpu=2, episode_idx=0):
    """
    PA method: learning a linear transformation per task to adapt the features to a discriminative space 
    on the support set during meta-testing
    """
    rho = 1.
    alpha = 0.5
    output_dim = input_dim
    n_layer = 12
    n_head = 6
    lam = 0.1
    n_hidden = input_dim // 2
    n_way = torch.unique(context_labels).shape[0]
    num_sample_per_cls = torch.zeros(n_way)

    for i in range(n_way):
        num_sample_per_cls[i] = (context_labels==i).float().sum()

    #vartheta = []
    #vartheta.append(torch.eye(output_dim, input_dim).to(device).requires_grad_(True))
    fc = nn.Linear(input_dim, num_class).to(device)

    if init_prefix is not None:
        prefix_weight = init_prefix.clone().to("cuda:0").requires_grad_(True)
    else:
        prefix_weight = torch.randn(n_way, output_dim).to("cuda:0").requires_grad_(True)
    control_trans = nn.Sequential(
                        nn.Linear(input_dim, n_hidden), #1024 * 512
                        nn.Tanh(),
                        nn.Linear(n_hidden, n_layer * 2 * input_dim)).requires_grad_(True).to("cuda:0")

    optim_list = [{'params':fc.parameters(), 'lr':lr}, 
            {'params':model.parameters(), 'lr':lr/2}, 
            {'params':control_trans.parameters(), 'lr':lr/2}, 
            {'params':prefix_weight, 'lr':lr/2}]

    if init_prefix is not None:
        loss_fn = PR(init_prefix, nepochs=max_iter).to("cuda:0")
        optim_list.append({'params':loss_fn.parameters(), 'lr':lr/2})
    optimizer = optim.AdamW(optim_list, eps=1e-4)

    #model.train()
    for i in range(max_iter):
        optimizer.zero_grad()

        prefix = control_trans(prefix_weight).view(n_way, n_layer, 2, n_head, input_dim//n_head)
        prefix = prefix.permute(1, 2, 3, 0, 4)
        prefix = prefix.unsqueeze(0).expand(2, -1, -1, -1, -1, -1)
        context_features = model(support_img, prefix=prefix, return_feat=True)
        context_logits = fc(context_features)
        loss,stat,_ = cross_entropy_loss(context_logits, context_labels)

        #wfrm
        grad = torch.autograd.grad(loss,context_features, retain_graph=True)[0]
        perturbation = norm(rho, grad)
        context_features_pert = context_features+perturbation
        context_features = F.normalize(context_features, p=2, dim=1)
        context_features_pert = F.normalize(context_features_pert, p=2, dim=1)
        context_logits = fc(context_features)
        context_logits_pert = fc(context_features_pert)
        loss1,stat,_ = cross_entropy_loss(context_logits, context_labels)
        loss2,stat,_ = cross_entropy_loss(context_logits_pert, context_labels)
        loss = (1-alpha)*loss1+alpha*loss2


        #selected_features = apply_selection(context_features, vartheta)
        #loss, stat, _ = prototype_loss(selected_features, context_labels,
        #                               selected_features, context_labels, distance=distance, T=T)


        if init_prefix is not None:
            dt_loss = loss_fn(prefix_weight, i)
            loss = loss + lam * dt_loss
        
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        prefix = control_trans(prefix_weight).view(n_way, n_layer, 2, n_head, input_dim//n_head)
        prefix = prefix.permute(1, 2, 3, 0, 4)
        prefix = prefix.unsqueeze(0).expand(2, -1, -1, -1, -1, -1)
        #context_features = model(support_img, prefix=prefix, return_feat=True)
        target_features = model(query_img, prefix=prefix, return_feat=True)
        #selected_context = apply_selection(context_features, vartheta)
        #selected_target = apply_selection(target_features, vartheta)
        selected_target = fc(target_features)

    return selected_target


def pa_vit(model, support_img, query_img, context_labels, target_labels, num_class, wfrm=False, init_prefix=None, max_iter=40, ad_opt='linear', lr=0.05, distance='cos', input_dim=512, T=1.0, num_gpu=2, episode_idx=0):
    """
    PA method: learning a linear transformation per task to adapt the features to a discriminative space 
    on the support set during meta-testing
    """
    rho = 0.01
    alpha = 0.5
    output_dim = input_dim
    n_layer = 12
    n_head = 6
    lam = 0.1
    n_hidden = input_dim // 2
    n_way = torch.unique(context_labels).shape[0]
    num_sample_per_cls = torch.zeros(n_way)

    for i in range(n_way):
        num_sample_per_cls[i] = (context_labels==i).float().sum()

    s = num_sample_per_cls.sum()
    if s%3==0:
        support_img = support_img[:-1]
        context_labels = context_labels[:-1]

    #vartheta = []
    #vartheta.append(torch.eye(output_dim, input_dim).to(device).requires_grad_(True))
    fc = nn.Linear(input_dim, num_class).to(device)

    optim_list = [{'params':fc.parameters(), 'lr':lr}]

    optimizer = optim.AdamW(optim_list, eps=1e-4)
    #optimizer = optim.SGD(fc.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

    #model.train()
    for i in range(max_iter):
        optimizer.zero_grad()

        context_features = model(support_img, prefix=None, return_feat=True)
        context_features.requires_grad = True
        context_logits = fc(context_features)
        loss,stat,_ = cross_entropy_loss(context_logits, context_labels)

        #wfrm
        if wfrm:
            grad = torch.autograd.grad(loss,context_features, retain_graph=True)[0]
            perturbation = norm(rho, grad)
            context_features_pert = context_features+perturbation
            #context_features = F.normalize(context_features, p=2, dim=1)
            #context_features_pert = F.normalize(context_features_pert, p=2, dim=1)
            context_logits = fc(context_features)
            context_logits_pert = fc(context_features_pert)
            loss1,stat,_ = cross_entropy_loss(context_logits, context_labels)
            loss2,stat,_ = cross_entropy_loss(context_logits_pert, context_labels)
            loss = (1-alpha)*loss1+alpha*loss2


        #selected_features = apply_selection(context_features, vartheta)
        #loss, stat, _ = prototype_loss(selected_features, context_labels,
        #                               selected_features, context_labels, distance=distance, T=T)
        
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        #context_features = model(support_img, prefix=prefix, return_feat=True)
        target_features = model(query_img, prefix=None, return_feat=True)
        #selected_context = apply_selection(context_features, vartheta)
        #selected_target = apply_selection(target_features, vartheta)
        selected_target = fc(target_features)

    return selected_target


def pa_proto(model, support_img, query_img, context_labels, target_labels, wfrm=False, init_prefix=None, max_iter=40, ad_opt='linear', lr=0.05, distance='cos', input_dim=512, T=1.0, num_gpu=2, episode_idx=0):
    """
    PA method: learning a linear transformation per task to adapt the features to a discriminative space 
    on the support set during meta-testing
    """
    rho = 0.05
    alpha = 0.5
    output_dim = input_dim
    n_layer = 12
    n_head = 6
    lam = 0.1
    n_hidden = input_dim // 2
    n_way = torch.unique(context_labels).shape[0]
    num_sample_per_cls = torch.zeros(n_way)

    for i in range(n_way):
        num_sample_per_cls[i] = (context_labels==i).float().sum()

    vartheta = []
    vartheta.append(torch.eye(output_dim, input_dim).to(device).requires_grad_(True))

    optim_list = [{'params':vartheta[0], 'lr':lr}]

    optimizer = optim.AdamW(optim_list, eps=1e-4)


    #model.train()
    for i in range(max_iter):
        optimizer.zero_grad()

        context_features = model(support_img, prefix=None, return_feat=True)
        context_features.requires_grad = True 
        selected_features = apply_selection(context_features, vartheta)
        loss, stat, _ = prototype_loss(selected_features, context_labels,
                                       selected_features, context_labels, distance=distance, T=T)
        
        if wfrm:
            grad = torch.autograd.grad(loss,context_features, retain_graph=True)[0]
            perturbation = norm(rho, grad)
            context_features_pert = context_features+perturbation
            #context_features = F.normalize(context_features, p=2, dim=1)
            #context_features_pert = F.normalize(context_features_pert, p=2, dim=1)
            selected_features_pert = apply_selection(context_features_pert, vartheta)
            loss_, stat, _ = prototype_loss(selected_features, context_labels,
                                       selected_features_pert, context_labels, distance=distance, T=T)
            loss = (1-alpha)*loss+alpha*loss_

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        context_features = model(support_img, prefix=None, return_feat=True)
        target_features = model(query_img, prefix=None, return_feat=True)
        selected_context = apply_selection(context_features, vartheta)
        selected_target = apply_selection(target_features, vartheta)

    return selected_context, selected_target


def pa_fgsm(model, support_img, query_img, context_labels, target_labels,return_fea=False, fgsm=False, lower_fea = 0, init_prefix=None, max_iter=40, ad_opt='linear', lr=0.05, distance='cos', input_dim=512, T=1.0, num_gpu=2, episode_idx=0):
    """
    PA method: learning a linear transformation per task to adapt the features to a discriminative space 
    on the support set during meta-testing
    """
    rho = 0.01
    alpha = 0.5
    output_dim = input_dim
    n_layer = 12
    n_head = 6
    lam = 0.1
    n_hidden = input_dim // 2
    n_way = torch.unique(context_labels).shape[0]
    num_sample_per_cls = torch.zeros(n_way)

    for i in range(n_way):
        num_sample_per_cls[i] = (context_labels==i).float().sum()

    vartheta = []
    if lower_fea == 0:
        vartheta.append(torch.eye(output_dim, input_dim).to(device).requires_grad_(True))
    else:
        vartheta.append(torch.eye(linear_dim, linear_dim).to(device).requires_grad_(True))
    if init_prefix is not None:
        prefix_weight = init_prefix.clone().to("cuda:0").requires_grad_(True)
    else:
        prefix_weight = torch.randn(n_way, output_dim).to("cuda:0").requires_grad_(True)
    control_trans = nn.Sequential(
                        nn.Linear(input_dim, n_hidden), #1024 * 512
                        nn.Tanh(),
                        nn.Linear(n_hidden, n_layer * 2 * input_dim)).requires_grad_(True).to("cuda:0")

    optim_list = [{'params':vartheta[0], 'lr':lr}, 
            {'params':model.parameters(), 'lr':lr/2}, 
            {'params':control_trans.parameters(), 'lr':lr/2}, 
            {'params':prefix_weight, 'lr':lr/2}]

    if init_prefix is not None:
        loss_fn = PR(init_prefix, nepochs=max_iter).to("cuda:0")
        optim_list.append({'params':loss_fn.parameters(), 'lr':lr/2})
    optimizer = optim.AdamW(optim_list, eps=1e-4)

    #model.train()
    for i in range(max_iter):
        optimizer.zero_grad()

        prefix = control_trans(prefix_weight).view(n_way, n_layer, 2, n_head, input_dim//n_head)
        prefix = prefix.permute(1, 2, 3, 0, 4)
        prefix = prefix.unsqueeze(0).expand(2, -1, -1, -1, -1, -1)
        context_features = model(support_img, prefix=prefix, return_feat=True)
        selected_features = apply_selection(context_features, vartheta)
        loss, stat, _ = prototype_loss(selected_features, context_labels,
                                       selected_features, context_labels, distance=distance, T=T)
        
        if fgsm:
            grad = torch.autograd.grad(loss,context_features, retain_graph=True)[0]
            perturbation = rho*torch.sign(grad)
            context_features_pert = context_features+perturbation
            #context_features = F.normalize(context_features, p=2, dim=1)
            #context_features_pert = F.normalize(context_features_pert, p=2, dim=1)
            selected_features_pert = apply_selection(context_features_pert, vartheta)
            loss_, stat, _ = prototype_loss(selected_features, context_labels,
                                       selected_features_pert, context_labels, distance=distance, T=T)
            loss = (1-alpha)*loss+alpha*loss_
        

        if init_prefix is not None:
            dt_loss = loss_fn(prefix_weight, i)
            loss = loss + lam * dt_loss
        
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        prefix = control_trans(prefix_weight).view(n_way, n_layer, 2, n_head, input_dim//n_head)
        prefix = prefix.permute(1, 2, 3, 0, 4)
        prefix = prefix.unsqueeze(0).expand(2, -1, -1, -1, -1, -1)
        context_features = model(support_img, prefix=prefix, return_feat=True)
        target_features = model(query_img, prefix=prefix, return_feat=True)
        selected_context = apply_selection(context_features, vartheta)
        selected_target = apply_selection(target_features, vartheta)

    if return_fea:
        return context_features, target_features, selected_context, selected_target
    return selected_context, selected_target