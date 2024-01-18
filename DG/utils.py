from torchvision import models
import collections
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, Compose, Normalize, ToTensor, RandomResizedCrop, \
    ColorJitter, Resize
import numpy as np
import random
import torch    

def get_pretrain(model):
    if model=='alexnet':
        model = models.alexnet(pretrained = True)

        d1 = model.state_dict()
        del d1['classifier.6.weight']
        del d1['classifier.6.bias']
        
        return d1
        
    if model=='resnet18':
        model = models.resnet18(pretrained = True)
        s_dic = model.state_dict()
        d_ = []
        for k,v in s_dic.items():
            if k.find('layer')>=0:
                k1 = list(k)
                k1.insert(7,'residual')
                k1 = ''.join(k1)
                d_.append((k1,v))
            else:
                d_.append((k,v))
        
        d1 = collections.OrderedDict(d_)
        return d1
        
        
def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_train = Compose([
                    RandomResizedCrop(224, scale=(0.8, 1.0)),
                    RandomHorizontalFlip(),
                    ColorJitter(.4, .4, .4, .4),
                    ToTensor(),
                    Normalize(mean=mean, std=std)
                ])
    transform_test = Compose([
                    Resize((224, 224)),
                    ToTensor(),
                    Normalize(mean=mean, std=std)
                ])
    return transform_train, transform_test
    
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    