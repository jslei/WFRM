import torch
import torch.nn as nn
from utils import get_transform
import torchvision.transforms as transforms
from os import path as osp
import torchvision
from PIL import Image
            
class PACS():
    """PACS.
    Statistics:
        - 4 domains: Photo (1,670), Art (2,048), Cartoon
        (2,344), Sketch (3,929).
        - 7 categories: dog, elephant, giraffe, guitar, horse,
        house and person.
    """
    # the following images contain errors and should be ignored
    _error_paths = ["sketch/dog/n02103406_4068-1.png"]

    def __init__(self,dataset_dir):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.image_dir = osp.join(self.dataset_dir, "images")
        self.split_dir = osp.join(self.dataset_dir, "splits")
        self.domains = ["art_painting", "cartoon", "photo", "sketch"]
        
        self.trainset = []
        self.valset = []
        self.testset = []
        for i in self.domains:
            self.trainset.append(mydataset(self._read_data([i], "train")))
            self.valset.append(mydataset(self._read_data([i], "crossval")))
            self.testset.append(mydataset(self._read_data([i], "test")))
            

    def _read_data(self, input_domains, split):

        for domain, dname in enumerate(input_domains):
            if split == "all":
                file_train = osp.join(
                    self.split_dir, dname + "_train_kfold.txt"
                )
                impath_label_list = self._read_split_pacs(file_train)
                file_val = osp.join(
                    self.split_dir, dname + "_crossval_kfold.txt"
                )
                impath_label_list += self._read_split_pacs(file_val)
            else:
                file = osp.join(
                    self.split_dir, dname + "_" + split + "_kfold.txt"
                )
                impath_label_list = self._read_split_pacs(file)

        return impath_label_list

    def _read_split_pacs(self, split_file):
        items = []

        with open(split_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                if impath in self._error_paths:
                    continue
                impath = osp.join(self.image_dir, impath)
                label = int(label) - 1
                tp_img = Image.open(impath).convert('RGB')

                items.append((tp_img, label))

        return items

class mydataset(torch.utils.data.Dataset):
    def __init__(self,dataset):
        super(mydataset, self).__init__()
        self.imgs_loader = dataset
        self.transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        
    def __len__(self):
        return len(self.imgs_loader)

    def __getitem__(self,i):
        img,label = self.imgs_loader[i]
        return self.transform(img),label
       
        
def VLCS():
    def __init__(self, dataset_dir):
    
        self.dataset_dir = dataset_dir
        self.domains = ['pascal','labelme','caltech','sun']
        
        transform_train, transform_test = get_transform()
        
        self.trainset = []
        self.valset = []
        self.testset = []
        for i in self.domains:
            self.trainset.append(torchvision.datasets.ImageFolder(self.dataset_dir+'/'+i.upper()+'/train',transform_train))
            self.valset.append(torchvision.datasets.ImageFolder(self.dataset_dir+'/'+i.upper()+'/crossval',transform_test))
            self.testset.append(torchvision.datasets.ImageFolder(self.dataset_dir+'/'+i.upper()+'/test',transform_test))
        
def OfficeHome():
    def __init__(self, dataset_dir):
    
        self.dataset_dir = dataset_dir
        self.domains = ['art','clipart','product','real_world']
        
        transform_train, transform_test = get_transform()
        
        self.trainset = []
        self.valset = []
        self.testset = []
        for i in self.domains:
            self.trainset.append(torchvision.datasets.ImageFolder(self.dataset_dir+'/'+i+'/train/',transform_train))
            self.valset.append(torchvision.datasets.ImageFolder(self.dataset_dir+'/'+i+'/val',transform_test))
            self.testset.append(torch.utils.data.ConcatDataset([
                torchvision.datasets.ImageFolder(self.dataset_dir+'/'+i+'/train/',transform_test),
                torchvision.datasets.ImageFolder(self.dataset_dir+'/'+i+'/val',transform_test)
            ]))