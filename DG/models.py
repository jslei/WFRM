import torch
import torch.nn as nn
import torch.nn.functional as F
         
class adapt_Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(adapt_Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if use_1x1conv:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),nn.BatchNorm2d(out_channels))
        else:
            self.downsample = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.downsample:
            X = self.downsample(X)
        return F.relu(Y + X)
    

    
class block(nn.Module):
    def __init__(self,in_channels, out_channels,first_block=False):
        super(block,self).__init__()
        if first_block:
            assert in_channels == out_channels 
            
        if not first_block:
            self.residual0=adapt_Residual(in_channels, out_channels, use_1x1conv=True, stride=2)
        else:
            self.residual0=adapt_Residual(in_channels, out_channels)
                
        self.residual1 = adapt_Residual(out_channels, out_channels)
                    
    def forward(self,X):
        Y = self.residual0(X)
        Y = self.residual1(Y)
        return Y
        

class resnet18(nn.Module):
    def __init__(self, num_class):
        super(resnet18,self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = block(in_channels=64, out_channels=64,first_block=True)
        self.layer2 = block(64, 128)
        self.layer3 = block(128, 256)
        self.layer4 = block(256, 512)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512,num_class)
            
    def forward(self,X):
        z = self.bn1(self.conv1(X))
        z = self.pool(self.relu(z))
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)    
        f = self.global_avg_pool(z)
        f = torch.flatten(f,1)
        y = self.linear(f)
        return y,f
        
    def predict(self,f):
        x = F.relu(f)
        return self.linear(x)
    
    def init_fc(self):
        self.linear.reset_parameters()
        
        
class AlexNet(nn.Module):
 
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
 
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
 
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),   
            nn.Linear(256 * 6 * 6, 4096),  
            nn.ReLU(inplace=True),
 
            nn.Dropout(0.5),
            nn.Linear(4096, 4096)
            )
         
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(4096, num_classes)
 
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)       
        x = torch.flatten(x, 1)  
        f = self.classifier(x)
        x = self.relu(f)
        x = self.fc(x)
        return x,f
    
    def predict(self,f):
        x = self.relu(f)
        return self.fc(x)
        
    def init_fc(self):
        self.fc.reset_parameters()