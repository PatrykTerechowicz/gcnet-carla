# -*- coding: utf-8 -*-
"""

Created on Tue Jan 19 09:59:12 2021

@author: Lab_admin
"""

#libraries
import torch
import torchvision
import torchvision.transforms as transforms
import progressbar
import cv2
from torch.optim import SGD, lr_scheduler
from pathlib import Path
#our imports,
import config
from read_carla_stereo import CarlaDataset
from gc_net import *



# ===Load Train data===
abs_path = Path(config.ds_paths['train'])
ds_train = CarlaDataset(root_dir=abs_path)
data_n0 = len(ds_train)
print("==>Found %d valid data samples" % data_n0)
# ===Create training model===
h=ds_train.dsize[1]
w=ds_train.dsize[0]
maxdisp=1*32 
net = GcNet(h,w,maxdisp)
net=torch.nn.DataParallel(net).cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ===Training settings===
max_epochs = config.train_conf['epochs']
lr = config.train_conf['lr']
momentum = config.train_conf['momentum']
schedule_lr = config.train_conf['schedule_lr']
lr_decay = config.train_conf['lr_decay']

optimizer = SGD(net.parameters(), lr=lr, momentum=momentum)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
# ===Define preprocessing===
preprocessing = transforms.Compose([transforms.ToTensor()])

# ===Training loop===
print("==>Starting training")
for i in range(1, max_epochs+1):
    print("====>Epoch: [%d] \n" % i)
    # Train data
    for j in progressbar.progressbar(range(data_n0)):
        imL, imR, imD, frame_id = ds_train[j]
        #preprocess
        imL = preprocessing(imL)
        imR = preprocessing(imR)
        result = net(torch.unsqueeze(imL, dim=0), torch.unsqueeze(imR, dim=0))
        

cv2.destroyAllWindows()