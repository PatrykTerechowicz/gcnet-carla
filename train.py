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
import numpy as np
from torch.optim import SGD, lr_scheduler
from pathlib import Path
from torch.autograd import Variable
#our imports,
import config
from read_carla_stereo import CarlaDataset, preprocess_disparity
from gc_net import *



# ===Load Train data===
abs_path = Path(config.ds_paths['train'])
ds_train = CarlaDataset(root_dir=abs_path, max_disp=32, h=128, w=256)
data_n0 = len(ds_train)
print("==>Found %d valid data samples" % data_n0)
# ===Create training model===
h=ds_train.dsize[1]
w=ds_train.dsize[0]
maxdisp=ds_train.max_disp
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
loss = torch.nn.L1Loss()
# ===Define preprocessing===
tsfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
to_tensor = transforms.ToTensor()
# ===Training loop===
loss_mul_list = []
for d in range(maxdisp):
    loss_mul_temp = Variable(torch.FloatTensor(np.ones([1, 1, h, w]) * d)).cuda()
    loss_mul_list.append(loss_mul_temp)
loss_mul = torch.cat(loss_mul_list, 1)
print("==>Starting training")
for i in range(1, max_epochs+1):
    print("====>Epoch: [%d] \n" % i)
    net.train()
    for j in progressbar.progressbar(range(data_n0)):
        net.zero_grad()
        optimizer.zero_grad()
        imL, imR, disp, frame_id = ds_train[j]
        #preprocess
        imL = tsfm(imL).cuda()
        imR = tsfm(imR).cuda()
        #disp = preprocess_disparity(disp, maxdisp)
        
        disp = to_tensor(disp)
        disp = disp.unsqueeze(0)
        disp = disp.cuda()
        x = net(torch.unsqueeze(imL, dim=0), torch.unsqueeze(imR, dim=0))
        result=torch.sum(x.mul(loss_mul),1)
        out = loss(result, disp)
        out.backward()
        optimizer.step()
    scheduler.step()
    

cv2.destroyAllWindows()