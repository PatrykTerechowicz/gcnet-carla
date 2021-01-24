# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 13:50:46 2021

@author: Lab_admin
"""
import torch
import torchvision
import torchvision.transforms as transforms
import time
from gc_net import *

# muszą być potęgi 2
h=128
w=256
maxdisp=1*32 #gc_net.py also need to change  must be a multiple of 32...maybe can cancel the outpadding of deconv
batch=1
net = GcNet(h,w,maxdisp)
#net=net.cuda()
net=torch.nn.DataParallel(net).cuda()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#testy
if __name__ == "__main__":
    total_time = 0
    for i in range(100):
        imL = torch.zeros([batch, 3, h, w], dtype=torch.float32) # normal_(mean=0, std=1, *, generator=None)
        imR = torch.zeros([batch, 3, h, w], dtype=torch.float32)
        imL.normal_(mean=0.5, std=0.5)
        imR.normal_(mean=0.5, std=0.5)
        tim1 = time.time()
        disp = net(imR, imR)
        disp = disp + 1
        tim2 = time.time()
        elapsed_time = tim2 - tim1
        total_time += elapsed_time
    print("Total elapsed time is %f" % total_time)
        