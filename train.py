# -*- coding: utf-8 -*-
"""

Created on Tue Jan 19 09:59:12 2021

@author: Lab_admin
"""

#libraries
import torch
import torchvision
import torchvision.transforms as transforms
import time
import pyyaml
from pathlib import Path
#our imports
from read_carla_stereo import CarlaDataset
from gc_net import *


h=1*128
w=1*128
maxdisp=1*32 #gc_net.py also need to change  must be a multiple of 32...maybe can cancel the outpadding of deconv
batch=1
net = GcNet(h,w,maxdisp)
#net=net.cuda()
net=torch.nn.DataParallel(net).cuda()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===Train data===
abs_path = Path(r"E:\Patryk_Terechowicz\Carla_20_01")
ds_train = CarlaDataset(root_dir=abs_path)
print("Found %d valid data samples" % len(ds))

# ===Training settings===
