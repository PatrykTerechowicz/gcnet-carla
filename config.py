# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:08:35 2021

@author: Lab_admin
"""
import yaml

conf_file = open("conf.yml")
conf_yml = yaml.load(conf_file)

train_conf = conf_yml['train-algorithm']
ds_paths = conf_yml['dataset-paths']