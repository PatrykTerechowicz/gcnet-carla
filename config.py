# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:08:35 2021

@author: Lab_admin
"""
import pyyaml
_instance = None
class Config:
    def __init__(self):
        if _instance:
            raise Exception("")
    def instance():
        global _instance
        if not _instance:
            _instance = Config()
        return _instance