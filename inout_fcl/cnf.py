# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:36:20 2023

@author: f_sca
"""



perc_filter = 0.3 #perc of train narrowing down (shrinking)
perc_exp=0.6   #expansion
aug = str(perc_filter)+'-'+str(perc_exp)

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True