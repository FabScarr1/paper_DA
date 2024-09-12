# -*- coding: utf-8 -*-
"""
Created on Mon May 30 11:08:20 2022

@author: f_sca
"""
import pandas as pd
import torch
import numpy as np
from datetime import datetime

def get_timewtamp():
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    return timestamp
    
def print_cuda_tensor( tns, timestamp, out_dir='./saved/', postfix='_post_df'):
    tns_np = tns.detach().cpu().numpy() #dfc is a named tuple 0 values 1 indices
    tns_df = pd.DataFrame(tns_np) #convert to a dataframe
    tns_df.to_csv(out_dir+timestamp+postfix, index=False, header=False ) #save to file

def print_cuda_tensor_3d( tns, timestamp, out_dir='./saved/', postfix='_post_df'):
    tns = tns.reshape(tns.shape[0]*tns.shape[1],tns.shape[2])
    tns_np = tns.detach().cpu().numpy() #dfc is a named tuple 0 values 1 indices
    tns_df = pd.DataFrame(tns_np) #convert to a dataframe
    tns_df.to_csv(out_dir+timestamp+postfix,index=False) #save to file
 
def print_tensor(tns, timestamp, out_dir='./saved/', postfix='_post_df'):
    tt = torch.tensor(tns)
    np.savetxt(out_dir+timestamp+postfix, tt)

def print_words_in_batch(w_in_b, timestamp, out_dir='./saved/', postfix='_post_df'):
    words_in_batch_df = pd.DataFrame(w_in_b) #convert to a dataframe
    words_in_batch_df.to_csv(out_dir+timestamp+postfix,index=False) #save to file
    
def print_list(lst, timestamp, out_dir='./saved/', postfix='_post_df'):
    with open(out_dir+timestamp+postfix, 'w') as fp:
        for item in lst:
            # write each item on a new line
            fp.write("%s\n" % item)

def print_list_tpl(lst, timestamp, out_dir='./saved/', postfix='_post_df'):
    with open(out_dir+timestamp+postfix, 'w') as fp:
        for item in lst:
            # write each item on a new line
            fp.write(str(item).replace('(','').replace(')','')+'\n')