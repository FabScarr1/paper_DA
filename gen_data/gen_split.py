# -*- coding: utf-8 -*-

import torch
import  torchvision
from torchvision import datasets, models, transforms
import numpy as np
import time
import os
from torch.utils.data.dataset import random_split
import sys
from PIL import Image
import random
import copy
import torchvision.transforms as T
import gc
import fs_print as fsp
from shutil import copyfile

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', default='./data/',
                        help='path to embeddingss')
#parser.add_argument('-e', '--exp_id', default=0.2, type=float,
#                        help='experiment id')
parser.add_argument('-c', '--num_classes', default=0, type=int,
                        help='number of classes.')
parser.add_argument('-r', '--random_state', default=0, type=int,
                        help='random_state.')

opt = parser.parse_args()
print(opt)

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
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object
#%%
#print(sys.argv)
#exp_id = sys.argv[1]
#random_state = int(sys.argv[2])

#%% seed
seed_everything(1)##random_state)

#%%
transforms1 = transforms.Compose([
    transforms.Resize((224, 224))           # Resize the image to 224x224 for VGG19
    #transforms.ToTensor(),                   # Convert the image to a tensor
    #transforms.Normalize(                    # Normalize using ImageNet's mean and std
    #    mean=[0.485, 0.456, 0.406],          # Mean values for R, G, B channels
    #    std=[0.229, 0.224, 0.225]            # Standard deviation for R, G, B channels
    #)
])


# data_dir = "/fs_part1/phd/data/Modern-Office-31/webcam" #amazon"

n_classes = opt.num_classes
# le tranform non mutano la diemnsione del dataset
dtst = datasets.ImageFolder(opt.data_path, transforms1)
#%%view the classes
dtst_dict = dtst.class_to_idx
print(dtst_dict)
#%% print first 4 elements
# viene fuori che è ordinato come nelle cartelle
# for i in range(3990, 4000):
#     x, y = dtst[i]
#     print( y)

#%% random split 819 6552 818

dtst_train, dtst_val, dtst_test = random_split(dtst,[0.8,0.1,0.1], generator=torch.Generator().manual_seed( int(opt.random_state) ))
#%%
print('dtst_train len: ',len(dtst_train))
print('dtst_test len: ',len(dtst_test))
#for i in range(10, 20):

l_pil_im =[]
l_pil_lbl=[]
#print(results.size())

timeflg = True

#%% Create dirs
root_path  = "./tmp_data"
ds_paths = ["train", "test", "valid"] 
for ds_path in ds_paths:
    path = os.path.join(root_path, ds_path) 
    if not os.path.exists(path):
        os.makedirs(path)
        
    for i in range(0, n_classes):
        os.makedirs(path+"/"+str(i))

#%% save orig train
for i in range(len(dtst_train)):
    im, lb = dtst_train[i]
    im.save('./tmp_data/train/'+str(lb)+'/'+str(i)+'.jpg')

#%% save test
for i in range(len(dtst_test)):
    im, lb = dtst_test[i]
    im.save('./tmp_data/test/'+str(lb)+'/'+str(i)+'.jpg')
        
#%% save valid
for i in range(len(dtst_val)):
    im, lb = dtst_val[i]
    im.save('./tmp_data/valid/'+str(lb)+'/'+str(i)+'.jpg')

#%% fix split problems if needed
for i in range(n_classes):
    dir_tst ='./tmp_data/test/'+str(i)
    dir_vld ='./tmp_data/valid/'+str(i)
    dir_trn ='./tmp_data/train/'+str(i)
    imgs_tst = os.listdir(dir_tst)
    imgs_vld = os.listdir(dir_vld)
    imgs_trn = os.listdir(dir_trn)
    print("i {} len tst {} vld {} trn {} ".format(i, len(imgs_tst),  len(imgs_vld), len(imgs_trn)))
   
    
    if len(imgs_tst)==0:
        if len(imgs_trn)>1:
            copyfile(dir_trn+'/'+imgs_trn[0], dir_tst+'/'+imgs_trn[0])
            print(' fixed tst fomr trn ', str(i))
        elif len(imgs_vld)>1:
            copyfile(dir_vld+'/'+imgs_vld[0], dir_tst+'/'+imgs_vld[0])
            print(' fixed tst fomr vld', str(i))
        else:
            print(dir_tst+'non puo essere corretta' )
            sys.exit()
        
    if len(imgs_vld)==0:
        if len(imgs_trn)>1:
            copyfile(dir_trn+'/'+imgs_trn[0], dir_vld+'/'+imgs_trn[0])
            print(' fixed vld fomr trn', str(i))
        elif len(imgs_tst)>1:
            copyfile(dir_tst+'/'+imgs_tst[0], dir_vld+'/'+imgs_tst[0])
            print(' fixed vld fomr tst', str(i))
        else:
            print(dir_vld+'non puo essere corretta' )
            sys.exit()

    if len(imgs_trn)==0:
        if len(imgs_vld)>1:
            copyfile(dir_vld+'/'+imgs_vld[0], dir_trn+'/'+imgs_vld[0])
            print(' fixed trn fomr vld')
        elif len(imgs_tst)>1:
            copyfile(dir_tst+'/'+imgs_tst[0], dir_trn+'/'+imgs_tst[0])
            print(' fixed trn fomr tst')
        else:
            print(dir_trn+'non puo essere corretta' )
            sys.exit()
print("GEN N FINISHED")
