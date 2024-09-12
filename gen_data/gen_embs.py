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
import random
from statistics import mean
import time
from datetime import datetime
import argparse

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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--embs_path', default='./data/',
                        help='path to embeddingss')
parser.add_argument('-s', '--strat', default='N',
                        help='N stands for No Augmentation')
parser.add_argument('-c', '--num_classes', default=0, type=int,
                        help='number of classes.')

opt = parser.parse_args()
print(opt)

print(sys.argv)
strat = opt.strat

#%% seed
seed_everything(1)##random_state)

path = strat
if not os.path.exists(path):
    os.makedirs(path)

transforms1 = transforms.Compose([
    transforms.Resize((224, 224)),           # Resize the image to 224x224 for VGG19
    transforms.ToTensor(),                   # Convert the image to a tensor
    transforms.Normalize(                    # Normalize using ImageNet's mean and std
        mean=[0.485, 0.456, 0.406],          # Mean values for R, G, B channels
        std=[0.229, 0.224, 0.225]            # Standard deviation for R, G, B channels
    )
])

n_classes=opt.num_classes
#./repo_data_webcam
data_dir = opt.embs_path+'/rs_data_'+strat+'/train'
data_valid =opt.embs_path+'/rs_data_'+strat+'/valid'
data_test = opt.embs_path+'/rs_data_'+strat+'/test'
print('dtst_train dir: ', data_dir)
print('dtst_test dir: ', data_valid)
print('dtst_test dir: ', data_test)

# le tranform non mutano la diemnsione del dataset
dtst_train = datasets.ImageFolder(data_dir, transforms1) 
dtst_valid = datasets.ImageFolder(data_valid, transforms1) 
dtst_test = datasets.ImageFolder(data_test, transforms1) 

#%%view the classes
dtst_dict = dtst_train.class_to_idx
print(dtst_dict)
#%% print first 4 elements
# viene fuori che è ordinato come nelle cartelle
# for i in range(3990, 4000):
#     x, y = dtst[i]
#     print( y)

print('dtst_train len: ',len(dtst_train))
print('dtst_valid len: ',len(dtst_valid))
print('dtst_test len: ',len(dtst_test))

#%% load vgg19
from torchvision.models import vgg19, VGG19_Weights

# https://discuss.pytorch.org/t/using-pretrained-vgg-16-to-get-a-feature-vector-from-an-image/76496/2
model = vgg19(pretrained = True)

default_weights = VGG19_Weights.DEFAULT

#prepare the model for inference
model = vgg19(weights=default_weights)

model.classifier = model.classifier[:-1] # vgg16_model.classifier is nn.Sequential block.

model.to(device) #cuda()
model.eval()
preprocess = default_weights.transforms()


l_pil_im =[]
l_pil_lbl=[]
#print(results.size())
added = 0

timeflg = True

lst_orig_tns =[]
lst_orig_vec =[]
lst_orig_lbls =[]

if timeflg: start_id = time.time()
#%% transform
t2PIL = transforms.ToPILImage()
toTensor = T.ToTensor()
print("Image processin Start")
for i in range(len(dtst_train)):
    
    img, label = dtst_train[i]
    file_name = dtst_train.imgs[i]
    if i % 100 == 0:
        print(i, " ", label)
    
    #print("type img: ", type(img))
    if(torchvision.transforms.functional.get_image_num_channels(img) != 1):
        proc_img = preprocess(img).unsqueeze(0)

        #extract the feature vector from the very last layer (before the softmax) and print the dimensions
        result =  model(proc_img.to(device)) # cuda())
        vector_last = result.squeeze(0)
        #https://stackoverflow.com/questions/57942487/how-to-convert-torch-tensor-to-pandas-dataframe

        lst_orig_tns.append(vector_last.detach().cpu())
        lst_orig_vec.append(vector_last.detach().cpu().numpy())
        lst_orig_lbls.append(label)        

#Training save
timestamp = datetime.now().strftime("%y%m%d-%H%M")
t_train_vec = torch.stack(lst_orig_tns)
fsp.print_list(lst_orig_lbls, timestamp, out_dir='./'+path+'/', postfix='_train_true_'+strat+'.csv')   
torch.save(t_train_vec,path+'/'+'t_train_'+strat+'.pt')

#%% centroids    
#%% https://stackoverflow.com/questions/49540922/convert-list-of-arrays-to-pandas-dataframe
import pandas as pd

df_orig =  pd.DataFrame(lst_orig_vec)
df_orig['lbl'] = lst_orig_lbls
print(df_orig.shape)
list_centr =[]
l_cl_sims=[]
for i in range(0,n_classes):
    df_orig_mean = df_orig[df_orig['lbl']==i].mean(axis=0)
    #print(df_orig_mean)
    print('mean shape:', df_orig_mean.shape)
    centroid = df_orig_mean[:-1]
    list_centr.append(centroid)

    #compute intra cluster similarity with centroi
    l_sims=[]
    df_clust = df_orig[df_orig['lbl']==i]
    df_clust_embs = df_clust.drop(columns=['lbl'])
    t_cen = torch.tensor(centroid)
    t_cl_embs = torch.tensor(df_clust_embs.values)
    print('df_clust_embs.shape: ',df_clust_embs.shape)
    for j in range(len(df_clust_embs)):
        sim = torch.dot(t_cl_embs[j].type(torch.DoubleTensor),t_cen)/(torch.norm(t_cl_embs[j].type(torch.DoubleTensor))*torch.norm(t_cen))
        l_sims.append(sim.item()) 
        
    l_cl_sims.append( mean(l_sims) )
    
    
#%% create centroids dataframe
df_centr = pd.DataFrame(list_centr)
#%% torch tensor from dataframe
t_centr = torch.tensor(df_centr.values)

torch.save(t_centr, path+'/'+'t_centr_'+strat+'.pt')
#%% example of similarity
table_sim = np.zeros((n_classes, n_classes))

for i in range(len(list_centr)):
    for j in range (len(list_centr)):
        if i==j:
            table_sim[i][j]=1.01
        else:        
            sim = torch.dot(t_centr[i],t_centr[j])/(torch.norm(t_centr[i])*torch.norm(t_centr[j]))
            table_sim[i][j]=sim.item()

print(table_sim)

# save to csv
df_table = pd.DataFrame(table_sim)
df_table.to_csv(path+'/'+'centr_sim_'+strat+'.csv',  header=False, index=False)

fsp.print_list(l_cl_sims, timestamp ,out_dir='./'+path+'/', postfix= '_cls_sims_'+strat+'.csv' )
fsp.print_list(lst_orig_lbls, timestamp ,out_dir='./'+path+'/', postfix= '_train_lbls_'+strat+'.csv' )

#TEST SAVING
lst_test_vec = []
lst_test_true = []
for i in range(len(dtst_test)):
 
    
    img, label = dtst_test[i]
    file_name = dtst_test.imgs[i]
    if i % 100 == 0:
        print(i, " ", label)
    
    #if(image_ID!=1581):
    #print("type img: ", type(img))
    if(torchvision.transforms.functional.get_image_num_channels(img) != 1):
        proc_img = preprocess(img).unsqueeze(0)

        #extract the feature vector from the very last layer (before the softmax) and print the dimensions
        result =  model(proc_img.to(device)) # cuda())
        vector_last = result.squeeze(0)
        #https://stackoverflow.com/questions/57942487/how-to-convert-torch-tensor-to-pandas-dataframe
#        lst_orig_vec.append((vector_last, label, file_name))
        lst_test_vec.append(vector_last.detach().cpu())
        lst_test_true.append(label)    

t_test_vec = torch.stack(lst_test_vec)
fsp.print_list(lst_test_true, timestamp, out_dir='./'+path+'/', postfix='_tst_true_'+strat+'.csv')   
torch.save(t_test_vec,path+'/'+'t_test_'+strat+'.pt')


#VALID SAVING
lst_valid_vec = []
lst_valid_true = []
for i in range(len(dtst_valid)):
     
    img, label = dtst_valid[i]
    file_name = dtst_valid.imgs[i]
    if i % 100 == 0:
        print(i, " ", label)
    
    #print("type img: ", type(img))
    if(torchvision.transforms.functional.get_image_num_channels(img) != 1):
        proc_img = preprocess(img).unsqueeze(0)

        #extract the feature vector from the very last layer (before the softmax) and print the dimensions
        result =  model(proc_img.to(device)) # cuda())
        vector_last = result.squeeze(0)
        #https://stackoverflow.com/questions/57942487/how-to-convert-torch-tensor-to-pandas-dataframe
#        lst_orig_vec.append((vector_last, label, file_name))
        lst_valid_vec.append(vector_last.detach().cpu())
        lst_valid_true.append(label)    

t_valid_vec = torch.stack(lst_valid_vec)
fsp.print_list(lst_valid_true, timestamp, out_dir='./'+path+'/', postfix='_vld_true_'+strat+'.csv')   
torch.save(t_valid_vec,path+'/'+'t_vld_'+strat+'.pt')
print("GEN EMBS FINISHED")
