# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 18:43:31 2023

@author: f_sca
"""

import torch
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.svm import SVC 
import os
import sys
import cnf
import glob 
import sys

import torch.optim as optim
from torch import nn
import tqdm
import copy
import argparse
import func4rotation as f4r

import time
timeflg = True
if timeflg: start_id = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-d1', '--data_path_dom1', default='./data/',
                        help='path to embeddingss of domain 1')
parser.add_argument('-d2', '--data_path_dom2', default='./data/',
                        help='path to embeddingss of domain 2')
#parser.add_argument('-e', '--exp_id', default=0.2, type=float,
#                        help='experiment id')
parser.add_argument('-c', '--num_classes', default=0, type=int,
                        help='number of classes.')
#parser.add_argument('-r', '--random_state', default=0, type=int,
#                        help='random_state.')
parser.add_argument('-r', '--ang_rad', default=0.1, type=float,
                        help='rotaion angle in radias.')
parser.add_argument('-p', '--print_pos', default=1, type=int,
                        help='Percentage of in or out movement.')
opt = parser.parse_args()
print(opt)


cnf.seed_everything(0)

print ('argument list', sys.argv)

#random_states=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'] 
random_states = ['1','2']

sd = 'aug'+'-'+str(round(opt.ang_rad*180/np.pi))  


itr=0
acc_l, prec, rec, f1 = [],[],[],[] 
acc_dom2_l, prec_dom2, rec_dom2, f1_dom2 = [],[],[],[] 

for rs in random_states:
    print('rs: {}'.format(rs))
    itr=itr+1
    data_dir = opt.data_path_dom1+'/N'+rs+'/' 
    data_dir1 = opt.data_path_dom2+'/N'+rs+'/' 
 
    t_train = torch.load(data_dir+'t_train_N'+rs+'.pt')
    t_test = torch.load(data_dir+'t_test_N'+rs+'.pt')
    t_valid = torch.load(data_dir+'t_vld_N'+rs+'.pt')
    t_centr_orig = torch.load(data_dir+'t_centr_N'+rs+'.pt')
    #dom2
    t_train_dom2 = torch.load(data_dir1+'t_train_N'+rs+'.pt')
    t_valid_dom2 = torch.load(data_dir1+'t_vld_N'+rs+'.pt')
    t_test_dom2_tmp = torch.load(data_dir1+'t_test_N'+rs+'.pt')
    t_test_dom2 = torch.cat((t_train_dom2, t_valid_dom2, t_test_dom2_tmp),0)

    print('t_train.shape {}'.format(t_train.shape))
    print('t_test.shape {}'.format(t_test.shape))
    print('t_centr_orig.shape {}'.format(t_centr_orig.shape))
    print('t_test_dom2.shape {}'.format(t_test_dom2.shape))
    
    train_lbls_file = glob.glob(data_dir+'*train_lbl*.csv')[0]
    test_lbls_file = glob.glob(data_dir+'*tst*.csv')[0]
    valid_lbls_file = glob.glob(data_dir+'*vld*.csv')[0]
    #dom2
    test_lbls_file_dom2_tmp = glob.glob(data_dir1+'*tst*.csv')[0]
    train_lbls_file_dom2 = glob.glob(data_dir1+'*train_tr*.csv')[0]
    vld_lbls_file_dom2 = glob.glob(data_dir1+'*vld*.csv')[0]
    
    train_lbls = pd.read_csv(train_lbls_file, header=None)
    test_lbls = pd.read_csv(test_lbls_file, header=None)
    vld_lbls = pd.read_csv(valid_lbls_file, header=None)
    y_test = test_lbls[0].values.tolist()
    y_test = torch.tensor(y_test)
    y_vld  = vld_lbls[0].values.tolist()
    y_vld  = torch.tensor(y_vld)
    #Dom2
    test_lbls_dom2_tmp = pd.read_csv(test_lbls_file_dom2_tmp, header=None)
    train_lbls_dom2 = pd.read_csv(train_lbls_file_dom2, header=None)
    vld_lbls_dom2 = pd.read_csv(vld_lbls_file_dom2, header=None)
    test_lbls_dom2 = pd.concat([train_lbls_dom2, vld_lbls_dom2, test_lbls_dom2_tmp])
    y_test_dom2=test_lbls_dom2[0].values.tolist()
    
    print('train_lbls.shape {}'.format(train_lbls.shape))
    print('test_lbls.shape {}'.format(test_lbls.shape))
    print('y_test_dom2.len {}'.format(len(y_test_dom2)))
    #%%
    n_classes = t_centr_orig.shape[0]
    print('n_classes {}'.format(n_classes))
    embs_dim = 4096
    
    train_df = pd.DataFrame(t_train)
    train_df['lbls']=train_lbls
    
    flt_trn = train_df
    flt_trn['lbls']= train_lbls
        
    flt_trn_exp = flt_trn
    print('flt_trn_exp.shape {}'.format(flt_trn_exp.shape))
    
    #%% directional augmentation 
    if opt.ang_rad != 0.0:
        angle = torch.tensor(opt.ang_rad).cuda() 
        #v_away = torch.tensor([1,embs_dim])
        for i in range(n_classes):
            print('class: ',i)
            c = t_centr_orig[i]
            c = c[None,:]
            flt_trn_lbls = flt_trn[ flt_trn['lbls']==i]    
            flt_trn_tmp = flt_trn_lbls.drop('lbls', axis=1)
            t_flt_trn = torch.tensor(flt_trn_tmp.values).cuda()
            
            vecs = t_flt_trn - torch.squeeze(c).cuda()

            l_vecs_rotated =[]
            for j in range(vecs.shape[0]): #for each vector
                #print('j:',j)
                v=vecs[j].type(torch.cuda.FloatTensor)
                v_perp = f4r.perpendicular_vector(v).cuda() #find vec_perp

                R = f4r.rotation_from_angle_and_plane(angle, v, v_perp) #find rotation

                v_rot = torch.matmul(R,v)
                l_vecs_rotated.append(v_rot.cpu())
                
            
            vecs_rotated = np.vstack((l_vecs_rotated))                    
            cl_p_away = c + vecs_rotated
            df = pd.DataFrame(cl_p_away)
            df['lbls'] = i
            
            flt_trn_exp = pd.concat([flt_trn_exp, df])
    else:
        print('No augmentation required')
        
    print('flt_trn_exp.shape {}'.format(flt_trn_exp.shape))            
    #%% TRaining
    
    X = flt_trn_exp.drop(['lbls'], axis=1).to_numpy()
    y = flt_trn_exp['lbls'].to_numpy()
    X_test = t_test.numpy()
    X_vld = t_valid.numpy()
    X_test_dom2 = t_test_dom2.numpy()
    
    # model.output.weight.dtype
    X = torch.from_numpy(X).type(torch.float32)
    X_test =  torch.from_numpy(X_test)
    X_vld =  torch.from_numpy(X_vld)
    y = torch.from_numpy(y)
    X_test_dom2 = torch.from_numpy(X_test_dom2)

    class Multiclass(nn.Module):
        def __init__(self):
            super().__init__()

            self.output = nn.Linear(4096,n_classes)
     
        def forward(self, x):
            #x = self.act(self.hidden(x))
            #x = self.act1(self.hidden1(x))
            x = self.output(x)
            return x
     
    # loss metric and optimizer
    model = Multiclass()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
     
    model.cuda()     
    # prepare model and training parameters
    n_epochs = 30
    batch_size = 32
    batches_per_epoch = len(X) // batch_size
     
    best_acc = - np.inf   # init to negative infinity
    best_weights = None
    train_loss_hist = []
    train_acc_hist = []
    vld_loss_hist = []
    vld_acc_hist = []
     
    # training loop
    for epoch in range(n_epochs):
        epoch_loss = []
        epoch_acc = []
        # set model in training mode and run through each batch
        model.train()
        with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
            bar.set_description(f"Epoch {epoch}")
            for i in bar:
                # take a batch
                start = i * batch_size
                X_batch = X[start:start+batch_size]
                y_batch = y[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch.cuda())
                loss = loss_fn(y_pred.cpu(), y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                #y_batch = y_batch[None, :]
                # compute and store metrics

                acc = (torch.argmax(y_pred.cpu(), 1) == y_batch.cpu()).float().mean()
                epoch_loss.append(float(loss))
                epoch_acc.append(float(acc))
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # set model in evaluation mode and run through the test set
        model.eval()
        y_pred = model(X_vld.cuda())
        ce = loss_fn(y_pred.cpu(), y_vld)
        #acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
        predicted = torch.argmax(y_pred, 1)
        #acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
        acc = (predicted.cpu() == y_vld).float().mean()
        ce = float(ce)
        acc = float(acc)
        train_loss_hist.append(np.mean(epoch_loss))
        train_acc_hist.append(np.mean(epoch_acc))
        vld_loss_hist.append(ce)
        vld_acc_hist.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%")

    #End classification part    
    #%% TST
    model.eval()
    y_pred_tst = model(X_test.cuda())
    y_pred = torch.argmax(y_pred_tst, 1)
    print('y_pred: ',y_pred)
    print('y_test: ',y_test)
    #dom2
    y_pred_dom2_out = model(X_test_dom2.cuda())
    y_pred_dom2 = torch.argmax(y_pred_dom2_out, 1)
    print('y_pred_dom2.shape: ',y_pred_dom2.shape)
    print('len y_test_dom2: ',len(y_test_dom2))
    
    
    #%% classifixation report
    import sklearn.metrics as metrics
    
    y_pred = y_pred.cpu().detach()
    #print(metrics.classification_report(y_test, y_pred, digits=4))    
    report = metrics.classification_report(y_test, y_pred,  digits=4, output_dict=True)
    report.update({"accuracy": {"precision": None, "recall": None, "f1-score": report["accuracy"], "support": report['macro avg']['support']}})
    crdf = pd.DataFrame(report).transpose() 
    
    
    fname = cnf.aug+'_'
    print(os.path.exists(fname+'.xlsx'))
    
    prec.append(metrics.precision_score(y_test, y_pred, average='macro'))
    acc_l.append(metrics.accuracy_score(y_test, y_pred))
    f1.append(metrics.f1_score(y_test, y_pred, average='macro' ))
    rec.append(metrics.recall_score(y_test, y_pred, average='macro' ))

    # dom2
    y_pred_dom2 = y_pred_dom2.cpu().detach()
    report_dom2 = metrics.classification_report(y_test_dom2, y_pred_dom2,  digits=4, output_dict=True)
    report_dom2.update({"accuracy": {"precision": None, "recall": None, "f1-score": report["accuracy"], "support": report['macro avg']['support']}})
    crdf_dom2 = pd.DataFrame(report_dom2).transpose() 

    
    fname_dom2 = cnf.aug+'_dom2_'
    print(os.path.exists(fname_dom2+'.xlsx'))
   
    #print ('f_ratio: {}'.format(f_ratio))
    
    prec_dom2.append(metrics.precision_score(y_test_dom2, y_pred_dom2, average='macro'))
    acc_dom2_l.append(metrics.accuracy_score(y_test_dom2, y_pred_dom2))
    f1_dom2.append(metrics.f1_score(y_test_dom2, y_pred_dom2, average='macro' ))
    rec_dom2.append(metrics.recall_score(y_test_dom2, y_pred_dom2, average='macro' ))

   
   
    fname_dom4 = cnf.aug+'_dom4_'
    print(os.path.exists(fname_dom4+'.xlsx'))
    
#%% for fineshed
mtr_df = pd.DataFrame(list(zip(acc_l, f1, prec, rec)), columns =['acc', 'f1', 'prec', 'rec'])
acc_df = pd.DataFrame(list(zip(acc_l)), columns =['Acc_'+sd])

#%%

if not os.path.exists('accuracy_'+'res.xlsx'):
    with pd.ExcelWriter('accuracy_'+'res.xlsx', engine="openpyxl") as writer:  
        acc_df.to_excel(writer, sheet_name='accuracy', startrow=2, startcol=5)     
else:
    with pd.ExcelWriter('accuracy_'+'res.xlsx', engine="openpyxl",
        mode='a', if_sheet_exists='overlay') as writer:  
        acc_df.to_excel(writer, sheet_name='accuracy', startrow=2, startcol=5++int(opt.print_pos), index=False)

#--- Dom2 --
        
mtr_df_dom2 = pd.DataFrame(list(zip(acc_dom2_l, f1_dom2, prec_dom2, rec_dom2)), columns =['acc', 'f1', 'prec', 'rec'])
    
acc_df_dom2 = pd.DataFrame(list(zip(acc_dom2_l)), columns =['acc_'+sd])

#%%

if not os.path.exists('accuracy_'+'res_dom2.xlsx'):
    with pd.ExcelWriter('accuracy_'+'res_dom2.xlsx', engine="openpyxl") as writer:  
        acc_df_dom2.to_excel(writer, sheet_name='accuracy', startrow=2, startcol=5)     
else:
    with pd.ExcelWriter('accuracy_'+'res_dom2.xlsx', engine="openpyxl",
        mode='a', if_sheet_exists='overlay') as writer:  
        acc_df_dom2.to_excel(writer, sheet_name='accuracy', startrow=2, startcol=5++int(opt.print_pos), index=False)





if timeflg: print('Finished in: {:.5f} sec\n'.format( (time.time() - start_id) ) )
    
