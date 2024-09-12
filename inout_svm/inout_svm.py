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
import argparse

import time
timeflg = True
if timeflg: start_id = time.time()

print ('argument list', sys.argv)

#random_states=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'] 
random_states = ['1','2']

parser = argparse.ArgumentParser()
parser.add_argument('-d1', '--data_path_dom1', default='./data/',
                        help='path to embeddingss of domain 1')
parser.add_argument('-d2', '--data_path_dom2', default='./data/',
                        help='path to embeddingss of domain 2')

parser.add_argument('-c', '--num_classes', default=0, type=int,
                        help='number of classes.')

parser.add_argument('-r', '--io_perc', default=0.1, type=float,
                        help='Percentage of in or out movement.')
parser.add_argument('-p', '--print_pos', default=1, type=int,
                        help='Percentage of in or out movement.')
opt = parser.parse_args()
print(opt)

sd = '_'+str(opt.io_perc) 

itr=0
acc, prec, rec, f1 = [],[],[],[] 
acc_dom2, prec_dom2, rec_dom2, f1_dom2 = [],[],[],[] 

for rs in random_states:
    print('rs: {}'.format(rs))
    itr=itr+1
    #data_dir = '/fs_part1/phd/batch_flw102_N_vgg/N'+rs+'/'
    #'../gen_data/domDslr
    data_dir = opt.data_path_dom1+'/N'+rs+'/'
    # '../gen_data/domSynthetic
    data_dir1 = opt.data_path_dom1 +'/N'+rs+'/'

    #data_dir='C:/Progetti/phd-simple-dir/data/flw102_vgg19_embs/N'+rs+'/'
    t_train = torch.load(data_dir+'t_train_N'+rs+'.pt')
    t_test = torch.load(data_dir+'t_test_N'+rs+'.pt')
    t_centr_orig = torch.load(data_dir+'t_centr_N'+rs+'.pt')
    
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
    test_lbls_file_dom2_tmp = glob.glob(data_dir1+'*tst*.csv')[0]
    train_lbls_file_dom2 = glob.glob(data_dir1+'*train_tr*.csv')[0]
    vld_lbls_file_dom2 = glob.glob(data_dir1+'*vld*.csv')[0]
    
    train_lbls = pd.read_csv(train_lbls_file, header=None)
    test_lbls = pd.read_csv(test_lbls_file, header=None)
    y_test=test_lbls[0].values.tolist()

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
    embs_dim = 4096
    
    train_df = pd.DataFrame(t_train)
    train_df['lbls']=train_lbls
    
    flt_trn = train_df
    flt_trn['lbls']=train_lbls
        
    flt_trn_exp = flt_trn
    print('flt_trn_exp.shape {}'.format(flt_trn_exp.shape))
    
    #%% directional augmentation 
    if opt.io_perc != 0.0:
        v_away = torch.tensor([1,embs_dim])
        for i in range(opt.num_classes):
            c = t_centr_orig[i]
            c = c[None,:]
            flt_trn_lbls = flt_trn[ flt_trn['lbls']==i]    
            flt_trn_tmp = flt_trn_lbls.drop('lbls', axis=1)
            t_flt_trn = torch.tensor(flt_trn_tmp.values)
            
            v = t_flt_trn - torch.squeeze(c)
            v_plus= v+v*cnf.perc_exp #vec from centroidds
            cl_p_away = c + v_plus 
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
    X_test_dom2 = t_test_dom2.numpy()
    
    #%% classification
    svc = SVC(kernel='linear', C=1, gamma=1)
    
    svc.fit(X, y)
    y_pred = svc.predict(X_test)
    
    print(y_pred)
    print(y_test)
    
    y_pred_dom2 = svc.predict(X_test_dom2)
    
    #%% classifixation report
    import sklearn.metrics as metrics
    
    #print(metrics.classification_report(y_test, y_pred, digits=4))    
    report = metrics.classification_report(y_test, y_pred,  digits=4, output_dict=True)
    report.update({"accuracy": {"precision": None, "recall": None, "f1-score": report["accuracy"], "support": report['macro avg']['support']}})
    crdf = pd.DataFrame(report).transpose() 
    
    prec.append(metrics.precision_score(y_test, y_pred, average='macro'))
    acc.append(metrics.accuracy_score(y_test, y_pred))
    f1.append(metrics.f1_score(y_test, y_pred, average='macro' ))
    rec.append(metrics.recall_score(y_test, y_pred, average='macro' ))

    # dom2
    report_dom2 = metrics.classification_report(y_test_dom2, y_pred_dom2,  digits=4, output_dict=True)
    report_dom2.update({"accuracy": {"precision": None, "recall": None, "f1-score": report_dom2["accuracy"], "support": report_dom2['macro avg']['support']}})
    crdf_dom2 = pd.DataFrame(report_dom2).transpose() 

    prec_dom2.append(metrics.precision_score(y_test_dom2, y_pred_dom2, average='macro'))
    acc_dom2.append(metrics.accuracy_score(y_test_dom2, y_pred_dom2))
    f1_dom2.append(metrics.f1_score(y_test_dom2, y_pred_dom2, average='macro' ))
    rec_dom2.append(metrics.recall_score(y_test_dom2, y_pred_dom2, average='macro' ))
    
#%% for fineshed
mtr_df = pd.DataFrame(list(zip(acc, f1, prec, rec)), columns =['acc', 'f1', 'prec', 'rec'])
acc_df = pd.DataFrame(list(zip(acc)), columns =['acc_'+sd])

#%%

if not os.path.exists('accuracy_'+'res.xlsx'):
    with pd.ExcelWriter('accuracy_'+'res.xlsx', engine="openpyxl") as writer:  
        acc_df.to_excel(writer, sheet_name='accuracy', startrow=2, startcol=5)     
else:
    with pd.ExcelWriter('accuracy_'+'res.xlsx', engine="openpyxl",
        mode='a', if_sheet_exists='overlay') as writer:  
        acc_df.to_excel(writer, sheet_name='accuracy', startrow=2, startcol=5++int(opt.print_pos), index=False)

#--- Dom2 --
        
mtr_df_dom2 = pd.DataFrame(list(zip(acc_dom2, f1_dom2, prec_dom2, rec_dom2)), columns =['acc', 'f1', 'prec', 'rec'])
    
acc_df_dom2 = pd.DataFrame(list(zip(acc_dom2)), columns =['acc_'+sd])

#%%

if not os.path.exists('accuracy_'+'res_dom2.xlsx'):
    with pd.ExcelWriter('accuracy_'+'res_dom2.xlsx', engine="openpyxl") as writer:  
        acc_df_dom2.to_excel(writer, sheet_name='accuracy', startrow=2, startcol=5)     
else:
    with pd.ExcelWriter('accuracy_'+'res_dom2.xlsx', engine="openpyxl",
        mode='a', if_sheet_exists='overlay') as writer:  
        acc_df_dom2.to_excel(writer, sheet_name='accuracy', startrow=2, startcol=5++int(opt.print_pos), index=False)


if timeflg: print('Finished in: {:.5f} sec\n'.format( (time.time() - start_id) ) )
    
