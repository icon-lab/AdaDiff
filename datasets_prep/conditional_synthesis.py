# Unreferenced code

import torch.utils.data
import numpy as np
import h5py
import random
import os
from brain_datasets import LoadDataSet

#This function loads paired t1, t2 pd-weighted images in ixi for coditional synthesis with class cond information
def CreateDatasetSynthesisClassCond(phase='train', dataset = 'IXI', shuffle = True):

    target_file='../datasets/recsynth/'+dataset+'_gaussian/T1_1_multi_synth_recon_'+str(phase)+'.mat'
    data_fs_t1=LoadDataSet(target_file)
    
    target_file='../datasets/recsynth/'+dataset+'_gaussian/T2_1_multi_synth_recon_'+str(phase)+'.mat'
    data_fs_t2=LoadDataSet(target_file)
    
    if dataset == 'IXI':
        target_file='../datasets/recsynth/'+dataset+'_gaussian/PD_1_multi_synth_recon_'+str(phase)+'.mat'
        data_fs_pd=LoadDataSet(target_file)    
        data_fs = np.concatenate((data_fs_t1,data_fs_t2,data_fs_pd),axis=1)

    elif dataset == 'BRATS':
        target_file='../datasets/recsynth/'+dataset+'_gaussian/T1c_1_multi_synth_recon_'+str(phase)+'.mat'
        data_fs_t1c=LoadDataSet(target_file)   
        target_file='../datasets/recsynth/'+dataset+'_gaussian/Flair_1_multi_synth_recon_'+str(phase)+'.mat'
        data_fs_flair=LoadDataSet(target_file)                  
        data_fs = np.concatenate((data_fs_t1, data_fs_t2, data_fs_t1c, data_fs_flair),axis=1)
        
    
    if shuffle:
        samples=list(range(data_fs.shape[0]))
        random.shuffle(samples)    
        data_fs = data_fs[samples,:]; 
    
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data_fs)) 
    
    return dataset 

#This function loads paired t1, t2 pd-weighted images in ixi for coditional synthesis

def CreateDatasetSynthesis(phase='train', dataset = 'IXI',source = 'T1'):

    target_file='../datasets/recsynth/'+dataset+'_gaussian/T1_1_multi_synth_recon_'+str(phase)+'.mat'
    data_fs_t1=LoadDataSet(target_file)
    
    target_file='../datasets/recsynth/'+dataset+'_gaussian/T2_1_multi_synth_recon_'+str(phase)+'.mat'
    data_fs_t2=LoadDataSet(target_file)
    if source=='T2':
        dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs_t1),torch.from_numpy(data_fs_t2))   
    if source=='T1':
        dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs_t2),torch.from_numpy(data_fs_t1))   
    return dataset 

#This function loads unpaired t1, t2 pd-weighted images in ixi for coditional synthesis
def CreateDatasetSynthesisUnpaired(phase ='train'):
    phase='train'

    target_file='../datasets/recsynth/IXI_gaussian/T1_1_multi_synth_recon_'+str(phase)+'.mat'
    data_fs_t1=LoadDataSet(target_file)
    
    target_file='../datasets/recsynth/IXI_gaussian/T2_1_multi_synth_recon_'+str(phase)+'.mat'
    data_fs_t2=LoadDataSet(target_file)
    #shuffling to make them unpaired
    samples=list(range(data_fs_t2.shape[0]))
    random.shuffle(samples)
    data_fs_t2=data_fs_t2[samples,:]

    dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs_t1),torch.from_numpy(data_fs_t2))   

    return dataset 