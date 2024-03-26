import torch.utils.data
import numpy as np, h5py
import random


#This function loads t1, t2, and pd-weighted images in ixi for uncoditional synthesis
def CreateDataset(phase='train'):

    target_file='/fast_storage/intern2/T1_1_multi_synth_recon_'+str(phase)+'.mat'
    f = h5py.File(target_file,'r') 
    data_fs_t1=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=1)
    data_fs_t1=data_fs_t1.astype(np.float32)        
    
    target_file='/fast_storage/intern2/T2_1_multi_synth_recon_'+str(phase)+'.mat'
    f = h5py.File(target_file,'r') 
    data_fs_t2=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=1)
    data_fs_t2=data_fs_t2.astype(np.float32)
    
    target_file='/fast_storage/intern2/PD_1_multi_synth_recon_'+str(phase)+'.mat'
    f = h5py.File(target_file,'r') 
    data_fs_pd=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=1)
    data_fs_pd=data_fs_pd.astype(np.float32)
   
    data_fs=np.concatenate((data_fs_t1[0:800,:],data_fs_t2[0:800,:],data_fs_pd[0:800,:]),axis=0)
    data_fs=np.pad(data_fs,((0,0),(0,0),(0,0),(52,52)))
    data_fs[data_fs<0]=0
    data_fs=(data_fs-0.5)/0.5
    dataset=[]
    dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs),torch.zeros([data_fs.shape[0],1]))   
    print(data_fs.shape)  

    return dataset 

def CreateDatasetClassCond(phase='train', shuffle = True):
    target_file='/fast_storage/intern2/T1_1_multi_synth_recon_'+str(phase)+'.mat'
    data_fs_t1=LoadDataSet(target_file)
    
    target_file='/fast_storage/intern2/T2_1_multi_synth_recon_'+str(phase)+'.mat'
    data_fs_t2=LoadDataSet(target_file)
    
    target_file='/fast_storage/intern2/PD_1_multi_synth_recon_'+str(phase)+'.mat'
    data_fs_pd=LoadDataSet(target_file)
    
    data_fs = np.concatenate((data_fs_t1[0:800,:],data_fs_t2[0:800,:],data_fs_pd[0:800,:]),axis=0)

    labels = np.zeros((2400,3), dtype = 'float32'); 
    labels[0:800,:] = np.asarray([1, 0, 0]); labels[800:1600,:] = np.asarray([0, 1, 0]); labels[1600:2400,:] = np.asarray([0, 0, 1])
      
    
    if shuffle:
        samples=list(range(data_fs.shape[0]))
        random.shuffle(samples)    
        data_fs = data_fs[samples,:]; labels = labels[samples,:]
    
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data_fs), torch.from_numpy(labels)) 
    
    return dataset 

#This function loads paired t1, t2 pd-weighted images in ixi for coditional synthesis with class cond information

def CreateDatasetSynthesisClassCond(phase='train', dataset = 'IXI', shuffle = True):
    target_file='/fast_storage/intern2/T1_1_multi_synth_recon_'+str(phase)+'.mat'
    data_fs_t1=LoadDataSet(target_file)
    
    target_file='/fast_storage/intern2/T2_1_multi_synth_recon_'+str(phase)+'.mat'
    data_fs_t2=LoadDataSet(target_file)
    
    if dataset == 'IXI':
        target_file='/fast_storage/intern2/PD_1_multi_synth_recon_'+str(phase)+'.mat'
        data_fs_pd=LoadDataSet(target_file)    
        data_fs = np.concatenate((data_fs_t1,data_fs_t2,data_fs_pd),axis=1)
    else:
        raise NotImplementedError('Dataset not implemented')
    
    if shuffle:
        samples=list(range(data_fs.shape[0]))
        random.shuffle(samples)    
        data_fs = data_fs[samples,:]; 
    
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data_fs)) 
    
    return dataset 

#This function loads paired t1, t2 pd-weighted images in ixi for coditional synthesis

def CreateDatasetSynthesis(phase='train', dataset = 'IXI',source = 'T1'):
    target_file='/fast_storage/intern2/T1_1_multi_synth_recon_'+str(phase)+'.mat'
    data_fs_t1=LoadDataSet(target_file)
    
    target_file='/fast_storage/intern2/T2_1_multi_synth_recon_'+str(phase)+'.mat'
    data_fs_t2=LoadDataSet(target_file)
    if source=='T2':
        dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs_t1),torch.from_numpy(data_fs_t2))   
    if source=='T1':
        dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs_t2),torch.from_numpy(data_fs_t1))   
    return dataset 

#This function loads unpaired t1, t2 pd-weighted images in ixi for coditional synthesis
def CreateDatasetSynthesisUnpaired(phase ='train'):
    phase='train'

    target_file='/fast_storage/intern2/T1_1_multi_synth_recon_'+str(phase)+'.mat'
    data_fs_t1=LoadDataSet(target_file)
    
    target_file='/fast_storage/intern2/T2_1_multi_synth_recon_'+str(phase)+'.mat'
    data_fs_t2=LoadDataSet(target_file)
    #shuffling to make them unpaired
    samples=list(range(data_fs_t2.shape[0]))
    random.shuffle(samples)
    data_fs_t2=data_fs_t2[samples,:]

    dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs_t1),torch.from_numpy(data_fs_t2))   

    return dataset 

#This function loads paired fully sampled and undersampled t1, t2 pd-weighted data along with undersampling masks in ixi for inference

def CreateDatasetReconstruction(phase='test', data = 'IXI',contrast= 'T1', R = 4):
    target_file='/fast_storage/intern2/'+contrast+'_'+str(R)+'_multi_synth_recon_'+str(phase)+'.mat'
    data_fs=LoadDataSet(target_file, 'data_fs')
    data_us=LoadDataSet(target_file, 'data_us', padding = False, Norm = False)        
    masks=LoadDataSet(target_file, 'us_masks', padding = False, Norm = False)                     

    dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs), torch.from_numpy(data_us), torch.from_numpy(masks))   

    return dataset 


#Dataset loading from load_dir and converintg to 256x256 
def LoadDataSet(load_dir, variable = 'data_fs', padding = True, Norm = True, res = 256):
    f = h5py.File(load_dir,'r') 
    if np.array(f[variable]).ndim==3:
        data=np.expand_dims(np.transpose(np.array(f[variable]),(0,2,1)),axis=1)
    else:
        data=np.transpose(np.array(f[variable]),(1,0,3,2))
    data=data.astype(np.float32) 
    if padding:
        pad_x = int((res-data.shape[2])/2); pad_y = int((res-data.shape[3])/2)
        data=np.pad(data,((0,0),(0,0),(pad_x, pad_x),(pad_y, pad_y)))   
    if Norm:    
        data=(data-0.5)/0.5      
    return data
