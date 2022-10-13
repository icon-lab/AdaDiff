import torch.utils.data
import numpy as np, h5py
import random


#This function loads t1, t2, and pd-weighted images in ixi for uncoditional synthesis
def CreateDataset(phase='train'):

    target_file='../datasets/recsynth/IXI_gaussian/T1_1_multi_synth_recon_'+str(phase)+'.mat'
    f = h5py.File(target_file,'r') 
    data_fs_t1=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=1)
    data_fs_t1=data_fs_t1.astype(np.float32)        
    
    target_file='../datasets/recsynth/IXI_gaussian/T2_1_multi_synth_recon_'+str(phase)+'.mat'
    f = h5py.File(target_file,'r') 
    data_fs_t2=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=1)
    data_fs_t2=data_fs_t2.astype(np.float32)
    
    target_file='../datasets/recsynth/IXI_gaussian/PD_1_multi_synth_recon_'+str(phase)+'.mat'
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
#This function loads t1, t2, and flair multi-coil images in fastMRI for uncoditional synthesis
def CreateDatasetMultiCoil(phase='train'):

    target_file='../datasets/fast_mri/brain/T1/T1_under_sampled_4x_multicoil_'+str(phase)+'.mat'
    data_fs_t1=LoadDataSetMultiCoil(target_file)       
    
    target_file='../datasets/fast_mri/brain/T2/T2_under_sampled_4x_multicoil_'+str(phase)+'.mat'
    data_fs_t2=LoadDataSetMultiCoil(target_file)
    
    target_file='../datasets/fast_mri/brain/FLAIR/FLAIR_under_sampled_4x_multicoil_'+str(phase)+'.mat'
    data_fs_flair=LoadDataSetMultiCoil(target_file)
   
    data_fs=np.concatenate((data_fs_t1[0:800,:],data_fs_t2[0:800,:],data_fs_flair[0:800,:]),axis=0)

    dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs),torch.zeros([data_fs.shape[0],1]))   
    print(data_fs.shape)  

    return dataset 

def CreateDatasetReconstructionMultiCoil(phase='test', contrast= 'T1', R = 4):

    target_file='../datasets/fast_mri/brain/'+contrast+'/'+contrast+'_under_sampled_'+str(R)+'x_multicoil_'+str(phase)+'.mat'
    data_fs=LoadDataSetMultiCoil(target_file, 'images_fs', padding = False, Norm = True, channel_cat = False)
    data_us=LoadDataSetMultiCoil(target_file, 'images_us', padding = False, Norm = True, channel_cat = False)        
    masks=LoadDataSetMultiCoil(target_file, 'map', padding = False, Norm = False, is_complex = False, channel_cat = False)                     
    coil_maps=LoadDataSetMultiCoil(target_file, 'coil_maps', padding = False, Norm = False, channel_cat = False)  
    
    dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs), torch.from_numpy(data_us), torch.from_numpy(masks), torch.from_numpy(coil_maps))   

    return dataset 

#This function loads t1, t2 pd-weighted images in ixi for uncoditional synthesis with class cond information

def CreateDatasetClassCond(phase='train', shuffle = True):

    target_file='../datasets/recsynth/IXI_gaussian/T1_1_multi_synth_recon_'+str(phase)+'.mat'
    data_fs_t1=LoadDataSet(target_file)
    
    target_file='../datasets/recsynth/IXI_gaussian/T2_1_multi_synth_recon_'+str(phase)+'.mat'
    data_fs_t2=LoadDataSet(target_file)
    
    target_file='../datasets/recsynth/IXI_gaussian/PD_1_multi_synth_recon_'+str(phase)+'.mat'
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

#This function loads paired fully sampled and undersampled t1, t2 pd-weighted data along with undersampling masks in ixi for inference

def CreateDatasetReconstruction(phase='test', data = 'IXI',contrast= 'T1', R = 4):

    target_file='../datasets/recsynth/'+data+'_gaussian/'+contrast+'_'+str(R)+'_multi_synth_recon_'+str(phase)+'.mat'
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

#Dataset loading from load_dir and converintg to 288 x 384
def LoadDataSetMultiCoil(load_dir, variable = 'images_fs', padding = True, Norm = True, res = [384, 288], slices = 10, is_complex = True, channel_cat = True):
    f = h5py.File(load_dir,'r') 
    if np.array(f[variable]).ndim==3:
        data=np.expand_dims(np.transpose(np.array(f[variable]),(0,1,2)),axis=1)
    else:
        data=np.transpose(np.array(f[variable]),(1,0,2,3))
        
    if is_complex:
        data  = data['real'] + 1j*data['imag']    
    else:
        data=data.astype(np.float32) 
        
    if Norm: 
        #normalize each subject    
        subjects=int(data.shape[0]/slices )
        data=np.split(data,subjects,axis=0)   
        data=[x/abs(x).max() for x in data] 
        data=np.concatenate(data,axis=0)       
    if channel_cat:    
        data  = np.concatenate((data.real, data.imag), axis=1)
    if padding:
        pad_x = int((res[0]-data.shape[2])/2); pad_y = int((res[1]-data.shape[3])/2)
        data=np.pad(data,((0,0),(0,0),(pad_x, pad_x),(pad_y, pad_y)))   
     
    return data