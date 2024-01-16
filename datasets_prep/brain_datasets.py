import torch.utils.data
import numpy as np
import h5py
import random
import os

BASE_IXI = '../datasets/recsynth/IXI_gaussian'
BASE_FASTMRI = '../datasets/fast_mri/brain/'
N_DATA = 800

def open_file(filename):
    f = h5py.File(filename, 'r')
    subdata_fs = np.expand_dims(np.transpose(np.array(f['data_fs']), (0,2,1)), axis=1)
    subdata_fs = subdata_fs.astype(np.float32)
    return subdata_fs

#This function loads t1, t2, and pd-weighted images in ixi for uncoditional synthesis
def CreateDataset(phase='train'):
    t1_file = os.path.join(BASE_IXI, f'T1_1_multi_synth_recon_{str(phase)}.mat')
    data_fs_t1 = open_file(t1_file)

    t2_file = os.path.join(BASE_IXI, f'T2_1_multi_synth_recon_{str(phase)}.mat')
    data_fs_t2 = open_file(t2_file)

    pd_file = os.path.join(BASE_IXI, f'PD_1_multi_synth_recon_{str(phase)}.mat')
    data_fs_pd = open_file(pd_file)

    data_fs=np.concatenate((data_fs_t1[0:N_DATA,:], data_fs_t2[0:N_DATA,:], data_fs_pd[0:N_DATA,:]),axis=0)
    data_fs=np.pad(data_fs, ((0,0),(0,0),(0,0),(52,52)))
    data_fs[data_fs<0] = 0
    data_fs = (data_fs-0.5)/0.5

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data_fs), torch.zeros([data_fs.shape[0],1]))   
    print(f'IXI data_fs shape: {data_fs.shape}')
    return dataset 

#This function loads t1, t2, and flair multi-coil images in fastMRI for uncoditional synthesis
def CreateDatasetMultiCoil(phase='train'):
    t1_file = os.path.join(BASE_FASTMRI, 'T1', f'T1_under_sampled_4x_multicoil_{str(phase)}.mat')
    data_fs_t1=LoadDataSetMultiCoil(t1_file)

    t2_file = os.path.join(BASE_FASTMRI, 'T2', f'T2_under_sampled_4x_multicoil_{str(phase)}.mat')
    data_fs_t2=LoadDataSetMultiCoil(t2_file)

    flair_file = os.path.join(BASE_FASTMRI, 'FLAIR', f'FLAIR_under_sampled_4x_multicoil_{str(phase)}.mat')
    data_fs_flair=LoadDataSetMultiCoil(flair_file)

    data_fs=np.concatenate((data_fs_t1[0:N_DATA,:],data_fs_t2[0:N_DATA,:],data_fs_flair[0:N_DATA,:]),axis=0)

    dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs),torch.zeros([data_fs.shape[0],1]))   
    print(f'fastMRI data_fs shape: {data_fs.shape}')
    return dataset 

def CreateDatasetReconstructionMultiCoil(phase='test', contrast= 'T1', R = 4):
    target_file = os.path.join(BASE_FASTMRI, contrast, f'contrast_undersampled_{str(R)}x_multicoil_{str(phase)}.mat')

    data_fs=LoadDataSetMultiCoil(target_file, 'images_fs', padding = False, Norm = True, channel_cat = False)
    data_us=LoadDataSetMultiCoil(target_file, 'images_us', padding = False, Norm = True, channel_cat = False)
    masks=LoadDataSetMultiCoil(target_file, 'map', padding = False, Norm = False, is_complex = False, channel_cat = False)
    coil_maps=LoadDataSetMultiCoil(target_file, 'coil_maps', padding = False, Norm = False, channel_cat = False)
    
    dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs), torch.from_numpy(data_us), torch.from_numpy(masks), torch.from_numpy(coil_maps))
    return dataset

#This function loads t1, t2 pd-weighted images in ixi for uncoditional synthesis with class cond information
def CreateDatasetClassCond(phase='train', shuffle = True):
    t1_file = os.path.join(BASE_IXI, f'T1_1_multi_synth_recon_{str(phase)}.mat')
    data_fs_t1 = LoadDataSet(t1_file)
    
    t2_file = os.path.join(BASE_IXI, f'T2_1_multi_synth_recon_{str(phase)}.mat')
    data_fs_t2 = LoadDataSet(t2_file)

    pd_file = os.path.join(BASE_IXI, f'PD_1_multi_synth_recon_{str(phase)}.mat')
    data_fs_pd = LoadDataSet(pd_file)

    data_fs = np.concatenate((data_fs_t1[0:800,:],data_fs_t2[0:800,:],data_fs_pd[0:800,:]),axis=0)

    labels = np.zeros((3*N_DATA, 3), dtype = 'float32'); 
    labels[0:N_DATA,:] = np.asarray([1, 0, 0])
    labels[N_DATA:2*N_DATA,:] = np.asarray([0, 1, 0])
    labels[2*N_DATA:3*N_DATA,:] = np.asarray([0, 0, 1])

    if shuffle:
        samples=list(range(data_fs.shape[0]))
        random.shuffle(samples)
        data_fs = data_fs[samples,:]
        labels = labels[samples,:]

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data_fs), torch.from_numpy(labels))
    return dataset

# This function loads paired fully sampled and undersampled t1, t2 pd-weighted data along with undersampling masks in ixi for inference
def CreateDatasetReconstruction(phase='test', contrast= 'T1', R = 4):
    target_file = os.path.join(BASE_IXI, f'{contrast}_{str(R)}_multi_synth_recon_{str(phase)}.mat')
    data_fs=LoadDataSet(target_file, 'data_fs')
    data_us=LoadDataSet(target_file, 'data_us', padding = False, Norm = False)
    masks=LoadDataSet(target_file, 'us_masks', padding = False, Norm = False)

    dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs), torch.from_numpy(data_us), torch.from_numpy(masks))
    return dataset 

# Dataset loading from load_dir and converting to 256x256
def LoadDataSet(load_dir, variable = 'data_fs', padding = True, Norm = True, res = 256):
    f = h5py.File(load_dir,'r')
    if np.array(f[variable]).ndim == 3:
        data = np.expand_dims(np.transpose(np.array(f[variable]),(0,2,1)),axis=1)
    else:
        data = np.transpose(np.array(f[variable]),(1,0,3,2))
    data = data.astype(np.float32)
    if padding:
        pad_x = int((res-data.shape[2])/2); pad_y = int((res-data.shape[3])/2)
        data=np.pad(data,((0,0),(0,0),(pad_x, pad_x),(pad_y, pad_y)))
    if Norm:
        data=(data-0.5)/0.5
    return data

# Dataset loading from load_dir and converintg to 288 x 384
def LoadDataSetMultiCoil(load_dir, variable = 'images_fs', padding = True, Norm = True, res = [384, 288], slices = 10, is_complex = True, channel_cat = True):
    f = h5py.File(load_dir,'r')

    if np.array(f[variable]).ndim == 3:
        data=np.expand_dims(np.transpose(np.array(f[variable]), (0,1,2)), axis=1)
    else:
        data=np.transpose(np.array(f[variable]), (1,0,2,3))

    if is_complex:
        data = data['real'] + 1j*data['imag']
    else:
        data = data.astype(np.float32)

    if Norm:
        #normalize each subject
        subjects = int(data.shape[0]/slices)
        data = np.split(data, subjects, axis=0)
        data = [x/abs(x).max() for x in data]
        data = np.concatenate(data, axis=0)
    if channel_cat:
        data = np.concatenate((data.real, data.imag), axis=1)
    if padding:
        pad_x = int((res[0]-data.shape[2])/2); pad_y = int((res[1]-data.shape[3])/2)
        data=np.pad(data,((0,0),(0,0),(pad_x, pad_x),(pad_y, pad_y)))
    return data