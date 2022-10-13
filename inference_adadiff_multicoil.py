import argparse
import torch
import numpy as np
import os
import torch.optim as optim
import torchvision
from utils.models.ncsnpp_generator_adagn import NCSNpp
from datasets_prep.brain_datasets import CreateDatasetReconstructionMultiCoil
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_mse as mse
from skimage.measure import compare_ssim as ssim  

        
def psnr_torch(img1, img2):
    #Peak Signal to Noise Ratio
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(img1.max() / torch.sqrt(mse))

        
#%% Diffusion coefficients 
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
def sample_posterior(coefficients, x_0,x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (1 - (t == 0).type(torch.float32))
        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
    sample_x_pos = p_sample(x_0, x_t, t)
    return sample_x_pos

def sample_from_model(coefficients, generator, n_time, x_init, fs, us, us_comb, mask, coil_maps, T, opt):
    x = us_comb
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)#.to(x.device)
            x_0 = generator(x, t_time, latent_z)
            x_0 = data_consistency(x_0, us, mask, coil_maps)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
        x_0 = generator(x, t_time, latent_z)
        x = x_0.detach()
    return x

def data_consistency(x, us, mask, coil_maps, reshape = True):
    mask=mask>0.5
    crop = transforms.CenterCrop((us.shape[-2],us.shape[-1]))
    pad_y = int ( ( x.shape[-2]-us.shape[-2])/2 )
    pad_x = int ( ( x.shape[-1]-us.shape[-1])/2 )
    pad = torch.nn.ZeroPad2d((pad_x, pad_x, pad_y, pad_y))
    if reshape:
        x = crop(x)
    x = torch.complex(x[:,[0],:], x[:,[1],:])  
    x = x * coil_maps
    x = fft2c(x) * ~mask + fft2c(us) * mask
    x = ifft2c(x)
    x = torch.sum(x * torch.conj(coil_maps), dim = 1, keepdim= True )
    if reshape:
        x = pad(x)
    x = torch.cat((x.real, x.imag), axis=1)
    return x

def data_consistency_loss(x, us, mask, coil_maps):
    mask=mask>0.5
    crop = transforms.CenterCrop((us.shape[-2],us.shape[-1]))
    x = torch.complex(x[:,[0],:], x[:,[1],:]) 
    if x.shape[-1] != us.shape[-1]:
        x = crop(x)
    x = x * coil_maps
    x_fft = fft2c(x) * mask
    us_fft =  fft2c(us) * mask                
    loss = F.l1_loss(x_fft, us_fft)
    return loss

def ifft2c(x, dim=((-2,-1)), img_shape=None):
    if not dim:
        dim = range(x.ndim)
    x = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=dim), s=img_shape, dim=dim), dim = dim)
    return x

def fft2c(x, dim=((-2,-1)), img_shape=None):
    if not dim:
        dim = range(x.ndim)
    x = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=dim), s=img_shape, dim=dim), dim = dim)
    return x

def load_checkpiont(checkpoint_dir, netG, device = 'cuda:0', trained = True):
    checkpoint_file = checkpoint_dir.format(args.dataset, args.exp)  
    checkpoint = torch.load(checkpoint_file, map_location=device)
    ckpt = checkpoint['netG_dict']  
    if trained:
        for key in list(ckpt.keys()):       
            ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()    

#%%
def sample_and_test(args):
    torch.manual_seed(42)
    gpu = args.local_rank
    device = torch.device('cuda:{}'.format(gpu))
    to_range_0_1 = lambda x: (x) / x.max()

    #loading dataset
    phase=args.phase
    dataset=CreateDatasetReconstructionMultiCoil(phase=phase, contrast = args.contrast, R = args.R )
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=4)
    args.attn_resolutions = tuple(args.attn_resolutions)
    #Initializing and loading network
    netG = NCSNpp(args).to(device)
    if args.exp !='DIP':
        checkpoint_file = '.../{}/{}/content.pth'
        load_checkpiont(checkpoint_file, netG, device = device)
    else:
        parent_dir = ".../{}".format(args.dataset)
        exp_path = os.path.join(parent_dir,args.exp)        
        
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)        
        content = {'netG_dict': netG.state_dict()}    
        torch.save(content, os.path.join(exp_path, args.contrast+'_content.pth'))   
        checkpoint_file = '.../{}/{}/'+args.contrast+'_content.pth'

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    T = get_time_schedule(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    save_dir = ".../{}/{}/".format(args.dataset,  args.exp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = save_dir + args.extra_string 
    print(save_dir)
    loss = np.zeros((len(data_loader),args.itr_inf))
    res =  [384, 288]    
    shape = data_loader.dataset[0][1].shape
    recons = np.zeros((len(data_loader), shape[-2],shape[-1]))
    if args.save_inter:
        recons_inter = np.zeros((int(args.itr_inf/100+1), len(data_loader), shape[-2],shape[-1]), dtype = np.float32)
    for iteration, (fs, us, mask, coil_maps) in enumerate(data_loader): 
        pad_y = int ( ( res[-2]-us.shape[-2])/2 )
        pad_x = int ( ( res[-1]-us.shape[-1])/2 )
        pad = torch.nn.ZeroPad2d((pad_x, pad_x, pad_y, pad_y))
        #make us complex
        us_comb = torch.sum(us * torch.conj(coil_maps), dim = 1, keepdim= True )
        us_comb = pad(us_comb)   
        us_comb = torch.cat((us_comb.real, us_comb.imag), dim=1)
        us_comb = us_comb.to(device)
        us = us.to(device)
        fs = torch.abs(fs.to(device))
        mask = mask.to(device)
        coil_maps = coil_maps.to(device)
        #set cropping window
        crop = transforms.CenterCrop((us.shape[-2],us.shape[-1]))
        x_t_1 = torch.randn_like(us_comb).to(device)
        #diffusion steps
        fake_sample_diff = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, fs, us, us_comb, mask, coil_maps, T,  args)
        us_comb = crop(us_comb)
        us_comb= torch.abs(torch.complex(us_comb[:,[0],:], us_comb[:,[1],:]))
        #optimization
        t_time = torch.zeros([1] , device=device)
        latent_z = torch.randn(1, args.nz, device=device)#.to(x.device)
        
        for ii in range(args.itr_inf):
            netG.zero_grad()    
            fake_sample = netG(fake_sample_diff, t_time, latent_z)
            lossDC =  data_consistency_loss(fake_sample, us, mask, coil_maps)
            fake_sample =  data_consistency(fake_sample, us, mask, coil_maps)
            fake_sample = torch.abs(torch.complex(fake_sample[:,[0],:], fake_sample[:,[1],:]))
            recon = crop(fake_sample).detach().cpu().numpy() ; recon = recon /recon.mean()
            target = fs.detach().cpu().numpy(); target = target/target.mean(); us_comb = us_comb/us_comb.mean();
            loss[iteration, ii] = psnr(target, recon, data_range=target.max()-target.min())
            lossDC.backward() 
            optimizerG.step()
            fake_sample = fake_sample.detach()
            if ii % 100 == 0 and args.save_inter:
                recons_inter[int(ii/100), iteration, :] = np.squeeze(crop(fake_sample).cpu().numpy())
        print('Iteration - {}'.format(iteration))
        recons[iteration, :] = np.squeeze(crop(fake_sample).cpu().numpy())
        np.save('{}{}_{}_{}_recons_{}_final.npy'.format(save_dir, args.contrast, phase, args.R, args.itr_inf), recons)                 
        if args.save_inter:
            np.save('{}{}_{}_{}_recons_{}_inter.npy'.format(save_dir, args.contrast, phase, args.R, args.itr_inf), recons_inter)   

        fake_sample = crop(fake_sample)
        fake_sample = torch.cat((us_comb/us_comb.mean(),fake_sample/fake_sample.mean(),fs/fs.mean()),axis=-1)
        torchvision.utils.save_image(fake_sample, '{}{}_{}_samples_{}.jpg'.format(save_dir, args.contrast, phase, iteration), normalize=True)
        if args.exp =='DIP':
            load_checkpiont(checkpoint_file, netG, device = device, trained = False)
        else:
            load_checkpiont(checkpoint_file, netG, device = device)    
        np.save('{}{}_{}_psnr_values_{}.npy'.format(save_dir, args.contrast, phase, args.itr_inf), loss)
        if args.reset_opt:    
            print('opt reset')
            optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))        
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('adadiff parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int,default=1000)
    parser.add_argument('--num_channels', type=int, default=3,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1,
                            help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                            help='beta_max for diffusion')
    parser.add_argument('--num_channels_dae', type=int, default=128,
                            help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                            help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', nargs='+', type=int, default=[16],
                            help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                            help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                            help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                            help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                            help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                            help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                            help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                            help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    #geenrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy', help='directory to real images for FID computation')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=200, help='sample generating batch size')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    #optimizaer parameters    
    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--itr_inf', type=int, default=100,
                            help='iterations for inference')
    parser.add_argument('--contrast', type=str, default='T1',
                            help='T1, T2 or PD')   
    parser.add_argument('--phase', type=str, default='val',
                            help='val or test')  
    parser.add_argument('--save_inter', type=str, default=False,
                            help='setting true woudl save intermediate results after 100 iterations') 
    parser.add_argument('--R', type=int, default=4,
                            help='acceleration rate')   
    parser.add_argument('--reset_opt', type=bool, default=False,
                            help='extra   string for save_dir')    
    parser.add_argument('--extra_string', type=str, default='',
                            help='extra   string for save_dir')        
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    sample_and_test(args)