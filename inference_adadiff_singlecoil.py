import argparse
import torch
import numpy as np
import os
import torch.optim as optim
import torchvision
from utils.models.ncsnpp_generator_adagn import NCSNpp
from datasets_prep.brain_datasets import CreateDatasetReconstruction
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.args_op import add_singlecoil_inference_args

def psnr(img1, img2):
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

class Diffusion_Coefficients():
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)
        
def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
      noise = torch.randn_like(x_start)
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise
    return x_t

def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t+1, x_start.shape) * noise
    return x_t, x_t_plus_one

def sample_from_model(coefficients, generator, n_time, x_init, fs, us, mask, T, opt):
    x = x_init
    x = -1 * torch.ones_like(x_init)
    x = data_consistency(x, us, mask)
    coeff = Diffusion_Coefficients(opt, x.device)
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)#.to(x.device)
            x_0 = generator(x, t_time, latent_z)
            x_0 = data_consistency(x_0, us, mask)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
        x_0 = generator(x, t_time, latent_z)
        x = x_0.detach()
    return x

def rand_sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)#.to(x.device)
            x_0 = generator(x, t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
    return x

def data_consistency(x, us, mask, range_adj = True, reshape = True):
    mask=mask>0.5
    crop = transforms.CenterCrop((us.shape[-2],us.shape[-1]))
    pad_y = int ( ( x.shape[-2]-us.shape[-2])/2 )
    pad_x = int ( ( x.shape[-1]-us.shape[-1])/2 )
    pad = torch.nn.ZeroPad2d((pad_x, pad_x, pad_y, pad_y))
    x = x * 0.5 + 0.5
    if reshape:
        x = crop(x)
    x = fft2c(x) * ~mask + fft2c(us) * mask
    x = torch.abs(ifft2c(x))
    if reshape:
        x = pad(x)
    if range_adj:    
        x = ( x - 0.5 ) / 0.5    
    return x

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

def data_consistency_loss(x, us, mask):
    mask=mask>0.5
    crop = transforms.CenterCrop((us.shape[-2],us.shape[-1]))
    x = x * 0.5 + 0.5
    if x.shape[-1] != us.shape[-1]:
        x = crop(x)
        x_fft = fft2c(x) * mask
        us_fft =  fft2c(us) * mask                
        loss = F.l1_loss(x_fft, us_fft)
    return loss

def load_checkpiont(checkpoint_dir, netG, device, trained = True, epoch_sel = False, epoch = 500):
    if epoch_sel:
        checkpoint_file = checkpoint_dir.format(args.dataset, args.exp, epoch)  
        checkpoint = torch.load(checkpoint_file, map_location=device)        
        ckpt = checkpoint 
    else:    
        checkpoint_file = checkpoint_dir.format(args.dataset, args.exp)  
        checkpoint = torch.load(checkpoint_file, map_location=device)               
        ckpt = checkpoint['netG_dict']
    if trained:
        for key in list(ckpt.keys()):       
            ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()    

def sample_and_test(args):
    torch.manual_seed(42)
    gpu = args.local_rank
    device = torch.device('cuda:{}'.format(gpu))
    to_range_0_1 = lambda x: (x + 1.) / 2.
    div_max = lambda x: x/x.max()
    div_mean = lambda x: x/x.mean()
    #loading dataset
    phase=args.phase
    dataset=CreateDatasetReconstruction(phase = phase, contrast = args.contrast , data = args.which_data, R = args.R)
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size,
                                               shuffle=args.shuffle,
                                               num_workers=4)
    #Initializing and loading network
    netG = NCSNpp(args).to(device)
    #if the networrk is pre-trained
    if args.exp !='DIP':
        #if a specific epoch is selected
        if args.epoch_sel:
            checkpoint_file = '../diffusion_test/saved_info/dd_gan/{}/{}/netG_{}.pth'
            load_checkpiont(checkpoint_file, netG, device = device, epoch_sel = True, epoch = args.epoch_id)
        #if the latest  
        else:
            checkpoint_file = '../diffusion_test/saved_info/dd_gan/{}/{}/content.pth'
            load_checkpiont(checkpoint_file, netG, device = device)
    #if the network is untrained, this part initilizes a random network and saves it
    else:
        parent_dir = "../diffusion_test/saved_info/dd_gan/{}".format(args.dataset)
        exp_path = os.path.join(parent_dir,args.exp)               
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)        
        content = {'netG_dict': netG.state_dict()}    
        torch.save(content, os.path.join(exp_path, args.contrast+'_content.pth'))   
        checkpoint_file = '../diffusion_test/saved_info/dd_gan/{}/{}/'+args.contrast+'_content.pth'

    #define optimizer for adaptation
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    #select a learning schedule
    if args.lr_schedule:
        if args.schedule=='cosine_anneal':
            schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.itr_inf, eta_min=1e-5)
        elif args.schedule=='onecycle':
            schedulerG = torch.optim.lr_scheduler.OneCycleLR(optimizerG, max_lr = args.lr_g , total_steps = args.itr_inf)
    T = get_time_schedule(args, device)
    #load coefficients of the diffusion model
    pos_coeff = Posterior_Coefficients(args, device)
    #saving directoy     
    save_dir = "../diffusion_test/saved_info/dd_gan/{}/{}/".format(args.dataset,  args.exp)
    #if the path doesnt exist create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #if weights are shared for slices within a subject
    save_dir = save_dir + args.extra_string 
    print(save_dir)
    #definve variables to save psnr values, and recosntructions
    loss = np.zeros((len(data_loader),args.itr_inf))
    shape = data_loader.dataset[0][1].shape
    recons = np.zeros((len(data_loader), shape[-2],shape[-1]))
    #if intermediate recsontructions (during adaptation) also needed to be saved
    if args.save_inter:
        recons_inter = np.zeros((int(args.itr_inf/100+1), len(data_loader), shape[-2],shape[-1]), dtype = np.float32)

    for iteration, (fs, us, mask) in enumerate(data_loader):
        if iteration == 21:
            break
        #make us complex
        us = us[:,[0],:]*np.exp(1j*(us[:,[1],:]*2*np.pi-np.pi))
        #move variables to device          
        us = us.to(device)
        fs = fs.to(device)
        mask = mask.to(device)
        #set cropping window, this is needed to crop to the original dimentions. loaded data are 256x256 which doesnt correspond to the original dimentions
        crop = transforms.CenterCrop((us.shape[-2],us.shape[-1]))
        fs = crop (fs)
        x_t_1 = torch.randn(args.batch_size, args.num_channels,args.image_size, args.image_size).to(device)
        # a - Diffusion steps
        fake_sample_diff = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, fs, us, mask, T,  args)
        # b - Optimization
        t_time = torch.zeros([1] , device=device)
        latent_z = torch.randn(1, args.nz, device=device)#.to(x.device)       
        for ii in range(args.itr_inf):
            #set gradient to zero
            netG.zero_grad()    
            #generate recon
            fake_sample = netG(fake_sample_diff, t_time, latent_z)
            #define DC loss
            lossDC =  data_consistency_loss(fake_sample, us, mask)
            #apply data consistency
            fake_sample =  crop(data_consistency(fake_sample, us, mask))
            fake_sample [fs==-1]= -1
            loss[iteration, ii] = psnr(div_mean(to_range_0_1(crop(fake_sample))), div_mean(to_range_0_1(fs)))
            #backward pass
            lossDC.backward() 
            #take step
            optimizerG.step()
            #if learning rate schedule is set
            if args.lr_schedule:
                schedulerG.step()
            fake_sample = fake_sample.detach()
            #save intermediate reconstruction every 100 steps
            if ii % 100 == 0 and args.save_inter:
                recons_inter[int(ii/100), iteration, :] = np.squeeze(to_range_0_1(crop(fake_sample)).cpu().numpy())
        #save final reconstruction        
        recons[iteration, :] = np.squeeze(to_range_0_1(crop(fake_sample)).cpu().numpy())
        np.save('{}{}_{}_{}_recons_{}_final.npy'.format(save_dir, args.contrast, phase, args.R, args.itr_inf), recons)    
        #save intermediate recons             
        if args.save_inter:
            np.save('{}{}_{}_{}_recons_{}_inter.npy'.format(save_dir, args.contrast, phase, args.R, args.itr_inf), recons_inter)   
        np.save('{}{}_{}_{}_psnr_{}_final.npy'.format(save_dir, args.contrast, phase, args.R, args.itr_inf), loss)      
        #if optimizer should be reset after every slices
        if args.reset_opt:            
            optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))        
            if args.lr_schedule:
                if args.schedule=='cosine_anneal':
                    print(args.schedule)
                    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.itr_inf, eta_min=1e-5)
                elif args.schedule=='onecycle':
                    print(args.schedule)
                    schedulerG = torch.optim.lr_scheduler.OneCycleLR(optimizerG, max_lr = args.lr_g , total_steps = args.itr_inf)

        print('PSNR {}'.format(psnr(div_mean(to_range_0_1(crop(fake_sample))), div_mean(to_range_0_1(fs)))))
        fake_sample =  data_consistency(fake_sample, us, mask)
        print('PSNR - DC {}'.format(psnr(div_mean(to_range_0_1(crop(fake_sample))), div_mean(to_range_0_1(fs)))))
        print('Iteration - {}'.format(iteration))
        #makes range 0 to 1
        fake_sample = to_range_0_1(fake_sample)        
        fs = to_range_0_1(fs)
        fake_sample = crop(fake_sample)
        us=torch.abs(us) 
        fake_sample = torch.cat((us,fake_sample,fs),axis=-1)
        torchvision.utils.save_image(fake_sample, '{}{}_{}_{}_samples_{}.jpg'.format(save_dir, args.contrast, phase, args.R, iteration), normalize=True)
        if args.exp =='DIP':
            load_checkpiont(checkpoint_file, netG, device = device, trained = False)
        else:
            if args.epoch_sel:
                load_checkpiont(checkpoint_file, netG, device = device, epoch_sel = True, epoch = args.epoch_id)
                print('Epoch select')
            else:
                load_checkpiont(checkpoint_file, netG, device = device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('adadiff parameters')
    parser = add_singlecoil_inference_args(parser)
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    sample_and_test(args)
