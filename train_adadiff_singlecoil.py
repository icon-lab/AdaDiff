import os
import shutil
import argparse
import torchvision
import torch
import numpy as np

from datasets_prep.brain_datasets import CreateDataset
from utils.args_op import add_singlecoil_train_args
from utils.models.discriminator import Discriminator_large
from utils.models.ncsnpp_generator_adagn import NCSNpp
from utils.EMA import EMA

PARENT_BASE_DIR = "../diffusion_test/saved_info/dd_gan"
MASTER_PORT = '6025'

def set_seed(seed_no):
    torch.manual_seed(seed_no)
    torch.cuda.manual_seed(seed_no)
    torch.cuda.manual_seed_all(seed_no)
    return

def prepare_dataset(rank, args):
    """ Create dataset. Return train sampler, data loader, and args. """
    dataset = CreateDataset()
    train_sampler = torch.utils.data.distributed.DistributedSampler(
                        dataset, num_replicas=args.world_size, rank=rank)
    data_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=args.batch_size, sampler=train_sampler,
                        shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    return train_sampler, data_loader, args

def initialize_models(args, device):
    """Initialize the Generator and Discriminator models."""
    netG = NCSNpp(args).to(device)
    netD = Discriminator_large(nc=2*args.num_channels, ngf=args.ngf,
                               t_emb_dim=args.t_emb_dim, act=torch.nn.LeakyReLU(0.2)).to(device)
    netG = torch.nn.parallel.DistributedDataParallel(netG, device_ids=[device])
    netD = torch.nn.parallel.DistributedDataParallel(netD, device_ids=[device])
    return netG, netD

def setup_optim_schedulers(netG, netD, args):
    """Setup the optimizers and schedulers for both models."""
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_epoch,
                                                            eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_epoch,
                                                            eta_min=1e-5)
    return optimizerG, optimizerD, schedulerG, schedulerD

def setup_experiment_directory(exp_path):
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        copy_source(__file__, exp_path)
        shutil.copytree('utils/models', os.path.join(exp_path, 'utils/models'))

def initialize_training_components(args, device):
    """ Initialize the training components. """
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)
    return coeff, pos_coeff, T

def load_checkpoint(exp_path, device, netG, optimizerG, schedulerG, netD, optimizerD, schedulerD):
    checkpoint_file = os.path.join(exp_path, 'content.pth')
    checkpoint = torch.load(checkpoint_file, map_location=device)
    netG.load_state_dict(checkpoint['netG_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG'])
    schedulerG.load_state_dict(checkpoint['schedulerG'])
    netD.load_state_dict(checkpoint['netD_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD'])
    schedulerD.load_state_dict(checkpoint['schedulerD'])
    print(f"* loaded checkpoint (epoch {checkpoint['epoch']})")
    return checkpoint['epoch'], checkpoint['global_step']

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))
            
def broadcast_params(params):
    for param in params:
        torch.distributed.broadcast(param.data, src=0)

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

def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
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

#%%
def train(rank, gpu, args):
    set_seed(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))

    batch_size = args.batch_size
    nz = args.nz #latent dimension
    args.dataset = 'brain'

    # Dataset preparation
    train_sampler, data_loader, args = prepare_dataset(rank, args)

    # Model Initialization
    netG, netD = initialize_models(args, device)

    broadcast_params(netG.parameters())
    broadcast_params(netD.parameters())

    # Optimizer and Scheduler Setup
    optimizerG, optimizerD, schedulerG, schedulerD = setup_optim_schedulers(netG, netD, args)
    exp_path = os.path.join(PARENT_BASE_DIR, args.dataset, args.exp)
    if rank == 0:
        setup_experiment_directory(exp_path)

    coeff, pos_coeff, T = initialize_training_components(args, device)

    if args.resume:
        init_epoch, global_step = load_checkpoint(exp_path, device, netG, optimizerG,
                                                  schedulerG, netD, optimizerD, schedulerD)
    else:
        init_epoch, global_step = 0, 0

    # Main Training Loop
    for epoch in range(init_epoch, args.num_epoch+1):
        train_sampler.set_epoch(epoch)
        for iteration, (x, y) in enumerate(data_loader):
            for p in netD.parameters():  
                p.requires_grad = True
            netD.zero_grad()
            #sample from p(x_0)
            real_data = x.to(device, non_blocking=True)
            x_t_1 = torch.randn_like(real_data)
            #sample t
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True
            # train with real
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)
            errD_real = torch.nn.functional.softplus(-D_real)
            errD_real = errD_real.mean()
            errD_real.backward(retain_graph=True)
            if args.lazy_reg is None:
                grad_real = torch.autograd.grad(
                            outputs=D_real.sum(), inputs=x_t, create_graph=True
                            )[0]
                grad_penalty = (
                                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()
                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad_real = torch.autograd.grad(
                            outputs=D_real.sum(), inputs=x_t, create_graph=True
                            )[0]
                    grad_penalty = (
                                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()

                    grad_penalty = args.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()

            # train with fake
            latent_z = torch.randn(batch_size, nz, device=device)
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            errD_fake = torch.nn.functional.softplus(output)
            errD_fake = errD_fake.mean()
            errD_fake.backward()
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            #update G
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            latent_z = torch.randn(batch_size, nz,device=device)
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            errG = torch.nn.functional.softplus(-output)
            errG = errG.mean()
            errG.backward()
            optimizerG.step()
            global_step += 1
            if iteration % 100 == 0:
                if rank == 0:
                    print('epoch {} iteration{}, G Loss: {}, D Loss: {}'.format(epoch,iteration, errG.item(), errD.item()))
        
        if not args.no_lr_decay:
            schedulerG.step()
            schedulerD.step()

        if rank == 0:
            if epoch % 10 == 0:
                torchvision.utils.save_image(x_pos_sample, os.path.join(exp_path, 'xpos_epoch_{}.png'.format(epoch)), normalize=True)
            
            x_t_1 = torch.randn_like(real_data)
            fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, T, args)
            torchvision.utils.save_image(fake_sample, os.path.join(exp_path, 'sample_discrete_epoch_{}.png'.format(epoch)), normalize=True)
            
            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                               'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                               'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                               'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
                    
                    torch.save(content, os.path.join(exp_path, 'content.pth'))
                
            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                    
                torch.save(netG.state_dict(), os.path.join(exp_path, 'netG_{}.pth'.format(epoch)))
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)

def setup_distributed_environment(rank, size, master_addr, master_port):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    torch.distributed.init_process_group(backend='nccl', init_method='env://',
                                         rank=rank, world_size=size)

def cleanup_distributed_environment():
    """ Cleanup the distributed environment. """
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

def train_process(rank, size, args):
    """ The training process to be executed on each distributed node/GPU. """
    setup_distributed_environment(rank, size, args.master_address, MASTER_PORT)

    gpu = args.local_rank
    torch.cuda.set_device(gpu)

    print(f"Training on rank {rank}, GPU {gpu}")
    train(rank, gpu, args)

    cleanup_distributed_environment()

def start_processes(args):
    """ Start training processes, one per distributed node/GPU. """
    args.world_size = args.num_nodes * args.num_process_per_node
    size = args.num_process_per_node
    processes = []

    if size == 1:
        print('starting in debug mode')
        train_process(0, size, args)
    else:
        torch.multiprocessing.set_start_method('spawn')
        for rank in range(size):
            args.local_rank = rank
            args.global_rank = rank + args.node_rank * args.num_process_per_node
            p = torch.multiprocessing.Process(target=train_process,
                                              args=(args.global_rank, args.world_size, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('adadiff parameters')
    parser = add_singlecoil_train_args(parser)
    args = parser.parse_args()
    start_processes(args)
