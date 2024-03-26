import argparse

def add_image_property_args(parser: argparse.ArgumentParser):
    # image properties
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--num_channels', type=int, default=3,
                        help='channels of image (3 for RGB, 1 for grayscale)')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='True if input data is [-1,1] scale, \
                        False if input data is [0,1] scale')
    return parser

def add_training_hyperparameter_args(parser: argparse.ArgumentParser):
        # training hyperparameters
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
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
    parser.add_argument('--attn_resolutions', default=(16,),
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
    parser.add_argument('--progressive', type=str, default='none',
                        choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual',
                        choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum',
                        choices=['sum', 'cat'],
                        help='progressive combine method.')
    parser.add_argument('--embedding_type', type=str, default='positional',
                        choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    return parser

def add_generator_hyperparameter_args(parser: argparse.ArgumentParser):
    parser.add_argument('--exp', default='experiment_cifar_default',
                        help='name of experiment')
    parser.add_argument('--dataset', default='brain',
                        choices=['cifar10', 'brain', 'brain_multi_coil'],
                        help='name of dataset')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    return parser

def add_singlecoil_train_args(parser: argparse.ArgumentParser):
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--resume', action='store_true',default=False)
    parser.add_argument('--num_epoch', type=int, default=500)

    parser = add_image_property_args(parser)
    parser = add_training_hyperparameter_args(parser)
    parser = add_generator_hyperparameter_args(parser)

    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--lr_g', type=float, default=1.6e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
    parser.add_argument('--no_lr_decay',action='store_true', default=False)
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--ema_decay', type=float, default=0.999, help='decay rate for EMA')
    parser.add_argument('--r1_gamma', type=float, default=1.0, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None, help='lazy regulariation')
    parser.add_argument('--save_content', action='store_true', default=False)
    parser.add_argument('--save_content_every', type=int, default=50,
                        help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=25,
                        help='save ckpt every x epochs')

    # DDP
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='number of nodes in multi-node env')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus per node')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='index of node. overriden if using DDP')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='index of gpu in a node. overriden if using DDP')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='IP address of master node')
    return parser

def add_singlecoil_inference_args(parser: argparse.ArgumentParser):
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False)
    parser.add_argument('--epoch_id', type=int, default=500) # in range [1, num_epoch]
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy', help='directory to real images for FID computation')

    parser = add_image_property_args(parser)
    parser = add_training_hyperparameter_args(parser)
    parser = add_generator_hyperparameter_args(parser)

    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    #optimizaer parameters    
    parser.add_argument('--lr_g', type=float, default=1.6e-4, help='learning rate g')
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
    parser.add_argument('--save_inter', type=bool, default=False,
                        help='setting true would save intermediate results after 100 iterations')      
    parser.add_argument('--R', type=int, default=4,
                        help='acceleration rate')  
    parser.add_argument('--extra_string', type=str, default='',
                        help='extra string for save_dir')      
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--reset_opt', type=bool, default=False)
    parser.add_argument('--lr_schedule', type=bool, default=False)
    parser.add_argument('--schedule', type=str, default='cosine_anneal')
    parser.add_argument('--epoch_sel', type=bool, default=False)
    parser.add_argument('--which_data', type=str, default='IXI',
                        help='which data to load from')     
    return parser
