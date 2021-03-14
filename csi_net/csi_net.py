import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

class Encoder(torch.nn.Module):
    """ encoder for CsiNet-Pro """
    def __init__(self, n_chan, H, W, latent_dim):
        super(Encoder, self).__init__()
        self.img_total = H*W
        self.n_chan = n_chan
        self.latent_dim = latent_dim
        self.enc_conv1 = nn.Conv2d(2, 16, 7, padding=3)
        self.bn_1 = nn.BatchNorm2d(16)
        self.enc_conv2 = nn.Conv2d(16, 8, 7, padding=3)
        self.bn_2 = nn.BatchNorm2d(8)
        self.enc_conv3 = nn.Conv2d(8, 4, 7, padding=3)
        self.bn_3 = nn.BatchNorm2d(4)
        self.enc_conv4 = nn.Conv2d(4, 2, 7, padding=3)
        self.bn_4 = nn.BatchNorm2d(2)
        self.enc_dense = nn.Linear(H*W*n_chan, latent_dim)

        # TODO: try different activation functions here (i.e., swish)
        self.activ = nn.LeakyReLU(0.1) # TODO: make sure slope matches TF slope

    def forward(self, x):
        x = self.activ(self.bn_1(self.enc_conv1(x)))
        x = self.activ(self.bn_2(self.enc_conv2(x)))
        x = self.activ(self.bn_3(self.enc_conv3(x)))
        x = self.activ(self.bn_4(self.enc_conv4(x)))
        x = torch.reshape(x, (x.size(0), -1,)) # TODO: verify -- does this return num samples in both channels?
        x = self.enc_dense(x)
        return x

class Decoder(torch.nn.Module):
    """ decoder for CsiNet-Pro """
    def __init__(self, n_chan, H, W, latent_dim):
        super(Decoder, self).__init__()
        self.H = H
        self.W = W
        self.img_total = H*W
        self.n_chan = n_chan
        self.dec_dense = nn.Linear(latent_dim, self.img_total*self.n_chan)
        self.dec_conv1 = nn.Conv2d(2, 16, 7, padding=3)
        self.bn_1 = nn.BatchNorm2d(16)
        self.dec_conv2 = nn.Conv2d(16, 8, 7, padding=3)
        self.bn_2 = nn.BatchNorm2d(8)
        self.dec_conv3 = nn.Conv2d(8, 4, 7, padding=3)
        self.bn_3 = nn.BatchNorm2d(4)
        self.dec_conv4 = nn.Conv2d(4, 2, 7, padding=3)

        self.activ = nn.LeakyReLU(0.1) # TODO: make sure slope matches TF slope
        self.out_activ = nn.Tanh()

    def forward(self, x):
        x = self.dec_dense(x)
        x = torch.reshape(x, (x.size(0), self.n_chan, self.H, self.W))
        x = self.activ(self.bn_1(self.dec_conv1(x)))
        x = self.activ(self.bn_2(self.dec_conv2(x)))
        x = self.activ(self.bn_3(self.dec_conv3(x)))
        return self.out_activ(self.dec_conv4(x))

class CsiNetQuant(nn.Module):
    """ CsiNet-Quant for csi estimation with entropy-based loss term """
    def __init__(self, encoder, decoder, quant, latent_dim, device=None, hard_sigma=1e6):
        super(CsiNetQuant, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quant = quant
        self.latent_dim = latent_dim
        self.device = device
        self.quant_bool = True
        self.training = True
        self.p_hist = torch.zeros(self.quant.L).to(self.device) # histogram estimate of probabilities
        # vals for annealing sigma

        # self.hard_sigma = OrderedDict({'sigma': torch.Tensor([hard_sigma]).to(device)})
        # self.hard_sigma = torch.Tensor([hard_sigma]).to(device)
        # self.hard_sigma = torch.nn.Parameter(torch.Tensor([hard_sigma]).to(device), requires_grad=False)
        self.hard_sigma = hard_sigma

        self.T_sigma = 11719 # timescale of annealing (batches)
        self.K_sigma = 10 # gain of annealing
        self.t = 0 # index of current iteration for annealing
        # self.p_hard = torch.zeros(b,self.quant.n_features,self.quant.L).to(self.device)

    def forward(self, H_in):
        """forward call for VAE"""
        h_enc = self.encoder(H_in)
        return self.decoder(self.quant(h_enc)) if self.quant_bool else self.decoder(h_enc)

    def crossentropy_loss(self, H_in, clip_val=1e-9):
        """ calculate crossentropy between soft/hard assignment probas """
        # temporarily store current quant_bool
        quant_mode_temp = self.quant.quant_mode
        self.quant.quant_mode = 2 # get softmax outputs
        q_soft = self.quant(self.encoder(H_in))
        b = q_soft.shape[0]
        H_idx = torch.argmax(q_soft, dim=2)
        for val in range(self.quant.L):
            num_val = torch.sum(H_idx == val)
            self.p_hist[val] = torch.true_divide(num_val, self.quant.n_features*b)
        p_hard = self.p_hist.view(1,1,self.quant.L).repeat(b,self.quant.n_features,1)
        p_hard = torch.clamp(p_hard, clip_val, 1.0)
        # store back original quant_mode, sigma values
        self.quant.quant_mode = quant_mode_temp
        entropy_loss = -torch.sum(q_soft * torch.log2(p_hard)) / (self.quant.n_features*b)
        return entropy_loss
    
    def calc_gap(self, mse_soft, H_hat_soft, H_in, hard_sigma=1e6, criterion=nn.MSELoss()):
        """
        running total of gap between soft/hard crossentropy
        H_hat_soft : soft quantization estimate from outer loop
        H_in : autoencoder input
        """
        # mse_soft = torch.mean(torch.pow(H_hat_soft - H_in, 2)) # assume we can pass this in
        # make sigma large -> hard quantization

        # sigma_temp = OrderedDict({'sigma': self.quant.state_dict()['sigma']})
        # sigma_temp = self.quant.sigma.data
        # sigma_temp = torch.nn.Parameter(self.quant.sigma.data, requires_grad=False)
        sigma_temp = self.quant.sigma

        # self.quant.load_state_dict(self.hard_sigma, strict=False)
        # self.quant.sigma.data = self.hard_sigma.data
        self.quant.sigma = self.hard_sigma

        H_hat_hard = self.forward(H_in)
        mse_hard = criterion(H_hat_hard, H_in)

        # self.quant.load_state_dict(sigma_temp, strict=False)
        # self.quant.sigma.data = sigma_temp
        self.quant.sigma = sigma_temp

        return mse_soft - mse_hard

    def update_gap(self, mse_soft, H_hat_soft, H_in, hard_sigma=1e6, criterion=nn.MSELoss()):
        self.gap_t = self.calc_gap(mse_soft, H_hat_soft, H_in, hard_sigma=hard_sigma, criterion=criterion)
        if self.t == 0:
            self.gap_0 = self.gap_t

    def anneal_sigma(self):
        self.e_g = self.gap_t + self.T_sigma / (self.T_sigma + self.t) * self.gap_0
# 
        # self.quant.load_state_dict(OrderedDict({'sigma':torch.add(self.quant.sigma,+ self.K_sigma*self.gap_t)}), strict=False)
        # self.quant.sigma.data += self.K_sigma*self.gap_t
        # self.quant.sigma = torch.nn.Parameter(torch.Tensor([self.quant.sigma.data + self.K_sigma*self.gap_t]).to(device), requires_grad=False)
        print(f"--- anneal_sigma pre-update - sigma={self.quant.sigma} ---")
        self.quant.sigma += self.K_sigma*self.gap_t
        self.quant.sigma = np.max([self.quant.sigma_eps, self.quant.sigma])
        print(f"--- anneal_sigma post-update - sigma={self.quant.sigma} ---")

        self.t += 1

class SoftQuantize(torch.nn.Module):
    def __init__(self, r, L, m, sigma=5, sigma_trainable=True):
        """
        Soft quantization layer based on Voronoi tesselation centers
        sigma = temperature for softmax
        r = dimensionality of autoencoder's latent features
        L = number of cluster centers
        m = dimensionality of cluster centers
        """
        super(SoftQuantize, self).__init__()
        self.sigma_trainable = sigma_trainable
        self.sigma = torch.nn.Parameter(data=torch.Tensor([sigma]), requires_grad=True) if sigma_trainable else sigma
        self.r = r
        self.L = L # num centers
        self.m = m # dim of centers
        self.n_features = int(r/m)
        self.softmax = nn.Softmax(dim=2)
        self.sigma_eps = 1e-4
        self.quant_mode = 0 # 0 = pass through (no quantization), 1 = soft quantization, 2 = return softmax outputs
        
    def init_centers(self, c):
        self.c = torch.nn.Parameter(data=c, requires_grad=True)
        self.quant_mode = 1

    def forward(self, x):
        """
        Take in latent features z, return soft assignment to clusters
        """
        b = x.shape[0]
        if self.quant_mode == 0:
            return x # no quantization in latent layer
        else:
            z = x.view(b, self.n_features, self.m, 1).repeat(1, 1, 1, self.L)
            c = self.c.view(1,1,self.m,self.L).repeat(b, self.n_features, 1, 1)
            sigma = self.sigma.relu() + self.sigma_eps if self.sigma_trainable else self.sigma
            # print(f"--- sigma: {sigma} ---")
            q = self.softmax(-sigma*torch.sum(torch.pow(z - c, 2), 2))
            c = self.c.view(1,self.m,self.L).repeat(b,1,1).transpose(2,1)
            out = torch.matmul(q, c)
            if self.quant_mode == 1:
                return out.view(b, self.r) # soft quantized outputs
            elif self.quant_mode == 2:
                return q # softmax outputs

def energy_loss(y_gt, y_hat):
    return torch.mean(torch.sum(torch.pow(y_gt - y_hat, 2), dim=1))

class AnnealingSchedule(object):
    """
    anneal a parameter based on current epoch
    """
    def __init__(self, param_range, epoch_limit=200):
        self.range = param_range
        self.limit = epoch_limit

    def get_param(self, epoch):
        ratio = epoch / self.limit
        return (self.range[1] - self.range[0]) * ratio if ratio <= 1 else self.range[1]

if __name__ == "__main__":
    import argparse
    import pickle
    import copy
    import sys
    sys.path.append("/home/mason/git/brat")
    from utils.NMSE_performance import renorm_H4, renorm_sphH4
    from utils.data_tools import dataset_pipeline_col, subsample_batches
    from utils.parsing import str2bool
    from utils.timing import Timer
    from utils.unpack_json import get_keys_from_json
    from utils.trainer import save_predictions, save_checkpoint_history
    from trainer import fit, score, profile_network

    # set up timers
    timers = {
             "fit_timer": Timer("Fit"),              
             "predict_timer": Timer("Predict"),
             "score_timer": Timer("Score")
             }

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug_flag", type=int, default=0, help="flag for toggling debugging mode")
    parser.add_argument("-g", "--gpu_num", type=int, default=0, help="number for torch device (cuda:gpu_num)")
    parser.add_argument("-p", "--pretrain_bool", type=str2bool, default=False, help="number for torch device (cuda:gpu_num)")
    parser.add_argument("-b", "--n_batch", type=int, default=20, help="number of batches to fit on (ignored during debug mode)")
    parser.add_argument("-l", "--dir", type=str, default=None, help="subdirectory for saving model, checkpoint, history")
    parser.add_argument("-e", "--env", type=str, default="indoor", help="environment (either indoor or outdoor)")
    parser.add_argument("-ep", "--epochs", type=int, default=10, help="number of epochs to train for")
    parser.add_argument("-tr", "--train_argv", type=str2bool, default=True, help="flag for toggling training")
    parser.add_argument("-sp", "--split", type=int, default=0, help="split of entire dataset. must be less than int(<total_num_files> / <n_batch>).")
    parser.add_argument("-t", "--n_truncate", type=int, default=32, help="value to truncate to along delay axis.")
    parser.add_argument("-ts", "--timeslot", type=int, default=0, help="timeslot which we are training (0-indexed).")
    parser.add_argument("-r", "--rate", type=int, default=512, help="number of elements in latent code (i.e., encoding rate)")
    parser.add_argument("-dt", "--data_type", type=str, default="norm_H4", help="type of dataset to train on (norm_H4, norm_sphH4)")
    parser.add_argument("-L", "--num_centers", type=int, default=12, help="Number of cluster centers for vector quantization")
    parser.add_argument("-m", "--dim_centers", type=int, default=8, help="Dimensions for cluster centers for vector quantization")
    opt = parser.parse_args()

    device = torch.device(f'cuda:{opt.gpu_num}' if torch.cuda.is_available() else 'cpu')
    print(f"--- Device is {device} ---")

    # dataset pipeline vars 
    if opt.data_type == "norm_H4":
        json_config = "../config/csinet-pro-indoor0001.json" if opt.env == "indoor" else "../config/csinet-pro-outdoor300.json"
    elif opt.data_type == "norm_sphH4":
        json_config = "../config/csinet-pro-indoor0001-sph.json" if opt.env == "indoor" else "../config/csinet-pro-outdoor300-sph-pow.json"
        # json_config = "../config/csinet-pro-quadriga-indoor0001-sph.json" if opt.env == "indoor" else "../config/csinet-pro-quadriga-outdoor300-sph.json"
    dataset_spec, minmax_file, img_channels, data_format, norm_range, T, network_name, base_pickle, n_delay, total_num_files, t1_power_file, subsample_prop, thresh_idx_path, diff_spec, init_sigma = get_keys_from_json(json_config, keys=["dataset_spec", "minmax_file", "img_channels", "data_format", "norm_range", "T", "network_name", "base_pickle", "n_delay", "total_num_files", "t1_power_file", "subsample_prop", "thresh_idx_path", "diff_spec", "init_sigma"])
    # aux_bool_list = get_keys_from_json(json_config, keys=["aux_bool"], is_bool=True)

    input_dim = (2,32,n_delay)
    epochs = 10 if opt.debug_flag else opt.epochs

    batch_num = 1 if opt.debug_flag else opt.n_batch # dataset batches
    M_1 = None # legacy holdover from CsiNet-LSTM
    aux_bool = False
    aux_size = 0
    # aux_bool = aux_bool_list[0] # dumb, but get_keys_from_json returns list

    batch_size, learning_rate = get_keys_from_json(json_config, keys=["batch_size", "learning_rate"])

    # load all data splits
    # data_train, data_val, data_test = dataset_pipeline(batch_num, opt.debug_flag, aux_bool, dataset_spec, M_1, T = T, img_channels = img_channels, img_height = input_dim[1], img_width = input_dim[2], data_format = data_format, idx_split=opt.split, n_truncate=opt.n_truncate, total_num_files=total_num_files+1)
    pow_diff, data_train, data_val = dataset_pipeline_col(opt.debug_flag, aux_bool, dataset_spec, diff_spec, aux_size, T = T, img_channels = input_dim[0], img_height = input_dim[1], img_width = input_dim[2], data_format = data_format, train_argv = opt.train_argv, subsample_prop=subsample_prop, thresh_idx_path=thresh_idx_path)

    # handle renorm data
    print('-> pre-renorm: data_val range is from {} to {} -- data_val.shape = {}'.format(np.min(data_val),np.max(data_val),data_val.shape))
    data_all = np.concatenate((data_train, data_val), axis=0)
    print(f"data_all.dtype: {data_all.dtype}")
    n_train, n_val = data_train.shape[0], data_val.shape[0]
    if norm_range == "norm_H4":
        data_all = renorm_H4(data_all, minmax_file)
    elif norm_range == "norm_sphH4":
        data_all = renorm_sphH4(data_all, minmax_file, t1_power_file, batch_num).astype(np.float32)
    data_train, data_val = data_all[:n_train], data_all[n_train:]
    print('-> post-renorm: data_val range is from {} to {} -- data_val.shape = {}'.format(np.min(data_val),np.max(data_val),data_val.shape))
    print(f"data_all.dtype: {data_all.dtype}")

    if opt.dir != None:
        base_pickle += "/" + opt.dir

    # cr_list = [512, 256, 128, 64, 32] # rates for different compression ratios
    cr_list = [opt.rate]
    for cr in cr_list:
        train_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_train).to(device), batch_size=batch_size, shuffle=True) if opt.train_argv else None
        valid_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_val).to(device), batch_size=batch_size)
        all_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_all).to(device), batch_size=batch_size)

        encoder = Encoder(input_dim[0], input_dim[1], opt.n_truncate, cr)
        decoder = Decoder(input_dim[0], input_dim[1], opt.n_truncate, cr)
        quant = SoftQuantize(cr, opt.num_centers, opt.dim_centers, sigma_trainable=False)
        csinet_quant = CsiNetQuant(encoder, decoder, quant, cr, device=device).to(device)

        pickle_dir = f"{base_pickle}/cr{cr}/t1"

        if not opt.pretrain_bool:
            csinet_quant.quant.quant_mode = 0 # pretrain with no latent quantization
            if opt.train_argv:
                model, checkpoint, history, optimizer, timers = fit(csinet_quant,
                                                                    train_ldr,
                                                                    valid_ldr,
                                                                    batch_num,
                                                                    epochs=epochs,
                                                                    timers=timers,
                                                                    json_config=json_config,
                                                                    debug_flag=opt.debug_flag,
                                                                    pickle_dir=pickle_dir,
                                                                    network_name=f"{network_name}-pretrain")


            [checkpoint, y_hat, y_test] = score(csinet_quant,
                                                all_ldr,
                                                data_all,
                                                batch_num,
                                                checkpoint,
                                                history,
                                                optimizer,
                                                timers=timers,
                                                json_config=json_config,
                                                debug_flag=opt.debug_flag,
                                                str_mod=f"CsiNetQuant CR={cr} (pretrain {epochs} epochs)",
                                                n_train=data_train.shape[0],
                                                pow_diff_t=pow_diff
                                                )
            if not opt.debug_flag:                
                save_checkpoint_history(checkpoint, history, optimizer, dir=pickle_dir, network_name=f"{network_name}-pretrain")

        else:
            print(f"--- Loading pretrained state_dict files for CsiNet-Quant --- ")
            # csinet_quant.quant.init_centers(torch.zeros(csinet_quant.quant.L, csinet_quant.quant.m).to(device))
            # csinet_quant.quant.load_state_dict(torch.load(f"{pickle_dir}/soft-quant-best-model.pt"))
            csinet_quant.load_state_dict(torch.load(f"{pickle_dir}/{network_name}-pretrain-best-model.pt"), strict=False)

        # use encoder to get train/validation codewords
        with torch.no_grad():
            enc_train = csinet_quant.encoder(torch.from_numpy(data_train).to(device))
            enc_valid = csinet_quant.encoder(torch.from_numpy(data_val).to(device))
        enc_train_ldr = torch.utils.data.DataLoader(enc_train, batch_size=batch_size, shuffle=True)
        enc_valid_ldr = torch.utils.data.DataLoader(enc_valid, batch_size=batch_size)
        c = np.random.randint(enc_train.shape[0], size=opt.num_centers)
        sampled_centers = enc_train.view(enc_train.shape[0]*int(opt.rate / opt.dim_centers), opt.dim_centers)[c,:]
        print(f"--- sampled_centers.shape: {sampled_centers.shape} ---")
        csinet_quant.quant.init_centers(sampled_centers)

        fit(csinet_quant.quant,
            enc_train_ldr,
            enc_valid_ldr,
            batch_num,
            epochs=epochs,
            timers=timers,
            criterion=energy_loss,
            json_config=json_config,
            debug_flag=opt.debug_flag,
            pickle_dir=pickle_dir,
            network_name="soft-quant")

        del enc_train_ldr, enc_valid_ldr

        encoder = csinet_quant.encoder
        decoder = csinet_quant.decoder
        # sigma = csinet_quant.quant.sigma.relu().item() + csinet_quant.quant.sigma_eps
        quant = SoftQuantize(cr, opt.num_centers, opt.dim_centers, sigma=csinet_quant.quant.sigma, sigma_trainable=False)

        quant.init_centers(csinet_quant.quant.c.data)
        # quant.load_state_dict(OrderedDict({'c': }))
        csinet_quant = CsiNetQuant(encoder, decoder, quant, cr, device=device).to(device) # remake network with non-trainable sigma

        csinet_quant.quant.quant_mode = 1 # train with quantization layer
        print(f"-> sigma: {csinet_quant.quant.sigma}")
        print(f"-> centers: {csinet_quant.quant.c}")

        # profile_network(csinet_quant,
        #                 train_ldr,
        #                 valid_ldr,
        #                 batch_num,
        #                 epochs=epochs,
        #                 json_config=json_config,
        #                 quant_bool=True
        #                )

        model, checkpoint, history, optimizer, timers = fit(csinet_quant,
                                                            train_ldr,
                                                            valid_ldr,
                                                            batch_num,
                                                            epochs=epochs,
                                                            timers=timers,
                                                            json_config=json_config,
                                                            debug_flag=opt.debug_flag,
                                                            pickle_dir=pickle_dir,
                                                            network_name=network_name,
                                                            quant_bool=True)

        [checkpoint, y_hat, y_test] = score(csinet_quant,
                                            all_ldr,
                                            data_all,
                                            batch_num,
                                            checkpoint,
                                            history,
                                            optimizer,
                                            timers=timers,
                                            json_config=json_config,
                                            debug_flag=opt.debug_flag,
                                            str_mod=f"CsiNetQuant CR={cr} (latent quantization)",
                                            n_train=data_train.shape[0],
                                            pow_diff_t=pow_diff
                                            )

        if not opt.debug_flag:                
            save_checkpoint_history(checkpoint, history, optimizer, dir=pickle_dir, network_name=network_name)
