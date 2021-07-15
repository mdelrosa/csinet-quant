import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
# import torchvision
# from torchvision import transforms
from collections import OrderedDict

import numpy as np
# import matplotlib.pyplot as plt
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
    def __init__(self, encoder, decoder, quant, latent_dim, batch_size=200, device=None, return_latent=False, quant_bool=False):
        super(CsiNetQuant, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quant = quant
        self.latent_dim = latent_dim
        self.device = device
        self.quant_bool = quant_bool
        self.training = True
        self.return_latent = return_latent
        # self.z_min_temp = torch.Tensor(1e6, requires_grad=False)
        # self.z_max_temp = torch.Tensor(-1e6, requires_grad=False)

    def make_p_hist(self):
        """ make histogram of unique levels in latent layer """
        self.p_hist = torch.zeros(self.quant.n_levels)
        self.p_vals = torch.arange(-1.0, 1.0, self.quant.res)

    def forward(self, H_in):
        """forward call for VAE"""
        h_enc = self.encoder(H_in)
        with torch.no_grad():
            self.z_max_temp = torch.max(h_enc)
            self.z_min_temp = torch.min(h_enc)
        if self.quant_bool:
            if self.return_latent:
                return self.quant(h_enc)
            else:
                return self.decoder(self.quant(h_enc))
        else:
            return self.decoder(h_enc)

    def crossentropy_loss(self, H_in, clip_val=1e-9):
        """ calculate entropy using p_hist """
        # temporarily store current quant_bool
        quant_mode_temp = self.quant.quant_mode
        self.quant.quant_mode = 2 # get softmax outputs
        z_quant = self.quant(self.encoder(H_in))
        b = z_quant.shape[0]

        # parallel impl
        self.p_hist = torch.true_divide(torch.sum(torch.sum(self.p_mask[:b, :] == H_idx.unsqueeze(1).repeat(1,self.quant.L,1), axis=0), axis=1), self.quant.n_features*b)
        p_hard = self.p_hist.view(1,1,self.quant.L).repeat(b,self.quant.n_features,1)
        p_hard = torch.clamp(p_hard, clip_val, 1.0)
        # store back original quant_mode, sigma values
        self.quant.quant_mode = quant_mode_temp
        entropy_loss = -torch.sum(q_soft * torch.log2(p_hard)) / (self.quant.n_features*b)
        return entropy_loss
    
class UniformQuantize(torch.nn.Module):
    def __init__(self, bits=16, device="cpu"):
        """
        Uniform quantization over minmax of encoder output
        """
        super(UniformQuantize, self).__init__()
        self.bits = bits
        self.device = device
        self.n_levels = 2**(bits) - 1
        self.res = 1/(self.n_levels)

        # verify size of histogram, values
        self.reset_hist()
        self.p_vals = torch.arange(0.0, 1.0+self.res, self.res).to(device)
        self.p_ones = torch.ones(self.p_vals.size(0)).to(device)

        # return normalized vs. denormalized latents
        self.return_denorm = True

    def reset_extrema(self, z_min, z_max):
        self.z_min = z_min
        self.z_max = z_max

    def reset_hist(self):
        """ set histogram counts to zero """
        self.p_hist = torch.zeros(self.n_levels+1).to(self.device)

    def update_hist(self, x):
        """ take quantized tensor (x), update counts in self.p_hist """
        with torch.no_grad():
            vals, counts = x.unique(return_counts=True)
            for i, val in enumerate(vals):
                temp = torch.isclose(self.p_vals, self.p_ones*val)
                idx = temp.nonzero(as_tuple=False).flatten()
                if idx.size(0) > 0:
                    j = idx[0]
                    self.p_hist[j] += counts[i]
                else:
                    print(f"val={val} not in p_vals")
    
    def get_entropy(self):
        """ return entropy based on hist"""
        M = torch.sum(self.p_hist) # total num of elements
        entropy = 0
        for p in self.p_hist:
            proba = p / M
            entropy -= proba*torch.log2(proba) if proba > 0 else 0
        return entropy
        
    def forward(self, x):
        """
        Uniform quantization on latent features 
        """
        x_scaled = (x - self.z_min) / (self.z_max - self.z_min) # scale to [0,1]
        x_quant = torch.fake_quantize_per_tensor_affine(x_scaled, self.res, 0, 0, self.n_levels) # quantize
        if not self.return_denorm:
            return x_quant
        x_denorm = x_quant * (self.z_max - self.z_min) + self.z_min # denormalize
        return x_denorm

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
    sys.path.append("/home/mdelrosa/git/brat")
    from utils.NMSE_performance import renorm_H4, renorm_sphH4
    from utils.data_tools import dataset_pipeline_col, subsample_batches
    from utils.parsing import str2bool
    from utils.timing import Timer
    from utils.unpack_json import get_keys_from_json
    from utils.trainer import save_predictions, save_checkpoint_history
    from trainer import fit, score, profile_network, update_histogram

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
    parser.add_argument("-p1", "--pretrain1_bool", type=str2bool, default=False, help="bool for performing pretrain stage 1 (autoencoder with no latent quantization)")
    parser.add_argument("-tr", "--train_bool", type=str2bool, default=True, help="flag for toggling training for soft-to-hard vector quantization")
    parser.add_argument("-lo", "--load_bool", type=str2bool, default=True, help="flag for toggling loading of soft-to-hard vector quantized model")
    parser.add_argument("-nb", "--n_batch", type=int, default=20, help="number of batches to fit on (ignored during debug mode)")
    parser.add_argument("-l", "--dir", type=str, default=None, help="subdirectory for saving model, checkpoint, history")
    parser.add_argument("-tl", "--tail_dir", type=str, default=None, help="subdirectory for saving model, checkpoint, history of SHVQ network")
    parser.add_argument("-e", "--env", type=str, default="outdoor", help="environment (either indoor or outdoor)")
    parser.add_argument("-ep", "--epochs", type=int, default=1000, help="number of epochs to train for")
    parser.add_argument("-ef", "--epochs_finetune", type=int, default=50, help="number of epochs to use for shvq finetuning")
    parser.add_argument("-sp", "--split", type=int, default=0, help="split of entire dataset. must be less than int(<total_num_files> / <n_batch>).")
    parser.add_argument("-t", "--n_truncate", type=int, default=32, help="value to truncate to along delay axis.")
    parser.add_argument("-r", "--rate", type=int, default=512, help="number of elements in latent code (i.e., encoding rate)")
    parser.add_argument("-dt", "--data_type", type=str, default="norm_sphH4", help="type of dataset to train on (norm_H4, norm_sphH4)")
    opt = parser.parse_args()

    device = torch.device(f'cuda:{opt.gpu_num}' if torch.cuda.is_available() else 'cpu')
    print(f"--- Device is {device} ---")

    # dataset pipeline vars 
    if opt.data_type == "norm_H4":
        json_config = "../config/csinet-pro-indoor0001.json" if opt.env == "indoor" else "../config/csinet-pro-outdoor300.json"
    elif opt.data_type == "norm_sphH4":
        json_config = "../config/csinet-pro-indoor0001-sph-pow.json" if opt.env == "indoor" else "../config/csinet-pro-outdoor300-sph-pow.json"
        # json_config = "../config/csinet-pro-quadriga-indoor0001-sph.json" if opt.env == "indoor" else "../config/csinet-pro-quadriga-outdoor300-sph.json"
    dataset_spec, minmax_file, img_channels, data_format, norm_range, T, network_name, base_pickle, n_delay, total_num_files, t1_power_file, subsample_prop, thresh_idx_path, diff_spec, init_sigma = get_keys_from_json(json_config, keys=["dataset_spec", "minmax_file", "img_channels", "data_format", "norm_range", "T", "network_name", "base_pickle", "n_delay", "total_num_files", "t1_power_file", "subsample_prop", "thresh_idx_path", "diff_spec", "init_sigma"])
    # aux_bool_list = get_keys_from_json(json_config, keys=["aux_bool"], is_bool=True)

    input_dim = (2,32,n_delay)

    batch_num = 1 if opt.debug_flag else opt.n_batch # dataset batches
    M_1 = None # legacy holdover from CsiNet-LSTM
    aux_bool = False
    aux_size = 0
    # aux_bool = aux_bool_list[0] # dumb, but get_keys_from_json returns list

    batch_size, learning_rate = get_keys_from_json(json_config, keys=["batch_size", "learning_rate"])

    # load all data splits
    # data_train, data_val, data_test = dataset_pipeline(batch_num, opt.debug_flag, aux_bool, dataset_spec, M_1, T = T, img_channels = img_channels, img_height = input_dim[1], img_width = input_dim[2], data_format = data_format, idx_split=opt.split, n_truncate=opt.n_truncate, total_num_files=total_num_files+1)
    pow_diff, data_train, data_val = dataset_pipeline_col(opt.debug_flag, aux_bool, dataset_spec, diff_spec, aux_size, T = T, img_channels = input_dim[0], img_height = input_dim[1], img_width = input_dim[2], data_format = data_format, train_argv = True, subsample_prop=subsample_prop, thresh_idx_path=thresh_idx_path)

    # handle renorm data
    print('-> pre-renorm: data_val range is from {} to {} -- data_val.shape = {}'.format(np.min(data_val),np.max(data_val),data_val.shape))
    data_all = np.concatenate((data_train, data_val), axis=0)
    n_train, n_val = data_train.shape[0], data_val.shape[0]
    if norm_range == "norm_H4":
        data_all = renorm_H4(data_all, minmax_file)
    elif norm_range == "norm_sphH4":
        data_all = renorm_sphH4(data_all, minmax_file, t1_power_file, batch_num, thresh_idx_path=thresh_idx_path).astype(np.float32)
    data_train, data_val = data_all[:n_train], data_all[n_train:]
    print('-> post-renorm: data_val range is from {} to {} -- data_val.shape = {}'.format(np.min(data_val),np.max(data_val),data_val.shape))

    if opt.dir != None:
        base_pickle += "/" + opt.dir

    # cr_list = [512, 256, 128, 64, 32] # rates for different compression ratios
    bits = 8
    cr_list = [opt.rate]
    for cr in cr_list:
        train_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_train).to(device), batch_size=batch_size, shuffle=True) 
        valid_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_val).to(device), batch_size=batch_size)

        encoder = Encoder(input_dim[0], input_dim[1], opt.n_truncate, cr)
        decoder = Decoder(input_dim[0], input_dim[1], opt.n_truncate, cr)
        quant = UniformQuantize(bits, device=device)

        csinet_quant = CsiNetQuant(encoder, decoder, quant, cr, device=device).to(device)

        pickle_dir = f"{base_pickle}/cr{cr}/t1"

        # csinet_quant.quant.quant_mode = 0 # pretrain with no latent quantization
        csinet_quant.quant_bool = False # train with quantization layer
        epochs = 1 if opt.debug_flag else opt.epochs # epochs for intial, non-quantized network performance
        if opt.pretrain1_bool:
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
        else:
            strs = ["checkpoint", "history"]
            load_dict = {}
            for str_i in strs:
                with open(f"{pickle_dir}/{network_name}-pretrain-{str_i}.pkl", "rb") as f:
                    load_dict[str_i] = pickle.load(f)
                    f.close()
            checkpoint, history = load_dict["checkpoint"], load_dict["history"]
            # optim_state_dict = torch.load(f"{pickle_dir}/{network_name}-pretrain-optimizer.pt")
            # for key, val in optim_state_dict.items():
            #     # print(f"{key}: {val.shape}")
            #     print(f"{key}: {type(val)}")
            optimizer = optim.Adam(csinet_quant.parameters(), lr=learning_rate)
        print(f"Loading best model weights from {pickle_dir}/{network_name}-pretrain-best-model.pt")
        csinet_quant.load_state_dict(torch.load(f"{pickle_dir}/{network_name}-pretrain-best-model.pt", map_location=device), strict=False)
        csinet_quant.quant.reset_extrema(checkpoint["best_z_min"], checkpoint["best_z_max"])

        del train_ldr, valid_ldr
        all_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_all).to(device), batch_size=batch_size)
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

        if not opt.debug_flag and opt.pretrain1_bool:                
            save_checkpoint_history(checkpoint, history, optimizer, dir=pickle_dir, network_name=f"{network_name}-pretrain")
        
        del all_ldr
        torch.cuda.empty_cache()

        if opt.tail_dir != None:
            pickle_dir += f"/{opt.tail_dir}" 
            print(f"--- pickle_dir with tail_dir: {pickle_dir} ---")

        # make histogram, bin edges
        # csinet_quant.make_p_hist()

        train_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_train).to(device), batch_size=batch_size, shuffle=True) 
        valid_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_val).to(device), batch_size=batch_size)
        # all_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_all).to(device), batch_size=batch_size)

        csinet_quant.quant_bool = True
        epochs = 1 if opt.debug_flag else opt.epochs_finetune # epochs for intial, non-quantized network performance
        if opt.train_bool:
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
                                                                quant_bool=True,
                                                                anneal_bool=True,
                                                                data_all=data_all 
                                                                )
            # csinet_quant.quant.sigma = checkpoint["best_sigma"]
            csinet_quant.load_state_dict(checkpoint["best_model"])
        elif opt.load_bool:
            model_weights_name = f"{pickle_dir}/{network_name}-best-model.pt"
            print(f"---- Loading best model from {model_weights_name} ---")
            csinet_quant.load_state_dict(torch.load(model_weights_name, map_location=device))
            csinet_quant.quant.quant_mode = 1 # train with quantization layer
            # checkpoint_name = f"{pickle_dir}/{network_name}-checkpoint.pkl"
            # with open(checkpoint_name, 'rb') as f:
            #     checkpoint = pickle.load(f)
            #     f.close()
            # csinet_quant.quant.sigma = checkpoint["best_sigma"].to(device)
            checkpoint = {}
        else:
            print(f"---- Model performance without soft-to-hard vector quantization ---")
            checkpoint = {"best_sigma": csinet_quant.quant.sigma, "latest_model": csinet_quant}
            history = {}
            optimizer = torch.optim.Adam(csinet_quant.parameters())

        del train_ldr, valid_ldr
        torch.cuda.empty_cache()
        all_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_all).to(device), batch_size=batch_size)
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
                                            str_mod=f"CsiNetQuant CR={cr} (uniform quantization)",
                                            n_train=data_train.shape[0],
                                            pow_diff_t=pow_diff,
                                            quant_bool=True
                                            )
        # history["init_nmse"] = init_nmse

        del all_ldr
        torch.cuda.empty_cache()

        if not opt.debug_flag and opt.train_bool:                
            save_checkpoint_history(checkpoint, history, optimizer, dir=pickle_dir, network_name=network_name)

        valid_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_val).to(device), batch_size=batch_size)
        csinet_quant.return_latent = True
        csinet_quant.quant.return_denorm = False
        update_histogram(csinet_quant,
                         valid_ldr,
                        #  data_all,
                         batch_num,
                         timers=timers,
                         json_config=json_config,
                         debug_flag=opt.debug_flag,
                         str_mod=f"CsiNetQuant CR={cr} (hard quantization)",
                         )

        print(f"-> Calculate entropy based on validation set <-")
        entropy = csinet_quant.quant.get_entropy()
        print(f"-> validation entropy ({opt.epochs_finetune} epochs): {entropy:4.3f}")

        del valid_ldr
        torch.cuda.empty_cache()