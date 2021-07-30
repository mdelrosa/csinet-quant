import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from collections import OrderedDict

import sys
sys.path.append("/home/mdelrosa/git/brat")
from models.latent_quantizers import SoftQuantize, energy_loss

import itertools
import numpy as np
# import matplotlib.pyplot as plt
from tqdm import trange, tqdm

class Encoder(torch.nn.Module):
    """ encoder for CsiNet-Pro """
    def __init__(self, n_chan, H, W, latent_chan=256, sample_factors=[4,2,2]):
        super(Encoder, self).__init__()
        self.img_total = H*W
        self.n_chan = n_chan
        self.sample_factors = sample_factors
        self.latent_chan = latent_chan

        self.enc_conv1 = nn.Conv2d(2, latent_chan, 9, padding=4)
        self.enc_down1 = nn.Conv2d(latent_chan, latent_chan, 3, padding=1, stride=sample_factors[0])
        self.bn_1 = nn.BatchNorm2d(latent_chan)
        self.enc_conv2 = nn.Conv2d(latent_chan, latent_chan, 5, padding=2)
        self.enc_down2 = nn.Conv2d(latent_chan, latent_chan, 3, padding=1, stride=sample_factors[1])
        self.bn_2 = nn.BatchNorm2d(latent_chan)
        self.enc_conv3 = nn.Conv2d(latent_chan, latent_chan, 5, padding=2)
        self.enc_down3 = nn.Conv2d(latent_chan, latent_chan, 3, padding=1, stride=sample_factors[2])
        self.bn_3 = nn.BatchNorm2d(latent_chan)

        # TODO: try different activation functions here (i.e., swish)
        self.activ = nn.LeakyReLU(0.1) # TODO: make sure slope matches TF slope

    def forward(self, x):
        x = self.activ(self.bn_1(self.enc_down1(self.enc_conv1(x))))
        x = self.activ(self.bn_2(self.enc_down2(self.enc_conv2(x))))
        x = self.activ(self.bn_3(self.enc_down3(self.enc_conv3(x))))
        return x

class Decoder(torch.nn.Module):
    """ decoder for CsiNet-Pro """
    def __init__(self, n_chan, H, W, latent_chan=256, sample_factors=[2,2,4]):
        super(Decoder, self).__init__()
        self.H = H
        self.W = W
        self.img_total = H*W
        self.n_chan = n_chan
        self.latent_chan = latent_chan
        self.sample_factors = sample_factors
        H_latent = int(H / np.prod(sample_factors))
        W_latent = int(W / np.prod(sample_factors))
        self.H_factors = [H_latent * np.prod(sample_factors[:i]) for i in range(1,len(sample_factors)+1)]
        self.W_factors = [W_latent * np.prod(sample_factors[:i]) for i in range(1,len(sample_factors)+1)]

        self.dec_up1 = nn.ConvTranspose2d(latent_chan, latent_chan, 3, padding=1, stride=sample_factors[0])
        self.dec_conv1 = nn.Conv2d(latent_chan, latent_chan, 5, padding=2)
        self.bn_1 = nn.BatchNorm2d(latent_chan)

        # residual layers
        self.dec_res1_conv1 = nn.Conv2d(latent_chan, latent_chan, 5, padding=2)
        self.res1_bn1 = nn.BatchNorm2d(latent_chan)
        self.dec_res1_conv2 = nn.Conv2d(latent_chan, latent_chan, 5, padding=2)
        self.res1_bn2 = nn.BatchNorm2d(latent_chan)
        self.dec_res2_conv1 = nn.Conv2d(latent_chan, latent_chan, 5, padding=2)
        self.res2_bn1 = nn.BatchNorm2d(latent_chan)
        self.dec_res2_conv2 = nn.Conv2d(latent_chan, latent_chan, 5, padding=2)
        self.res2_bn2 = nn.BatchNorm2d(latent_chan)

        self.dec_up2 = nn.ConvTranspose2d(latent_chan, latent_chan, 3, padding=1, stride=sample_factors[1])
        self.dec_conv2 = nn.Conv2d(latent_chan, latent_chan, 5, padding=2)
        self.bn_2 = nn.BatchNorm2d(latent_chan)
        self.dec_up3 = nn.ConvTranspose2d(latent_chan, latent_chan, 3, padding=1, stride=sample_factors[2])
        self.dec_conv3 = nn.Conv2d(latent_chan, n_chan, 9, padding=4)

        self.activ = nn.LeakyReLU(0.1) # TODO: make sure slope matches TF slope
        self.out_activ = nn.Tanh()

    def forward(self, x):
        x = self.activ(self.bn_1(self.dec_conv1(self.dec_up1(x, output_size=(x.size(0), self.latent_chan, self.H_factors[0], self.W_factors[0])))))
        y = x # x = identity, y = residual connection

        # residual blocks
        z = self.activ(self.res1_bn1(self.dec_res1_conv1(y)))
        z = self.activ(self.res1_bn2(self.dec_res1_conv2(z)))
        y = z + y
        z = self.activ(self.res2_bn1(self.dec_res2_conv1(y)))
        z = self.activ(self.res2_bn2(self.dec_res2_conv2(z)))
        y = z + y

        x = y + x # residual connection around both residual blocks
        x = self.activ(self.bn_2(self.dec_conv2(self.dec_up2(x, output_size=(x.size(0), self.latent_chan, self.H_factors[1], self.W_factors[1])))))
        x = self.out_activ(self.dec_conv3(self.dec_up3(x, output_size=(x.size(0), self.n_chan, self.H_factors[2], self.W_factors[2]))))
        return x
        # return self.out_activ(self.dec_up3(self.dec_conv3(x), output_size=(x.size(0), self.latent_chan, self.H_factors[2], self.W_factors[2])))

class DeepCMCQuant(nn.Module):
    """ DeepCMC for csi estimation with SHVQ and entropy-based loss term """
    def __init__(self, encoder, decoder, quant, latent_dim, K_sigma=1, T_sigma=11719, batch_size=200, device=None, hard_sigma=1e6):
        super(DeepCMCQuant, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quant = quant
        self.latent_dim = latent_dim
        self.device = device
        self.batch_size = batch_size
        self.quant_bool = True
        self.training = True
        self.return_latent = False

        self.p_hist = torch.zeros(self.quant.L).to(self.device) # histogram estimate of probabilities
        self.p_mask = torch.arange(self.quant.L).reshape(1,self.quant.L,1).repeat(batch_size,1,self.quant.n_features).to(self.device)

        # vals for annealing sigma
        self.hard_sigma = hard_sigma

        self.T_sigma = T_sigma # timescale of annealing (batches)
        self.K_sigma = K_sigma # gain of annealing
        self.t = 0 # index of current iteration for annealing

        # self.p_hist = torch.zeros(self.quant.L).to(self.device) # histogram estimate of probabilities
        # self.p_mask = torch.arange(self.quant.L).reshape(1,self.quant.L,1).repeat(batch_size,1,self.quant.n_features).to(self.device)
        # self.p_hard = torch.zeros(batch_size,self.quant.L,self.quant.n_features).to(self.device) # to save on inference time, we will store forward passes with hard quantization here

    def forward(self, H_in):
        """forward call for DeepCMC-SoftQuant"""
        h_enc = self.encoder(H_in)
        if self.quant_bool:
            h_size = h_enc.size()
            x = h_enc.view(self.batch_size,-1) # flatten latent layer
            x = self.quant(x).view(h_size) # quantize, reshape
            return self.decoder(x)
        elif self.return_latent:
            x = h_enc.view(self.batch_size,-1) # flatten latent layer
            return x
        else:
            return self.decoder(h_enc)

    def crossentropy_loss(self, H_in, clip_val=1e-9):
        """ calculate crossentropy between soft/hard assignment probas """
        # temporarily store current quant_bool
        quant_mode_temp = self.quant.quant_mode
        self.quant.quant_mode = 2 # get softmax outputs
        h_size = H_in.size()
        x = self.encoder(H_in).view(self.batch_size,-1) # flatten latent layer
        q_soft = self.quant(x)
        b = q_soft.shape[0]
        H_idx = torch.argmax(q_soft, dim=2)

        # parallel impl
        self.p_hist = torch.true_divide(torch.sum(torch.sum(self.p_mask[:b, :] == H_idx.unsqueeze(1).repeat(1,self.quant.L,1), axis=0), axis=1), self.quant.n_features*b)
        p_hard = self.p_hist.view(1,1,self.quant.L).repeat(b,self.quant.n_features,1)
        p_hard = torch.clamp(p_hard, clip_val, 1.0)
        # store back original quant_mode, sigma values
        self.quant.quant_mode = quant_mode_temp
        entropy_loss = -torch.sum(q_soft * torch.log2(p_hard)) / (self.quant.n_features*b)
        return entropy_loss
    
    def calc_gap(self, mse_soft, H_in, criterion=nn.MSELoss()):
        """
        running total of gap between soft/hard crossentropy
        H_hat_soft : soft quantization estimate from outer loop
        H_in : autoencoder input
        """
        # mse_soft = torch.mean(torch.pow(H_hat_soft - H_in, 2)) # assume we can pass this in
        # make sigma large -> hard quantization
        sigma_temp = self.quant.sigma

        self.quant.sigma = self.quant.hard_sigma
        # quant_mode_temp = self.quant.quant_mode
        # self.quant.quant_mode = 3

        H_hat_hard = self.forward(H_in)
        mse_hard = criterion(H_hat_hard, H_in)

        self.quant.sigma = sigma_temp
        # self.quant.quant_mode = quant_mode_temp

        return mse_hard - mse_soft

    def update_gap(self, mse_soft, H_hat_soft, H_in, criterion=nn.MSELoss()):
        self.gap_t = self.calc_gap(mse_soft, H_in, criterion=criterion)
        if self.t == 0:
            self.gap_0 = self.gap_t

    def anneal_sigma(self):
        self.e_g = self.gap_t + self.T_sigma / (self.T_sigma + self.t) * self.gap_0
        self.quant.sigma += self.K_sigma*self.gap_t
        # self.quant.sigma = np.max([self.quant.sigma_eps, self.quant.sigma])

        self.t += 1

    
if __name__ == "__main__":
    import argparse
    import pickle
    import copy

    from utils.NMSE_performance import renorm_H4, renorm_sphH4
    from utils.data_tools import dataset_pipeline_col, CSIDataset
    from utils.transforms import CircularShift
    from utils.parsing import str2bool
    from utils.timing import Timer
    from utils.unpack_json import get_keys_from_json
    from utils.trainer import save_predictions, save_checkpoint_history
    from trainer import fit, score

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
    parser.add_argument("-p2", "--pretrain2_bool", type=str2bool, default=True, help="bool for performing pretrain stage 2 (training initial centers)")
    parser.add_argument("-tr", "--train_bool", type=str2bool, default=True, help="flag for toggling training for soft-to-hard vector quantization")
    parser.add_argument("-po", "--preload_bool", type=str2bool, default=True, help="flag for toggling loading of pretrained model (no quant)")
    parser.add_argument("-lo", "--load_bool", type=str2bool, default=True, help="flag for toggling loading of finetuned, quantized model")
    parser.add_argument("-th", "--train_hard_bool", type=str2bool, default=False, help="flag for fine-tuning training on hard vector quantization)")
    parser.add_argument("-nb", "--n_batch", type=int, default=20, help="number of batches to fit on (ignored during debug mode)")
    parser.add_argument("-b", "--beta", type=float, default=1e-5, help="hyperparam for mse loss")
    parser.add_argument("-a", "--alpha", type=float, default=1.0, help="hyperparam for entropy loss")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5, help="learning rate for optimizer")
    parser.add_argument("-l", "--dir", type=str, default=None, help="subdirectory for saving model, checkpoint, history")
    parser.add_argument("-tl", "--tail_dir", type=str, default=None, help="subdirectory for saving model, checkpoint, history of SHVQ network")
    parser.add_argument("-e", "--env", type=str, default="outdoor", help="environment (either indoor or outdoor)")
    parser.add_argument("-ep", "--epochs", type=int, default=1000, help="number of epochs to train for")
    parser.add_argument("-ef", "--epochs_finetune", type=int, default=50, help="number of epochs to use for shvq finetuning")
    parser.add_argument("-sp", "--split", type=int, default=0, help="split of entire dataset. must be less than int(<total_num_files> / <n_batch>).")
    parser.add_argument("-t", "--n_truncate", type=int, default=32, help="value to truncate to along delay axis.")
    parser.add_argument("-ts", "--timeslot", type=int, default=0, help="timeslot which we are training (0-indexed).")
    parser.add_argument("-r", "--rate", type=int, default=1024, help="number of elements in latent code (i.e., encoding rate)")
    parser.add_argument("-dt", "--data_type", type=str, default="norm_sphH4", help="type of dataset to train on (norm_H4, norm_sphH4)")
    parser.add_argument("-L", "--num_centers", type=int, default=256, help="Number of cluster centers for vector quantization")
    parser.add_argument("-m", "--dim_centers", type=int, default=4, help="Dimensions for cluster centers for vector quantization")
    parser.add_argument("-ec", "--epochs_centers", type=int, default=1000, help="Epochs for pretrain2 (cluster center initialization)")
    parser.add_argument("-K", "--K_sigma", type=int, default=100, help="Gain for sigma annealing")
    parser.add_argument("-c", "--circ_proba", type=float, default=0.5, help="Probability of circular shift transform")
    opt = parser.parse_args()


    device = torch.device(f'cuda:{opt.gpu_num}' if torch.cuda.is_available() else 'cpu')
    print(f"--- Device is {device} ---")

    # dataset pipeline vars 
    if opt.data_type == "norm_H4":
        json_config = "../config/deepcmc-soft-indoor0001.json" if opt.env == "indoor" else "../config/deepcmc-soft-outdoor300.json"
    elif opt.data_type == "norm_sphH4":
        json_config = "../config/deepcmc-soft-indoor0001-sph-pow.json" if opt.env == "indoor" else "../config/deepcmc-soft-outdoor300-sph-pow.json"
        # json_config = "../config/csinet-pro-quadriga-indoor0001-sph.json" if opt.env == "indoor" else "../config/csinet-pro-quadriga-outdoor300-sph.json"

    dataset_spec, minmax_file, img_channels, data_format, norm_range, T, base_pickle, n_delay, total_num_files, t1_power_file, subsample_prop, thresh_idx_path, diff_spec, network_name = get_keys_from_json(json_config, keys=["dataset_spec", "minmax_file", "img_channels", "data_format", "norm_range", "T", "base_pickle", "n_delay", "total_num_files", "t1_power_file", "subsample_prop", "thresh_idx_path", "diff_spec", "network_name"])
    # aux_bool_list = get_keys_from_json(json_config, keys=["aux_bool"], is_bool=True)
    # network_name = "deepcmc"

    input_dim = (2,n_delay,32)

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
        data_all = renorm_sphH4(data_all, minmax_file, t1_power_file, thresh_idx_path=thresh_idx_path).astype(np.float32)
    data_train, data_val = data_all[:n_train], data_all[n_train:]
    print('-> post-renorm: data_val range is from {} to {} -- data_val.shape = {}'.format(np.min(data_val),np.max(data_val),data_val.shape))

    if opt.dir != None:
        base_pickle += "/" + opt.dir

    # cr_list = [512, 256, 128, 64, 32] # rates for different compression ratios
    # cr_list = [opt.rate]
    # for cr in cr_list:

    # random circular shift of data
    axs = 2 # data = (n_batch, n_chan, n_ang, n_del); shift along angular axis
    circ_shift = CircularShift(axs, opt.circ_proba)

    train_dataset = CSIDataset(torch.from_numpy(data_train), transform=circ_shift, device=device)
    valid_dataset = CSIDataset(torch.from_numpy(data_val), device=device)

    train_ldr = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    valid_ldr = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    # train_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_train).to(device), batch_size=batch_size, shuffle=True) 
    # valid_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_val).to(device), batch_size=batch_size)

    encoder = Encoder(input_dim[0], input_dim[1], opt.n_truncate)
    decoder = Decoder(input_dim[0], input_dim[1], opt.n_truncate)
    quant = SoftQuantize(opt.rate, opt.num_centers, opt.dim_centers, sigma_trainable=False, device=device)

    deepcmc = DeepCMCQuant(encoder, decoder, quant, opt.rate, device=device).to(device)

    # pickle_dir = f"{base_pickle}/t1"
    pickle_dir = f"{base_pickle}"

    # deepcmc.quant.quant_mode = 0 # pretrain with no latent quantization
    deepcmc.quant_bool = False # train without quantization layer
    deepcmc.noise_bool = True
    epochs = 1 if opt.debug_flag else opt.epochs # epochs for intial, non-quantized network performance
    if opt.pretrain1_bool:
        # if opt.preload_bool:
        #     deepcmc, checkpoint, history, optimizer, pretrained_epochs = load_pretrained(deepcmc, json_config, pickle_dir, epochs=epochs, device=device, lr=opt.learning_rate)
        # else:
            # pretrained_epochs = 0
            # optimizer = history = checkpoint = None
        model, checkpoint, history, optimizer, timers = fit(deepcmc,
                                                            train_ldr,
                                                            valid_ldr,
                                                            batch_num,
                                                            epochs=epochs,
                                                            timers=timers,
                                                            json_config=json_config,
                                                            debug_flag=opt.debug_flag,
                                                            pickle_dir=pickle_dir,
                                                            beta=opt.beta,
                                                            network_name=f"{network_name}-pretrain",
                                                            # pretrained_epochs=pretrained_epochs,
                                                            # checkpoint=checkpoint,
                                                            # history=history,
                                                            # optimizer=optimizer,
                                                            # lr=opt.learning_rate
                                                            )
                                                            # network_name=f"{network_name}-pretrain")
    elif opt.preload_bool:
        strs = ["checkpoint", "history"]
        load_dict = {}
        for str_i in strs:
            with open(f"{pickle_dir}/{network_name}-pretrain-{str_i}.pkl", "rb") as f:
                load_dict[str_i] = pickle.load(f)
                f.close()
        checkpoint, history = load_dict["checkpoint"], load_dict["history"]
        optimizer = optim.Adam(deepcmc.parameters(), lr=learning_rate)
        opt_state_dict = torch.load(f"{pickle_dir}/{network_name}-pretrain-optimizer.pt")
        optimizer.load_state_dict(opt_state_dict)
        print(f"Loading best model weights from {pickle_dir}/{network_name}-pretrain-best-model.pt")
        deepcmc.load_state_dict(torch.load(f"{pickle_dir}/{network_name}-pretrain-best-model.pt", map_location=device), strict=False)
        # deepcmc.quant.reset_extrema(checkpoint["best_z_min"], checkpoint["best_z_max"])
    else:
        print(f"---- Model performance without pretraining ---")
        checkpoint = {"latest_model": deepcmc}
        history = {}
        optimizer = torch.optim.Adam(deepcmc.parameters())

    del train_ldr, valid_ldr, train_dataset, valid_dataset
    # del train_ldr, valid_ldr
    all_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_all).to(device), batch_size=batch_size)
    [checkpoint, y_hat, y_test] = score(deepcmc,
                                        all_ldr,
                                        data_all,
                                        batch_num,
                                        checkpoint,
                                        history,
                                        optimizer,
                                        timers=timers,
                                        json_config=json_config,
                                        debug_flag=opt.debug_flag,
                                        str_mod=f"DeepCMC (pretrain {epochs} epochs)",
                                        n_train=data_train.shape[0],
                                        pow_diff_t=pow_diff
                                        )

    if not opt.debug_flag and opt.pretrain1_bool:                
        save_checkpoint_history(checkpoint, history, optimizer, dir=pickle_dir, network_name=f"{network_name}-pretrain")
        
    del all_ldr
    torch.cuda.empty_cache()

    # --- center pretraining ---

    if opt.pretrain2_bool:
        # use encoder to get train/validation codewords
        # encoder = deepcmc.encoder
        # quant = deepcmc.quant
        # del decoder, deepcmc
        # torch.cuda.empty_cache()

        train_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_train).to(device), batch_size=batch_size) 
        valid_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_val).to(device), batch_size=batch_size)

        enc_train = torch.zeros(size=(data_train.shape[0], opt.rate))
        enc_valid = torch.zeros(size=(data_val.shape[0], opt.rate))

        ldrs  = [train_ldr, valid_ldr]
        encs  = {"training": enc_train, "validation": enc_valid}
        descs = ["training", "validation"]

        deepcmc.return_latent = True
        with torch.no_grad():
            # enc_train = csinet_quant.encoder(torch.from_numpy(data_train).to(device))
            # enc_valid = csinet_quant.encoder(torch.from_numpy(data_val).to(device))
            # enc_train = encoder(torch.from_numpy(data_train).to(device))
            for ldr, desc in zip(ldrs,descs):
                i_e = 0
                for i, data_tuple in enumerate(tqdm(ldr, desc=f"Encoder {desc} data"), 0):
                    if len(data_tuple) != 2:
                        data_batch = data_tuple
                    else:
                        aux_batch, data_batch = data_tuple
                        aux_input = torch.autograd.Variable(aux_batch)
                    h_input = torch.autograd.Variable(data_batch)
                    optimizer.zero_grad()
                    model_in = h_input if len(data_tuple) != 2 else [aux_input, h_input]
                    bs = model_in.size(0)
                    i_s = i_e
                    i_e += bs
                    encs[desc][i_s:i_e,:] = deepcmc(model_in).cpu()
            enc_train, enc_valid = encs["training"], encs["validation"]
            # foo = torch.from_numpy(data_train).to(device)
            # enc_train = deepcmc(foo)
            # # enc_train = encoder(foo)
            # del foo
            # torch.cuda.empty_cache()
            # # enc_valid = encoder(torch.from_numpy(data_val).to(device))
            # foo = torch.from_numpy(data_val).to(device)
            # enc_valid = deepcmc(foo)
            # # enc_train = encoder(foo)
            # del foo
            # torch.cuda.empty_cache()
        # del encoder, decoder, csinet_quant
        enc_train_ldr = torch.utils.data.DataLoader(enc_train, batch_size=batch_size, shuffle=True)
        enc_valid_ldr = torch.utils.data.DataLoader(enc_valid, batch_size=batch_size)
        c = np.random.randint(enc_train.shape[0], size=opt.num_centers)
        sampled_centers = enc_train.view(enc_train.shape[0]*int(opt.rate / opt.dim_centers), opt.dim_centers)[c,:]
        # print(f"--- sampled_centers.shape: {sampled_centers.shape} ---")
        quant.init_centers(sampled_centers)
        del sampled_centers
        torch.cuda.empty_cache()

        epochs = 1 if opt.debug_flag else opt.epochs_centers # epochs for intial, non-quantized network performance
        fit(quant,
            enc_train_ldr,
            enc_valid_ldr,
            batch_num,
            epochs=opt.epochs_centers,
            timers=timers,
            criterion=energy_loss,
            json_config=json_config,
            debug_flag=opt.debug_flag,
            pickle_dir=pickle_dir,
            network_name="soft-quant")

        del enc_train_ldr, enc_valid_ldr
        torch.cuda.empty_cache()

    encoder = Encoder(input_dim[0], input_dim[1], opt.n_truncate)
    decoder = Decoder(input_dim[0], input_dim[1], opt.n_truncate)
    # encoder = csinet_quant.encoder
    # decoder = csinet_quant.decoder
    # sigma = csinet_quant.quant.sigma.relu().item() + csinet_quant.quant.sigma_eps
    quant = SoftQuantize(opt.rate, opt.num_centers, opt.dim_centers, sigma=quant.sigma, sigma_trainable=False, device=device)


    # quant.load_state_dict(OrderedDict({'c': }))
    deepcmc = DeepCMCQuant(encoder, decoder, quant, opt.rate, K_sigma=opt.K_sigma, device=device).to(device) # remake network with non-trainable sigma
    deepcmc.load_state_dict(torch.load(f"{pickle_dir}/{network_name}-pretrain-best-model.pt", map_location=device), strict=False)
    deepcmc.quant.init_centers(torch.zeros(deepcmc.quant.L, deepcmc.quant.m).to(device))
    deepcmc.quant.load_state_dict(torch.load(f"{pickle_dir}/soft-quant-best-model.pt", map_location=device))

    deepcmc.quant.quant_mode = 1 # train with quantization layer
    print(f"-> sigma: {deepcmc.quant.sigma}")
    print(f"-> centers: {deepcmc.quant.c}")

    checkpoint = {}
    all_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_all).to(device), batch_size=batch_size)
    [checkpoint, y_hat, y_test] = score(deepcmc,
                                        all_ldr,
                                        data_all,
                                        batch_num,
                                        checkpoint,
                                        history,
                                        optimizer,
                                        timers=timers,
                                        json_config=json_config,
                                        debug_flag=opt.debug_flag,
                                        str_mod=f"DeepCMCQuant CR={opt.rate} (after training centers, sigma={deepcmc.quant.sigma:4.3f})",
                                        n_train=data_train.shape[0],
                                        pow_diff_t=pow_diff,
                                        quant_bool=True
                                        )

    init_nmse = checkpoint["best_nmse"] # save nmse from checkpoint
    print(f"--- after center pretraining, init_nmse={init_nmse} ---")
    del all_ldr
    torch.cuda.empty_cache()

    if opt.tail_dir != None:
        pickle_dir += f"/{opt.tail_dir}" 
        print(f"--- pickle_dir with tail_dir: {pickle_dir} ---")

    train_dataset = CSIDataset(torch.from_numpy(data_train), transform=circ_shift, device=device)
    valid_dataset = CSIDataset(torch.from_numpy(data_val), device=device)

    train_ldr = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    valid_ldr = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    # train_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_train).to(device), batch_size=batch_size, shuffle=True) 
    # valid_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_val).to(device), batch_size=batch_size)
    # all_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_all).to(device), batch_size=batch_size)

    epochs = 1 if opt.debug_flag else opt.epochs_finetune # epochs for intial, non-quantized network performance
    if opt.train_bool:
        model, checkpoint, history, optimizer, timers = fit(deepcmc,
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
                                                            beta=opt.beta,
                                                            data_all=data_all 
                                                            )
        deepcmc.quant.sigma = checkpoint["best_sigma"]
        deepcmc.load_state_dict(checkpoint["best_model"])
    elif opt.load_bool:
        model_weights_name = f"{pickle_dir}/{network_name}-best-model.pt"
        print(f"---- Loading best model from {model_weights_name} ---")
        deepcmc.load_state_dict(torch.load(model_weights_name, map_location=device))
        deepcmc.quant.quant_mode = 1 # train with quantization layer
        checkpoint_name = f"{pickle_dir}/{network_name}-checkpoint.pkl"
        with open(checkpoint_name, 'rb') as f:
            checkpoint = pickle.load(f)
            f.close()
        deepcmc.quant.sigma = checkpoint["best_sigma"].to(device)
    else:
        print(f"---- Model performance without soft-to-hard vector quantization ---")
        checkpoint = {"best_sigma": deepcmc.quant.sigma, "latest_model": deepcmc}
        history = {}
        optimizer = torch.optim.Adam(deepcmc.parameters())

    del train_dataset, valid_dataset, train_ldr, valid_ldr
    # del train_ldr, valid_ldr
    torch.cuda.empty_cache()
    all_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_all).to(device), batch_size=batch_size)
    [checkpoint, y_hat, y_test] = score(deepcmc,
                                        all_ldr,
                                        data_all,
                                        batch_num,
                                        checkpoint,
                                        history,
                                        optimizer,
                                        timers=timers,
                                        json_config=json_config,
                                        debug_flag=opt.debug_flag,
                                        str_mod=f"DeepCMCQuant CR={opt.rate} (best soft-quantization with sigma={checkpoint['best_sigma']:4.3f})",
                                        n_train=data_train.shape[0],
                                        pow_diff_t=pow_diff,
                                        quant_bool=True
                                        )
    history["init_nmse"] = init_nmse

    if not opt.debug_flag and opt.train_bool:                
        save_checkpoint_history(checkpoint, history, optimizer, dir=pickle_dir, network_name=network_name)

    del all_ldr
    torch.cuda.empty_cache()

    # epochs = 1 if opt.debug_flag else opt.epochs_finetune # epochs for intial, non-quantized network performance
    # if opt.train_bool:
    #     model, checkpoint, history, optimizer, timers = fit(deepcmc,
    #                                                         train_ldr,
    #                                                         valid_ldr,
    #                                                         batch_num,
    #                                                         epochs=epochs,
    #                                                         timers=timers,
    #                                                         json_config=json_config,
    #                                                         debug_flag=opt.debug_flag,
    #                                                         pickle_dir=pickle_dir,
    #                                                         network_name=network_name,
    #                                                         quant_bool=True,
    #                                                         data_all=data_all,
    #                                                         beta=opt.beta,
    #                                                         # optimizer=optimizer
    #                                                         )
    #     # deepcmc.quant.sigma = checkpoint["best_sigma"]
    #     deepcmc.load_state_dict(checkpoint["best_model"])
    # elif opt.load_bool:
    #     model_weights_name = f"{pickle_dir}/{network_name}-best-model.pt"
    #     print(f"---- Loading best model from {model_weights_name} ---")
    #     deepcmc.load_state_dict(torch.load(model_weights_name, map_location=device))
    #     deepcmc.quant.quant_mode = 1 # train with quantization layer
    #     checkpoint_name = f"{pickle_dir}/{network_name}-checkpoint.pkl"
    #     with open(checkpoint_name, 'rb') as f:
    #         checkpoint = pickle.load(f)
    #         f.close()
    #     deepcmc.quant.reset_extrema(checkpoint["best_z_min"], checkpoint["best_z_max"])
    #     # deepcmc.quant.sigma = checkpoint["best_sigma"].to(device)
    #     # checkpoint = {}
    # else:
    #     print(f"---- Model performance without quantization finetuning  ---")
    #     checkpoint = {"latest_model": deepcmc}
    #     history = {}
    #     optimizer = torch.optim.Adam(deepcmc.parameters())

    # del train_ldr, valid_ldr
    # torch.cuda.empty_cache()
    # all_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_all).to(device), batch_size=batch_size)
    # [checkpoint, y_hat, y_test] = score(deepcmc,
    #                                     all_ldr,
    #                                     data_all,
    #                                     batch_num,
    #                                     checkpoint,
    #                                     history,
    #                                     optimizer,
    #                                     timers=timers,
    #                                     json_config=json_config,
    #                                     debug_flag=opt.debug_flag,
    #                                     str_mod=f"DeepCMC (uniform quantization)",
    #                                     n_train=data_train.shape[0],
    #                                     pow_diff_t=pow_diff,
    #                                     quant_bool=True
    #                                     )
    # # history["init_nmse"] = init_nmse

    # del all_ldr
    # torch.cuda.empty_cache()

    # if not opt.debug_flag and opt.train_bool:                
    #     save_checkpoint_history(checkpoint, history, optimizer, dir=pickle_dir, network_name=network_name)

    # valid_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_val).to(device), batch_size=batch_size)
    # deepcmc.return_latent = True
    # deepcmc.quant.return_denorm = False
    # update_histogram(deepcmc,
    #                     valid_ldr,
    #                 #  data_all,
    #                     batch_num,
    #                     timers=timers,
    #                     json_config=json_config,
    #                     debug_flag=opt.debug_flag,
    #                     str_mod=f"DeepCMC (hard quantization)",
    #                     )

    # print(f"-> Calculate entropy based on validation set <-")
    # entropy = deepcmc.quant.get_entropy()
    # print(f"-> validation entropy ({opt.epochs_finetune} epochs): {entropy:4.3f}")

    # del valid_ldr
    # torch.cuda.empty_cache()