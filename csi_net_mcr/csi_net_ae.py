import unicodedata
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from collections import OrderedDict
from csi_net import Encoder, Decoder, CsiNetQuant, SoftQuantize, energy_loss

import numpy as np
# import matplotlib.pyplot as plt
from tqdm import trange, tqdm

def make_alphabet(L):
    """ contruct alphabet of length L, remove whitespace and control characters """
    true_i = 0
    alphabet = ''
    while(len(alphabet) < L):
        while(str.isspace(chr(true_i)) or unicodedata.category(chr(true_i))[0] == "C"):
            true_i += 1
        alphabet += chr(true_i)
        print(f"i={len(alphabet)} -> true_i={true_i} -> {chr(true_i)}")
        true_i += 1
    return alphabet

class CsiNetQuantArithmeticEncoding(CsiNetQuant):

    # def __init__(self, encoder, decoder, quant, latent_dim, K_sigma=1, T_sigma=11719, batch_size=200, device=None, hard_sigma=1e6):
    def __init__(self, encoder, decoder, quant, latent_dim, **kwargs):
        """
        Add ArithmeticEncoding to CsiNet-Quant arch
        sigma = temperature for softmax
        r = dimensionality of autoencoder's latent features
        L = number of cluster centers
        m = dimensionality of cluster centers
        """
        # CsiNetQuant.__init__(self, encoder, decoder, quant, latent_dim, K_sigma=K_sigma, T_sigma=T_sigma, batch_size=batch_size, device=device, hard_sigma=hard_sigma)
        CsiNetQuant.__init__(self, encoder, decoder, quant, latent_dim, **kwargs)
        self.reset_center_counts()
        self.entropy_encoding = False
        self.count_centers = True
        self.alphabet = make_alphabet(self.quant.L)
        self.vect_return_symbol = np.vectorize(self.return_symbol)
        self.message_bit_lens = []
        self.max_msg_size = int(self.quant.n_features / 32)
        self.all_bit_counts = []

    def reset_center_counts(self):
        """ reset center counts for arithmetic coding """
        self.H_counts = torch.zeros(self.quant.L).to(self.device) # keep track of counts for centers
    
    def make_freq_table(self):
        H_counts = self.H_counts.cpu().numpy() 
        freq_table = {}
        for i, count in enumerate(H_counts):
            freq_table[self.alphabet[i]] = int(count) + 1 # avoid 0 counts; improves stability
        self.freq_table = freq_table
        self.AE = pyae.ArithmeticEncoding(frequency_table=self.freq_table)

    def print_avg_bits_per_message(self, ci=0.05):
        """ print average number of bits per message """
        all_bits = np.sort(self.all_bit_counts)
        idx_lo, idx_hi = int(len(self.all_bit_counts)*(ci/2)), int(len(self.all_bit_counts)*(1 - ci/2))
        bits_lo, bits_hi = all_bits[idx_lo], all_bits[idx_hi]
        N_inp_pix = self.encoder.img_total * self.encoder.n_chan * 32
        N_inp = self.encoder.img_total * self.encoder.n_chan 
        perc = 100*(1-ci)
        print(f"--- Average bits per message: {np.mean(all_bits)} with {perc:2.1f}% ci ({bits_lo}, {bits_hi}) ---")
        print(f"--- Average bits per input bit: {np.mean(self.all_bit_counts) / N_inp_pix} with {perc:2.1f}% ci ({bits_lo / N_inp_pix}, {bits_hi / N_inp_pix})---")
        print(f"--- Average bits per input pixel: {np.mean(self.all_bit_counts) / N_inp} with {perc:2.1f}% ci ({bits_lo / N_inp}, {bits_hi / N_inp})---")

    def forward(self, H_in):
        """forward call for VAE"""
        h_enc = self.encoder(H_in)
        q = self.quant(h_enc)
        b = h_enc.shape[0]
        H_idx = torch.argmax(q, dim=2)
        if self.count_centers:
            self.p_mask[:b, :] == H_idx.unsqueeze(1).repeat(1,self.quant.L,1)
            self.H_counts += torch.sum(torch.sum(self.p_mask[:b, :] == H_idx.unsqueeze(1).repeat(1,self.quant.L,1), axis=0), axis=1) # count indices, store back to H_counts
        elif self.entropy_encoding:
            q_encoded = self.int_to_alphabet(H_idx) # TODO: perform all this on device side w/ torch tensors
            # messages are encoded per batch
            q_decoded = []
            bit_counts = []
            for i, original_msg_full in enumerate(q_encoded):
                preamble_str = f"-> #{i}:"
                idx = 0
                decoded_msg_full = ""
                bits_full = 0
                while idx < len(original_msg_full):
                    idx_e = min([idx+self.max_msg_size, self.quant.n_features])
                    original_msg = original_msg_full[idx:idx_e] 
                    encoded_msg, encoder, interval_min_value, interval_max_value = self.AE.encode(msg=original_msg, 
                                                                                            probability_table=self.AE.probability_table)
                    # print(f"{preamble_str} Encoded Message: {encoded_msg}")
                    # encode
                    binary_code, encoder_binary = self.AE.encode_binary(float_interval_min=interval_min_value,
                                                                float_interval_max=interval_max_value)
                    # print("The binary code is: {binary_code}".format(binary_code=binary_code))
                    # decode the message
                    decoded_msg, decoder = self.AE.decode(encoded_msg=encoded_msg, 
                                                    msg_length=len(original_msg),
                                                    probability_table=self.AE.probability_table)
                    decoded_msg = "".join(decoded_msg)
                    q_decoded.append(decoded_msg)
                    decoded_msg_full += decoded_msg
                    idx += self.max_msg_size
                    bits_full += len(binary_code)

                bit_counts.append(bits_full)
                print(f"{preamble_str} Message Decoded Successfully? {original_msg_full == decoded_msg_full} with total bit length={bits_full}")
            # TODO: alphabet_to_int impl
            self.all_bit_counts += bit_counts
            # return self.decoder(q) if self.quant_bool else self.decoder(h_enc)
    
    def int_to_alphabet(self, q):
        """ take (n_batch, n_features) tensor, convert to (n_batch,) tensor with alphabet-encoded str """
        # q_symbols = q.detach().numpy()
        q_symbols = q.cpu().numpy()
        q_symbols = self.vect_return_symbol(q_symbols, self.alphabet)
        q_messages = []
        for array in q_symbols:
            q_messages.append("".join(array))
        return q_messages

    def return_symbol(self, x, alphabet):
        """ return x-th symbol from alphabet """
        return alphabet[x]




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
    from trainer import fit, score, count_centers, arithmetic_encoding, profile_network

    sys.path.append("/home/mdelrosa/git/ArithmeticEncodingPython")
    import pyae

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
    parser.add_argument("-lo", "--load_bool", type=str2bool, default=True, help="flag for toggling loading of soft-to-hard vector quantized model")
    parser.add_argument("-th", "--train_hard_bool", type=str2bool, default=False, help="flag for fine-tuning training on hard vector quantization)")
    parser.add_argument("-nb", "--n_batch", type=int, default=20, help="number of batches to fit on (ignored during debug mode)")
    parser.add_argument("-b", "--beta", type=float, default=1e-5, help="hyperparam for entropy loss")
    parser.add_argument("-l", "--dir", type=str, default=None, help="subdirectory for saving model, checkpoint, history")
    parser.add_argument("-tl", "--tail_dir", type=str, default=None, help="subdirectory for saving model, checkpoint, history of SHVQ network")
    parser.add_argument("-e", "--env", type=str, default="outdoor", help="environment (either indoor or outdoor)")
    parser.add_argument("-t", "--n_truncate", type=int, default=32, help="value to truncate to along delay axis.")
    parser.add_argument("-ts", "--timeslot", type=int, default=0, help="timeslot which we are training (0-indexed).")
    parser.add_argument("-r", "--rate", type=int, default=512, help="number of elements in latent code (i.e., encoding rate)")
    parser.add_argument("-dt", "--data_type", type=str, default="norm_sphH4", help="type of dataset to train on (norm_H4, norm_sphH4)")
    parser.add_argument("-L", "--num_centers", type=int, default=256, help="Number of cluster centers for vector quantization")
    parser.add_argument("-m", "--dim_centers", type=int, default=4, help="Dimensions for cluster centers for vector quantization")
    parser.add_argument("-ec", "--epochs_centers", type=int, default=1000, help="Epochs for pretrain2 (cluster center initialization)")
    parser.add_argument("-K", "--K_sigma", type=int, default=100, help="Gain for sigma annealing")
    parser.add_argument("-N", "--n_valid", type=int, default=1000, help="Number of samples in validation set for assessing arithmetic encoding")
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
        data_all = renorm_sphH4(data_all, minmax_file, t1_power_file, thresh_idx_path=thresh_idx_path).astype(np.float32)
    data_train, data_val = data_all[:n_train], data_all[n_train:]
    print('-> post-renorm: data_val range is from {} to {} -- data_val.shape = {}'.format(np.min(data_val),np.max(data_val),data_val.shape))

    if opt.dir != None:
        base_pickle += "/" + opt.dir

    # cr_list = [512, 256, 128, 64, 32] # rates for different compression ratios
    cr_list = [opt.rate]
    for cr in cr_list:

        pickle_dir = f"{base_pickle}/cr{cr}/t1"

        encoder = Encoder(input_dim[0], input_dim[1], opt.n_truncate, cr)
        decoder = Decoder(input_dim[0], input_dim[1], opt.n_truncate, cr)
        quant = SoftQuantize(cr, opt.num_centers, opt.dim_centers, sigma_trainable=False, device=device)

        csinet_quant = CsiNetQuant(encoder, decoder, quant, cr, K_sigma=opt.K_sigma, device=device).to(device)
        # csinet_quant = CsiNetQuantArithmeticEncoding(encoder, decoder, quant, cr, K_sigma=opt.K_sigma, device=device).to(device) # remake network with non-trainable sigma
        csinet_quant.load_state_dict(torch.load(f"{pickle_dir}/{network_name}-pretrain-best-model.pt", map_location=device), strict=False)
        csinet_quant.quant.init_centers(torch.zeros(csinet_quant.quant.L, csinet_quant.quant.m).to(device))
        csinet_quant.quant.load_state_dict(torch.load(f"{pickle_dir}/soft-quant-best-model.pt"))

        if opt.tail_dir != None:
            pickle_dir += f"/{opt.tail_dir}" 
            print(f"--- pickle_dir with tail_dir: {pickle_dir} ---")
        if opt.load_bool:
            model_weights_name = f"{pickle_dir}/{network_name}-best-model.pt"
            print(f"---- Loading best model from {model_weights_name} ---")
            csinet_quant.load_state_dict(torch.load(model_weights_name), strict=False)
            # csinet_quant.quant.quant_mode = 1 # train with quantization layer
            checkpoint_name = f"{pickle_dir}/{network_name}-checkpoint.pkl"
            with open(checkpoint_name, 'rb') as f:
                checkpoint = pickle.load(f)
                f.close()
            csinet_quant.quant.sigma = checkpoint["best_sigma"].to(device)

        print(f"-> sigma: {csinet_quant.quant.sigma}")
        print(f"-> centers: {csinet_quant.quant.c}")

        # transfer encoder, decoder, and soft quant layer into csinet_quant_ae
        csinet_quant_ae = CsiNetQuantArithmeticEncoding(csinet_quant.encoder, csinet_quant.decoder, csinet_quant.quant, cr, K_sigma=opt.K_sigma, device=device).to(device) # remake network with non-trainable sigma
        csinet_quant_ae.quant.quant_mode = 2 # train with quantization layer

        optimizer = optim.Adam(csinet_quant_ae.parameters(), lr=learning_rate)
        checkpoint = {}
        history = {}

        # train_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_train).to(device), batch_size=batch_size) 
        valid_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_val).to(device), batch_size=batch_size)
        count_centers(csinet_quant_ae,
                      valid_ldr,
                      batch_num,
                      timers=timers,
                      json_config=json_config,
                      debug_flag=opt.debug_flag,
                      str_mod=f"CsiNetQuant CR={cr} (after training centers, sigma={csinet_quant.quant.sigma:4.3f})",
                      n_train=data_train.shape[0],
                      pow_diff_t=pow_diff,
                      quant_bool=True
                      )
        del valid_ldr

        # make freq table based on counts
        csinet_quant_ae.make_freq_table()

        print(f"--- Calculating avg number of bits/message under arithmetic encoding ({opt.n_valid} validation samples) ---")
        valid_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_val[:opt.n_valid]).to(device), batch_size=batch_size)
        csinet_quant_ae.entropy_encoding = True
        csinet_quant_ae.count_centers = False
        arithmetic_encoding(csinet_quant_ae,
                            valid_ldr,
                            batch_num,
                            timers=timers,
                            json_config=json_config,
                            debug_flag=opt.debug_flag,
                            str_mod=f"CsiNetQuant CR={cr} (after training centers, sigma={csinet_quant.quant.sigma:4.3f})",
                            n_train=data_train.shape[0],
                            pow_diff_t=pow_diff,
                            quant_bool=True
                            )

        csinet_quant_ae.print_avg_bits_per_message()

        # with open(f"{pickle_dir}/"):
        #     pickle.dump()

        # init_nmse = checkpoint["best_nmse"] # save nmse from checkpoint
        # print(f"--- after center pretraining, init_nmse={init_nmse} ---")
        del valid_ldr

        # # profile_network(csinet_quant,
        # #                 train_ldr,
        # #                 valid_ldr,
        # #                 batch_num,
        # #                 epochs=epochs,
        # #                 json_config=json_config,
        # #                 quant_bool=True
        # #                )

        # if opt.tail_dir != None:
        #     pickle_dir += f"/{opt.tail_dir}" 
        #     print(f"--- pickle_dir with tail_dir: {pickle_dir} ---")

        # torch.cuda.empty_cache()
        # train_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_train).to(device), batch_size=batch_size, shuffle=True) 
        # valid_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_val).to(device), batch_size=batch_size)
        # # all_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_all).to(device), batch_size=batch_size)

        # epochs = 1 if opt.debug_flag else opt.epochs_finetune # epochs for intial, non-quantized network performance
        # if opt.train_bool:
        #     model, checkpoint, history, optimizer, timers = fit(csinet_quant,
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
        #                                                         anneal_bool=True,
        #                                                         beta=opt.beta,
        #                                                         data_all=data_all 
        #                                                         )
        #     csinet_quant.quant.sigma = checkpoint["best_sigma"]
        #     csinet_quant.load_state_dict(checkpoint["best_model"])
        # elif opt.load_bool:
        #     model_weights_name = f"{pickle_dir}/{network_name}-best-model.pt"
        #     print(f"---- Loading best model from {model_weights_name} ---")
        #     csinet_quant.load_state_dict(torch.load(model_weights_name))
        #     csinet_quant.quant.quant_mode = 1 # train with quantization layer
        #     checkpoint_name = f"{pickle_dir}/{network_name}-checkpoint.pkl"
        #     with open(checkpoint_name, 'rb') as f:
        #         checkpoint = pickle.load(f)
        #         f.close()
        #     csinet_quant.quant.sigma = checkpoint["best_sigma"]
        # else:
        #     print(f"---- Model performance without soft-to-hard vector quantization ---")
        #     checkpoint = {"best_sigma": csinet_quant.quant.sigma, "latest_model": csinet_quant}
        #     history = {}
        #     optimizer = torch.optim.Adam(csinet_quant.parameters())

        # del train_ldr, valid_ldr
        # all_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_all).to(device), batch_size=batch_size)
        # [checkpoint, y_hat, y_test] = score(csinet_quant,
        #                                     all_ldr,
        #                                     data_all,
        #                                     batch_num,
        #                                     checkpoint,
        #                                     history,
        #                                     optimizer,
        #                                     timers=timers,
        #                                     json_config=json_config,
        #                                     debug_flag=opt.debug_flag,
        #                                     str_mod=f"CsiNetQuant CR={cr} (best soft-quantization with sigma={checkpoint['best_sigma']:4.3f})",
        #                                     n_train=data_train.shape[0],
        #                                     pow_diff_t=pow_diff,
        #                                     quant_bool=True
        #                                     )
        # history["init_nmse"] = init_nmse

        # if not opt.debug_flag and opt.train_bool:                
        #     save_checkpoint_history(checkpoint, history, optimizer, dir=pickle_dir, network_name=network_name)

        # del all_ldr
        # train_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_train).to(device), batch_size=batch_size, shuffle=True) 
        # valid_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_val).to(device), batch_size=batch_size)

        # csinet_quant.quant.quant_mode = 3
        # if opt.train_hard_bool:
        #     model, checkpoint, history, optimizer, timers = fit(csinet_quant,
        #                                                         train_ldr,
        #                                                         valid_ldr,
        #                                                         batch_num,
        #                                                         epochs=epochs,
        #                                                         timers=timers,
        #                                                         json_config=json_config,
        #                                                         debug_flag=opt.debug_flag,
        #                                                         pickle_dir=pickle_dir,
        #                                                         network_name=f"{network_name}-hard",
        #                                                         quant_bool=True,
        #                                                         anneal_bool=False)
        # # else:
        # #     model_weights_name = f"{pickle_dir}/{network_name}-best-model-hard.pt"
        # #     print(f"---- Loading best hard quantized model from {model_weights_name} ---")
        # #     csinet_quant.load_state_dict(torch.load(model_weights_name))

        # del train_ldr, valid_ldr
        # all_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_all).to(device), batch_size=batch_size)
        # [checkpoint, y_hat, y_test] = score(csinet_quant,
        #                                     all_ldr,
        #                                     data_all,
        #                                     batch_num,
        #                                     checkpoint,
        #                                     history,
        #                                     optimizer,
        #                                     timers=timers,
        #                                     json_config=json_config,
        #                                     debug_flag=opt.debug_flag,
        #                                     str_mod=f"CsiNetQuant CR={cr} (hard quantization)",
        #                                     n_train=data_train.shape[0],
        #                                     pow_diff_t=pow_diff,
        #                                     key_mod="_hard",
        #                                     quant_bool=True
        #                                     )

        # # if not opt.debug_flag:                
        # #     save_checkpoint_history(checkpoint, history, optimizer, dir=pickle_dir, network_name=f"{network_name}-hard")

        # del all_ldr