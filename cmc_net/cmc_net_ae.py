import unicodedata
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from collections import OrderedDict
from cmc_net import Encoder, Decoder, DeepCMC, UniformQuantize

import itertools
import numpy as np
# import matplotlib.pyplot as plt
from tqdm import trange, tqdm

import sys
sys.path.append("/home/mdelrosa/git/ArithmeticEncodingPython")
import pyae

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


class DeepCMCArithmeticEncoding(DeepCMC):
    """ DeepCMC for csi estimation with arithmetic encoding """
    def __init__(self, encoder, decoder, quant, **kwargs):
        DeepCMC.__init__(self, encoder, decoder, quant, **kwargs)
        # arithmetic encoding setup
        self.entropy_encoding = True
        self.alphabet = make_alphabet(self.quant.n_levels+1)
        # self.alphabet = make_alphabet(self.quant.n_levels)
        self.vect_return_symbol = np.vectorize(self.return_symbol)
        self.message_bit_lens = []
        self.n_features = int(self.encoder.img_total * self.encoder.latent_chan / np.prod(self.encoder.sample_factors))
        # print(self.max_msg_size)
        # self.max_msg_size = int(self.n_features / 32)
        self.max_msg_size = 4
        self.all_bit_counts = []


    def forward(self, H_in):
        """ forward call for DeepCMC with AE """
        h_enc = self.encoder(H_in)
        b = h_enc.shape[0]

        if self.return_latent:
            return h_enc

        q = self.quant(h_enc)
        # H_idx = torch.argmax(q, dim=2)
        if self.update_hist: # assume 
            assert(self.quant.return_denorm == False)
            self.quant.update_hist(q) 
            # self.p_mask[:b, :] == H_idx.unsqueeze(1).repeat(1,self.quant.L,1)
            # self.H_counts += torch.sum(torch.sum(self.p_mask[:b, :] == H_idx.unsqueeze(1).repeat(1,self.quant.L,1), axis=0), axis=1) # count indices, store back to H_counts
        elif self.entropy_encoding:
            q = torch.reshape(q, (q.size(0), -1))
            q_idx = torch.bucketize(q, boundaries=self.quant.p_vals) # assume normalized latents are returned
            q_encoded = self.int_to_alphabet(q_idx) # TODO: perform all this on device side w/ torch tensors
            # messages are encoded per batch
            q_decoded = []
            bit_counts = []
            for i, original_msg_full in enumerate(q_encoded):
                # print(f"#{i}: entropy encoding")
                preamble_str = f"-> #{i}:"
                idx = 0
                decoded_msg_full = ""
                bits_full = 0
                while idx < len(original_msg_full):
                    idx_e = min([idx+self.max_msg_size, self.n_features])
                    # print(f"idx: {idx} - idx_e: {idx_e}")
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
                    # print(f"{preamble_str} idx = {idx} - idx_e = {idx_e} - original_msg = {original_msg} - decoded_msg = {decoded_msg}")

                bit_counts.append(bits_full)
                print(f"{preamble_str} Message Decoded Successfully? {original_msg_full == decoded_msg_full} with total bit length={bits_full}")
                # print(f"{preamble_str} original_msg_full is {original_msg_full}")
                # print(f"{preamble_str} decoded_msg_full is  {decoded_msg_full}")
            # TODO: alphabet_to_int impl
            self.all_bit_counts += bit_counts
            # return self.decoder(q) if self.quant_bool else self.decoder(h_enc)

    def make_freq_table(self):
        # H_counts = self.H_counts.cpu().detach().numpy() 
        H_counts = self.quant.p_hist.cpu().numpy() 
        freq_table = {}
        print(f"len(H_counts): {len(H_counts)} - len(self.alphabet): {len(self.alphabet)}")
        for i, count in enumerate(H_counts):
            # print(f"--- #{i} - alphabet[{i}]={self.alphabet[i]} - count={count} ---")
            freq_table[self.alphabet[i]] = int(count) + 1 # avoid 0 counts; improves stability
        self.freq_table = freq_table
        # print(self.freq_table)
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
        x = int(x)
        # print(f"-> x={x} <-")
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
    from trainer import update_extrema, update_histogram, arithmetic_encoding

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
    parser.add_argument("-dt", "--data_type", type=str, default="norm_sphH4", help="type of dataset to train on (norm_H4, norm_sphH4)")
    parser.add_argument("-q", "--quant_bits", type=int, default=8, help="number of quantization bits for uniform quantization")
    parser.add_argument("-N", "--n_valid", type=int, default=1000, help="Number of samples in validation set for assessing arithmetic encoding")
    opt = parser.parse_args()

    device = torch.device(f'cuda:{opt.gpu_num}' if torch.cuda.is_available() else 'cpu')
    print(f"--- Device is {device} ---")

    # dataset pipeline vars 
    if opt.data_type == "norm_H4":
        json_config = "../config/deepcmc-indoor0001.json" if opt.env == "indoor" else "../config/deepcmc-outdoor300.json"
    elif opt.data_type == "norm_sphH4":
        json_config = "../config/deepcmc-indoor0001-sph-pow.json" if opt.env == "indoor" else "../config/deepcmc-outdoor300-sph-pow.json"
        # json_config = "../config/csinet-pro-quadriga-indoor0001-sph.json" if opt.env == "indoor" else "../config/csinet-pro-quadriga-outdoor300-sph.json"

    dataset_spec, minmax_file, img_channels, data_format, norm_range, T, base_pickle, n_delay, total_num_files, t1_power_file, subsample_prop, thresh_idx_path, diff_spec, network_name = get_keys_from_json(json_config, keys=["dataset_spec", "minmax_file", "img_channels", "data_format", "norm_range", "T", "base_pickle", "n_delay", "total_num_files", "t1_power_file", "subsample_prop", "thresh_idx_path", "diff_spec", "network_name"])
    # aux_bool_list = get_keys_from_json(json_config, keys=["aux_bool"], is_bool=True)
    network_name = "deepcmc"

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

    train_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_train).to(device), batch_size=batch_size, shuffle=True) 
    valid_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_val).to(device), batch_size=batch_size)

    bits = opt.quant_bits
    encoder = Encoder(input_dim[0], input_dim[1], opt.n_truncate)
    decoder = Decoder(input_dim[0], input_dim[1], opt.n_truncate)
    quant = UniformQuantize(bits, device=device)

    deepcmc = DeepCMC(encoder, decoder, quant, device=device).to(device)

    # load model weights
    # pickle_dir = f"{base_pickle}/t1"
    pickle_dir = f"{base_pickle}"

    # strs = ["checkpoint", "history"]
    # load_dict = {}
    # for str_i in strs:
    #     with open(f"{pickle_dir}/{network_name}-pretrain-{str_i}.pkl", "rb") as f:
    #         load_dict[str_i] = pickle.load(f)
    #         f.close()
    # checkpoint, history = load_dict["checkpoint"], load_dict["history"]
    # optimizer = optim.Adam(deepcmc.parameters(), lr=learning_rate)
    # opt_state_dict = torch.load(f"{pickle_dir}/{network_name}-pretrain-optimizer.pt")
    # optimizer.load_state_dict(opt_state_dict)

    print(f"Loading best model weights from {pickle_dir}/{network_name}-noise-best-model.pt")
    deepcmc.load_state_dict(torch.load(f"{pickle_dir}/{network_name}-noise-best-model.pt", map_location=device), strict=False)
    # deepcmc.quant.reset_extrema(checkpoint["best_z_min"], checkpoint["best_z_max"])

    deepcmc_ae = DeepCMCArithmeticEncoding(deepcmc.encoder, deepcmc.decoder, deepcmc.quant, device=device).to(device) # remake network with non-trainable sigma
    deepcmc_ae.quant.quant_mode = 2 # train with quantization layer

    optimizer = optim.Adam(deepcmc_ae.parameters(), lr=learning_rate)
    checkpoint = {}
    history = {}

    # get extrema on validation set
    deepcmc_ae.return_latent = True
    deepcmc_ae.entropy_encoding = False 
    deepcmc_ae.update_hist = False

    valid_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_val).to(device), batch_size=batch_size)
    update_extrema(deepcmc_ae,
                    valid_ldr,
                    timers=timers
                    )

    print(f"--- min: {deepcmc_ae.quant.z_min} - max: {deepcmc_ae.quant.z_max} ---")

    # # update histogram on validation set
    deepcmc_ae.update_hist = True
    deepcmc_ae.quant.return_denorm = False # for calculating histogram
    deepcmc_ae.return_latent = False
    deepcmc_ae.entropy_encoding = False 

    deepcmc_ae.quant.reset_hist()

    # valid_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_val).to(device), batch_size=batch_size)
    update_histogram(deepcmc_ae,
                    valid_ldr,
                    batch_num,
                    timers=timers,
                    json_config=json_config,
                    debug_flag=opt.debug_flag,
                    str_mod=f"DeepCMC (pre-loaded weights)",
                    n_train=data_train.shape[0],
                    pow_diff_t=pow_diff,
                    quant_bool=True
                    )
    del valid_ldr

    # arithmetic encoding on validation set
    deepcmc_ae.make_freq_table()

    deepcmc_ae.entropy_encoding = True
    deepcmc_ae.quant.return_denorm = False # for calculating histogram
    deepcmc_ae.return_latent = False
    deepcmc_ae.update_hist = False

    valid_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_val[:opt.n_valid]).to(device), batch_size=batch_size)

    arithmetic_encoding(deepcmc_ae,
                        valid_ldr,
                        batch_num,
                        timers=timers,
                        json_config=json_config,
                        debug_flag=opt.debug_flag,
                        str_mod=f"DeepCMC (pre-loaded weights)",
                        n_train=data_train.shape[0],
                        pow_diff_t=pow_diff,
                        quant_bool=True
                        )

    deepcmc_ae.print_avg_bits_per_message()