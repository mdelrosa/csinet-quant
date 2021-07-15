import sys
import copy
from numpy.lib.function_base import kaiser
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch import nn, optim, autograd
from torch.utils.checkpoint import checkpoint_sequential
from collections import OrderedDict

sys.path.append("/home/mason/git/brat")
from utils.NMSE_performance import get_NMSE, denorm_H3, denorm_H4, denorm_sphH4
from utils.data_tools import dataset_pipeline, subsample_batches, split_complex, load_pow_diff
from utils.unpack_json import get_keys_from_json

def fit(model, train_ldr, valid_ldr, batch_num, schedule=None, criterion=nn.MSELoss(), epochs=10, timers=None, json_config=None, torch_type=torch.float, debug_flag=True, pickle_dir=".", input_type="split", patience=1000, network_name=None, quant_bool=False, lr=None, beta=1e-5, alpha=1.0, l2_weight=1e-12, data_all=None, timeslot=0,history=None,checkpoint=None,optimizer=None,pretrained_epochs=0):
    # pull out timers
    fit_timer = timers["fit_timer"] 

    # load hyperparms
    network_name = get_keys_from_json(json_config, keys=["network_name"])[0] if network_name == None else network_name
    batch_size, minmax_file, norm_range, thresh_idx_path = get_keys_from_json(json_config, keys=["batch_size", "minmax_file", "norm_range", "thresh_idx_path"])

    # criterion = nn.MSELoss()
    # TODO: if we use lr_schedule, then do we need to use SGD instead? 
    learning_rate, lr_quant = get_keys_from_json(json_config, keys=['learning_rate', 'learning_rate_quant'])
    # lr = lr_quant if quant_bool else lr
    lr = learning_rate if type(lr) == type(None) else lr

    if type(optimizer) == type(None):
        print(f"--- Learning rate for network is {lr:2.1E} ---")
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # TODO: Load in epoch
    if type(checkpoint) == type(None):
        checkpoint = {
                        "latest_model": None,
                        # "latest_z_min": None,
                        # "latest_z_max": None,
                        "latest_epoch": None,
                        "best_model": None,
                        # "best_z_min": None,
                        # "best_z_max": None,
                        "best_epoch": 0,
                        "best_mse": None,
                        "best_nmse": None,
                        # "optimizer_state": None,
        }

    if type(history) == type(None):
        history = {
                        "train_loss": np.zeros(epochs),
                        "test_loss": np.zeros(epochs),
                        # "z_min": np.zeros(epochs),
                        # "z_max": np.zeros(epochs),
                    }

        if quant_bool or model.noise_bool:
            history["train_mse"] = np.zeros(epochs)
            history["train_entropy"] = np.zeros(epochs)
            history["test_mse"] = np.zeros(epochs)
            history["test_entropy"] = np.zeros(epochs)

        if type(data_all) != None:
            history["nmse_denorm"] = np.zeros(epochs)
            history["mse_denorm"] = np.zeros(epochs)

    best_test_loss = None
    epochs_no_improvement = 0
    # grace_period = 1000
    # anneal_range = [0.0, 1.0]
    # beta_annealer = AnnealingSchedule(anneal_range, epoch_limit=grace_period)
    # clip_val = 1e5 # tuning this

    autograd.set_detect_anomaly(True)

    n_train = 0
    true_epochs = pretrained_epochs + epochs
    with fit_timer:
        for i_epoch in range(epochs):
            epoch = pretrained_epochs + i_epoch # adjust epoch based on pretraining
            # z_min, z_max = 1e6, -1e6
            train_loss = 0
            train_mse = 0
            train_entropy = 0
            model.training = True
            desc_str = f"Epoch #{epoch+1}" 
            for i, data_tuple in enumerate(tqdm(train_ldr, desc=desc_str), 0):
                if len(data_tuple) != 2:
                    data_batch = data_tuple
                else:
                    aux_batch, data_batch = data_tuple
                    aux_input = autograd.Variable(aux_batch)
                h_input = autograd.Variable(data_batch)
                optimizer.zero_grad()
                model_in = h_input if len(data_tuple) != 2 else [aux_input, h_input]
                # dec = checkpoint_sequential(model, 3, model_in)
                # dec = model(model_in)
                # batch_size_i = dec.size(0)
                # mse = criterion(dec, h_input)
                if quant_bool:
                    z_quant, dec = model(model_in)
                    mse = criterion(dec, h_input)
                    train_mse += mse
                    if type(model.quant.z_max) != type(None):
                        # z_quant = (z_quant - model.quant.z_min) / (model.quant.z_max - model.quant.z_min) # cast to range [0,1]
                        # print(f"normalized -- range z_quant: {torch.min(z_quant)} to {torch.max(z_quant)}")
                        model.quant.reset_hist()
                        model.quant.update_hist(z_quant)
                        entropy = model.quant.get_entropy()
                        train_entropy += entropy
                        loss_i = mse+beta*entropy
                    else:
                        loss_i = mse
                elif model.noise_bool:
                    p_y, dec = model(model_in)
                    mse = criterion(dec, h_input)
                    train_mse += mse
                    entropy = model.entropy_loss(p_y)
                    train_entropy += entropy
                    # print(f"beta: {beta} - mse: {mse} - entropy: {entropy}")
                    loss_i = beta*mse+alpha*entropy
                else:
                    dec = model(model_in)
                    mse = criterion(dec, h_input)
                    loss_i = mse
                batch_size_i = dec.size(0)
                train_loss += loss_i
                loss_i.backward()
                optimizer.step()
                # if quant_bool and anneal_bool:
                #     with torch.no_grad():
                #         model.update_gap(mse, dec, h_input)
                #         model.anneal_sigma()
                #     train_gap += model.gap_t
                
                # update exterma in latent layer
                # z_min = np.min([z_min, model.z_min_temp])
                # z_max = np.max([z_max, model.z_max_temp])

                if (i != 0):
                    tqdm.write(f"\033[A                                                         \033[A")
                # choose string based on quantization, annealing
                if (quant_bool and model.quant.z_max != None) or (model.noise_bool):
                    tqdm_str = f"Epoch #{epoch+1}/{true_epochs}: Training loss: {loss_i:.5E} - mse: {mse.data:.5E} - entropy: {entropy.data:.5E}"
                else:
                    tqdm_str = f"Epoch #{epoch+1}/{true_epochs}: Training loss: {mse.data:.5E}" 
                tqdm.write(tqdm_str)
                if epoch == 0:
                    n_train += batch_size_i

            # post training step, dump to checkpoint
            checkpoint["model"] = copy.deepcopy(model).to("cpu").state_dict()
            # optimizer_state = copy.deepcopy(optimizer.state_dict())
            checkpoint["latest_epoch"] = epoch
            history["train_loss"][epoch] = train_loss.detach().to("cpu").numpy() / (i+1) if not quant_bool else train_loss / (i+1)
            if (quant_bool and type(model.quant.z_max) != type(None)) or model.noise_bool:
                history["train_mse"][epoch] = train_mse / (i+1)
                history["train_entropy"][epoch] = train_entropy / (i+1)
            #     if anneal_bool:
            #         history["train_gap"][epoch] = train_gap.detach().to("cpu").numpy() / (i+1)

            # validation step
            # model.training = False # optionally check just the MSE performance during eval
            with torch.no_grad():
                if type(data_all) != type(None):
                    y_hat = torch.zeros(data_all.shape, dtype=torch_type).to("cpu")
                    y_test = torch.zeros(data_all.shape, dtype=torch_type).to("cpu")
                    print(f"-> y_hat.shape: {y_hat.shape} ")
                test_loss = 0
                test_mse = 0
                test_entropy = 0
                test_gap = 0
                # for i, data_batch in enumerate(valid_ldr):
                for i, data_tuple in enumerate(valid_ldr):
                    # inputs = autograd.Variable(data_batch).float()
                    if len(data_tuple) != 2:
                        data_batch = data_tuple
                    # elif len(data_tuple) == 2:
                    else:
                        aux_batch, data_batch = data_tuple
                        aux_input = autograd.Variable(aux_batch)
                    h_input = autograd.Variable(data_batch)
                    optimizer.zero_grad()
                    model_in = h_input if len(data_tuple) != 2 else [aux_input, h_input]
                    # dec = checkpoint_sequential(model, 3, model_in)
                    # dec = model(model_in)
                    # mse = criterion(dec, h_input)
                    # test_loss += mse

                    # update exterma in latent layer
                    # z_min = np.min([z_min, model.z_min_temp])
                    # z_max = np.max([z_max, model.z_max_temp])

                    if quant_bool:
                        z_quant, dec = model(model_in)
                        mse = criterion(dec, h_input)
                        test_mse += mse
                        if type(model.quant.z_max) != type(None):
                            # z_quant = (z_quant - model.quant.z_min) / (model.quant.z_max - model.quant.z_min) # cast to range [0,1]
                            # print(f"normalized -- range z_quant: {torch.min(z_quant)} to {torch.max(z_quant)}")
                            model.quant.reset_hist()
                            model.quant.update_hist(z_quant)
                            entropy = model.quant.get_entropy()
                            test_entropy += entropy
                            loss_i = mse+beta*entropy
                        else:
                            loss_i = mse
                    elif model.noise_bool:
                        p_y, dec = entropy = model(model_in)
                        mse = criterion(dec, h_input)
                        test_mse += mse
                        entropy = model.entropy_loss(p_y)
                        test_entropy += entropy
                        loss_i = beta*mse+alpha*entropy
                    else:
                        dec = model(model_in)
                        batch_size_i = dec.size(0)
                        mse = criterion(dec, h_input)
                        loss_i = mse
                    test_loss += loss_i
                    batch_size_i = dec.size(0)

                    if type(data_all) != type(None):
                        idx_s = i*batch_size
                        idx_e = min((i+1)*batch_size, y_hat.shape[0])
                        y_hat[n_train+idx_s:n_train+idx_e,:,:,:] = dec.to("cpu")
                        y_test[n_train+idx_s:n_train+idx_e,:,:,:] = h_input.to("cpu")
                    loss_i = mse
                history["test_loss"][epoch] = test_loss.detach().to("cpu").numpy() / (i+1) if not quant_bool else test_loss / (i+1)

                if type(data_all) != type(None):
                    # calculate nmse
                    if norm_range == "norm_H4":
                        y_hat_denorm = denorm_H4(y_hat.to("cpu").detach().numpy(),minmax_file)
                        y_test_denorm = denorm_H4(y_test.to("cpu").detach().numpy(),minmax_file)
                    elif norm_range == "norm_sphH4":
                        t1_power_file = get_keys_from_json(json_config, keys=["t1_power_file"])[0]
                        y_hat_denorm = denorm_sphH4(y_hat.detach().numpy(),minmax_file, t1_power_file, batch_num, timeslot=timeslot, thresh_idx_path=thresh_idx_path)
                        y_test_denorm = denorm_sphH4(y_test.detach().numpy(),minmax_file, t1_power_file, batch_num, timeslot=timeslot, thresh_idx_path=thresh_idx_path)
                    y_test_denorm, y_hat_denorm = y_test_denorm[n_train:,:,:,:], y_hat_denorm[n_train:,:,:,:]
                    y_hat_denorm = y_hat_denorm[:,0,:,:] + 1j*y_hat_denorm[:,1,:,:]
                    y_test_denorm = y_test_denorm[:,0,:,:] + 1j*y_test_denorm[:,1,:,:]
                    y_shape = y_test_denorm.shape
                    mse_denorm, nmse_denorm = get_NMSE(y_hat_denorm, y_test_denorm, return_mse=True, n_ang=y_shape[1], n_del=y_shape[2]) # one-step prediction -> estimate of single timeslot
                    # print(f"-> {str_mod} - truncate | NMSE = {nmse:5.3f} | MSE = {mse:.4E}")
                    history["nmse_denorm"][epoch] = nmse_denorm
                    history["mse_denorm"][epoch] = mse_denorm

                if (quant_bool and type(model.quant.z_max) != type(None)) or model.noise_bool:
                    history["test_mse"][epoch] = test_mse / (i+1)
                    history["test_entropy"][epoch] = test_entropy / (i+1)
                    # if anneal_bool:
                    #     history["test_gap"][epoch] = test_gap.detach().to("cpu").numpy() / (i+1)
                # if epoch >= grace_period:
                if type(best_test_loss) == type(None) or best_test_loss > history["test_loss"][epoch]:
                    best_test_loss = history["test_loss"][epoch]
                    checkpoint["best_epoch"] = epoch
                    checkpoint["best_model"] = copy.deepcopy(model).to("cpu").state_dict()
                    # checkpoint["best_z_min"] = z_min
                    # checkpoint["best_z_max"] = z_max
                    epochs_no_improvement = 0
                    # if quant_bool:
                        # checkpoint["best_sigma"] = model.quant.sigma
                    if not debug_flag:
                        torch.save(checkpoint["best_model"], f"{pickle_dir}/{network_name}-best-model.pt")
                    print(f"Epoch #{epoch+1}/{true_epochs}: Test loss: {history['test_loss'][epoch]:.5E} -- New best epoch: {epoch+1}")
                elif epochs_no_improvement < patience:
                    epochs_no_improvement += 1
                    print(f"Epoch #{epoch+1}/{true_epochs}: Test loss: {history['test_loss'][epoch]:.5E} -- Test loss did not improve. Best epoch: #{checkpoint['best_epoch']+1}")
                else:
                    model.load_state_dict(checkpoint["best_model"])
                    # model.quant.sigma = checkpoint["best_sigma"]
                    print(f"Epoch #{epoch+1}/{true_epochs}: Test loss: {history['test_loss'][epoch]:.5E} -- Test loss did not improve for {patience} epochs. Loading best epoch #{checkpoint['best_epoch']+1}")
                    break
                # else:
                #     # don't track best epoch until grace period has expired
                #     print(f"Epoch #{epoch+1}/{epochs}: Test loss: {history['test_loss'][epoch]:.5E}. Grace period is {grace_period} epochs.")
                checkpoint["latest_model"] = copy.deepcopy(model).to("cpu").state_dict()
                # checkpoint["latest_z_min"] = z_min
                # checkpoint["latest_z_max"] = z_max
                if quant_bool or model.noise_bool:
                    if type(data_all) != type(None):
                        val_str = f"Epoch #{epoch+1}/{true_epochs}: Training (loss: {history['train_loss'][epoch]:4.3E} | mse: {history['train_mse'][epoch]:4.3E} | entropy: {history['train_entropy'][epoch]:4.3E}) Test (loss: {history['test_loss'][epoch]:4.3E} | mse: {history['test_mse'][epoch]:4.3E} | entropy: {history['test_entropy'][epoch]:4.3E} | nmse_denorm: {nmse_denorm:5.3f} dB)"
                    else:
                        val_str = f"Epoch #{epoch+1}/{true_epochs}: Training (loss: {history['train_loss'][epoch]:4.3E} | mse: {history['train_mse'][epoch]:4.3E} | entropy: {history['train_entropy'][epoch]:4.3E}) Test (loss: {history['test_loss'][epoch]:4.3E} | mse: {history['test_mse'][epoch]:4.3E} | entropy: {history['test_entropy'][epoch]:4.3E})"
                else:
                    val_str = f"Epoch #{epoch+1}/{true_epochs}: Training loss: {history['train_loss'][epoch]:4.3E} -- Test loss: {history['test_loss'][epoch]:4.3E}"
                tqdm.write(val_str)
            # history["z_min"][epoch] = z_min
            # history["z_max"][epoch] = z_max
            # model.quant.reset_extrema(z_min, z_max)
            # print(f"--- extrema: z_min={z_min}, z_max={z_max} ---")

    # print(f"--- checkpoint['best_sigma'] = {checkpoint['best_sigma']:4.3f} ---")
    return [model, checkpoint, history, optimizer, timers]

def score(model, valid_ldr, data_val, batch_num, checkpoint, history, optimizer, timeslot=0, err_dict=None, timers=None, json_config=None, debug_flag=True, str_mod="", torch_type=torch.float, n_train=0, pow_diff_t=None, key_mod="", quant_bool=False):
    """
    take model, predict on valid_ldr, score
    currently scores a spherically normalized dataset
    """

    # pull out timers
    predict_timer = timers["predict_timer"]
    score_timer = timers["score_timer"]

    batch_size, minmax_file, norm_range, thresh_idx_path = get_keys_from_json(json_config, keys=["batch_size", "minmax_file", "norm_range", "thresh_idx_path"])

    test_entropy = 0
    # z_min, z_max = 1e6, -1e6

    with predict_timer:
        # y_hat = torch.zeros(data_val.shape).to(device)
        # y_test = torch.zeros(data_val.shape).to(device)
        model.training = False
        model.eval()
        with torch.no_grad():
            y_hat = torch.zeros(data_val.shape, dtype=torch_type).to("cpu")
            y_test = torch.zeros(data_val.shape, dtype=torch_type).to("cpu")
            for i, data_tuple in enumerate(valid_ldr):
                # inputs = autograd.Variable(data_batch).float()
                if len(data_tuple) != 2:
                    data_batch = data_tuple
                # elif len(data_tuple) == 2:
                else:
                    aux_batch, data_batch = data_tuple
                    aux_input = autograd.Variable(aux_batch)
                h_input = autograd.Variable(data_batch)
                model_in = h_input if len(data_tuple) != 2 else [aux_input, h_input]
            # for i, data_batch in enumerate(valid_ldr):
            #     # inputs = autograd.Variable(data_batch).float()
            #     inputs = autograd.Variable(data_batch)
                optimizer.zero_grad()
                idx_s = i*batch_size
                idx_e = min((i+1)*batch_size, y_hat.shape[0])

                if model.quant_bool:
                    z_quant, dec = model(model_in)
                    if type(model.quant.z_max) != type(None):
                        z_quant = (z_quant - model.quant.z_min) / (model.quant.z_max - model.quant.z_min) # cast to range [0,1]
                        model.quant.reset_hist()
                        model.quant.update_hist(z_quant)
                        entropy = model.quant.get_entropy()
                        test_entropy += entropy
                elif model.noise_bool:
                    p_y, dec = model(model_in)
                    # mse = criterion(dec, h_input)
                    # test_mse += mse
                    entropy = model.entropy_loss(p_y)
                    test_entropy += entropy
                    # loss_i = beta*mse+entropy
                else:
                    dec = model(model_in).to("cpu")
                y_hat[idx_s:idx_e,:,:,:] = dec.to("cpu")
                y_test[idx_s:idx_e,:,:,:] = h_input.to("cpu")
                # z_min = np.min([z_min, model.z_min_temp])
                # z_max = np.max([z_max, model.z_max_temp])
    if (model.quant_bool and type(model.quant.z_max) != type(None)) or model.noise_bool:
        test_entropy = test_entropy / (i+1)
    # model.quant.reset_extrema(z_min, z_max)
    # print(f"--- extrema: z_min={z_min}, z_max={z_max} ---")

    # for markovnet, we add "addend" to the error to get our actual estimates
    if type(err_dict) != type(None):            
        print("--- err_dict passed in - calculating estimates with error added back --- ")
        y_hat = y_hat * err_dict["M"] + err_dict["hat"]
        y_test = y_test * err_dict["M"] + err_dict["hat"]

    # score model - account for spherical normalization
    with score_timer:
        if y_hat.shape[1] == 1:
            y_hat = torch.cat((y_hat.real, y_hat.imag), 1)
            y_test = torch.cat((y_test.real, y_test.imag), 1)
        print('-> pre denorm: y_hat range is from {} to {}'.format(np.min(y_hat.detach().numpy()), np.max(y_hat.detach().numpy())))
        print('-> pre denorm: y_test range is from {} to {}'.format(np.min(y_test.detach().numpy()),np.max(y_test.detach().numpy())))
        if norm_range == "norm_H3":
            y_hat_denorm = denorm_H3(y_hat.detach().numpy(),minmax_file)
            y_test_denorm = denorm_H3(y_test.detach().numpy(),minmax_file)
        elif norm_range == "norm_H4":
            y_hat_denorm = denorm_H4(y_hat.to("cpu").detach().numpy(),minmax_file)
            y_test_denorm = denorm_H4(y_test.to("cpu").detach().numpy(),minmax_file)
        elif norm_range == "norm_sphH4":
            t1_power_file = get_keys_from_json(json_config, keys=["t1_power_file"])[0]
            y_hat_denorm = denorm_sphH4(y_hat.detach().numpy(),minmax_file, t1_power_file, batch_num, timeslot=timeslot, thresh_idx_path=thresh_idx_path)
            y_test_denorm = denorm_sphH4(y_test.detach().numpy(),minmax_file, t1_power_file, batch_num, timeslot=timeslot, thresh_idx_path=thresh_idx_path)
        # predicted on pooled data -- split out validation set
        print('-> post denorm: y_hat range is from {} to {}'.format(np.min(y_hat_denorm),np.max(y_hat_denorm)))
        print('-> post denorm: y_test range is from {} to {}'.format(np.min(y_test_denorm),np.max(y_test_denorm)))
        y_test_denorm, y_hat_denorm = y_test_denorm[n_train:,:,:,:], y_hat_denorm[n_train:,:,:,:]
        y_hat_denorm = y_hat_denorm[:,0,:,:] + 1j*y_hat_denorm[:,1,:,:]
        y_test_denorm = y_test_denorm[:,0,:,:] + 1j*y_test_denorm[:,1,:,:]
        y_shape = y_test_denorm.shape
        mse, nmse = get_NMSE(y_hat_denorm, y_test_denorm, return_mse=True, n_ang=y_shape[1], n_del=y_shape[2]) # one-step prediction -> estimate of single timeslot
        print(f"-> {str_mod} - truncate | NMSE = {nmse:5.3f} | MSE = {mse:.4E}")
        checkpoint[f"best_nmse{key_mod}"] = nmse
        checkpoint[f"best_mse{key_mod}"] = mse

        if type(pow_diff_t) != type(None): 
            mse, nmse = get_NMSE(y_hat_denorm, y_test_denorm, return_mse=True, n_ang=y_shape[1], n_del=y_shape[2], pow_diff_timeslot=pow_diff_t[n_train:]) # one-step prediction -> estimate of single timeslot
            print(f"-> {str_mod} - all | NMSE = {nmse:5.3f} | MSE = {mse:.4E}")
            checkpoint[f"best_nmse_full{key_mod}"] = nmse
            checkpoint[f"best_mse_full{key_mod}"] = mse

        if model.quant_bool:
            # bpps = test_entropy / (np.prod(model.encoder.sample_factors)) # bits per pixel
            bpps = test_entropy / model.encoder.img_total # bits per pixel
            print(f"-> best_entropy={test_entropy:4.3E} - bpps={bpps:4.3f}")
            checkpoint["best_entropy"] = test_entropy
            checkpoint["bpps"] = bpps
        elif model.noise_bool:
            # bpps = test_entropy / (np.prod(model.encoder.sample_factors)) # bits per pixel
            print(f"-> best_entropy={test_entropy:4.3E}")
            checkpoint["best_entropy"] = test_entropy
            # checkpoint["bpps"] = bpps

    return [checkpoint, y_hat, y_test]

def update_extrema(model, data_ldr, timers=None):
    """
    take model, update histogram estimates based on loader
    """
    assert(model.return_latent == True)
    # pull out timers
    predict_timer = timers["predict_timer"]

    model.quant.reset_hist()

    z_min, z_max = 1e6, -1e6
    with predict_timer:
        # y_hat = torch.zeros(data_val.shape).to(device)
        # y_test = torch.zeros(data_val.shape).to(device)
        model.training = False
        model.eval()
        with torch.no_grad():
            for i, data_tuple in enumerate(data_ldr):
                # inputs = autograd.Variable(data_batch).float()
                if len(data_tuple) != 2:
                    data_batch = data_tuple
                # elif len(data_tuple) == 2:
                else:
                    aux_batch, data_batch = data_tuple
                    aux_input = autograd.Variable(aux_batch)
                h_input = autograd.Variable(data_batch)
                # model_in = h_input if len(data_tuple) != 2 else [aux_input, h_input]
                y_hat = model(h_input)
                z_min, z_max = torch.min(y_hat), torch.max(y_hat)

            model.quant.reset_extrema(z_min, z_max)

def update_histogram(model, valid_ldr, batch_num, timeslot=0, err_dict=None, timers=None, json_config=None, debug_flag=True, str_mod="", torch_type=torch.float, n_train=0, pow_diff_t=None, key_mod="", quant_bool=False, entropy_coding=False, count_centers=False):
    """
    take model, predict on valid_ldr, score
    count occurrences for each center
    """
    # pull out timers
    predict_timer = timers["predict_timer"]
    score_timer = timers["score_timer"]

    batch_size, minmax_file, norm_range, thresh_idx_path = get_keys_from_json(json_config, keys=["batch_size", "minmax_file", "norm_range", "thresh_idx_path"])

    test_entropy = 0

    model.quant.mode = 2

    with predict_timer:
        # y_hat = torch.zeros(data_val.shape).to(device)
        # y_test = torch.zeros(data_val.shape).to(device)
        model.training = False
        model.eval()
        with torch.no_grad():
            # y_hat = torch.zeros(data_val.shape, dtype=torch_type).to("cpu")
            # y_test = torch.zeros(data_val.shape, dtype=torch_type).to("cpu")
            for i, data_tuple in enumerate(valid_ldr):
                # inputs = autograd.Variable(data_batch).float()
                if len(data_tuple) != 2:
                    data_batch = data_tuple
                else:
                    aux_batch, data_batch = data_tuple
                    aux_input = autograd.Variable(aux_batch)
                h_input = autograd.Variable(data_batch)
                model_in = h_input if len(data_tuple) != 2 else [aux_input, h_input]
                y_hat = model(model_in)

    print(f"model.quant.p_hist: {model.quant.p_hist}")

def arithmetic_encoding(model, valid_ldr, batch_num, timeslot=0, err_dict=None, timers=None, json_config=None, debug_flag=True, str_mod="", torch_type=torch.float, n_train=0, pow_diff_t=None, key_mod="", quant_bool=False, entropy_coding=False, count_centers=False):
    """
    take model, predict on valid_ldr, score
    model = 
    inference with arithmetic encoding in latent layer
    """
    # pull out timers
    predict_timer = timers["predict_timer"]
    score_timer = timers["score_timer"]

    batch_size, minmax_file, norm_range, thresh_idx_path = get_keys_from_json(json_config, keys=["batch_size", "minmax_file", "norm_range", "thresh_idx_path"])

    test_entropy = 0

    model.quant.mode = 2

    with predict_timer:
        # y_hat = torch.zeros(data_val.shape).to(device)
        # y_test = torch.zeros(data_val.shape).to(device)
        model.training = False
        model.eval()
        with torch.no_grad():
            # y_hat = torch.zeros(data_val.shape, dtype=torch_type).to("cpu")
            # y_test = torch.zeros(data_val.shape, dtype=torch_type).to("cpu")
            for i, data_tuple in enumerate(valid_ldr):
                # inputs = autograd.Variable(data_batch).float()
                if len(data_tuple) != 2:
                    data_batch = data_tuple
                # elif len(data_tuple) == 2:
                else:
                    aux_batch, data_batch = data_tuple
                    aux_input = autograd.Variable(aux_batch)
                h_input = autograd.Variable(data_batch)
                model_in = h_input if len(data_tuple) != 2 else [aux_input, h_input]
                y_hat = model(model_in)

    # print(f"model.message_bit_lens: mean={np.mean(csinet_quant.message_bit_lens)} - min={np.min(csinet_quant.message_bit_lens)} - max={np.max(csinet_quant.message_bit_lens)} ")

def profile_network(model, train_ldr, valid_ldr, batch_num, schedule=None, criterion=nn.MSELoss(), epochs=10, timers=None, json_config=None, quant_bool=True):
    lr = get_keys_from_json(json_config, keys=['learning_rate'])[0] 
    beta = get_keys_from_json(json_config, keys=['beta'])[0] # hyperparam -- entropy weight 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    with autograd.profiler.profile(record_shapes=True, use_cuda=True) as prof:
        with autograd.profiler.record_function("model_inference"):
            for epoch in range(epochs):
                train_loss = 0
                model.training = True
                for i, data_tuple in enumerate(tqdm(train_ldr, desc=f"Epoch #{epoch+1}"), 0):
                    # inputs = autograd.Variable(data_batch).float()
                    # print(f"len(data_tuple): {len(data_tuple)}")
                    if len(data_tuple) != 2:
                        data_batch = data_tuple
                    # elif len(data_tuple) == 2:
                    else:
                        aux_batch, data_batch = data_tuple
                        aux_input = autograd.Variable(aux_batch)
                    h_input = autograd.Variable(data_batch)
                    optimizer.zero_grad()
                    model_in = h_input if len(data_tuple) != 2 else [aux_input, h_input]
                    dec = model(model_in)
                    mse = criterion(dec, h_input)
                    if quant_bool:
                        loss_i = mse+beta*model.crossentropy_loss(model_in)
                    else:
                        loss_i = mse
                    train_loss += loss_i
                    loss_i.backward()
                    optimizer.step()
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

def load_pretrained(model, json_config, pickle_dir, epochs=0, device="cpu", lr=None):

    network_name, learning_rate = get_keys_from_json(json_config, keys=["network_name", "learning_rate"])

    lr = learning_rate if type(lr) == type(None) else lr
    print(f"--- Learning rate for pretrained network is {lr:2.1E} ---")

    strs = ["checkpoint", "history"]
    load_dict = {}
    for str_i in strs:
        with open(f"{pickle_dir}/{network_name}-pretrain-{str_i}.pkl", "rb") as f:
            load_dict[str_i] = pickle.load(f) # if str_i == "history" else torch.load(f, map_location=device)
            f.close()
    checkpoint, history = load_dict["checkpoint"], load_dict["history"]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    opt_state_dict = torch.load(f"{pickle_dir}/{network_name}-pretrain-optimizer.pt")
    optimizer.load_state_dict(opt_state_dict)
    print(f"Loading best model weights from checkpoint['latest_model']")
    model_state = torch.load(f"{pickle_dir}/{network_name}-pretrain-model.pt", map_location=device)
    model.load_state_dict(model_state, strict=False)
    # print(f"Loading best model weights from {pickle_dir}/{network_name}-noise-best-model.pt")
    # model.load_state_dict(torch.load(f"{pickle_dir}/{network_name}-noise-best-model.pt", map_location=device), strict=False)

    if epochs > 0:
        history_new = {}
        for key, val in history.items():
            # add to <epochs> of zeros to each list in history
	        history_new[key] = np.concatenate([val, np.zeros(epochs)])
        pretrained_epochs = len(val)
        history = history_new

    return [model, checkpoint, history, optimizer, pretrained_epochs]