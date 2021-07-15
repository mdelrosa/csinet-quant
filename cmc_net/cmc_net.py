import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from collections import OrderedDict

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

class MultipleUnivariateDensity(nn.Module):
    def __init__(self, l, K, r, device="cpu"):
        """
        l = latent dimension of density
        K = depth
        r = dimension of autoencoder latent vector
        """
        super(MultipleUnivariateDensity, self).__init__()
        self.K = K
        for i in range(1,K+1):
            j = 1 if i == 1 else l
            k = 1 if i == K else l
            setattr(self, f"f_{i}", g_cumul_k_multi(i,K,j,k,r,device=device))

    def forward(self, x):
        for i in range(1, self.K+1):
            x = getattr(self, f"f_{i}")(x)
        return x

class UnivariateDensity(nn.Module):
    def __init__(self, l, K, p_idx, device="cpu"):
        """
        l = latent dimension 
        K = depth
        p_idx = index of univariate density in latent dimension
        """
        super(UnivariateDensity, self).__init__()
        self.K = K
        self.p_idx = p_idx
        for i in range(1,K+1):
            j = 1 if i == 1 else l
            k = 1 if i == K else l
            setattr(self, f"f_{p_idx}_{i}", g_cumul_k(i,K,j,k,device=device))
        # self.f_1 = g_cumul_k(1,K,1,l)
        # self.f_2 = g_cumul_k(2,K,l,l)
        # self.f_3 = g_cumul_k(3,K,l,l)
        # self.f_K = g_cumul_k(K,K,l,1)

    def forward(self, x):
        for i in range(1, self.K+1):
            x = getattr(self, f"f_{self.p_idx}_{i}")(x)
        return x
        # return self.f_K(self.f_3(self.f_2(self.f_1(x))))

class MultivariateDensity(nn.Module):
    def __init__(self, r, l, K, device="cpu"):
        super(MultivariateDensity, self).__init__()
        self.device = device
        self.r = r
        self.y = None
        self.device_factory()
        # print(f"self.device_list: {self.device_list}")
        for i in range(1,r+1):
            setattr(self, f"p_{i}", UnivariateDensity(l, K, i, device=self.device_list[i-1]).to(self.device_list[i-1]))
        self.densities = [getattr(self, f"p_{i+1}") for i in range(self.r)]

    
    def device_factory(self, num=8):
        """ create torch device list up to num """
        self.device_list = [torch.device(f"cuda:{i}") for i in range(num)]
        n = int(self.r / len(self.device_list))
        self.device_list = list(itertools.chain.from_iterable(itertools.repeat(x, n) for x in self.device_list))

    def forward(self, x):
        bs = x.size(0)
        # warning: this is probably really slow. vectorize this?
        # print(f"x.size(): {x.size()}")
        # if type(self.y) == type(None):
        #     self.y = torch.zeros(size=(x.size(0), self.r), device=self.device)

        # # apply p_i for each element with for loop
        # for i in range(self.r):
        #     temp = x[:,i].clone().unsqueeze(1) # reshape
        #     # print(f"#{i}: slice temp.size(): {temp.size()}")
        #     temp = getattr(self, f"p_{i+1}")(temp) # p_i
        #     # print(f"#{i}: p_i temp.size(): {temp.size()}")
        #     # x[:,i] = temp.squeeze() # write back
        #     self.y[:,i] = temp.squeeze()
        #     # x[:,i] = temp # write back
        #     # x[:,i] = getattr(self, f"p_{i}")(temp)

        # apply p_i for each element with list comprehension
        # return torch.cat([f(x[:,i].unsqueeze(1)) for i, f in enumerate(self.densities)], axis=1)

        # use parallel_apply from torch.nn.parallel lib
        print(f'multivariate input: {x.size()}')
        # x = list(torch.unbind(x,1))
        x = torch.transpose(x,0,1)
        print(f'transposed input: {x.size()}')
        # x = torch.nn.parallel.scatter(x, [self.device]*len(x))
        x = torch.nn.parallel.scatter(x, self.device_list)
        print(f'pre parallel_apply: {len(x)} - first element size: {x[0].size()}')
        # x = torch.nn.parallel.parallel_apply(self.densities, x, devices=[self.device]*len(x))
        x = torch.nn.parallel.parallel_apply(self.densities, x, devices=self.device_list)
        print(f'post parallel_apply: {len(x)} - first element size: {x[0].size()}')
        x = torch.nn.parallel.gather(x, self.device)
        x = torch.reshape(x, (self.r, bs))
        print(f'post gather: {x.size()} - first element size: {x[0].size()}')
        # print(f'post gather: {len(x)}')
        x = torch.transpose(x,0,1)
        # x = torch.cat(x, axis=1)
        print(f'transposed output: {x.size()}')
        return x

        # self.y = torch.cat([f(x[:,i].unsqueeze(1)) for i, f in enumerate(self.densities)], axis=1)
        # return x
        # return self.y.clone()

class DeepCMC(nn.Module):
    """ DeepCMC for csi estimation with entropy-based loss term """
    def __init__(self, encoder, decoder, quant, batch_size=200, device=None):
        super(DeepCMC, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quant = quant
        self.device = device
        self.quant_bool = True
        self.noise_bool = True
        self.return_latent = False
        self.training = True
        self.batch_size = batch_size

        self.make_prior()

        # self.p_hist = torch.zeros(self.quant.L).to(self.device) # histogram estimate of probabilities
        # self.p_mask = torch.arange(self.quant.L).reshape(1,self.quant.L,1).repeat(batch_size,1,self.quant.n_features).to(self.device)
        # self.p_hard = torch.zeros(batch_size,self.quant.L,self.quant.n_features).to(self.device) # to save on inference time, we will store forward passes with hard quantization here

    def make_prior(self, r_k=3):
        """
        scale hyperprior for estimating latent density 
        """
        K = 4
        k = int(self.encoder.img_total / np.prod(self.encoder.sample_factors)**2)
        c = self.encoder.latent_chan * k

        # self.multivar = MultivariateDensity(c, 3, K, device=self.device)
        # self.univar = UnivariateDensity(r_k, K, 0)
        self.multivar = MultipleUnivariateDensity(r_k, K, c, device=self.device)
        # def __init__(self, l, K, r, device="cpu"):

        self.c = c
        self.uni = torch.distributions.uniform.Uniform(torch.Tensor([-0.5]).to(self.device), torch.Tensor([0.5]).to(self.device))

    def cumulative(self, x):
        return self.f_K(self.f_3(self.f_2(self.f_1(x))))

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
            quant = self.quant(h_enc)
            if self.return_latent:
                return quant
            else:
                return [quant, self.decoder(quant)]
        elif self.noise_bool:
            # self.uni = torch.rand(size=h_enc.size())
            # h_enc += torch.rand(size=h_enc.size(), device=device) - 0.5
            noise = self.uni.sample(h_enc.size()).squeeze()
            h_enc = h_enc + noise
            # p_y = self.cumulative(h_enc.view(enc_shape[0], self.c) + 0.5) - self.cumulative(h_enc.view(enc_shape[0], self.c) - 0.5)
            # p_y = self.multivar(h_enc.view(enc_shape[0], self.c) + 0.5) - self.multivar(h_enc.view(enc_shape[0], self.c) - 0.5)

            # multivariate - using list comprehension (see MultivariateDensity)
            # p_y = self.multivar(torch.reshape(h_enc, (h_enc.size(0), -1)) + 0.5) - self.multivar(torch.reshape(h_enc, (h_enc.size(0), -1)) - 0.5)

            # multivariate - using parallel_apply (see MultivariateDensity)
            # p_positive = self.multivar(torch.reshape(h_enc, (h_enc.size(0), -1))+0.5)
            # p_negative = self.multivar(torch.reshape(h_enc, (h_enc.size(0), -1))-0.5)

            # multivariate - using H.size = (batch_size, latent_size, ...) (see MultipleUnivariateDensity)
            p_positive = self.multivar(torch.reshape(h_enc, (h_enc.size(0), -1))+0.5)
            p_negative = self.multivar(torch.reshape(h_enc, (h_enc.size(0), -1))-0.5)

            # univariate mapped over elements in latent layer
            # p_positive = torch.cat(list(map(self.univar, torch.unbind(torch.reshape(h_enc, (h_enc.size(0), -1))+0.5, 1))), axis=1)
            # p_negative = torch.cat(list(map(self.univar, torch.unbind(torch.reshape(h_enc, (h_enc.size(0), -1))-0.5, 1))), axis=1)

            p_y = p_positive - p_negative

            return [p_y, self.decoder(h_enc)]
        else:
            if self.return_latent:
                return h_enc
            return self.decoder(h_enc)

    def entropy_loss(self, p_y, clip_val=1e-6):
        # print(f"p_y.shape: {p_y.shape}")
        entropy = -torch.sum(p_y * torch.log2(torch.clip(p_y, min=clip_val, max=1.0)), axis = 1) / self.encoder.img_total
        # print(f"entropy.shape: {entropy.shape}")
        entropy_loss = torch.mean(entropy)
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
        self.p_vals = torch.arange(0.0, 1.0+self.res, self.res, device=device)
        self.p_ones = torch.ones(self.p_vals.size(0), device=device)
        # self.p_vals = torch.arange(0.0, 1.0+self.res, self.res).to(device)
        # self.p_ones = torch.ones(self.p_vals.size(0)).to(device)

        # return normalized vs. denormalized latents
        self.return_denorm = True

    def reset_extrema(self, z_min, z_max):
        self.z_min = z_min
        self.z_max = z_max

    def reset_hist(self):
        """ set histogram counts to zero """
        # self.p_hist = torch.zeros(self.n_levels, device=self.device)
        self.p_hist = torch.zeros(self.n_levels+1).to(self.device)

    def update_hist(self, x):
        """ take quantized tensor (x), update counts in self.p_hist """
        with torch.no_grad():
            vals, counts = x.unique(return_counts=True)
            # print(f"--- len(counts)={len(counts)} - len(self.p_hist): {len(self.p_hist)} ---")
            for i, val in enumerate(vals):
                temp = torch.isclose(self.p_vals, self.p_ones*val)
                idx = temp.nonzero(as_tuple=False).flatten()
                if idx.size(0) > 0:
                    j = idx[0]
                    self.p_hist[j] += counts[i]
                else:
                    print(f"val={val} not in p_vals")
        # print(f"in update_hist, self.p_hist: {self.p_hist}")
    
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

class g_cumul_k(torch.nn.Module):
    """
    scale hyperprior based on analytic evaluation of probability 
    see 6.1 of "Ballé, Johannes, et al. "Variational image compression with a scale hyperprior." arXiv preprint arXiv:1802.01436 (2018)."
    """
    def __init__(self, k, K, dim_in, dim_out, device="cpu"):
        super(g_cumul_k, self).__init__()
        self.x_inner = None
        self.device = device
        self.k = k
        self.K = K
        # H = torch.empty(dim_out, dim_in, device=device)
        # torch.nn.init.xavier_normal_(H)
        H = torch.normal(0,1,size=(dim_out, dim_in))
        a = torch.normal(0,1,size=(dim_out,))
        b = torch.normal(0,1,size=(dim_out,))
        # H = torch.normal(0,1,size=(dim_out, dim_in), device=self.device)
        # a = torch.normal(0,1,size=(dim_out,), device=self.device)
        # b = torch.normal(0,1,size=(dim_out,), device=self.device)
        self.H = torch.nn.Parameter(data=H, requires_grad=True).to(device)
        self.a = torch.nn.Parameter(data=a, requires_grad=True).to(device)
        self.b = torch.nn.Parameter(data=b, requires_grad=True).to(device)
        self.softplus = torch.nn.Softplus()
        self.tanh = torch.nn.Tanh()
    
    def forward(self, x):
        # a = torch.tanh(self.a).unsqueeze(0).repeat(x.size(0),1)
        # H = self.softplus(self.H).unsqueeze(0).repeat(x.size(0),1,1)        
        # b = self.b.unsqueeze(0).repeat(x.size(0),1)
        # a = self.tanh(self.a)
        # H = self.softplus(self.H)
        # b = self.b
        # if type(self.x_inner) == type(None):
        #     self.x_inner = torch.zeros(size=b.size(), device=self.device)
        
        # print(f"g_cumul - init - x.size(): {x.size()}")
        # if self.k == 1:
        #     print(f"k = 1")
        if x.size(0) == 1:
            # print(f"x.size(0) = 1")
            # first input has first input os size = (1,200); squeeze and unsqueeze to yield (200,1)
            x = x.squeeze().unsqueeze(1).unsqueeze(2)
        if len(x.size()) == 1:
            # print(f"len(x.size()) = 1")
            x = x.unsqueeze(1).unsqueeze(1)
        if len(x.size()) == 2:
            # print(f"len(x.size()) = 1")
            x = x.unsqueeze(2)
        # print(f"g_cumul - reshape - x.size(): {x.size()} - H.size(): {self.H.size()}")
        # elif len(x.size()) == 2:
        #     print(f"len(x.size()) = 2")
        #     x = x.unsqueeze(2)
        # print(f"g_cumul - unsqueezed - x.size(): {x.size()}")
        # temp_1 = torch.matmul(self.softplus(self.H.clone()), x.unsqueeze(2)).squeeze() 
        # x = torch.matmul(self.softplus(self.H), x.unsqueeze(2)).squeeze() 
        x = torch.matmul(self.softplus(self.H), x).squeeze() 
        # print(f"g_cumul - matmul - x.size(): {x.size()}")
        # temp_1 = torch.matmul(H, x).squeeze() 
        # print(f"g_cumul - addends - temp_1.size(): {temp_1.size()} - b.size(): {b.size()}")
        # self.x_inner = temp_1.squeeze() + b  if len(temp_1.size()) > 1 else temp_1.squeeze().unsqueeze(1) + b
        # x = x.squeeze() + b  if len(x.size()) > 1 else x.squeeze().unsqueeze(1) + b
        x = x.squeeze() + self.b  if len(x.size()) > 1 else x.squeeze().unsqueeze(1) + self.b
        # print(f"g_cumul - add - x.size(): {x.size()}")
        # temp = torch.matmul(H, x).squeeze() + b  # if H.size(1) > 1 else torch.matmul(H, x).unsqueeze(1)
        # print(f"g_cumul - unsqueezed - x_inner.size(): {self.x_inner.size()}")
        # temp = torch.matmul(H, x).squeeze() + b  if H.size(1) > 1 else torch.matmul(H, x).view(x.size(0),1)
        # self.x_inner = temp + b
        if (self.k == self.K):
            # y = torch.sigmoid(self.x_inner)
            y = torch.sigmoid(x)
        else:
            # y = torch.mul(a, torch.tanh(self.x_inner)) + self.x_inner
            # y = torch.mul(a, torch.tanh(x)) + x
            y = torch.mul(self.tanh(self.a), torch.tanh(x)) + x
        # print(f"g_cumul - out - y.size(): {y.size()}")
        return y

class g_cumul_k_multi(torch.nn.Module):
    """
    scale hyperprior based on analytic evaluation of probability 
    see 6.1 of "Ballé, Johannes, et al. "Variational image compression with a scale hyperprior." arXiv preprint arXiv:1802.01436 (2018)."
    """
    def __init__(self, k, K, dim_in, dim_out, dim_r, device="cpu"):
        super(g_cumul_k_multi, self).__init__()
        self.device = device
        self.k = k
        self.K = K
        # H = torch.empty(dim_out, dim_in, device=device)
        # torch.nn.init.xavier_normal_(H)
        H = torch.normal(0,1,size=(dim_r, dim_out, dim_in), requires_grad=True, device=device)
        a = torch.normal(0,1,size=(dim_r, dim_out,), requires_grad=True, device=device)
        b = torch.normal(0,1,size=(dim_r, dim_out,), requires_grad=True, device=device)
        self.H = torch.nn.Parameter(data=H, requires_grad=True).to(device)
        self.a = torch.nn.Parameter(data=a, requires_grad=True).to(device)
        self.b = torch.nn.Parameter(data=b, requires_grad=True).to(device)
        self.softplus = torch.nn.Softplus()
        self.tanh = torch.nn.Tanh()
    
    def forward(self, x):
        # print(f"g_cumul - input - x.size(): {x.size()} - H.size(): {self.H.size()}") # elif len(x.size()) == 2:
        if self.k == 1:
            # x = x.unsqueeze(2).unsqueeze(3)
            x = torch.reshape(x, x.size()+(1,1))
        elif self.k > 1:
            # ???
            # x = x.unsqueeze(3)
            x = torch.reshape(x, x.size()+(1,))
        # print(f"g_cumul - reshape - x.size(): {x.size()} - H.size(): {self.H.size()}") # elif len(x.size()) == 2:
        x = torch.matmul(self.softplus(self.H), x).squeeze() 
        # print(f"g_cumul - matmul - x.size(): {x.size()} - self.b.size(): {self.b.size()}")
        x = x.squeeze() + self.b  if self.k < self.K else x.squeeze() + self.b.squeeze()
        # print(f"g_cumul - add - x.size(): {x.size()}")
        if (self.k == self.K):
            y = torch.sigmoid(x)
        else:
            y = torch.mul(self.tanh(self.a), self.tanh(x)) + x
        # print(f"g_cumul - out - y.size(): {y.size()}")
        return y

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
    from trainer import fit, score, update_extrema, load_pretrained

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

    train_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_train).to(device), batch_size=batch_size, shuffle=True) 
    valid_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_val).to(device), batch_size=batch_size)

    bits = opt.quant_bits
    encoder = Encoder(input_dim[0], input_dim[1], opt.n_truncate)
    decoder = Decoder(input_dim[0], input_dim[1], opt.n_truncate)
    quant = UniformQuantize(bits, device=device)

    deepcmc = DeepCMC(encoder, decoder, quant, device=device).to(device)

    # pickle_dir = f"{base_pickle}/t1"
    pickle_dir = f"{base_pickle}"

    # deepcmc.quant.quant_mode = 0 # pretrain with no latent quantization
    deepcmc.quant_bool = False # train without quantization layer
    deepcmc.noise_bool = True
    epochs = 1 if opt.debug_flag else opt.epochs # epochs for intial, non-quantized network performance
    if opt.pretrain1_bool:
        if opt.preload_bool:
            deepcmc, checkpoint, history, optimizer, pretrained_epochs = load_pretrained(deepcmc, json_config, pickle_dir, epochs=epochs, device=device, lr=opt.learning_rate)
        else:
            pretrained_epochs = 0
            optimizer = history = checkpoint = None
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
                                                            alpha=opt.alpha,
                                                            network_name=f"{network_name}-noise",
                                                            pretrained_epochs=pretrained_epochs,
                                                            checkpoint=checkpoint,
                                                            history=history,
                                                            optimizer=optimizer,
                                                            lr=opt.learning_rate
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
        # optim_state_dict = torch.load(f"{pickle_dir}/{network_name}-pretrain-optimizer.pt")
        # for key, val in optim_state_dict.items():
        #     # print(f"{key}: {val.shape}")
        #     print(f"{key}: {type(val)}")
        optimizer = optim.Adam(deepcmc.parameters(), lr=learning_rate)
        opt_state_dict = torch.load(f"{pickle_dir}/{network_name}-pretrain-optimizer.pt")
        optimizer.load_state_dict(opt_state_dict)
        print(f"Loading best model weights from {pickle_dir}/{network_name}-noise-best-model.pt")
        deepcmc.load_state_dict(torch.load(f"{pickle_dir}/{network_name}-noise-best-model.pt", map_location=device), strict=False)
        # deepcmc.quant.reset_extrema(checkpoint["best_z_min"], checkpoint["best_z_max"])
    else:
        print(f"---- Model performance without pretraining ---")
        checkpoint = {"latest_model": deepcmc}
        history = {}
        optimizer = torch.optim.Adam(deepcmc.parameters())

    del train_ldr, valid_ldr
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

    if opt.tail_dir != None:
        pickle_dir += f"/{opt.tail_dir}" 
        print(f"--- pickle_dir with tail_dir: {pickle_dir} ---")

    # hard quantization
    train_ldr = torch.utils.data.DataLoader(torch.from_numpy(data_train).to(device), batch_size=batch_size, shuffle=True) 

    # first, return latents and update z_min, z_max
    deepcmc.noise_bool = False
    deepcmc.return_latent = True

    update_extrema(deepcmc, train_ldr, timers=timers)

    del train_ldr
    torch.cuda.empty_cache()

    deepcmc.return_latent = False
    deepcmc.quant_bool = True

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
                                        str_mod=f"DeepCMC ({bits}-bit uniform quant)",
                                        n_train=data_train.shape[0],
                                        pow_diff_t=pow_diff
                                        )

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