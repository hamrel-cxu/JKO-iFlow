import os
import argparse
import yaml
import math
import matplotlib.pyplot as plt
import scipy
import pickle
import numpy as np 
import torch
import torch.nn as nn
import torchdiffeq as tdeq
import time
from PIL import Image
from argparse import Namespace
from scipy.stats import gaussian_kde
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpu = torch.cuda.device_count()
mult_gpu = False if num_gpu < 2 else True

def inf_train_gen(img_name, data_size):
    def gen_data_from_img(image_mask, train_data_size):
        def sample_data(train_data_size):
            inds = np.random.choice(
                int(probs.shape[0]), int(train_data_size), p=probs)
            m = means[inds] 
            samples = np.random.randn(*m.shape) * std + m 
            return samples
        img = image_mask
        h, w = img.shape
        xx = np.linspace(-4, 4, w)
        yy = np.linspace(-4, 4, h)
        xx, yy = np.meshgrid(xx, yy)
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        means = np.concatenate([xx, yy], 1) # (h*w, 2)
        img = img.max() - img
        probs = img.reshape(-1) / img.sum() 
        std = np.array([8 / w / 2, 8 / h / 2])
        full_data = sample_data(train_data_size)
        return full_data
    image_mask = np.array(Image.open(f'{img_name}.png').rotate(
        180).transpose(0).convert('L'))
    dataset = gen_data_from_img(image_mask, data_size)
    return dataset

def get_e_ls(out, num_e):
    e_ls = []
    for i in range(num_e):
        e_ls.append(torch.randn_like(out).to(device))
    return e_ls

def divergence_approx(out, x, e_ls=[], t = None, net = None):
    approx_tr_dzdx_ls = []
    Jac_norm_ls = []
    for e in e_ls:
        sigma0, d = 0.01, Xdim_flow
        if 'sigma0' in args_yaml['training']:
            sigma0 = args_yaml['training']['sigma0'] 
        sigma = sigma0 / torch.sqrt(torch.tensor(d)).float()
        out_e = net(x+sigma*e.float(),t)
        e_dzdx = (out_e - out)/sigma
        Jac_norm = torch.zeros(x.shape[0], 1).to(device)
        Jac_norm_ls.append(Jac_norm)
        e_dzdx_e = e_dzdx * e
        approx_tr_dzdx_ls.append(e_dzdx_e.view(x.shape[0], -1).sum(dim=1, keepdim=True))
    approx_tr_dzdx_out = torch.cat(approx_tr_dzdx_ls, dim=1).mean(dim=1)
    Jac_norm_out = torch.cat(Jac_norm_ls, dim=1).mean(dim=1)
    return approx_tr_dzdx_out, Jac_norm_out

def divergence_bf(dx, x):
    sum_diag = 0.
    for i in range(x.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(),
                                        x, create_graph=True)[0][:, i]
    return sum_diag.view(x.shape[0], 1)

class FCnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        hid_dims = tuple(map(int, config.hid_dims.split("-")))
        self.layer_dims_in = (Xdim_flow,) + hid_dims
        self.layer_dims_out = hid_dims + (Xdim_flow,)
        self.build_layers()

    def build_layers(self):
        self.layers = []
        for layer_in, layer_out in zip(self.layer_dims_in, self.layer_dims_out):
            self.layers.append(nn.Linear(layer_in, layer_out))
            if layer_out != Xdim_flow:
                self.layers.append(nn.Softplus(beta=20))
        self.layers = nn.Sequential(*self.layers)
            
    def forward(self, x, t):
        return self.layers(x)

class ODEFunc(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.div_bf = False 

    def forward(self, t, x):
        def odefunc_wrapper(t, x):
            x, _, _ = x
            x = x.float()
            if self.logpx:
                if self.fix_e_ls:
                    if self.e_ls is None:
                        self.e_ls = get_e_ls(x, self.num_e)
                else:
                    self.e_ls = get_e_ls(x, self.num_e)
                if self.div_bf:
                    with torch.set_grad_enabled(True):
                        x.requires_grad_(True)
                        t.requires_grad_(True)
                        out = self.model(x,t)
                        divf = divergence_bf(out, x).to(device)
                        Jac_norm_out = torch.zeros_like(divf).to(device)
                else:
                    out = self.model(x,t)
                    divf, Jac_norm_out = divergence_approx(out, x, self.e_ls,
                                                        t = t, net = self.model)
            else:
                divf = torch.zeros(x.shape[0]).to(device)
                Jac_norm_out = torch.zeros_like(divf)
                out = self.model(x,t)
            return out, -divf, Jac_norm_out
        return odefunc_wrapper(t, x)

class CNF(nn.Module):
    def __init__(self, odefunc):
        super(CNF, self).__init__()
        self.odefunc = odefunc

    def forward(self, x, args, reverse=False, test=False, mult_gpu=False):
        self.odefunc.logpx = True
        integration_times = torch.linspace(
            args.Tk_1, args.Tk, args.num_int_pts+1).to(device)
        if test:
            self.odefunc.logpx = False  
        if reverse:
            integration_times = torch.flip(integration_times, [0])
        self.odefunc.num_e = args.num_e
        dlogpx = torch.zeros(x.shape[0]).to(device)
        dJacnorm = torch.zeros(x.shape[0]).to(device)
        self.odefunc.e_ls = None 
        self.odefunc.fix_e_ls = args.fix_e_ls
        self.odefunc.counter = 0
        if args.use_NeuralODE is False:
            predz, dlogpx, dJacnorm = tdeq.odeint(
                self.odefunc, (x, dlogpx, dJacnorm), integration_times, method=args.int_mtd,
                rtol = args.rtol, atol = args.atol)
        else:
            predz, dlogpx, dJacnorm = tdeq.odeint_adjoint(
                self.odefunc, (x, dlogpx, dJacnorm), integration_times, method=args.int_mtd,
                rtol = args.rtol, atol = args.atol)
        if mult_gpu:
            return predz[-1], dlogpx[-1], dJacnorm[-1]
        else:
            return predz, dlogpx, dJacnorm

def get_config(style):
    if style == 'tree' or style == 'rose':
        config = Namespace(
            hid_dims = '128-128-128'
        )
    else:
        raise ValueError(f'Unknown style {style}')
    return config

def default_CNF_structure(config):
    model = FCnet(config).to(device)
    odefunc = ODEFunc(model).to(device)
    CNF_ = CNF(odefunc).to(device)
    return CNF_

def FlowNet_forward(xinput, CNF, ls_args_CNF,
                    block_now,
                    reverse = False, test = True,
                    return_full = False):
    if block_now == 0:
        return xinput, 0
    else:
        ls_args_CNF = ls_args_CNF[:block_now]
        with torch.no_grad():
            predz_ls, dlogpx_ls = [], []
            if reverse:
                ls_args_CNF = list(reversed(ls_args_CNF))
            for i, args_CNF in enumerate(ls_args_CNF):
                predz, dlogpx, _ = CNF(xinput, args_CNF, 
                                    reverse = reverse, test = test,
                                    mult_gpu = mult_gpu)
                if mult_gpu:
                    if i == 0:
                        predz_ls.append(xinput)
                        dlogpx_ls.append(torch.zeros(xinput.shape[0]).to(device))
                    predz_ls.append(predz)
                    dlogpx_ls.append(dlogpx)
                    xinput = predz
                else:
                    xinput = predz[-1]
                    if i == 0:
                        predz_ls.append(predz)
                        dlogpx_ls.append(dlogpx)
                    else:
                        predz_ls.append(predz[1:])
                        dlogpx_ls.append(dlogpx[1:])
        if mult_gpu is False:
            predz_ls = torch.cat(predz_ls, dim=0)
            dlogpx_ls = torch.cat(dlogpx_ls, dim=0)
        else:
            predz_ls = torch.stack(predz_ls, dim=0)
            dlogpx_ls = torch.stack(dlogpx_ls, dim=0)
        if return_full:
            return predz_ls, dlogpx_ls
        else:
            return predz_ls[-1], dlogpx_ls[-1]

def l2_norm_sqr(input, return_full = False):
    if len(input.size()) > 2:
        norms = 0.5*input.view(input.shape[0], -1).pow(2).sum(axis=1)
    else:
        norms = 0.5*input.pow(2).sum(axis=1)
    if return_full:
            return norms
    else:
        return norms.mean()

def plt_losses_at_block(ls_all, args, window_size = 50):
    titlesize = 20
    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    errs = np.array(ls_all)
    msize = 0.5
    def convolve(x):
        if len(x) <= window_size:
            return x
        else:
            return scipy.signal.convolve(x, np.ones(window_size)/window_size, 
                                         mode='valid', method = 'fft')[window_size:]
    num_losses = len(errs[:, 1].flatten())
    if num_losses > window_size:
        xaxis = np.arange(window_size, num_losses+1)[window_size:]
    else:
        xaxis = np.arange(num_losses)
    ax[0].plot(xaxis, convolve(errs[:, 1].flatten()), '-o', markersize=msize, color='blue')
    ax[0].set_title(r'W2: $W_2^2(f([t_{k-1}, t_k]))/h_k$', fontsize=titlesize)  
    ax[1].plot(xaxis, convolve(errs[:, 2].flatten()), '-o', markersize=msize, color='blue')
    ax[1].set_title(r'V: $V(X(t_k))/2$', fontsize=titlesize)
    ax[2].plot(xaxis, convolve(errs[:, 3].flatten()), '-o', markersize=msize, color='blue')
    ax[2].set_title(r'Div: $-\int_{t_{k-1}}^{t_k} \nabla \cdot f(X(s),s)ds$', fontsize=titlesize)
    ax[3].plot(xaxis, convolve(errs[:, 4].flatten()), '-o', markersize=msize, color='blue')
    ax[3].set_title(r'Jac: $\int_{t_{k-1}}^{t_k} ||\nabla_{X(s)} f(X(s),s)||^2_F ds$', fontsize=titlesize)
    ax[-1].plot(xaxis, convolve(errs[:, 0].flatten()), '-o', markersize=msize, color='blue')
    ax[-1].set_title('Sum of all', fontsize=titlesize)
    fig.suptitle(
        f'Training metrics for block {args.block_now}', y=0.98, fontsize=titlesize)
    for a in ax.flatten():
        a.set_xlabel('Num batches/loops', fontsize=titlesize)   
    fig.tight_layout()
    plt.show()
    plt.close()
    return fig

def check_inv_err(self, nsamples = 500):
    with torch.no_grad():
        Xtest = self.X_test[torch.randperm(self.X_test.shape[0])[:nsamples]].to(device)
        Xtest_raw = Xtest.clone()
        if block_id > 1:
            for self_mod in self_ls_prev:
                Xtest, _ = FlowNet_forward(Xtest, self_mod.CNF, self_mod.ls_args_CNF,
                                          self_mod.block_now, reverse = False,
                                          return_full = False)
        Zhat, _ = FlowNet_forward(Xtest, self.CNF, self.ls_args_CNF, self.block_now,
                                    reverse = False, test = True,
                                    return_full = False)
        Xback, _ = FlowNet_forward(Zhat, self.CNF, self.ls_args_CNF, self.block_now,
                                    reverse = True, test = True,
                                    return_full = False)
        if block_id > 1:
            for self_mod in reversed(self_ls_prev):
                Xback, _ = FlowNet_forward(Xback, self_mod.CNF, self_mod.ls_args_CNF,
                                          self_mod.block_now, reverse = True,
                                          return_full = False)
        abs_err = l2_norm_sqr(Xback-Xtest_raw)
        print(f'--Test absolute MSE ||X-Finv(F(X))|| is {abs_err.item():.2e}')
    return abs_err.item()

def move_over_blocks(self, reverse = False, nte = 1000):
    with torch.no_grad():
        if reverse:
            Xtest = torch.randn(nte, Xdim_flow).to(device)
        else:
            Xtest = self.X_test.to(device)
        if block_id > 1 and reverse is False:
            Zhat_ls_prev, Zout = [Xtest], Xtest
            for self_mod in self_ls_prev:
                Zout, _ = FlowNet_forward(Zout, self_mod.CNF, self_mod.ls_args_CNF,
                                          self_mod.block_now, reverse = False,
                                          return_full = False)
                Zhat_ls_prev.append(Zout)
            Xtest = Zout
        Zhat_ls, _ = FlowNet_forward(Xtest, self.CNF, self.ls_args_CNF,
                                     self.block_now,
                                     reverse = reverse,
                                     return_full = True)
        if block_id > 1 and reverse is True:
            Xhat_ls_prev, Xout = [], Zhat_ls[-1]
            for self_mod in reversed(self_ls_prev):
                Xout, _ = FlowNet_forward(Xout, self_mod.CNF, self_mod.ls_args_CNF,
                                          self_mod.block_now, reverse = True,
                                          return_full = False)
                Xhat_ls_prev.append(Xout)
        if mult_gpu:
            ids = range(len(Zhat_ls))
        else:
            ids = torch.linspace(0, Zhat_ls.shape[0]-1, self.block_now+1).long()
        Zhat_ls = [Zhat_ls[i] for i in ids]
        if block_id > 1:
            Zhat_ls = Zhat_ls_prev + Zhat_ls[1:] if reverse is False else Zhat_ls + Xhat_ls_prev
    return Zhat_ls 

def plot_W2_movement(self, num_fig = 500):
    with torch.no_grad():
        Xtest = self.X_test[:num_fig]
        if block_id > 1:
            W2_prev_blocks = []
            for self_mod in self_ls_prev:
                Xtest_, _ = FlowNet_forward(Xtest, self_mod.CNF, self_mod.ls_args_CNF,
                                          self_mod.block_now, reverse = False,
                                          return_full = False)
                diff = Xtest_ - Xtest
                W2_sqr = 0.5*diff.view(diff.shape[0], -1).pow(2).sum(dim=1).mean()
                W2_prev_blocks.append(W2_sqr.item())
                Xtest = Xtest_
        Zhat_ls, _ = FlowNet_forward(Xtest, self.CNF, self.ls_args_CNF,
                                     self.block_now,
                                     return_full = True)
        if mult_gpu:
            ids = range(len(Zhat_ls))
        else:
            ids = torch.linspace(0, Zhat_ls.shape[0]-1, self.block_now+1).long()
        Zhat_ls = Zhat_ls[ids]
        Diff_Zhat = Zhat_ls[1:] - Zhat_ls[:-1]
        W2_sqr = 0.5*Diff_Zhat.view(Diff_Zhat.shape[0], -1).pow(2).sum(dim=1)/num_fig
        print(f'W2 =\n {W2_sqr.cpu().detach().numpy()}')
        fig, ax = plt.subplots(1,1, figsize = (8,4))
        ax.plot(range(1,len(W2_sqr)+1), W2_sqr.cpu().detach().numpy(), 'o-')
        ax.set_title(r'W2(k)=$0.5\mathbb{E}_{\tilde{x}\sim p_{k-1}} ||\tilde{x}(\tilde{t}_k)-\tilde{x}(\tilde{t}_{k-1})||^2$')
        if block_id > 1:
            W2_last = Zhat_ls[-1] - Zhat_ls[0]
            W2_last = 0.5*W2_last.view(W2_last.shape[0], -1).pow(2).sum(dim=1).mean()
            W2_prev_blocks.append(W2_last.item())
            print(f'W2 over all blocks: {W2_prev_blocks}')
            fig1, ax1 = plt.subplots(1,1, figsize = (8,4))
            ax1.plot(range(1,len(W2_prev_blocks)+1), W2_prev_blocks, 'o-')
            ax1.set_title('W2 at each block')
        else:
            fig1 = None
    return fig, fig1, W2_sqr

def JKO_loss_func(xinput, model, ls_args_CNF):
    num_rk4 = len(ls_args_CNF)
    loss_div_tot, loss_Jac_tot = 0, 0
    xinput_ = xinput.clone()
    for k in range(num_rk4):
        args = ls_args_CNF[k]
        predz, dlogpx, lossJacnorm = model(xinput, args, reverse = False, test = False,
                                           mult_gpu = mult_gpu) 
        if mult_gpu:
            xpk = predz
            loss_div_tot += dlogpx.mean()
            loss_Jac_tot += args_training.lam_jac * lossJacnorm.mean()
        else:
            xpk = predz[-1]
            loss_div_tot += dlogpx[-1].mean()
            loss_Jac_tot += args_training.lam_jac * lossJacnorm[-1].mean()
        xinput = xpk
    raw_movement = l2_norm_sqr(xpk - xinput_)
    loss_W2_tot = raw_movement/delta_tk
    loss_V_tot = l2_norm_sqr(xpk) # V(x(T))
    return loss_V_tot, loss_div_tot, loss_W2_tot, loss_Jac_tot, raw_movement

def push_samples_forward(data_loader, self):
    X = []
    for xsample in data_loader:
        xsample = xsample[0]
        xpushed, _ = FlowNet_forward(xsample.to(device), self.CNF, 
                                    self.ls_args_CNF, self.block_now,
                                    reverse = False, test = True, 
                                    return_full = False)
        X.append(xpushed)
    X = torch.cat(X, dim=0)
    return X

def on_off(self, on = True):
    for a in self.ls_args_CNF:
        a.int_mtd = 'dopri5' if on else 'rk4'

def load_prev_CNFs():
    self_ls_prev = []
    for b in range(1, block_id):
        self_prev = Namespace()
        filepath = os.path.join(master_dir, f'{prefix}{b}.pth')
        checkpoint = torch.load(filepath)
        self_prev.CNF = default_CNF_structure(config = vfield_config)
        self_prev.CNF.load_state_dict(checkpoint['model'])
        self_prev.ls_args_CNF = checkpoint['ls_args_CNF']
        on_off(self_prev, on = True)
        self_prev.block_now = len(self_prev.ls_args_CNF)
        self_ls_prev.append(self_prev)
    return self_ls_prev

def loop_data_loader(dataloader):
    data_iterator = iter(dataloader)
    while True:
        try:
            yield next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)

def add_diffuse(x):
    eps = 1e-3 
    t, dt = eps, eps
    beta_min, beta_max = 0.1, 20
    beta_t = beta_min + t*(beta_max - beta_min)
    dt_term = -0.5*beta_t*x*dt
    dw = np.sqrt(dt)*torch.randn_like(x)
    dw_term = np.sqrt(beta_t)*dw
    dx = dt_term + dw_term
    return x + dx

def pdist(sample_1, sample_2, norm=2):
    return torch.cdist(sample_1, sample_2, p=norm)

class MMDStatistic:
    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        self.a00 = 1. / (n_1 * n_1)
        self.a11 = 1. / (n_2 * n_2)
        self.a01 = - 1. / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, alphas, ret_matrix=False):
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = pdist(sample_12, sample_12, norm=2)
        mmd_dict = {}
        for alpha in alphas:
            kernels = torch.exp(- alpha * distances**2)
            k_1 = kernels[:self.n_1, :self.n_1]
            k_2 = kernels[self.n_1:, self.n_1:]
            k_12 = kernels[:self.n_1, self.n_1:]
            mmd = self.a00 * k_1.sum() + self.a11 * k_2.sum() + 2 * self.a01 * k_12.sum()
            mmd_dict[alpha.item()] = f'{mmd.item():.2e}' 
        if ret_matrix:
            return mmd_dict, kernels
        else:
            return mmd_dict

def get_MMD(X, Xhat, nmax = 1000, alpha_ls = [0.5]):
    X, Xhat = torch.from_numpy(X).to(device), torch.from_numpy(Xhat).to(device)
    nmax1 = min(nmax, X.shape[0])
    nmax2 = min(nmax, Xhat.shape[0])
    X = X[torch.randperm(X.shape[0])[:nmax1]]
    Xhat = Xhat[torch.randperm(Xhat.shape[0])[:nmax2]]
    print(X.shape, Xhat.shape)
    distances = pdist(X,X)
    dist_median = torch.median(distances)
    gamma = 0.1*dist_median
    alpha_ls = [0.5/gamma**2]
    alpha_ls = torch.tensor(alpha_ls).to(device)
    mtd = MMDStatistic(nmax1, nmax2)
    mmd_dict = mtd(X, Xhat, alpha_ls)
    return mmd_dict

def helper(on = True):
    if on:
        self.CNF.module.odefunc.div_bf = True
        if block_id > 1:
            for self_mod in self_ls_prev:
                on_off(self_mod, on = False)
                self_mod.CNF.odefunc.div_bf = True
    else:
        self.CNF.module.odefunc.div_bf = False
        if block_id > 1:
            for self_mod in self_ls_prev:
                on_off(self_mod, on = True)
                self_mod.CNF.odefunc.div_bf = False

parser = argparse.ArgumentParser(description='Load hyperparameters from a YAML file.')
parser.add_argument('--JKO_config', default = 'configs/JKO_rose.yaml', type=str, help='Path to the YAML file')
args_parsed = parser.parse_args()
with open(args_parsed.JKO_config, 'r') as file:
    args_yaml = yaml.safe_load(file)
    print(yaml.dump(args_yaml, default_flow_style=False))

if __name__ == '__main__':
    block_idxes = args_yaml['training']['block_idxes']
    for block_id in block_idxes:
        vfield_style = args_yaml['CNF']['vfield_style']
        folder_suffix = args_yaml['eval']['folder_suffix']
        master_dir = f'results/JKO_{vfield_style}{folder_suffix}'
        os.makedirs(master_dir, exist_ok=True)
        prefix = 'block' 
        common_name = f'{prefix}{block_id}'
        filepath = os.path.join(master_dir, common_name + '.pth')
        directory = os.path.join(master_dir, common_name)
        os.makedirs(directory, exist_ok=True) 
        filename = os.path.join(directory, common_name)
        self = Namespace() 
        print(f'#### Training block {block_id} ####')
        print('########################## Data part ##########################')
        Xdim_flow = args_yaml['data']['Xdim_flow'] # After encoding
        batch_size = args_yaml['training']['batch_size'] 
        ntr, nte = args_yaml['training']['ntr'], args_yaml['training']['nte']
        xraw = inf_train_gen(f'img_{vfield_style}', ntr)
        xte = inf_train_gen(f'img_{vfield_style}', nte)
        xraw = torch.from_numpy(xraw).float().to(device)
        xte = torch.from_numpy(xte).float().to(device)
        self.X_test = xte
        if block_id > 1:
            common_name_data = f'{prefix}{block_id-1}'
            filename_data = os.path.join(master_dir, common_name_data + '_Xpushed.pkl')
            Xtrain_pushed = pickle.load(open(filename_data, 'rb'))
            train_loader_raw = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(Xtrain_pushed),
                batch_size=batch_size, shuffle=True)
        else:
            train_loader_raw = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(xraw),
                batch_size=batch_size, shuffle=True)
        train_loader_raw_tr = loop_data_loader(train_loader_raw)
        print('########################## CNF flow setup ##########################')
        vfield_config = get_config(style = vfield_style)
        self.CNF = default_CNF_structure(config = vfield_config)
        total_params = sum(p.numel() for p in self.CNF.parameters())
        print(f'######## Number of parameters in CNF: {total_params/1e3}K ########')
        common_args_CNF = Namespace(
            int_mtd = 'rk4',
            num_e = 1, 
            num_int_pts = 1, 
            fix_e_ls = True, 
            use_NeuralODE = True, 
            rtol = 1e-5,
            atol = 1e-5
        )
        S = args_yaml['CNF']['S_ls'][block_id-1] 
        hk_blocks = args_yaml['CNF']['hk_blocks']
        hk_b = 1
        delta_tk = hk_blocks[block_id-1]
        hk_ls = np.array([hk_b/S] * S)
        self.ls_args_CNF = []
        for i in range(S):
            args_CNF_now = Namespace(**vars(common_args_CNF))
            hk_sub = hk_ls[i]
            args_CNF_now.Tk_1 = 0 if i == 0 else np.sum(hk_ls[:i])
            args_CNF_now.Tk = args_CNF_now.Tk_1 + hk_sub
            self.ls_args_CNF.append(args_CNF_now)  
        args_CNF_ = Namespace(**vars(args_CNF_now))
        args_CNF_.Tk_1 = 0
        args_CNF_.Tk = 1         
        for i, a in enumerate(self.ls_args_CNF):
            print(f'##### Sub-Interval {i+1}: [{a.Tk_1}, {a.Tk}], h_k = {a.Tk - a.Tk_1}, m_k = {a.num_int_pts}')
            print(f'Penalty delta_tk at block {block_id} is {delta_tk}')
        print('Done instantiating CNF and CNF args')
        self.block_now = len(self.ls_args_CNF)
        print('########################## Training args ##########################')
        load_checkpoint = args_yaml['training']['load_checkpoint']
        args_training = Namespace(
            tot_iters = args_yaml['training']['tot_iters'], 
            lr = args_yaml['training']['lr'], 
            load_checkpoint = load_checkpoint, 
            iter_start = 0,
            lam_jac = 0, 
        )
        override_default = True 
        optimizer = torch.optim.Adam(self.CNF.parameters(), lr=args_training.lr)
        print('########################## Resume from checkpoint (or not) ##########################')
        self_ls_prev = []
        if block_id > 1:
            print(f'############ Loaded previous CNFs ############')
            self_ls_prev = load_prev_CNFs()
            assert len(self_ls_prev) == block_id - 1
            if args_yaml['training']['warm_start']:
                self.CNF.load_state_dict(self_ls_prev[-1].CNF.state_dict())
                print(f'############ Warm start from {block_id-1} parameter ############')
        if args_training.load_checkpoint and os.path.exists(filepath):
            checkpt = torch.load(filepath)
            self.CNF.load_state_dict(checkpt['model'])
            args_training = checkpt['args']
            args_training.load_checkpoint = True
            self.loss_at_block = checkpt['loss_at_block']
            optimizer.load_state_dict(checkpt['optimizer'])
            print(f'Starting at batch # {args_training.iter_start+1}')
        else:
            self.loss_at_block = []
            print('Starting from batch # 0')
        self.CNF = torch.nn.DataParallel(self.CNF)
        print(self.CNF)
        if override_default:
            args_training.tot_iters = args_yaml['training']['tot_iters']
            print(f'############ Train until {args_training.tot_iters} batches ############')
        print('########################## Start training ##########################')
        while args_training.iter_start < args_training.tot_iters:
            i = args_training.iter_start
            start = time.time()
            on_off(self, on = False) 
            xsub = next(train_loader_raw_tr)[0]
            if block_id == 1 and 'add_diffuse' in args_yaml['training']:
                xsub = add_diffuse(xsub)
            optimizer.zero_grad()
            loss_V, loss_div, loss_W2, loss_Jac, _ = JKO_loss_func(xsub, self.CNF, self.ls_args_CNF)
            loss = loss_V + loss_div + loss_W2 + loss_Jac
            if np.isnan(loss.item()):
                raise ValueError('NaN encountered.')
            loss.backward()
            if args_yaml['training']['clip_grad']:
                _ = torch.nn.utils.clip_grad_norm_(self.CNF.parameters(), 1.0)
            optimizer.step()
            args_training.iter_start += 1 
            current_loss = [loss.item(), loss_W2.item(), loss_V.item(), loss_div.item(), loss_Jac.item()]
            self.loss_at_block.append(current_loss)
            if args_training.iter_start % 100 == 0:
                print(f'Iter {args_training.iter_start} with {batch_size} batches done, took {time.time() - start:.2f} seconds')
            viz_freq = args_yaml['eval']['viz_freq']
            max_iter = args_training.tot_iters - 1
            sdict = {'model': self.CNF.module.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'args': args_training,
                     'ls_args_CNF': self.ls_args_CNF,
                     'loss_at_block': self.loss_at_block}
            if i % viz_freq == 0 or i == max_iter:
                print(f'######### Evaluate at iter {i+1}')    
                on_off(self, on = True)
                abs_err = check_inv_err(self, nsamples = 1000)
                X_traj = move_over_blocks(self, reverse = True, nte = nte)
                Z_traj = move_over_blocks(self, reverse = False, nte = nte)
                X = self.X_test.clone().cpu().numpy()
                Zhat = Z_traj[-1].cpu().numpy()
                Xhat = X_traj[-1].cpu().numpy()
                Z = torch.randn_like(Z_traj[-1]).cpu().numpy()
                fig, ax = plt.subplots(1,4, figsize = (20,5))
                s=0.01
                ax[0].scatter(X[:,0], X[:,1], s = s)
                ax[1].scatter(Xhat[:,0], Xhat[:,1], s = s)
                ax[2].scatter(Z[:,0], Z[:,1], s = s)
                ax[3].scatter(Zhat[:,0], Zhat[:,1], s = s)
                ax[0].set_title('X')
                ax[1].set_title('Xhat')
                ax[2].set_title('Z')
                ax[3].set_title('Zhat')
                for a in [ax[2], ax[3]]:
                    a.set_xlim([-4,4])
                    a.set_ylim([-4,4])
                fig.tight_layout()
                filename_gen = filename+f'_XZhat_iter{args_training.iter_start}.png'
                fig.savefig(filename_gen)
                fig_W2, fig_W2_all, W2_sqr = plot_W2_movement(self, num_fig = 1000)
                filename_W2 = filename+f'_W2movement_iter{args_training.iter_start}.png'
                fig_W2.savefig(filename_W2)
                if block_id > 1:
                    filename_W2_all = filename+f'_W2movement_all_iter{args_training.iter_start}.png'
                    fig_W2_all.savefig(filename_W2_all)
                args_for_viz = Namespace(block_now = f'JKO discrete block{block_id}')
                fig_loss_block = plt_losses_at_block(self.loss_at_block, args_for_viz) # Loss at this block
                filename_loss = filename+f'_loss_iter{args_training.iter_start}.png'
                fig_loss_block.savefig(filename_loss)
                plt.close('all')
                on_off(self, on = False)
                torch.cuda.empty_cache()
                if abs_err > 1000 and i > 100:
                    raise ValueError('Inverse error is too large, something is wrong')
                torch.save(sdict, filepath)
                helper(on = True)
                dlogpx_full = 0
                Xtest = self.X_test.clone().to(device)
                for self_mod in self_ls_prev + [self]:
                    dlogpx = 0
                    predz, dlogpx_, _ = self_mod.CNF(Xtest, args_CNF_, reverse = False, test = False, mult_gpu = mult_gpu)
                    Xtest = predz if mult_gpu else predz[-1]
                    dlogpx += dlogpx_ if mult_gpu else dlogpx_[-1]
                    dlogpx_full += dlogpx
                logrhoXtoZ = dlogpx_full.mean()
                constant = -Xdim_flow/2 * math.log(2*math.pi)
                logrhoZ = -l2_norm_sqr(Xtest) + constant
                logrhoX = logrhoZ - logrhoXtoZ
                print(f'Test NLL is {-logrhoX.item()} at iter {i+1}')
                helper(on = False)
            if i == max_iter:
                on_off(self, on = True)
                nmax = 10000
                X_traj = move_over_blocks(self, reverse = True, nte=nmax)
                Xhat = X_traj[-1].cpu().numpy()
                mmd_dict = get_MMD(X, Xhat, nmax = nmax)
                print(mmd_dict)
                on_off(self, on = False)
                Xtrain_pushed = push_samples_forward(train_loader_raw, self)
                print(f'##### Shape of Xtrain_pushed is {Xtrain_pushed.shape} #####')
                filename_data = filepath.split('.pth')[0] + '_Xpushed.pkl'
                pickle.dump(Xtrain_pushed, open(filename_data, 'wb'))