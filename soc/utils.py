import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tinyimagenet import TinyImageNetDataset
from moons2d import Moons2D
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import math
import os
from itertools import chain, combinations
import matplotlib.pyplot as plt

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2507, 0.2507, 0.2507)

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def get_loaders(dir_, batch_size, dataset_name='cifar10', normalize=True):
    if dataset_name == 'cifar10':
        dataset_func = datasets.CIFAR10
    elif dataset_name == 'cifar100':
        dataset_func = datasets.CIFAR100
    elif dataset_name == 'tinyimagenet':
        dataset_func = TinyImageNetDataset
    elif dataset_name in ['moons2d', '01moons2d']:
        dataset_func = Moons2D
    
    if dataset_name == 'tinyimagenet':
        if normalize:
            train_transform = transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),
            ])
        else:
            train_transform = transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        train_dataset = dataset_func(
            dir_, mode='train', transform=train_transform, download=False, preload=False)
        test_dataset = dataset_func(
            dir_, mode='val', transform=test_transform, download=False, preload=False)
    elif dataset_name in['moons2d', '01moons2d']:
        center = (dataset_name == '01moons2d')
        train_dataset = Moons2D(2000, noise=0.0, uniform_noise=0.1, center=center)
        test_dataset = Moons2D(20000, noise=0.0, uniform_noise=0.1, center=center)
    else:
        if normalize:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        train_dataset = dataset_func(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = dataset_func(
            dir_, train=False, transform=test_transform, download=True)
        
    num_workers = 4
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader, train_dataset

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def evaluate_pgd(test_loader, model, attack_iters, restarts, limit_n=float("inf")):
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            if n >= limit_n:
                break
    return pgd_loss/n, pgd_acc/n

def attack_pgd_l2(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def evaluate_pgd_l2(test_loader, model, attack_iters, restarts, limit_n=float("inf")):
    epsilon = (36 / 255.) / std
    alpha = epsilon/5.
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd_l2(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            if n >= limit_n:
                break
    return pgd_loss/n, pgd_acc/n

def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n

def ortho_certificates(output, class_indices, L):
    # raise RuntimeError

    batch_size = output.shape[0]
    batch_indices = torch.arange(batch_size)
    
    onehot = torch.zeros_like(output).cuda()
    onehot[torch.arange(output.shape[0]), class_indices] = 1.
    output_trunc = output - onehot*1e6

    output_class_indices = output[batch_indices, class_indices]
    output_nextmax = torch.max(output_trunc, dim=1)[0]
    output_diff = output_class_indices - output_nextmax
    return output_diff/(math.sqrt(2)*L)

def lln_certificates(output, class_indices, last_layer, L):
    batch_size = output.shape[0]
    batch_indices = torch.arange(batch_size)
    
    onehot = torch.zeros_like(output).cuda()
    onehot[batch_indices, class_indices] = 1.
    output_trunc = output - onehot*1e6    
        
    if hasattr(last_layer, 'last_weight'):
        lln_weight = last_layer.last_weight
        with torch.no_grad():
            istraining = last_layer.training
            last_layer.train(False)
            L_mlp = last_layer.getK(num_iter=1)
            lln_weight *= L_mlp
            last_layer.train(istraining)
    elif hasattr(last_layer, 'lln_weight'):
        lln_weight = last_layer.lln_weight
    else:
        lln_weight = last_layer.weight
    lln_weight_indices = lln_weight[class_indices, :]
    lln_weight_diff = lln_weight_indices.unsqueeze(1) - lln_weight.unsqueeze(0)
    lln_weight_diff_norm = torch.norm(lln_weight_diff, dim=2, p=2)
    # assert torch.sum(torch.abs(lln_weight_diff_norm-torch.norm(lln_weight_diff, dim=2))) < 1.e-6
    # print(lln_weight_diff_norm)
    lln_weight_diff_norm = lln_weight_diff_norm + onehot
    # print(lln_weight_diff_norm)

    output_class_indices = output[batch_indices, class_indices]
    output_diff = output_class_indices.unsqueeze(1) - output_trunc
    # print('f_t(x)-f_i(x)', output_diff[0])
    # print('W_t-W_i', L, lln_weight_diff_norm[0])
    all_certificates = output_diff/(lln_weight_diff_norm*L)
    # print('R(x)', all_certificates[0])
    return torch.min(all_certificates, dim=1)[0]

def evaluate_certificates(test_loader, model, L, lln_certificate=True, epsilon=36.):
    losses_list = []
    certificates_list = []
    correct_list = []
    model.eval()

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y, reduction='none')
            losses_list.append(loss)

            output_max, output_amax = torch.max(output, dim=1)
            
            if lln_certificate:
                certificates = lln_certificates(output, output_amax, model.last_layer, L)
            else:
                certificates = ortho_certificates(output, output_amax, L)
                
            correct = (output_amax==y)
            certificates_list.append(certificates)
            correct_list.append(correct)
            
        losses_array = torch.cat(losses_list, dim=0).cpu().numpy()
        certificates_array = torch.cat(certificates_list, dim=0).cpu().numpy()
        correct_array = torch.cat(correct_list, dim=0).cpu().numpy()
    return losses_array, correct_array, certificates_array


from cayley_ortho_conv import Cayley, CayleyLinear
from block_ortho_conv import BCOP
from skew_ortho_conv import SOC

conv_mapping = {
    'standard': nn.Conv2d,
    'soc': SOC,
    'bcop': BCOP,
    'cayley': Cayley
}

from custom_activations import MaxMin, HouseHolder, HouseHolder_Order_2

activation_dict = {
    'relu': F.relu,
    'swish': F.silu,
    'sigmoid': F.sigmoid,
    'tanh': F.tanh,
    'softplus': F.softplus,
    'maxmin': MaxMin()
}

def activation_mapping(activation_name, channels=None):
    if activation_name == 'hh1':
        assert channels is not None, channels
        activation_func = HouseHolder(channels=channels)
    elif activation_name == 'hh2':
        assert channels is not None, channels
        activation_func = HouseHolder_Order_2(channels=channels)
    else:
        activation_func = activation_dict[activation_name]
    return activation_func

def parameter_lists(model):
    conv_params = []
    activation_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'activation' in name:
                activation_params.append(param)
            elif 'conv' in name:
                conv_params.append(param)
            else:
                other_params.append(param)
    return conv_params, activation_params, other_params


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class nf(float):
    def __repr__(self):
        if self != 0:
            s = f'K{self:.1f}'
            return f'K{self:.0f}' if s[-1] == '0' else s
        else:
            s = f'{self:.1f}'
            return f'{self:.0f}' if s[-1] == '0' else s

class DecisionBoundaryPlot(object):
    def __init__(self, xlim, ylim, nsamples, pairwise=False, 
        levels=[-1.0, -0.5, -0.25, -0.15, 0, 0.15, 0.25, 0.5, 1.0], 
        data_std=1., save_path=None, color_non_robust=True, 
        attr_K_pos=(-1.45, 1.25), attr_K_fontsize=24, attr_K_ha='left',
        attr_vra_pos=(2.4, -0.95), attr_vra_fontsize=24, attr_vra_ha='right',
        *args, **kwargs):
        super(DecisionBoundaryPlot, self).__init__(*args, **kwargs)
        self.xlim = xlim
        self.ylim = ylim
        self.dx = xlim[1]-xlim[0]
        self.dy = ylim[1]-ylim[0]
        self.nsamples = nsamples
        self.pairwise = pairwise
        self.levels = levels
        self.data_K = 1
        self.save_path = save_path
        assert save_path is not None and os.path.exists(save_path)
        self.color_non_robust = color_non_robust
        # self.data_K = 1./min(data_std)
        # assert len(levels) == 9

        self.attr_K_pos = attr_K_pos
        self.attr_K_fontsize = attr_K_fontsize
        self.attr_K_ha = attr_K_ha
        self.attr_vra_pos = attr_vra_pos
        self.attr_vra_fontsize = attr_vra_fontsize
        self.attr_vra_ha = attr_vra_ha

    def forward(self, K, model, train_dataset, val_loader, epoch, Kh=None):
        dataset = train_dataset
        loader = val_loader
        model_device = torch.ones(1).cuda().device

        x = np.linspace(self.xlim[0], self.xlim[1], int(np.sqrt(self.nsamples)))
        y = np.linspace(self.ylim[0], self.ylim[1], int(np.sqrt(self.nsamples)))
        X, Y = np.meshgrid(x, y)
        
        inputs = torch.Tensor(np.stack([X.ravel(), Y.ravel()], axis=1)).to(model_device)
        outputs = []

        for batch_inp, _ in loader:
            val_bs = batch_inp.shape[0]
            break

        with torch.no_grad():
            for bs_idx in range(0,inputs.shape[0],val_bs):
                batch_inp = inputs[bs_idx:bs_idx+val_bs]

                output = model(batch_inp)
                outputs.append(output)

        outputs = torch.cat(outputs, dim=0)

        margins = dict()

        with torch.no_grad():
            if hasattr(model.last_layer, 'lln_weight'):
                W = model.last_layer.lln_weight
            else:
                W = model.last_layer.weight
            y_j = outputs

            kW = K * W
            Kij = torch.cdist(kW, kW, p=2)
            if not self.pairwise:
                Kij = K * torch.norm(W, p=2, dim=1)

            nclasses = W.shape[0]
            if self.pairwise:
                combinations = [set_ for set_ in powerset(range(nclasses)) if len(set_) == 2]

                for class0, class1 in combinations:
                    y_diff = y_j[:,class0] - y_j[:,class1]  # N x Classes
                    ratios = y_diff / Kij[class0, class1]
                    ratio = ratios #.min(dim=1)[0]
                    margins[(class0, class1)] = ratios.cpu().numpy()
            else:
                combinations = list(range(nclasses))
                Ki = Kij
                for classj in combinations:
                    y_diff = y_j[:, classj]
                    ratios = y_diff / Ki[classj]
                    ratio = ratios #.min(dim=1)[0]
                    margins[(classj, classj)] = ratios.cpu().numpy()

        # fig = plt.figure(0)
        plt.style.use('seaborn')
        fig, axes = plt.subplots(len(combinations),1, figsize=(5, 3.5*len(margins)))
        if len(combinations) == 1:
            axes = [axes]

        # mmin, mmax = margins.min(), margins.max()
        # absvmax = max(abs(mmin), abs(mmax))
        # norm = mpl.colors.Normalize(vmin=-absvmax, vmax=absvmax)
        # CS_0=plt.contour(X, Y, (outputs[:,0]-outputs[:,1]).cpu().numpy().reshape(*X.shape), 
        #     levels=[0],
        #     colors=['b'])

        train_dat = np.vstack([dataset[i][0] for i in range(len(dataset))])
        # print(train_dat.max(0))
        labels = dataset.labels

        with torch.no_grad():
            correct_prediction = []
            batch_inp, targets = torch.from_numpy(train_dat).float(), torch.from_numpy(labels)
            output = model(batch_inp.cuda())
            prediction = output.max(dim=1)[1]
            correct_prediction.append(prediction.eq(targets.cuda()))
            correct_prediction = torch.cat(correct_prediction, dim=0).cpu().numpy()

        # colors = np.linspace(1.0, 0, len(self.levels)//2-1)
        colors = ['0.25'] * (len(self.levels)//2-1)
        colors = colors + ['gray', 'k', 'gray'] + colors[::-1]

        for i, (key, decision_margin) in enumerate(margins.items()):
            CS=axes[i].contour(X, Y, decision_margin.reshape(*X.shape), 
                levels=[nf(val) for val in self.levels],
                colors=colors)
                # colors=['1.0', '0.5', '0.25', '0.15', 'b', '0.15', '0.25', '0.5', '1.0'])
            axes[i].clabel(CS, inline=True, fontsize=10)
            # plt.clabel(CS_0, inline=True, fontsize=10)
            # axes[i].set_xlabel('$x_1$')
            # axes[i].set_ylabel('$x_2$')

            for classidx in range(nclasses):
                alpha = 1.0
                if classidx not in key:
                    alpha = 0.3

                axes[i].scatter(x=train_dat[labels==classidx][:,0], y=train_dat[labels==classidx][:,1], alpha=alpha, s=20.0)
                # if classidx in key:
                #     axes[i].scatter(x=train_dat[correct_prediction&(labels==classidx)][:,0], y=train_dat[correct_prediction&(labels==classidx)][:,1], s=2.0, c='k', alpha=alpha)

            # paint incorrect/non-robust samples red
            eps_idx = abs(len(self.levels)//2 + 1)
            robust_samples = torch.abs(output[:, key[0]] - output[:, key[1]])/Kij[key[0], key[1]] > self.levels[eps_idx]
            correct_and_robust = correct_prediction & robust_samples.cpu().numpy()
            axes[i].scatter(x=train_dat[~correct_and_robust][:,0], y=train_dat[~correct_and_robust][:,1], s=20.0, c='#bf484b', alpha=alpha)

            vra = correct_and_robust.mean()
            axes[i].text(self.attr_vra_pos[0], self.attr_vra_pos[1], '$VRA: {:.1f}\%$'.format(vra*100.0), ha=self.attr_vra_ha, fontsize=self.attr_vra_fontsize)
            # axes[i].text(self.attr_K_pos[0], self.attr_K_pos[1], '$K_{{{},{}}}={:.1f}$'.format(key[0], key[1], Kij[key[0], key[1]]), ha=self.attr_K_ha, fontsize=self.attr_K_fontsize)
            if Kh is None:
                Kh = Kij[key[0], key[1]]

            # # if CLip is used:
            # 
            # if 'knll' in lc.loss and (isinstance(lc.loss['knll'].loss, losses.HingeLoss)):
            #     Kh = lc.loss['knll'].loss.K
            # elif 'knll' in lc.loss and (hasattr(lc.loss['knll'], 'sigma')):
            #     knll = lc.loss['knll']
            #     if knll.sigma is None:
            #         cdf_left = knll.err_quantile/2.
            #         cdf_right = (1.0-knll.err_quantile) + (knll.err_quantile)/2.
            #         Q = knll.icdf(cdf_right) - knll.icdf(cdf_left)
            #         #sigma = max(knll.Di / Q)
            #         sigma = knll.Di / Q
            #     else:
            #         sigma = knll.sigma
            #     Kh = 1./sigma            

            axes[i].text(self.attr_K_pos[0], self.attr_K_pos[1], '$K^{{(h\\circ f)}}={:.1f}$'.format(Kh), ha=self.attr_K_ha, fontsize=self.attr_K_fontsize)

            # if self.pairwise:
            #     name = '%d-%d'%key
            #     axes[i].set_title('{} :: $K_{{{},{}}}={:.1f}$'.format(name, key[0], key[1], Kij[key[0], key[1]]))
            # else:
            #     name = '%d'%key[0]
            #     axes[i].set_title('{} :: $K_{{{}}}={:.1f}$'.format(name, key[0], Kij[key[0]]))

        
        import matplotlib
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(os.path.join(self.save_path, 'epoch_{}.pdf'.format(epoch))) as pp:
            plt.tight_layout()
            pp.savefig(facecolor='white')

        # return dict(margins=MatplotlibFigure(fig))