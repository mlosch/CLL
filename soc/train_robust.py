import argparse
import copy
import logging
import os
import time
import math
from shutil import copyfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp

from lip_convnets import LipConvNet, MLP, MoonsLipConvNet
from utils import *

# torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    
    # Training specifications
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--gamma', default=0., type=float, help='gamma for certificate regularization')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O2'],
        help='O0 is FP32 training and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')


    # Model architecture specifications
    parser.add_argument('--conv-layer', default='soc', type=str, choices=['bcop', 'cayley', 'soc'], 
                        help='BCOP, Cayley, SOC convolution')
    parser.add_argument('--init-channels', default=32, type=int)
    parser.add_argument('--activation', default='maxmin', choices=['maxmin', 'hh1', 'hh2'], 
                        help='Activation function')
    parser.add_argument('--block-size', default=1, type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12], 
                        help='size of each block')
    parser.add_argument('--lln', action='store_true', help='set last linear to be linear and normalized')
    parser.add_argument('--fc_mlp', type=int, nargs='*', default=[], help='multi-layered final classification module')


    # Loss specification
    parser.add_argument('--loss_type', default='CrossEntropyLoss', type=str, choices=['CrossEntropyLoss', 'BCELoss', 'CLL', 'CLL_BCE'])
    parser.add_argument('--calib_epsilon', type=float)
    parser.add_argument('--calib_err_quantile', default=0.1, type=float)
    parser.add_argument('--calib_err_quantile_scheduled', type=float, nargs=3, default=[])
    parser.add_argument('--calib_add_epsilon', default=0, type=float)
    parser.add_argument('--calib_onesided', action='store_true')
    parser.add_argument('--detach_K', action='store_true', default=False)
    parser.add_argument('--lambda_', type=float, default=0)
    parser.add_argument('--L_multiplier', type=float, default=1.0)


    # Optimizer specification
    parser.add_argument('--optim', default='SGD', type=str, choices=['SGD', 'AdamW'])
    
    # Dataset specifications
    parser.add_argument('--data-dir', default='./cifar-data', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'tinyimagenet', 'moons2d', '01moons2d'], 
                        help='dataset to use for training')
    
    # Other specifications
    parser.add_argument('--epsilon', default=36, type=int)
    parser.add_argument('--out-dir', default='LipConvnet', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()

def init_model(args):
    if args.dataset in ['moons2d', '01moons2d']:
        if len(args.fc_mlp) > 0:
            model = MoonsLipConvNet(args.conv_layer, args.activation, lln=args.lln, fc_mlp=args.fc_mlp, L_multiplier=args.L_multiplier, soc_fc=args.loss_type=='CrossEntropyLoss')
        else:
            model = MoonsLipConvNet(args.conv_layer, args.activation, lln=args.lln, L_multiplier=args.L_multiplier, soc_fc=args.loss_type=='CrossEntropyLoss')
    else:
        model = LipConvNet(args.conv_layer, args.activation, init_channels=args.init_channels, 
                           block_size = args.block_size, num_classes=args.num_classes, 
                           lln=args.lln, L_multiplier=args.L_multiplier, fc_mlp=args.fc_mlp, soc_fc=args.loss_type=='CrossEntropyLoss')
    print(model)
    return model

def robust_statistics(losses_arr, correct_arr, certificates_arr, 
                      epsilon_list=[36., 72., 108., 144., 180., 216.]):
    mean_loss = np.mean(losses_arr)
    mean_acc = np.mean(correct_arr)
    mean_certs = (certificates_arr * correct_arr).sum()/correct_arr.sum()
    
    robust_acc_list = []
    for epsilon in epsilon_list:
        robust_correct_arr = (certificates_arr > (epsilon/255.)) & correct_arr
        robust_acc = robust_correct_arr.sum()/robust_correct_arr.shape[0]
        robust_acc_list.append(robust_acc)
    return mean_loss, mean_acc, mean_certs, robust_acc_list

class PolynomialScheduler(nn.Module):
    def __init__(self, start_val, end_val, power):
        super(PolynomialScheduler, self).__init__()
        self.start_val = start_val
        self.end_val = end_val
        self.power = power
        self.val = start_val

    def update(self, curr_iter, max_iter):
        scale_factor = (1 - float(curr_iter) / max_iter) ** self.power
        self.val = self.end_val + (self.start_val-self.end_val) * scale_factor
        return self.val

class CalibratedLoss(nn.Module):
    def __init__(self, epsilon, err_quantile_schedule, add_epsilon, onesided, model, L, detach_K=False, lambda_=0.0):
        super(CalibratedLoss, self).__init__()
        self.epsilon = epsilon
        self.err_quantile = PolynomialScheduler(*err_quantile_schedule)
        self.add_epsilon = add_epsilon
        self.detach_K = detach_K
        self.lambda_ = lambda_
        self.onesided = onesided
        self.fc_mlp = None
        if isinstance(model.last_layer, MLP):
            self.fc_weight = lambda : model.last_layer.last_weight
            self.fc_mlp = lambda : model.last_layer
        elif hasattr(model.last_layer, 'lln_weight'):
            self.fc_weight = lambda : model.last_layer.lln_weight
        else:
            self.fc_weight = lambda : model.last_layer.weight
        self.L = L

class CLL(CalibratedLoss):
    def __init__(self, *args, **kwargs):
        super(CLL, self).__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, prediction, target):
        kW = self.fc_weight()
        kW_t = kW[target]
        kW_tj = kW_t[:,:,None] - kW.transpose(1,0).unsqueeze(0)
        K_tj = torch.norm(kW_tj, dim=1, p=2)
        K_tj = torch.scatter(K_tj, dim=1, index=target.unsqueeze(1), value=1.0)

        L = self.L()

        if self.fc_mlp is not None:
            L = L * self.fc_mlp().getK(num_iter=5)

        K_tj = K_tj * L

        icdf = lambda y: np.log(y/(1-y))
        err_quantile = self.err_quantile.val
        cdf_left = err_quantile/2.
        cdf_right = (1.0-err_quantile) + (err_quantile)/2.
        Q = icdf(cdf_right) - icdf(cdf_left)
        sigma = (2.*self.epsilon) / Q

        if self.detach_K:
            K = K_tj.detach()
        else:
            K = K_tj

        wx_i = prediction
        wx_t = torch.gather(wx_i, dim=1, index=target.unsqueeze(1))
        wx_it = wx_i - wx_t
        wx_it = (1./K)*wx_it

        with torch.no_grad():
            if self.onesided:
                delta = torch.full_like(prediction, self.add_epsilon)
                delta.scatter_(dim=1, index=target.unsqueeze(1), value=0)
            else:
                delta = torch.full_like(prediction, self.add_epsilon)
                delta.scatter_(dim=1, index=target.unsqueeze(1), value=-self.add_epsilon)
            # delta = torch.zeros_like(prediction)
            # delta.scatter_(dim=1, index=target.unsqueeze(1), value=-self.add_epsilon)

        wx_it = (wx_it + delta) / sigma

        if self.lambda_ != 0:
            return self.loss(wx_it, target) + self.lambda_*(K_tj.max()**2)  #+ self.lambda_*(K_tj.max()**2)
        else:
            return self.loss(wx_it, target)

class CLL_BCE(CalibratedLoss):
    def __init__(self, *args, **kwargs):
        super(CLL_BCE, self).__init__(*args, **kwargs)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, prediction, target):
        Kfc = torch.norm(self.fc_weight(), dim=1, p=2)
        Kfc = Kfc * self.L

        icdf = lambda y: np.log(y/(1-y))
        err_quantile = self.err_quantile.val
        cdf_left = err_quantile/2.
        cdf_right = (1.0-err_quantile) + (err_quantile)/2.
        Q = icdf(cdf_right) - icdf(cdf_left)
        sigma = (2.*self.epsilon) / Q

        if self.detach_K:
            K = Kfc.detach()
        else:
            K = Kfc

        with torch.no_grad():
            if self.onesided:
                delta = torch.full_like(prediction, self.add_epsilon)
                delta.scatter_(dim=1, index=target.unsqueeze(1), value=0)
            else:
                delta = torch.full_like(prediction, self.add_epsilon)
                delta.scatter_(dim=1, index=target.unsqueeze(1), value=-self.add_epsilon)
        prediction = (prediction + delta) / (K * self.sigma)

        onehot = torch.zeros_like(prediction)
        onehot.scatter_(dim=1, index=target.unsqueeze(1), src=torch.ones_like(target).unsqueeze(1).float())

        if self.lambda_ != 0:
            return self.loss(prediction, onehot) + self.lambda_*(Kfc.max()**2)
        else:
            return self.loss(prediction, onehot)



def main():
    args = get_args()
    
    if args.conv_layer == 'cayley' and args.opt_level == 'O2':
        raise ValueError('O2 optimization level is incompatible with Cayley Convolution')

    args.out_dir += '_' + str(args.dataset) 
    args.out_dir += '_' + str(args.block_size) 
    args.out_dir += '_' + str(args.conv_layer)
    args.out_dir += '_' + str(args.init_channels)
    args.out_dir += '_' + str(args.activation)
    args.out_dir += '_' + str(args.optim)
    args.out_dir += '_cr' + str(args.gamma)
    if args.lln:
        args.out_dir += '_lln'
    if args.loss_type in ['CLL', 'CLL_BCE']:
        args.out_dir += '_calib'+args.loss_type.replace('Calibrated', '').replace('Loss', '')
        args.out_dir += 'eps{:.2f}'.format(args.calib_epsilon)
        if args.calib_err_quantile_scheduled:
            args.out_dir += 'q{:.2e}to{:.2e}p{:.1f}'.format(*args.calib_err_quantile_scheduled)
        else:
            args.out_dir += 'q{:.2e}'.format(args.calib_err_quantile)
        args.out_dir += 'add{:.2f}'.format(args.calib_add_epsilon)
        if args.calib_onesided:
            args.out_dir += '-onesided_'
        if args.lambda_ != 0:
            args.out_dir += 'lam{:.2e}'.format(args.lambda_)
        if args.detach_K:
            args.out_dir += 'detachK'
    if args.L_multiplier != 1.0:
        args.out_dir += '-Lm{:.2f}'.format(args.L_multiplier)

    if args.seed != 0:
        args.out_dir += '-seed{}'.format(args.seed)

    
    if os.path.exists(args.out_dir):
        raise RuntimeError('Configuration already exists at {}'.format(args.out_dir))

    os.makedirs(args.out_dir, exist_ok=True)
    code_dir = os.path.join(args.out_dir, 'code')
    os.makedirs(code_dir, exist_ok=True)
    for f in os.listdir('./'):
        src = os.path.join('./', f)
        dst = os.path.join(code_dir, f)
        if os.path.isfile(src):
            if f[-3:] == '.py' or f[-3:] == '.sh':
                copyfile(src, dst)
    
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, 'output.log'))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader, train_dataset = get_loaders(args.data_dir, args.batch_size, args.dataset)
    std = cifar10_std
    if args.dataset == 'cifar10':
        args.num_classes = 10    
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'tinyimagenet':
        args.num_classes = 200
    elif args.dataset == 'moons2d' or args.dataset == '01moons2d':
        args.num_classes = 2
        std = [1, 1]
    else:
        raise Exception('Unknown dataset')

    # Lipschitz constant of input normalization
    std = torch.tensor(std).cuda()
    # L = 1/torch.max(std)

    # Evaluation at early stopping
    model = init_model(args).cuda()
    model.train()

    getL = lambda : model.L_multiplier.detach().prod().item() * (1./torch.max(std))

    # if args.dataset == 'moons2d':
    #     L *= args.L_multiplier ** len(model.layers)
    # else:
    #     L *= args.L_multiplier ** (args.block_size * 5)
    print('Lipschitz constant of data normalization and initial L_multipler: ', getL())

    conv_params, activation_params, other_params = parameter_lists(model)
    if args.conv_layer == 'soc':
        if args.optim == 'SGD':
            opt = torch.optim.SGD([
                            {'params': activation_params, 'weight_decay': 0.},
                            {'params': (conv_params + other_params), 'weight_decay': args.weight_decay}
                        ], lr=args.lr_max, momentum=args.momentum)
        elif args.optim == 'AdamW':
            opt = torch.optim.AdamW([
                            {'params': activation_params, 'weight_decay': 0.},
                            {'params': (conv_params + other_params), 'weight_decay': args.weight_decay}
                        ], lr=args.lr_max)
        else:
            raise NotImplementedError
    else:
        if args.optim == 'SGD':
            opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, 
                                  weight_decay=0.)
        else:
            raise NotImplementedError
        
    amp_enabled = False
    if args.dataset not in ['moons2d', '01']:
        amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
        if args.opt_level == 'O2':
            amp_args['master_weights'] = True
        model, opt = amp.initialize(model, opt, **amp_args)
        amp_enabled = True

    if args.loss_type == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_type == 'BCELoss':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss_type == 'CLL':
        if len(args.calib_err_quantile_scheduled) == 0:
            args.calib_err_quantile_scheduled = [args.calib_err_quantile, args.calib_err_quantile, 1.0]
        criterion = CLL(args.calib_epsilon, args.calib_err_quantile_scheduled, args.calib_add_epsilon, args.calib_onesided, model=model, L=getL)
    elif args.loss_type == 'CLL_BCE':
        if len(args.calib_err_quantile_scheduled) == 0:
            args.calib_err_quantile_scheduled = [args.calib_err_quantile, args.calib_err_quantile, 1.0]
        criterion = CLL_BCE(args.calib_epsilon, args.calib_err_quantile_scheduled, args.calib_add_epsilon, args.calib_onesided, model=model, L=getL)

    lr_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps // 2, 
        (3 * lr_steps) // 4], gamma=0.1)
    
    best_model_path = os.path.join(args.out_dir, 'best.pth')
    last_model_path = os.path.join(args.out_dir, 'last.pth')
    last_opt_path = os.path.join(args.out_dir, 'last_opt.pth')
    
    # Training
    eps_test = [36., 72., 108., 144., 180., 216.]
    if args.loss_type.startswith('Calibrated'):
        eps_test.append(args.calib_epsilon*255.)

    prev_robust_acc = 0.
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t Train Loss \t Train Acc \t Test Loss \t ' + 
                'Test Acc \t VRA (36) \t VRA (72) \t VRA (108) \t VRA (eps) \t Kfc')
    print('Epoch \t Seconds \t LR \t Train Loss \t Train Acc \t Test Loss \t ' + 
                'Test Acc \t VRA (36) \t VRA (72) \t VRA (108) \t VRA (eps) \t Kfc')
    for epoch in range(args.epochs):
        model.train()
        start_epoch_time = time.time()
        train_loss = 0
        train_cert = 0
        train_robust = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            
            output = model(X)
            curr_correct = (output.max(1)[1] == y)
            L = getL()
            if args.loss_type == 'CrossEntropyLoss' and not args.lln:
                curr_cert = ortho_certificates(output, y, L)
            else:
                curr_cert = lln_certificates(output, y, model.last_layer, L)
            # if args.lln:
            #     curr_cert = lln_certificates(output, y, model.last_layer, L)
            # else:
            #     curr_cert = ortho_certificates(output, y, L)
                
            ce_loss = criterion(output, y)
            loss = ce_loss - args.gamma * F.relu(curr_cert).mean()

            opt.zero_grad()
            if amp_enabled:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            opt.step()

            train_loss += ce_loss.item() * y.size(0)
            train_cert += (curr_cert * curr_correct).sum().item()
            train_robust += ((curr_cert > (args.epsilon/255.)) * curr_correct).sum().item()
            train_acc += curr_correct.sum().item()
            train_n += y.size(0)
            scheduler.step()

            # update scheduler
            if isinstance(criterion, CalibratedLoss):
                val = criterion.err_quantile.update(epoch*len(train_loader)+i, args.epochs*len(train_loader))
                # print(epoch*len(train_loader)+i, val)
            
        # print(model.L_multiplier)
        L = getL()
        # Check current test accuracy of model
        losses_arr, correct_arr, certificates_arr = evaluate_certificates(test_loader, model, L, lln_certificate=not (args.loss_type == 'CrossEntropyLoss' and not args.lln))
        
        test_loss, test_acc, test_cert, test_robust_acc_list = robust_statistics(
            losses_arr, correct_arr, certificates_arr, eps_test)
        
        robust_acc = test_robust_acc_list[0]
        if (robust_acc >= prev_robust_acc):
            torch.save(model.state_dict(), best_model_path)
            prev_robust_acc = robust_acc
            best_epoch = epoch

        if args.loss_type == 'CrossEntropyLoss' and not args.lln:
            with torch.no_grad():
                Kfc = 1.0 * L
        else:
            if args.lln:
                Wfc = model.last_layer.lln_weight
            else:
                Wfc = model.last_layer.weight
            
            with torch.no_grad():
                Kfc = 0
                for i in range(Wfc.shape[0]):
                    for j in range(i+1, Wfc.shape[0]):
                        Wdiff = (Wfc[i] - Wfc[j]).float()
                        Kfc = max(Kfc, torch.norm(Wdiff, p=2).item())
                Kfc *= L
        # Kfc = torch.cdist(Wfc, Wfc, p=2).max()

        epoch_time = time.time()
        lr = scheduler.get_last_lr()[0]
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.2f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n, 
            test_loss, test_acc, test_robust_acc_list[0], test_robust_acc_list[1], 
            test_robust_acc_list[2], test_robust_acc_list[-1], Kfc)
        print('%d \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.2f'%(
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n, 
            test_loss, test_acc, test_robust_acc_list[0], test_robust_acc_list[1], 
            test_robust_acc_list[2], test_robust_acc_list[-1], Kfc))
        
        torch.save(model.state_dict(), last_model_path)
        
        trainer_state_dict = { 'epoch': epoch, 'optimizer_state_dict': opt.state_dict()}
        torch.save(trainer_state_dict, last_opt_path)

        if args.dataset in ['moons2d', '01moons2d'] and (epoch%10)==0:
            model.eval()
            xlim = (-1.5, 2.5)
            ylim = (-1.0, 1.5)
            attr_K_pos=(-1.45, 1.25)
            attr_vra_pos=(2.4, -0.95)
            if args.dataset == '01moons2d':
                xlim = (0, 1)
                ylim = (0, 1)
                attr_K_pos=(-0.05, 0.75)
                attr_vra_pos=(0.95, 0.05)

            Kh = None
            if args.loss_type == 'CLL':
                p = args.calib_err_quantile
                cdf_left = p/2.
                cdf_right = (1.0-p) + p/2.
                icdf = lambda y: np.log(y/(1-y))
                Q = icdf(cdf_right) - icdf(cdf_left)
                sigma = (2*args.calib_epsilon) / Q
                Kh = 1./sigma

            db = DecisionBoundaryPlot(xlim, ylim, 1000000, pairwise=True, levels=[-0.15, 0, 0.15], save_path=args.out_dir, attr_K_pos=attr_K_pos, attr_vra_pos=attr_vra_pos)
            db.forward(L, model, train_dataset, test_loader, epoch, Kh=Kh)
        
    train_time = time.time()

    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)
    print('Total train time: %.4f minutes', (train_time - start_train_time)/60)
    
    
    # Evaluation at best model (early stopping)
    model_test = init_model(args).cuda()
    model_test.load_state_dict(torch.load(best_model_path))
    model_test.float()
    model_test.eval()
        
    start_test_time = time.time()
    losses_arr, correct_arr, certificates_arr = evaluate_certificates(test_loader, model_test, L)
    total_time = time.time() - start_test_time
    
    test_loss, test_acc, test_cert, test_robust_acc_list = robust_statistics(
        losses_arr, correct_arr, certificates_arr, eps_test)
    
    logger.info('Best Epoch \t Test Loss \t Test Acc \t Test Robust (36) \t Test Robust (72) \t Test Robust (108) \t Test Robust (eps) \t Mean Cert \t Test Time')
    logger.info('%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f', best_epoch, test_loss, test_acc,
                                                        test_robust_acc_list[0], test_robust_acc_list[1], 
                                                        test_robust_acc_list[2], test_robust_acc_list[-1], total_time)

    # Evaluation at last model
    model_test.load_state_dict(torch.load(last_model_path))
    model_test.float()
    model_test.eval()

    start_test_time = time.time()
    losses_arr, correct_arr, certificates_arr = evaluate_certificates(test_loader, model_test, L)
    total_time = time.time() - start_test_time
    
    test_loss, test_acc, test_cert, test_robust_acc_list = robust_statistics(
        losses_arr, correct_arr, certificates_arr, eps_test)
    
    logger.info('Last Epoch \t Test Loss \t Test Acc \t Test Robust (36) \t Test Robust (72) \t Test Robust (108) \t Test Robust (eps) \t Mean Cert \t Test Time')
    logger.info('%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f', epoch, test_loss, test_acc,
                                                        test_robust_acc_list[0], test_robust_acc_list[1], 
                                                        test_robust_acc_list[2], test_robust_acc_list[-1], total_time)

if __name__ == "__main__":
    main()


