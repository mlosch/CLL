import os
# import csv
import logging
import time

from collections import OrderedDict
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ConvexPotentialLayerNetwork, NormalizedModel
from layers import LinearNormalized, PoolingLinear
import data as data
from util import get_loaders

class InputSpaceEvaluationBase(nn.Module):
    def __init__(self, data_mean, data_std, **kwargs):
        super(InputSpaceEvaluationBase, self).__init__()
        self.data_mean = torch.Tensor(data_mean)
        self.data_std = torch.Tensor(data_std)

    def _init_optimizer(self, params, optim_cfg):
        cfg = dict(optim_cfg)
        optim = torch.optim.__dict__[cfg.pop('type')](params, **cfg)
        return optim

    def _remove_normalization(self, x):
        # remove image normalization
        if x.device != self.data_mean.device:
            self.data_mean = self.data_mean.to(x.device)
            self.data_std = self.data_std.to(x.device)
        if len(x.shape) == 4:
            std = self.data_std[None, :, None, None]
            mean = self.data_mean[None, :, None, None]
        elif len(x.shape) == 2:
            std = self.data_std[None, :]
            mean = self.data_mean[None, :]
        else:
            raise RuntimeError(str(x.shape))
        x = x * std + mean
        return x

    def _normalize(self, x):
        if x.device != self.data_mean.device:
            self.data_mean = self.data_mean.to(x.device)
            self.data_std = self.data_std.to(x.device)
        if len(x.shape) == 4:
            std = self.data_std[None, :, None, None]
            mean = self.data_mean[None, :, None, None]
        elif len(x.shape) == 2:
            std = self.data_std[None, :]
            mean = self.data_mean[None, :]
        x = (x - mean) / std
        return x

class LowerLipschitzBoundEstimation(InputSpaceEvaluationBase):
    def __init__(self, n_samples, batch_size, optimizer_cfg, max_iter, dataset, input_norm_correction=1.0, input_min_val=0, input_max_val=1, **kwargs):
        super(LowerLipschitzBoundEstimation, self).__init__(**kwargs)
        assert batch_size % 2 == 0, 'batch_size must be multiple of 2'
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.logger = None
        self.dataset_key = dict(val='val_dataset', train='train_dataset')[dataset]
        self.input_norm_correction = input_norm_correction
        self.input_min_val = input_min_val
        self.input_max_val = input_max_val

        assert 'type' in optimizer_cfg
        self.optimizer_cfg = optimizer_cfg

        self._times = []

    def lbo_loss(self, model, inputs):    
        outputs = model(self._normalize(inputs))   
        # outputs, _ = self._forward_pass(model, self._normalize(inputs))
        prediction_values, predictions = outputs.max(dim=1)

        N = inputs.shape[0] // 2
        X1, X2 = inputs[:N], inputs[N:]
        y1, y2 = outputs[:N], outputs[N:]

        j = predictions[:N]

        y1_j = prediction_values[:N]
        y2_j = torch.gather(y2, dim=1, index=j.unsqueeze(1))

        margin1 = y1_j.unsqueeze(1) - y1
        margin2 = y2_j - y2

        if len(X1.shape) == 4:
            # print(torch.norm(X1-X2, p=2, dim=[1,2,3])[:10])
            L = torch.abs(margin1 - margin2) / torch.norm(X1-X2, p=2, dim=[1,2,3]).unsqueeze(1)
        else:
            ynorm = torch.abs(margin1 - margin2)
            Xnorm = torch.norm((X1-X2).view(X1.shape[0],-1), p=2, dim=-1)
            # Xnorm = Xnorm[Xnorm>0]
            # ynorm = ynorm[Xnorm>0]
            L = ynorm / (Xnorm.unsqueeze(1)+1.e-9)
        L = L * self.input_norm_correction

        loss = -L.max(dim=1)[0]

        return loss

    def forward(self, model, K, dataset, epoch, logger):
        model_device = torch.ones(1).cuda().device
        self.logger = logger
        # self.logger = kwargs.get('logger', None)
        # dataset = kwargs[self.dataset_key]
        # epoch = kwargs['epoch']

        all_inputs, all_labels = [], []

        im_inds = torch.randperm(len(dataset))
        
        for i in range(min(len(dataset), self.n_samples*2)):
            image, label = dataset[im_inds[i]]
            all_inputs.append(image.unsqueeze(0))
            all_labels.append(label)

        max_loss = None

        # model = kwargs['lipschitz_computer']
        # K = model.lipschitz_estimate(num_iter=-1, update=False)
        # Kfc = model._modules[model.classifier_layer].estimate(num_iter=0)

        if isinstance(model.last_last, LinearNormalized):
            Wfc = model.last_last.weight / torch.norm(model.last_last.weight, dim=1, keepdim=True)
        elif isinstance(model.last_last, PoolingLinear):
            # if model.last_last.agg != 'trunc':
            #     raise NotImplementedError('Not implemented aggregation type in PoolingLinear: {}'.format(model.last_last.agg))
            # # in case of truncation, we need the weight matrix of the layer before

            # last = model.layers_linear._modules[str(self.depth_linear-1)]
            Wfc = None #last.weights[:model.num_classes, :]
        elif isinstance(model.last_last, nn.Linear):
            Wfc = model.last_last.weight
        else:
            raise NotImplementedError

        if Wfc is None:
            Kfc = np.sqrt(2)
        else:
            with torch.no_grad():
                Kfc = 0
                for i in range(Wfc.shape[0]):
                    for j in range(i+1, Wfc.shape[0]):
                        Wdiff = (Wfc[i] - Wfc[j]).float()
                        Kfc = max(Kfc, torch.norm(Wdiff, p=2).item())
        K = K * Kfc

        for j, bs_idx in enumerate(range(0, self.n_samples*2, self.batch_size)):
            inputs = torch.cat(all_inputs[bs_idx:bs_idx+self.batch_size], dim=0).to(model_device)
            labels = torch.Tensor(all_labels[bs_idx:bs_idx+self.batch_size]).to(model_device)

            inputs = self._remove_normalization(inputs)
            # print(inputs[0])

            inputs.requires_grad = True

            optimizer = self._init_optimizer([inputs], self.optimizer_cfg)

            for i in range(self.max_iter):
                loss = self.lbo_loss(model, inputs)
                loss.sum().backward()
                # loss.sum().backward()

                L = -loss

                if max_loss is None:
                    max_loss = L.max().detach()
                else:
                    with torch.no_grad():
                        max_loss = torch.max(max_loss, L.max().detach())

                optimizer.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    # mean = self.data_mean
                    # std = self.data_std
                    inputs.clamp_(self.input_min_val,self.input_max_val)
                    # inputs[:,0].clamp_(-mean[0]/std[0],(1-mean[0])/std[0])
                    # inputs[:,1].clamp_(-mean[1]/std[1],(1-mean[1])/std[1])
                    # inputs[:,2].clamp_(-mean[2]/std[2],(1-mean[2])/std[2])

                i_iter = 100
                if i % i_iter == 0:
                    if self.logger is not None:
                        self._times.append(time.time())
                        sum_max_iter = self.max_iter * ((self.n_samples*2)/self.batch_size)
                        cur_iter = j*self.max_iter + i
                        dt_iter = np.nan
                        if len(self._times) > 1:
                            times = np.array(self._times)
                            dts = (times[1:] - times[:-1])
                            dt_iter = np.mean(dts)/i_iter
                        eta = (sum_max_iter - cur_iter) * dt_iter
                        # self.logger.info('LowerLipschitzBoundEstimation: [{}/{}] Lower bound estimate: {:.4f}  ETA: {:.0f}h:{:.0f}m:{:.0f}s'.format(cur_iter, sum_max_iter, max_loss.item(), eta//3600, (eta-(eta//3600)*3600)//60, eta%60))
                        # print('LowerLipschitzBoundEstimation: [{}/{}] Lower bound estimate: {:.4f}  ETA: {:.0f}h:{:.0f}m:{:.0f}s'.format(cur_iter, sum_max_iter, max_loss.item(), eta//3600, (eta-(eta//3600)*3600)//60, eta%60))
                        self.logger.info('LowerLipschitzBoundEstimation: [{}/{}] Lower bound estimate: {:.4f}, K: {:.4f}, Tightness {:.4f}, ETA: {:.0f}h:{:.0f}m:{:.0f}s'.format(cur_iter, sum_max_iter, max_loss.item(), K, (max_loss/K).detach().item(), eta//3600, (eta-(eta//3600)*3600)//60, eta%60))
                        print('LowerLipschitzBoundEstimation: [{}/{}] Lower bound estimate: {:.4f}, K: {:.4f}, Tightness {:.4f}, ETA: {:.0f}h:{:.0f}m:{:.0f}s'.format(cur_iter, sum_max_iter, max_loss.item(), K, (max_loss/K).detach().item(), eta//3600, (eta-(eta//3600)*3600)//60, eta%60))

            model = model.float()

        if self.logger is not None:
            self.logger.info('LowerLipschitzBoundEstimation: [Done] Lower bound estimate: {:.4f}, K: {:.4f}, Tightness {:.4f}'.format(max_loss.item(), K, (max_loss/K).detach().item()))
            print('LowerLipschitzBoundEstimation: [Done] Lower bound estimate: {:.4f}, K: {:.4f}, Tightness {:.4f}'.format(max_loss.item(), K, (max_loss/K).detach().item()))

        results = dict(lower_bound=max_loss, bound_tightness=(max_loss/K).detach())
        return results


def init_model(args, mean, std):
    model = ConvexPotentialLayerNetwork(depth=args.depth,
                                             depth_linear=args.depth_linear,
                                             num_classes=args.num_classes, conv_size=args.conv_size,
                                             num_channels=args.num_channels,
                                             n_features=args.n_features,
                                             use_lln=args.lln,
                                             use_vanilla_linear=args.loss_type!='default')

    model = NormalizedModel(model, mean, std)
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)

    # data args
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--norm_input', action='store_true')

    # model args
    parser.add_argument('--depth', type=int, default=20)
    parser.add_argument('--depth-linear', type=int, default=7)
    parser.add_argument('--num-channels', type=int, default=30)
    parser.add_argument('--n-features', type=int, default=2048)
    parser.add_argument('--conv-size', type=int, default=5)
    parser.add_argument('--lln', action='store_true')
    parser.add_argument('--block_lin', default='cpl', help='choose between {cpl, nn.Linear}')
    parser.add_argument('--loss_type', type=str, default='default')

    # parser.add_argument('--conv-layer', default='soc', type=str, choices=['bcop', 'cayley', 'soc'], 
    #                     help='BCOP, Cayley, SOC convolution')
    # parser.add_argument('--init-channels', default=32, type=int)
    # parser.add_argument('--activation', default='maxmin', choices=['maxmin', 'hh1', 'hh2'], 
    #                     help='Activation function')
    # parser.add_argument('--block-size', default=1, type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12], 
    #                     help='size of each block')
    # parser.add_argument('--lln', action='store_true', help='set last linear to be linear and normalized')
    # parser.add_argument('--data-dir', default='./cifar-data', type=str)
    # parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'tinyimagenet'], 
    #                     help='dataset to use for training')
    # parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O2'],
    #     help='O0 is FP32 training and O2 is "Almost FP16" Mixed Precision')
    # parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
    #     help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    args = parser.parse_args()

    assert os.path.exists(args.checkpoint)
    args.out_dir = os.path.dirname(args.checkpoint)

    logfile = os.path.join(args.out_dir, 'lobo.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)
    
    batch_size = 200
    if args.dataset == "cifar10":
        mean = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
        std = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)
        args.num_classes = 10
    else:
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
        args.num_classes = 100

    if not args.norm_input:
        std = (1., 1., 1.)
    train_loader, test_loader = get_loaders(args.data_dir, batch_size, args.dataset, mean=mean, std=std)

    # data_getter = data.DataClass(args.dataset, batch_size=batch_size)
    # train_batches, test_batches = data_getter()
    # train_loader, test_loader = train_batches.dataloader, test_batches.dataloader


    model = init_model(args, mean, std).cuda()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)
    model = model.module.model # unpack from DataParallel and NormalizedModel
    print(model)
    model.eval()

    #__init__(self, n_samples, batch_size, optimizer_cfg, max_iter, dataset, input_norm_correction=1.0, input_min_val=0, input_max_val=1, **kwargs):
    #n_samples: 4000
    #batch_size: 1000
    #max_iter: 1000
    # input_norm_correction: 0.2
    #optimizer_cfg:
    lobo = LowerLipschitzBoundEstimation(
        n_samples=4000,
        batch_size=batch_size,
        optimizer_cfg=dict(type='Adam', lr=3.e-4, weight_decay=0),
        max_iter=2000,
        dataset='train',
        data_mean=mean, data_std=std)

    L = 1/min(std)
    lobo.forward(model, L, train_loader.dataset, 0, logger)
