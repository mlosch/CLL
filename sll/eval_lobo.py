import os
import sys
import logging
import warnings
import argparse
import time
import glob

from os.path import exists, realpath, join
from collections import OrderedDict
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# from model import ConvexPotentialLayerNetwork, NormalizedModel
# from layers import LinearNormalized, PoolingLinear
import core.data as data

from core import utils
from core.models.model import NormalizedModel, LipschitzNetwork
from core.models.layers import LinearNormalized
from core.data.readers import readers_config
from core.evaluate import Evaluator

warnings.filterwarnings("ignore")

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
    def __init__(self, config, n_samples, batch_size, optimizer_cfg, max_iter, dataset, input_norm_correction=1.0, input_min_val=0, input_max_val=1, **kwargs):
        super(LowerLipschitzBoundEstimation, self).__init__(**kwargs)
        assert batch_size % 2 == 0, 'batch_size must be multiple of 2'
        self.config = config
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
        outputs = model(inputs)
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
            L = torch.abs(margin1 - margin2) / torch.norm(X1-X2, p=2, dim=[1,2,3]).unsqueeze(1) #.detach()
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
        
        logging.info('Prepping dataset ...')
        for i in range(min(len(dataset), self.n_samples*2)):
            image, label = dataset[im_inds[i]]
            all_inputs.append(image.unsqueeze(0))
            all_labels.append(label)
        logging.info('Done')

        max_loss = None

        # model = kwargs['lipschitz_computer']
        # K = model.lipschitz_estimate(num_iter=-1, update=False)
        # Kfc = model._modules[model.classifier_layer].estimate(num_iter=0)

        if self.config.last_layer == 'lln':
        # if isinstance(model.last_last, LinearNormalized):
            Wfc = model.module.model.last_last.weight / torch.norm(model.module.model.last_last.weight, dim=1, keepdim=True)
        elif self.config.last_layer == 'vanilla':
            Wfc = model.module.model.last_last.weight
        elif self.config.last_layer == 'padding_linear':
            Wfc = None #last.weights[:model.num_classes, :]
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
                        #Kfc = max(Kfc, torch.norm(Wdiff, p=2).item())
                        Kfc = max(Kfc, torch.sqrt(torch.sum(Wdiff**2)).item())
        K = K * Kfc
        logging.info('Kfc={}. K={}'.format(Kfc, K))

        for j, bs_idx in enumerate(range(0, self.n_samples*2, self.batch_size)):
            inputs = torch.cat(all_inputs[bs_idx:bs_idx+self.batch_size], dim=0).to(model_device)
            labels = torch.Tensor(all_labels[bs_idx:bs_idx+self.batch_size]).to(model_device)

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
                        # print('LowerLipschitzBoundEstimation: [{}/{}] Lower bound estimate: {:.4f}, K: {:.4f}, Tightness {:.4f}, ETA: {:.0f}h:{:.0f}m:{:.0f}s'.format(cur_iter, sum_max_iter, max_loss.item(), K, (max_loss/K).detach().item(), eta//3600, (eta-(eta//3600)*3600)//60, eta%60))

            model = model.float()

        if self.logger is not None:
            self.logger.info('LowerLipschitzBoundEstimation: [Done] Lower bound estimate: {:.4f}, K: {:.4f}, Tightness {:.4f}'.format(max_loss.item(), K, (max_loss/K).detach().item()))
            # print('LowerLipschitzBoundEstimation: [Done] Lower bound estimate: {:.4f}, K: {:.4f}, Tightness {:.4f}'.format(max_loss.item(), K, (max_loss/K).detach().item()))

        results = dict(lower_bound=max_loss, bound_tightness=(max_loss/K).detach())
        return results


def load_ckpt(config, model):
  # if ckpt_path is None:
  checkpoints = glob.glob(join(config.train_dir, "checkpoints", "model.ckpt-*.pth"))
  # print(checkpoints)
  get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
  ckpt_name = sorted([ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)[-1]
  print(ckpt_name)
  ckpt_path = join(config.train_dir, "checkpoints", ckpt_name)
  checkpoint = torch.load(ckpt_path)
  new_checkpoint = {}
  for k, v in checkpoint['model_state_dict'].items():
    if 'alpha' not in k:
      new_checkpoint[k] = v
  model.load_state_dict(new_checkpoint)
  print('Epoch', checkpoint['epoch'])
  return model


def main(config):
  folder = config.train_dir.split('/')[-1]

  cudnn.benchmark = True

  # create a mesage builder for logging
  message = utils.MessageBuilder()
  # Setup logging & log the version.
  utils.setup_logging(config, 0)

  ngpus = torch.cuda.device_count()
  if ngpus:
    batch_size = config.batch_size * ngpus
  else:
    batch_size = config.batch_size

  # load reader
  Reader = readers_config[config.dataset]
  reader = Reader(config, batch_size, False, is_training=False)
  config.means = reader.means

  mean, std = reader.means, reader.stds

  # load model
  model = LipschitzNetwork(config, reader.n_classes)
  model = NormalizedModel(model, mean, std)
  model = torch.nn.DataParallel(model)
  model = model.cuda()

  train_loader, _ = reader.load_dataset()
  model = load_ckpt(config, model)
  # model.module.mean.zero_()

  model.eval()

  # evaluator = Evaluator(config)
  # evaluator.message = message
  # evaluator.batch_size = batch_size
  # evaluator.reader = reader
  # evaluator.config.means = reader.means
  # evaluator.model = model

  # for eps in [36, 72, 108, 255]:
  #   if config.last_layer == 'lln' or config.last_layer == 'vanilla':
  #     accuracy, certified = evaluator.eval_certified_lln(eps)
  #   else:
  #     accuracy, certified = evaluator.eval_certified(eps)

  lobo = LowerLipschitzBoundEstimation(
        config,
        n_samples=4000,
        batch_size=batch_size,
        optimizer_cfg=dict(type='Adam', lr=3.e-4, weight_decay=0),
        max_iter=2000,
        dataset='train',
        data_mean=mean, data_std=std)

  L = 1/min(std)
  print('Dataset K={}'.format(L))
  lobo.forward(model, L, train_loader.dataset, 0, logging)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')

  # parameters training or eval
  # parser.add_argument('ckpt', type=str, help="Checkpoint file")

  parser.add_argument("--train_dir", type=str, help="Name of the training directory.")
  parser.add_argument("--data_dir", type=str, help="Name of the data directory.")
  parser.add_argument("--dataset", type=str,  default='cifar10', help="Dataset to use")

  parser.add_argument("--shift_data", type=bool, default=True, help="Shift dataset with mean.")
  parser.add_argument("--normalize_data", action='store_true', help="Normalize dataset.")

  # parameters of the architectures
  parser.add_argument("--model-name", type=str)
  parser.add_argument("--batch_size", type=int, default=200)
  parser.add_argument("--depth", type=int, default=30)
  parser.add_argument("--num_channels", type=int, default=30)
  parser.add_argument("--depth_linear", type=int, default=5)
  parser.add_argument("--n_features", type=int, default=2048)
  parser.add_argument("--conv_size", type=int, default=5)
  parser.add_argument("--init", type=str, default='xavier_normal')

  parser.add_argument("--first_layer", type=str, default="padding_channels")
  parser.add_argument("--last_layer", type=str, default="pooling_linear")

  parser.add_argument("--ngpus", type=int, default=1)
  parser.add_argument("--logging_verbosity", type=str, default='INFO', help="Level of verbosity of the logs")


  # parse all arguments 
  config = parser.parse_args()
  config.cmd = f"python3 {' '.join(sys.argv)}"
  config.mode = 'lobo'
  config.unconstrained_layer_Ks = False
  config.mlp = False
  config.liplin = False

  def override_args(config, depth, num_channels, depth_linear, n_features):
    config.depth = depth
    config.num_channels = num_channels
    config.depth_linear = depth_linear
    config.n_features = n_features
    return config

  if config.model_name == 'small':
    config = override_args(config, 20, 45, 7, 2048)
  elif config.model_name == 'medium':
    config = override_args(config, 30, 60, 10, 2048)
  elif config.model_name == 'large':
    config = override_args(config, 50, 90, 10, 2048)
  elif config.model_name == 'xlarge':
    config = override_args(config, 70, 120, 15, 2048)
  elif config.model_name is None and \
      not all([config.depth, config.num_channels, config.depth_linear, config.n_features]):
    ValueError("Choose --model-name 'small' 'medium' 'large' 'xlarge'")

  # process argments
  eval_mode = ['certified', 'attack']
  if config.data_dir is None:
    config.data_dir = os.environ.get('DATADIR', None)
  if config.data_dir is None:
    ValueError("the following arguments are required: --data_dir")
  os.makedirs('./trained_models', exist_ok=True)
  path = realpath('./trained_models')
  if config.train_dir is None:
    ValueError("--train_dir must be defined.")
  # config.train_dir = f'{path}/{config.train_dir}'
  # elif config.mode == 'train' and config.train_dir is not None:
  #   config.train_dir = f'{path}/{config.train_dir}'
  #   os.makedirs(config.train_dir, exist_ok=True)
  #   os.makedirs(f'{config.train_dir}/checkpoints', exist_ok=True)
  # elif config.mode in eval_mode and config.train_dir is not None:
  #   config.train_dir = f'{path}/{config.train_dir}'
  # elif config.mode in eval_mode and config.train_dir is None:
  #   ValueError("--train_dir must be defined.")

  main(config)
