import os
import sys
import time
import random
import datetime
import pprint
import socket
import logging
import glob
from os.path import join, exists
from contextlib import nullcontext

from core import utils
from core.models.model import NormalizedModel, LipschitzNetwork
from core.models.layers import LinearNormalized, PoolingLinear
from core.data.readers import readers_config

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist


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
        self.err_quantile = err_quantile_schedule
        self.add_epsilon = add_epsilon
        self.detach_K = detach_K
        self.lambda_ = lambda_
        self.onesided = onesided
        if isinstance(model.last_last, PoolingLinear):
            self.fc_weight = None
        elif isinstance(model.last_last, LinearNormalized):
            self.fc_weight = lambda : model.last_last.weight / torch.norm(model.last_last.weight, dim=1, keepdim=True)
        elif isinstance(model.last_last, nn.Linear):
            self.fc_weight = lambda : model.last_last.weight
        else:
            raise NotImplementedError('Unknown layer type {}'.format(model.last_last))

        if model.unconstrained_layer_Ks:
          self.model_Ks = lambda : model
        else:
          self.model_Ks = None

        if len(model.soft_constrained_layers) > 0:
          self.hidden_Ks = model.soft_constrained_layers
        else:
          self.hidden_Ks = None

        self.L = L
        print(f'Lambda: {lambda_}, L: {L}')

class CLL(CalibratedLoss):
    def __init__(self, *args, **kwargs):
        super(CLL, self).__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, prediction, target):
        kW = self.fc_weight()
        target = target.cuda()
        prediction = prediction.cuda()
        kW_t = kW[target]
        kW_tj = kW_t[:,:,None] - kW.transpose(1,0).unsqueeze(0)
        K_tj = torch.norm(kW_tj, dim=1, p=2)
        K_tj = torch.scatter(K_tj, dim=1, index=target.unsqueeze(1), value=1.0)

        L = self.L

        if self.hidden_Ks:
          pK = 1.0
          for liplayer in self.hidden_Ks:
            pK *= liplayer.estimate(num_iter=1, update=self.training)
          K_tj = K_tj * L * pK

        if self.model_Ks is not None:
          K_tj = K_tj * L * torch.prod(self.model_Ks().Ks)
        K_tj = K_tj * L

        icdf = lambda y: np.log(y/(1-y))
        err_quantile = self.err_quantile #.val
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

        wx_it = (wx_it + delta) / sigma

        log_probs = F.log_softmax(wx_it, dim=1) #*K
        return F.nll_loss(log_probs, target) + self.lambda_*(K_tj.max()**2)


class Trainer:
  """A Trainer to train a PyTorch."""

  def __init__(self, config):
    self.config = config

  def _load_state(self):
    # load last checkpoint
    checkpoints = glob.glob(join(self.train_dir, 'checkpoints', 'model.ckpt-*.pth'))
    get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
    checkpoints = sorted(
      [ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)
    path_last_ckpt = join(self.train_dir, 'checkpoints', checkpoints[-1])
    self.checkpoint = torch.load(path_last_ckpt, map_location=self.model.device)
    self.model.load_state_dict(self.checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
    self.saved_ckpts.add(self.checkpoint['epoch'])
    epoch = self.checkpoint['epoch']
    if self.local_rank == 0:
      logging.info('Loading checkpoint {}'.format(checkpoints[-1]))

  def _save_ckpt(self, step, epoch, final=False, best=False):
    """Save ckpt in train directory."""
    freq_ckpt_epochs = self.config.save_checkpoint_epochs
    if (epoch % freq_ckpt_epochs == 0 and self.is_master \
        and epoch not in self.saved_ckpts) \
         or (final and self.is_master) or best:
      prefix = "model" if not best else "best_model"
      ckpt_name = f"{prefix}.ckpt-{step}.pth"
      ckpt_path = join(self.train_dir, 'checkpoints', ckpt_name)
      if exists(ckpt_path) and not best: return 
      self.saved_ckpts.add(epoch)
      state = {
        'epoch': epoch,
        'global_step': step,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        # 'scheduler': self.scheduler.state_dict()
      }
      logging.debug("Saving checkpoint '{}'.".format(ckpt_name))
      torch.save(state, ckpt_path)


  # @record
  def __call__(self):
    """Performs training and evaluation
    """
    cudnn.benchmark = True

    self.train_dir = self.config.train_dir
    self.ngpus = self.config.ngpus

    # job_env = submitit.JobEnvironment()
    self.rank = 0 #int(os.environ['RANK'])
    self.local_rank = 0 #int(os.environ['LOCAL_RANK'])
    self.num_nodes = 1 #int(os.environ['LOCAL_WORLD_SIZE']) 
    self.num_tasks = 1 #int(os.environ['WORLD_SIZE'])
    self.is_master = bool(self.rank == 0)

    if self.config.resume:
      checkpoint = torch.load(self.config.resume)
      self.config.start_epoch = checkpoint['epoch']

    # Setup logging
    utils.setup_logging(self.config, self.rank)

    logging.info(self.rank)
    logging.info(self.local_rank)
    logging.info(self.num_nodes)
    logging.info(self.num_tasks)

    self.message = utils.MessageBuilder()
    # print self.config parameters
    if self.local_rank == 0:
      logging.info(self.config.cmd)
      pp = pprint.PrettyPrinter(indent=2, compact=True)
      logging.info(pp.pformat(vars(self.config)))
    # print infos
    if self.local_rank == 0:
      logging.info(f"PyTorch version: {torch.__version__}.")
      logging.info(f"NCCL Version {torch.cuda.nccl.version()}")
      logging.info(f"Hostname: {socket.gethostname()}.")

    # ditributed settings
    self.world_size = 1
    self.is_distributed = False
    if self.num_nodes > 1 or self.num_tasks > 1:
      self.is_distributed = True
      self.world_size = self.num_nodes * self.ngpus
    if self.num_nodes > 1:
      logging.info(
        f"Distributed Training on {self.num_nodes} nodes")
    elif self.num_nodes == 1 and self.num_tasks > 1:
      logging.info(f"Single node Distributed Training with {self.num_tasks} tasks")
    else:
      assert self.num_nodes == 1 and self.num_tasks == 1
      logging.info("Single node training.")

    if not self.is_distributed:
      self.batch_size = self.config.batch_size * self.ngpus
    else:
      self.batch_size = self.config.batch_size

    self.global_batch_size = self.batch_size * self.world_size
    logging.info('World Size={} => Total batch size {}'.format(
      self.world_size, self.global_batch_size))

    torch.cuda.set_device(self.local_rank)

    # load dataset
    Reader = readers_config[self.config.dataset]
    self.reader = Reader(self.config, self.batch_size, self.is_distributed, is_training=True)
    if self.local_rank == 0:
      logging.info(f"Using dataset: {self.config.dataset}")

    # load reader
    self.reader_val = Reader(self.config, self.batch_size, False, is_training=False)
    self.config.means = self.reader.means

    # load model
    self.model = LipschitzNetwork(self.config, self.reader.n_classes)
    self.model = NormalizedModel(self.model, self.reader.means, self.reader.stds)
    self.model = self.model.cuda()
    nb_parameters = np.sum([p.numel() for p in self.model.parameters() if p.requires_grad])
    logging.info(f'Number of parameters to train: {nb_parameters}')

    # setup distributed process if training is distributed 
    # and use DistributedDataParallel for distributed training
    if self.is_distributed:
      utils.setup_distributed_training(self.world_size, self.rank)
      self.model = DistributedDataParallel(
        self.model, device_ids=[self.local_rank], output_device=self.local_rank)
      if self.local_rank == 0:
        logging.info('Model defined with DistributedDataParallel')
    else:
      self.model = nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

    # resume model, if applicable
    if self.config.resume:
      self.model.load_state_dict(checkpoint['model_state_dict'])
      logging.info(f'Succesfully loaded state_dict from {self.config.resume}.')
    else:
      logging.info(f'No resume flag set. Training from scratch.')

    # define set for saved ckpt
    self.saved_ckpts = set([0])

    data_loader, sampler = self.reader.load_dataset()
    if sampler is not None:
      assert sampler.num_replicas == self.world_size

    if self.is_distributed:
      n_files = sampler.num_samples
    else:
      n_files = self.reader.n_train_files

    # define optimizer
    self.optimizer = utils.get_optimizer(self.config, self.model.parameters())

    if self.config.resume:
      self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      # self.config.warmup_scheduler = checkpoint['epoch'] / self.config.epochs

    # define learning rate scheduler
    num_steps = self.config.epochs * (self.reader.n_train_files // self.global_batch_size)
    self.scheduler, self.warmup = utils.get_scheduler(self.optimizer, self.config, num_steps) 
    if self.config.warmup_scheduler is not None:
      logging.info(f"Warmup scheduler on {self.config.warmup_scheduler*100:.0f}% of training")

    # define the loss
    if self.config.loss == 'cll':
      L = 1.0
      model = self.model.module.model
      self.criterion = CLL(self.config.cll_eps, self.config.cll_p, self.config.cll_delta, False, model, L, detach_K=True, lambda_=self.config.cll_lambda)
    else:
      self.criterion = utils.get_loss(self.config)

    if self.local_rank == 0:
      logging.info("Number of files on worker: {}".format(n_files))
      logging.info("Start training")

    # training loop
    start_epoch, global_step = 0, 0
    self.best_checkpoint = None
    self.best_accuracy = None
    self.best_accuracy = [0., 0.]

    if self.config.resume:
      start_epoch = checkpoint['epoch']
      global_step = checkpoint['global_step']

    for epoch_id in range(start_epoch, self.config.epochs):
      if self.is_distributed:
        sampler.set_epoch(epoch_id)
      for n_batch, data in enumerate(data_loader):
        if global_step == 2 and self.is_master:
          start_time = time.time()
        epoch = (int(global_step) * self.global_batch_size) / self.reader.n_train_files
        self.one_step_training(data, epoch, global_step)
        self._save_ckpt(global_step, epoch_id)
        if global_step == 20 and self.is_master:
          self._print_approximated_train_time(start_time)

        global_step += 1

      if self.is_master:
        logging.info('Evaluate:')
        if self.config.loss == 'cll':
          self.eval_certified_lln([36, 72, 108, 255])
        else:
          if self.config.last_layer == 'lln':
            self.eval_certified_lln([36, 72, 108, 255])
          else:
            self.eval_certified([36, 72, 108, 255])

    self._save_ckpt(global_step, epoch_id, final=True)
    logging.info("Done training -- epoch limit reached.")

  def filter_parameters(self):
    conv_params, linear_params = [], []
    for name, params in self.model.named_parameters():
      if 'weight' in name.lower() and params.dim() == 4:
        conv_params.append(params)
      elif 'weight' in name.lower() and params.dim() == 2:
        linear_params.append(params)
      elif 'bias' in name.lower():
        conv_params.append(params)
    return conv_params, linear_params

  def compute_gradient_norm(self):
    grad_norm = 0.
    for name, p in self.model.named_parameters():
      if p.grad is None: continue
      norm = p.grad.detach().data.norm(2)
      grad_norm += norm.item()**2
    grad_norm = grad_norm**0.5
    return grad_norm

  def _print_approximated_train_time(self, start_time):
    total_steps = self.reader.n_train_files * self.config.epochs / self.global_batch_size
    total_seconds = total_steps * ((time.time() - start_time) / 18)
    n_days = total_seconds // 86400
    n_hours = (total_seconds % 86400) / 3600
    logging.info(
      'Approximated training time: {:.0f} days and {:.1f} hours'.format(
        n_days, n_hours))

  def _to_print(self, step):
    frequency = self.config.frequency_log_steps
    if frequency is None:
      return False
    return (step % frequency == 0 and self.local_rank == 0) or \
        (step == 1 and self.local_rank == 0)

  def process_gradients(self, step):
    if self.config.gradient_clip_by_norm:
      if step == 0 and self.local_rank == 0:
        logging.info("Clipping Gradient by norm: {}".format(
          self.config.gradient_clip_by_norm))
      torch.nn.utils.clip_grad_norm_(
        self.model.parameters(), self.config.gradient_clip_by_norm)
    elif self.config.gradient_clip_by_value:
      if step == 0 and self.local_rank == 0:
        logging.info("Clipping Gradient by value: {}".format(
          self.config.gradient_clip_by_value))
      torch.nn.utils.clip_grad_value_(
        self.model.parameters(), self.config.gradient_clip_by_value)

  def one_step_training(self, data, epoch, step):
    self.model.train()
    self.optimizer.zero_grad()

    batch_start_time = time.time()
    images, labels = data
    images, labels = images.cuda(), labels.cuda()

    if step == 0 and self.local_rank == 0:
      logging.info(f'images {images.shape}')
      logging.info(f'labels {labels.shape}')

    outputs = self.model(images)
    if step == 0 and self.local_rank == 0:
      logging.info(f'outputs {outputs.shape}')

    loss = self.criterion(outputs, labels)
    loss.backward()
    self.process_gradients(step)
    self.optimizer.step()
    with self.warmup.dampening() if self.warmup else nullcontext():
      self.scheduler.step(step)

    seconds_per_batch = time.time() - batch_start_time
    examples_per_second = self.batch_size / seconds_per_batch
    examples_per_second *= self.world_size

    if self._to_print(step):
      lr = self.optimizer.param_groups[0]['lr']
      self.message.add("epoch", epoch, format="4.2f")
      self.message.add("step", step, width=5, format=".0f")
      self.message.add("lr", lr, format=".6f")
      self.message.add("loss", loss, format=".4f")
      if self.config.print_grad_norm:
        grad_norm = self.compute_gradient_norm()
        self.message.add("grad", grad_norm, format=".4f")
      self.message.add("imgs/sec", examples_per_second, width=5, format=".0f")

      if self.config.loss == 'cll':
        accuracy, certified = utils.eval_certified_lln(self.model, self.reader)
      else:
        if not self.config.last_layer == 'lln':
          accuracy, certified = utils.eval_certified(self.model, self.reader)
        else:
          accuracy, certified = utils.eval_certified_lln(self.model, self.reader)

      self.message.add("acc", accuracy*100, format=".1f")
      self.message.add("Cra0.141", certified[0]*100, format=".1f")
      self.message.add("Cra0.282", certified[1]*100, format=".1f")
      self.message.add("Cra0.424", certified[2]*100, format=".1f")
      # with torch.no_grad():
      #   K = 1.0
      #   for liplayer in self.model.module.model.soft_constrained_layers:
      #     K *= liplayer.estimate(1, update=False)
      #   last_weight = self.model.module.model.last_last.weight
      #   if isinstance(self.model.module.model.last_last, LinearNormalized):
      #     normalized_weight = F.normalize(last_weight, p=2, dim=1)
      #   else:
      #     normalized_weight = last_weight

      # self.message.add("K", )

      logging.info(self.message.get_message())

  @torch.no_grad()
  def eval_certified(self, epsilons):
    epsilons_float = [eps / 255 for eps in epsilons]
    self.model.eval()
    running_accuracy = 0
    running_certified = [0]*len(epsilons)
    running_inputs = 0
    lip_cst = 1.
    if self.model.module.model.unconstrained_layer_Ks:
      lip_cst = torch.prod(self.model.module.model.Ks).item()
    elif len(self.model.module.model.soft_constrained_layers) > 0:
      for liplayer in self.model.module.model.soft_constrained_layers:
        lip_cst *= liplayer.estimate(1000, update=True)
    data_loader, _ = self.reader_val.load_dataset()
    for batch_n, data in enumerate(data_loader):
      inputs, labels = data
      inputs, labels = inputs.cuda(), labels.cuda()
      outputs = self.model(inputs)
      predicted = outputs.argmax(axis=1)
      correct = outputs.max(1)[1] == labels
      margins = torch.sort(outputs, 1)[0]
      for eps_i, eps_float in enumerate(epsilons_float):
        certified = (margins[:, -1] - margins[:, -2]) > np.sqrt(2.) * lip_cst * eps_float
        running_certified[eps_i] += torch.sum(correct & certified).item()
      running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
      running_inputs += inputs.size(0)
    accuracy = running_accuracy / running_inputs
    certified = torch.Tensor(running_certified) / running_inputs
    # self.message.add('eps', [eps, 255], format='.0f')
    # self.message.add('eps', eps_float, format='.5f')
    self.message.add('Evaluation :: accuracy', accuracy, format='.5f')
    for eps_i, eps_float in enumerate(epsilons_float):
      self.message.add('Cra{:.3f}'.format(eps_float), certified[eps_i], format='.5f')
    logging.info(self.message.get_message())
    return accuracy, certified

  @torch.no_grad()
  def eval_certified_lln(self, epsilons):
    epsilons_float = [eps / 255 for eps in epsilons]
    self.model.eval()
    running_accuracy = 0
    running_certified = [0]*len(epsilons)
    running_inputs = 0
    lip_cst = 1.
    if self.model.module.model.unconstrained_layer_Ks:
      lip_cst = torch.prod(self.model.module.model.Ks).item()
    elif len(self.model.module.model.soft_constrained_layers) > 0:
      for liplayer in self.model.module.model.soft_constrained_layers:
        lip_cst *= liplayer.estimate(1000, update=True)
    data_loader, _ = self.reader_val.load_dataset()
    last_weight = self.model.module.model.last_last.weight
    if isinstance(self.model.module.model.last_last, LinearNormalized):
      normalized_weight = F.normalize(last_weight, p=2, dim=1)
    else:
      normalized_weight = last_weight
    for batch_n, data in enumerate(data_loader):
      inputs, labels = data
      inputs, labels = inputs.cuda(), labels.cuda()
      outputs = self.model(inputs)
      predicted = outputs.argmax(axis=1)
      correct = outputs.max(1)[1] == labels
      margins, indices = torch.sort(outputs, 1)
      margins = margins[:, -1][:, None] - margins[: , 0:-1]
      for idx in range(margins.shape[0]):
        margins[idx] /= torch.norm(
          normalized_weight[indices[idx, -1]] - normalized_weight[indices[idx, 0:-1]], dim=1, p=2)
      margins, _ = torch.sort(margins, 1)

      for eps_i, eps_float in enumerate(epsilons_float):
        certified = margins[:, 0] > eps_float * lip_cst
        running_certified[eps_i] += torch.sum(correct & certified).item()
      
      running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
      running_inputs += inputs.size(0)
    accuracy = running_accuracy / running_inputs
    certified = torch.Tensor(running_certified) / running_inputs
    self.message.add('Evaluation :: accuracy', accuracy, format='.5f')
    for eps_i, eps_float in enumerate(epsilons_float):
      self.message.add('Cra{:.3f}'.format(eps_float), certified[eps_i], format='.5f')
    logging.info(self.message.get_message())
    return accuracy, certified
