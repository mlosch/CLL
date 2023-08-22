import os
import torch
import torch.optim as optim
import torch.nn as nn
from model import ConvexPotentialLayerNetwork, NormalizedModel
import util
import numpy as np
import torch.backends.cudnn as cudnn
import time
import data as data
from layers import PoolingLinear, LinearNormalized
import logging

cudnn.benchmark = True

logger = logging.getLogger(__name__)


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
        if isinstance(model.last_last, PoolingLinear):
            self.fc_weight = None
        elif isinstance(model.last_last, LinearNormalized):
            self.fc_weight = lambda : model.last_last.weight / torch.norm(model.last_last.weight, dim=1, keepdim=True)
        elif isinstance(model.last_last, nn.Linear):
            self.fc_weight = lambda : model.last_last.weight
        else:
            raise NotImplementedError('Unknown layer type {}'.format(model.last_last))
        self.L = L
        logger.info(f'Lambda: {lambda_}, L: {L}')

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

        wx_it = (wx_it + delta) / sigma

        if self.lambda_ != 0:
            return self.loss(wx_it, target) + self.lambda_*(K_tj.max()**2)
        else:
            return self.loss(wx_it, target)

class Trainer:

    def __init__(self, config):
        self.cuda = True
        self.seed = config.seed
        self.lr = config.lr
        self.lr_scheduler_type = config.lr_scheduler
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.wd = config.weight_decay
        self.depth = config.depth
        self.depth_linear = config.depth_linear
        self.save_dir = config.save_dir
        self.conv_size = config.conv_size
        self.num_channels = config.num_channels
        self.n_features = config.n_features
        self.loss_type = config.loss_type
        if self.loss_type == 'default':
            self.loss_params = config.margin
        else:
            self.loss_params = dict(
                epsilon=config.clip_epsilon, 
                err_quantile_schedule=(config.clip_p, config.clip_p, 1.0), 
                add_epsilon=config.clip_delta, 
                onesided=config.clip_onesided,
                detach_K=config.clip_detach,
                lambda_=config.clip_lambda)
        
        self.lln = config.lln
        self.dataset = config.dataset
        self.norm_input = config.norm_input

        # logging.basicConfig(
        #     format='%(message)s',
        #     level=logging.INFO,
        #     filename=os.path.join(config.save_dir, 'output.log'))
        # logger.info(config)

    def set_everything(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # logger.setLevel(logging.INFO)
        # handler = logging.StreamHandler()
        # fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
        # handler.setFormatter(logging.Formatter(fmt))
        # logger.addHandler(handler)
        logging.basicConfig(
            format='%(message)s',
            level=logging.INFO,
            filename=os.path.join(self.save_dir, 'output.log'))

        torch.manual_seed(self.seed)

        # Init dataset
        if self.dataset == "c10":
            self.mean = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
            self.std = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)
            num_classes = 10
        else:
            self.mean = (0.5071, 0.4865, 0.4409)
            self.std = (0.2673, 0.2564, 0.2762)
            num_classes = 100
        self.num_classes = num_classes

        if not self.norm_input:
            self.std = (1., 1., 1.)

        self.data = data.DataClass(self.dataset, batch_size=self.batch_size)
        self.train_batches, self.test_batches = self.data()

        ## Init model
        self.model = ConvexPotentialLayerNetwork(depth=self.depth,
                                                 depth_linear=self.depth_linear,
                                                 num_classes=num_classes, conv_size=self.conv_size,
                                                 num_channels=self.num_channels,
                                                 n_features=self.n_features,
                                                 use_lln=self.lln,
                                                 use_vanilla_linear=self.loss_type!='default')

        self.model = NormalizedModel(self.model, self.mean, self.std)
        self.model = nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
        self.model = self.model.cuda()
        logger.info(self.model)

        ## Init optimizer
        lr_steps = self.epochs * len(self.train_batches)
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=self.wd, lr=self.lr)

        if self.lr_scheduler_type == 'triangular':
            self.lr_scheduler = util.TriangularLRScheduler(self.optimizer, lr_steps, self.lr)
        elif self.lr_scheduler_type == 'multisteplr':
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[lr_steps // 2, (3 * lr_steps) // 4], gamma=0.1)
        else:
            raise NotImplementedError(f'Unknown lr_scheduler value {self.lr_scheduler_type}')

        if self.loss_type == 'default':
            self.criterion = lambda yhat, y: util.margin_loss(yhat, y, self.loss_params)
            logger.info(f"default margin loss with param {self.loss_params} with lr = {self.lr}")
        elif self.loss_type == 'CLL':
            logger.info(f"CLL margin loss with params {self.loss_params}")
            loss_params = self.loss_params
            loss_params['model'] = self.model.module.model
            loss_params['L'] = 1./np.min(self.std)
            self.criterion = CLL(**loss_params)
        else:
            raise NotImplementedError(f'Unknown loss type {self.loss_type}')

        logger.info(f"number of gpus: {torch.cuda.device_count()}")

    def __call__(self):
        self.set_everything()
        acc_best = 0
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            self.train(epoch)
            logger.info(f'Epoch {epoch}: training 1 epoch -> compute eval')

            self.time_epoch = time.time() - start_time
            logger.info(f"Time epoch: {self.time_epoch}")

            acc, _ = self.test(epoch)
            if (acc > acc_best):
                acc_best = acc
                torch.save(self.model.state_dict(), f"{self.save_dir}/best_model.pt")

        torch.save(self.model.state_dict(), f"{self.save_dir}/last_model.pt")

    def train(self, epoch):
        self.model.train()
        for batch_idx, batch in enumerate(self.train_batches):
            self.lr_scheduler.step(epoch * len(self.train_batches) + (batch_idx + 1))
            images, target = batch['input'], batch['target']
            predictions = self.model(images)
            loss = self.criterion(predictions.cpu(), target.cpu())
            loss = loss.cuda()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % 100 == 0:

                last_last = self.model.module.model.last_last
                if self.lln:
                    Wfc = torch.nn.functional.normalize(last_last.weight, p=2, dim=1)
                elif isinstance(last_last, PoolingLinear):
                    if last_last.agg != 'trunc':
                        raise NotImplementedError('Not implemented aggregation type in PoolingLinear: {}'.format(last_last.agg))
                    # in case of truncation, we need the weight matrix of the layer before

                    last_last = self.model.module.model.layers_linear._modules[str(self.depth_linear-1)]
                    Wfc = last_last.weights[:self.num_classes, :]
                else:
                    Wfc = last_last.weight
                
                with torch.no_grad():
                    Kfc = 0
                    for i in range(Wfc.shape[0]):
                        for j in range(i+1, Wfc.shape[0]):
                            Wdiff = (Wfc[i] - Wfc[j]).float()
                            Kfc = max(Kfc, torch.norm(Wdiff, p=2).item())
                    L = 1. / np.min(self.std)
                    Kfc *= L

                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Kfc {:.3f}'.format(
                    epoch, batch_idx * len(images), 50000,
                           100. * batch_idx / len(self.train_batches), loss.item(), Kfc.item()))
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Kfc {:.3f}'.format(
                    epoch, batch_idx * len(images), 50000,
                           100. * batch_idx / len(self.train_batches), loss.item(), Kfc.item()))

    def test(self, epoch):
        if self.loss_type=='default' and not self.lln:
            acc, cert_acc, _, _, _ = util.certified_accuracy(self.test_batches, self.model,
                                                             lip_cst=1. / np.min(self.std), eps=36. / 255)
        else:
            acc, cert_acc, _, _, _ = util.certified_accuracy_linear(self.test_batches, self.model,
                                                                lip_cst=1. / np.min(self.std), eps=36. / 255, use_lln=self.lln)
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
        logger.info(f"Epoch {epoch}: Accuracy : {acc}, certified accuracy: {cert_acc}, lr = {lr}\n")
        print(f"Epoch {epoch}: Accuracy : {acc}, certified accuracy: {cert_acc}, lr = {lr}\n")
        return acc, cert_acc

    def eval_final(self, eps=36. / 255):

        self.model.eval()
        # acc_cert
        lip_cst = 1. / np.min(self.std)
        if self.loss_type=='default' and not self.lln:
            acc, cert_acc, _, _, _ = util.certified_accuracy(self.test_batches, self.model, lip_cst=lip_cst, eps=eps)
        else:
            acc, cert_acc, _, _, _ = util.certified_accuracy_linear(self.test_batches, self.model, lip_cst=lip_cst,
                                                                eps=eps, use_lln=self.lln)
        # autoattack
        with torch.no_grad():
            acc_auto = util.test_auto_attack(self.model, self.test_batches, eps=eps)

        # acc pgd
        eps_iter = 2. * eps / 10.
        if eps == 0:
            acc_pgd = acc
        else:
            acc_pgd = util.test_pgd_l2(self.model, self.test_batches, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                       eps=eps, nb_iter=10, eps_iter=eps_iter, rand_init=True, clip_min=0.0,
                                       clip_max=1.0, targeted=False)
        logger.info(f"Final for epsilon={eps}: Accuracy : {acc}, certified accuracy: {cert_acc}, "
              f"autoattack accuracy: {acc_auto}, pgd attack accuracy: {acc_pgd}")
        print(f"Final for epsilon={eps}: Accuracy : {acc}, certified accuracy: {cert_acc}, "
              f"autoattack accuracy: {acc_auto}, pgd attack accuracy: {acc_pgd}")
