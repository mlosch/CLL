# Calibrated Lipschitz-Margin Loss (CLL)
Official repository for "Certified Robust Models with Slack Control and Large Lipschitz Constants", to be published at GCPR 2023 (in press)

This repository integrates our proposed CLL-loss into 4 different code bases. These base repositories don't need to be cloned, but can be found here:
- [GloRo (Globally-Robust Neural Networks)](https://github.com/klasleino/gloro/tree/master)
- [SOC (Skew Orthogonal Convolutions)](https://github.com/singlasahil14/SOC/tree/main)
- [CPL (Convex Potential Layers)](https://github.com/MILES-PSL/Convex-Potential-Layer/tree/main)
- [SLL (SDP-based Lipschitz Layers)](https://github.com/araujoalexandre/Lipschitz-SLL-Networks/tree/main)


## Requirements
- python 3.8
- pytorch 1.6.0
- tensorboard
- apex


## On the basis of [GloRo](https://github.com/klasleino/gloro/tree/master) (4C3F, 6C2F, 8C2F).

- To run main experiments on 4C3F for GloRo and CLL
  
  `cd unconstrained; bash config/cifar10_4C3F_gloro_eps_sweep/run_all.sh`
 
  `cd unconstrained; bash config/cifar10_4C3F_cll_eps_sweep/run_all.sh`
 
  `cd unconstrained; bash config/cifar10_4C3F_cll_random_seeds/run_all.sh`

- To run main experiments on 6C2F for CLL
  
  `cd unconstrained; bash config/cifar10_6C2f_cll_random_seeds/run_all.sh`

- To run main experiments on 8C2F for CLL
  
  `cd unconstrained; bash config/tiny-imagenet_8C2F_cll_random_seeds/run_all.sh`
  
  `cd unconstrained; bash config/tiny-imagenet_8C2F_cll_slack_sweep/run_all.sh`

## To run two-moons experiments for GloRo and CLL:
- Visualizations are automatically placed in `exp/moons_gloro_eps_sweep` and `exp/moons_cll_eps_sweep`

  `cd unconstrained; bash config/moons_gloro_eps_sweep/run_all.sh`

  `cd unconstrained; bash config/moons_cll_eps_sweep/run_all.sh`

### All training and validation statistics are logged in terminal as well as Tensorboard. To open Tensorboard for CLL on CIFAR-10:
`tensorboard --logdir unconstrained/exp/cifar10_4C3F_cll_eps_sweep`

### Method references:
- CLL is defined in `unconstrained/model/losses.py` in class `CLL`
- GloRo is defined in `unconstrained/model/losses.py` in class `GloroLoss`

- Power iterations are defined in `unconstrained/model/lipschitz_model.py`:
  - `LipschitzLayerComputer` (abstract class)
  - `Conv2dLipschitzComputer`
  - `LinearLipschitzComputer`

- Power iteration convergence after each training epoch is defined in `unconstrained/model/lipschitz_model.py`:
  - `class LipschitzModel`, function `post_training_hook()`

## On the basis of [SOC](https://github.com/singlasahil14/SOC/tree/main) (LipConv).

- To train CLL on CIFAR-10

  `cd soc; bash train_CLL_on_cifar10.sh`

- To train CLL on Tiny-ImageNet

  `cd soc; bash train_CLL_on_tinyimagenet.sh`

- To train SOC on Tiny-ImageNet

  `cd soc; bash train_soc_on_tinyimagenet.sh`


## On the basis of [CPL](https://github.com/MILES-PSL/Convex-Potential-Layer/tree/main) (XL).

- To train/evaluate (CPL-)XL with CLL on CIFAR-10/CIFAR-100:

  `cd cpl; bash train_CLL_on_cifar10.sh`

  `cd cpl; bash train_CLL_on_cifar100.sh`


## On the basis of [SLL](https://github.com/araujoalexandre/Lipschitz-SLL-Networks/tree/main) (XL).

- To train/evaluate (SLL-)XL with CLL on CIFAR-100:

  `cd sll; python train_cifar100_cll.py; bash train_cifar100_cll.sh`

- To evaluate the lower bound of trained models:

  `cd sll; bash eval_lobo_cifar100_cll.sh`
