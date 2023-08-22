import sys
import torch
import torch.nn as nn
import model
import model.layers as layers
import model.losses as losses
import model.metrics as metrics
import model.evaluations as evaluations
import lib.scheduler as scheduler
import lib.util as util


def build(config):
    arch_type = config['architecture']['type']

    if arch_type.startswith('torchvision'):
        # model_class = ...
        raise NotImplementedError
    else:
        model_class = model.__dict__[arch_type]
        if model_class is None:
            raise RuntimeError('Architecture type {} not listed in model definitions'.format(arch_type))

    compile_schedulers(config)

    instance = model_class(**config['architecture'].get('kwargs', {}))
    print(instance)

    if 'layers' in config['architecture']:
        build_layers(instance, config)
    match_config_layers(instance, config)
    if 'loss' in config:
        build_losses(instance, config)
    if 'metrics' in config:
        build_metrics(instance, config)
    if 'evaluations' in config and config['evaluations'] is not None:
        build_evaluations(instance, config)

    return instance


def compile_schedulers(config, prefix=''):
    for name in config.keys():
        value = config[name]
        if name == 'type' and type(value) is str and value.startswith('scheduler.'):
            sched_type = value.replace('scheduler.', '')
            sched_class = scheduler.__dict__[sched_type]
            kwargs = dict(config)
            kwargs.pop('type')
            kwargs['name'] = prefix[:-1] # strip last 'dot' .
            instance = sched_class(**kwargs)
            return instance
        elif isinstance(value, dict):
            ret = compile_schedulers(value, prefix=name+'.')
            if ret is not None:
                # replace value with scheduler
                config[name] = ret
    return None


def match_config_layers(model, config):
    remove_names = []
    append_items = []
    for name in config.keys():
        value = config[name]

        if isinstance(value, dict):
            match_config_layers(model, value)
        elif name.startswith('MATCH_LAYER_'):
            new_name = name.replace('MATCH_LAYER_', '')
            append_items.append((new_name, model._modules[value]))
            remove_names.append(name)

    for name in remove_names:
        del config[name]

    for name, value in append_items:
        config[name] = value


def build_layers(model, config):
    for name, layer_def in config['architecture']['layers'].items():

        try:
            layer_type = layer_def.pop('type')
            if layer_type.startswith('nn.'):
                layer_module = nn.__dict__[layer_type.replace('nn.','')]
                if 'args' in layer_def:
                    layer = layer_module(*layer_def.pop('args'), **layer_def)
                else:
                    layer = layer_module(**layer_def)
            else:
                layer_module = layers.__dict__[layer_type]

                if layer_type == 'AddGloroAbsentLogit':
                    layer_def['output_module'] = getattr(model, layer_def.pop('output_layer'))
                    layer_def['lipschitz_computer'] = model.lipschitz_estimate

                if 'args' in layer_def:
                    layer = layer_module(*layer_def.pop('args'), **layer_def)
                else:
                    layer = layer_module(**layer_def)

            setattr(model, name, layer)
        except Exception as e:
            print('Exception at creating layer {}'.format(name))
            raise e


    if 'initialization' in config['architecture']:
        print('Initializing weights {}'.format(config['architecture']['initialization']))
        util.init_weights(model, 
            conv=config['architecture']['initialization'].get('conv', 'kaiming'),
            batchnorm=config['architecture']['initialization'].get('batchnorm', 'normal'),
            linear=config['architecture']['initialization'].get('linear', 'kaiming'),
            )


def build_loss(model, loss_cfg):
    assert 'type' in loss_cfg

    loss_type = loss_cfg.pop('type')
    if loss_type.startswith('nn.'):
        loss_class = nn.__dict__[loss_type.replace('nn.','')]
    else:
        loss_class = losses.__dict__[loss_type]

    if loss_type == 'GloroLoss':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'LipschitzIntegratedCosineLoss':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'LipschitzLagrangeCosineLoss':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'LipschitzLagrangeLoss':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'LipschitzCrossEntropyLoss':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'MarginLipschitzCrossEntropyLoss':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'LipschitzPairwiseCELoss':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'KeepLipschitzMarginCELoss':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'KeepLipschitzMarginCELossV2':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'KeepLipschitzMarginCELossV3':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'SupportClipCE':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'TradesClipCE':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'KeepPairwiseLipschitzMarginCELoss':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'KeepPairwiseLipschitzMarginBCELoss':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'KeepLipschitzMarginHingeLoss':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'KeepLipschitzMarginBCELoss':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'KeepLipschitzMarginHardBCELoss':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'KeepLipschitzMarginCosineBCELoss':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'GlobalLipschitzDecay':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'GlobalLipschitzTarget':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate
    elif loss_type == 'MarginRatioLoss':
        loss_cfg['lipschitz_computer'] = model.lipschitz_estimate

    try:
        loss = loss_class(**loss_cfg)
    except Exception as e:
        print()
        print('##################################################')
        print('#')
        print('#  Error at '+loss_type)
        print('#')
        print('##################################################')
        raise e

    return loss

def build_losses(model, config):
    container = nn.ModuleDict()

    for name, loss_cfg in config['loss'].items():
        loss = build_loss(model, loss_cfg)

        container[name] = loss

    model.loss = container


def build_metrics(model, config):
    container = nn.ModuleDict()

    for name, metric_cfg in config['metrics'].items():
        assert 'type' in metric_cfg

        metric_type = metric_cfg.pop('type')
        metric_class = metrics.__dict__[metric_type]

        if metric_type == 'LipCosRobustAccuracy':
            metric_cfg['lipcosloss'] = model.loss[metric_cfg['lipcosloss']]
        elif metric_type == 'RobustAccuracy' or metric_type == 'RobustAccuracyV2':
            metric_cfg['model'] = model

        metric = metric_class(**metric_cfg)
        container[name] = metric

    model.metrics = container


def build_evaluations(model, config):
    container = nn.ModuleDict()

    for name, eval_cfg in config['evaluations'].items():
        assert 'type' in eval_cfg
        if 'eval_freq'  not in eval_cfg:
            raise RuntimeError("Evaluation configuration {} requires definition of 'eval_freq' in epochs.".format(name))

        # ------------------------------------------------------------------
        # quick fix to get the right reference into loss construction
        if 'loss_type' in eval_cfg and isinstance(eval_cfg['loss_type'], dict):
            if eval_cfg['loss_type']['type'] == 'LipschitzCrossEntropyLoss':
                eval_cfg['loss_type']['lipschitz_computer'] = model.lipschitz_estimate
            if eval_cfg['loss_type']['type'] == 'MarginLipschitzCrossEntropyLoss':
                eval_cfg['loss_type']['lipschitz_computer'] = model.lipschitz_estimate
        # ------------------------------------------------------------------

        eval_class = evaluations.__dict__[eval_cfg.pop('type')]

        evaluation = eval_class(**eval_cfg)
        container[name] = evaluation

    model.evaluations = container