import inspect
from collections import OrderedDict

import torch.nn as nn
import torch

from lib import util

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.evaluations = {}
        self.metrics = {}
        self.loss = {}
        self.metric_argspecs = {}
        self.loss_argspecs = {}

    def pre_training_hook(self, **kwargs):
        return {}

    def post_training_hook(self, **kwargs):
        return {}

    def post_training_iteration_hook(self, **kwargs):
        return {}

    def evaluate(self, **kwargs):
        self.eval()
        results = OrderedDict()
        kwargs['model'] = self._forward_pass
        kwargs['lipschitz_computer'] = self
        for name, module in self.evaluations.items():
            # kwargs['logger'].info('{}, {}, {}, {}'.format(name, kwargs['epoch'], module.eval_freq, str(kwargs['epoch'] % module.eval_freq)))
            if (type(module.eval_freq) is list and kwargs['epoch'] in module.eval_freq) or \
                (type(module.eval_freq) is not list and (kwargs['epoch'] % module.eval_freq) == 0):
                if 'logger' in kwargs:
                    kwargs['logger'].info('{}:'.format(name))
                result = module(name=name, **kwargs)

                if result is not None:
                    result = {name: result}
                    results = util.merge_dicts(results, result)
        return results

    def _assemble_function_inputs(self, argspec_collection, f_name, f, outputs, inputs, target):
        """
        Given a function f, matches its signature with arguments listed in (outputs, input, target)
        Returns a dict object that can be passed to f directly f(**assembly)
        """
        #if isinstance(f, nn.NLLLoss) or isinstance(f, nn.CrossEntropyLoss) or isinstance(f, nn.BCELoss):
        if isinstance(f, nn.modules.loss._Loss):
            assembly = dict(input=outputs['prediction'], target=target)
            return assembly

        if f_name not in argspec_collection:
            if hasattr(f, 'inputs'):
                required = f.inputs
            else:
                required = []
                for arg_name in inspect.signature(f.forward).parameters.keys():
                    required.append(arg_name)

            argspec_collection[f_name] = required
        required = argspec_collection[f_name]

        assembly = {}

        for var in required:
            if var == 'target':
                assembly[var] = target
            elif var == 'input':
                assembly[var] = inputs
            elif var == 'model':
                assembly[var] = self
            else:
                value = outputs
                for subvar in var.split('.'):
                    value = value[subvar]
                assembly[var] = value

        return assembly

    def _forward_pass(self, x, y=None, **kwargs):
        """
        Generates all outputs when forwarding input through the network
        """
        outputs = {}
        outputs['layer_output'] = OrderedDict()
        for name, child in self.named_children():
            if name != 'loss' and name != 'metrics' and name != 'evaluations':
                x = child(x)
                outputs['layer_output'][name] = x

        outputs['prediction'] = x
        return outputs

    def forward(self, x, y=None, **kwargs):
        return_outputs = kwargs.get('return_outputs', False)
        outputs = self._forward_pass(x, y, **kwargs)

        # -----------------------------
        # Losses
        losses = {}
        for name, loss_f in self.loss.items():
            f_kwargs = self._assemble_function_inputs(self.loss_argspecs, name, loss_f, outputs, x, y)

            losses[name] = loss_f(**f_kwargs)
        if len(losses) > 0:
            outputs['loss'] = losses

        # -----------------------------
        # Metrics
        metrics = {}
        for name, metric in self.metrics.items():
            f_kwargs = self._assemble_function_inputs(self.metric_argspecs, name, metric, outputs, x, y)
            results = metric(**f_kwargs)
            if results is not None:
                metrics[name] = results
        if len(metrics) > 0:
            outputs['metric'] = metrics
        
        if not return_outputs:
            del outputs['layer_output']

        return outputs