from torch import nn
import numpy as np


class Scheduler(object):
    def __init__(self, start_val, end_val=None, name='Unnamed_Scheduler'):
        self.val = start_val
        self.start_val = start_val
        if end_val is None:
            self.end_val = start_val
        else:
            self.end_val = end_val
        self.name = name

    def update(self, *args, **kwargs):
        raise NotImplementedError


class ConstantScheduler(Scheduler):
    def update(self, *args, **kwargs):
        return self.val

    def __repr__(self):
        #return 'Constant({})'.format(self.val)
        return str(self.val)


class MultiStepScheduler(Scheduler):
    def __init__(self, start_val, gamma, milestones, name=''):
        super(MultiStepScheduler, self).__init__(start_val, None, name)
        self.gamma = gamma
        self.milestones = milestones

    def update(self, curr_iter, max_iter):
        val = self.start_val
        for milestone in self.milestones:
            if curr_iter >= milestone:
                val *= self.gamma
        self.val = val
        return self.val

    def __repr__(self):
        return 'MultiStep(start={}, gamma={}, milestones={})'.format(self.start_val, self.gamma, self.milestones)


class PolynomialScheduler(Scheduler):
    def __init__(self, start_val, end_val, power, max_iter=None, offset_iter=None, name=''):
        super(PolynomialScheduler, self).__init__(start_val, end_val, name)
        self.power = power
        self.max_iter = max_iter
        self.offset_iter = offset_iter

    def update(self, curr_iter, max_iter):
        if self.offset_iter is not None:
            curr_iter = max(0, curr_iter-self.offset_iter)
            max_iter -= self.offset_iter

        if self.max_iter is not None:
            curr_iter = min(curr_iter, self.max_iter)
            max_iter = min(max_iter, self.max_iter)

        scale_factor = (1 - float(curr_iter) / max_iter) ** self.power
        self.val = self.end_val + (self.start_val-self.end_val) * scale_factor
        return self.val

    def __repr__(self):
        return 'Polynomial(start={}, end={}, power={}, maxiter={}, offset={})'.format(self.start_val, self.end_val, self.power, self.max_iter, self.offset_iter)


class LogarithmicScheduler(Scheduler):
    def __init__(self, *args, max_iter=None, **kwargs):
        super(LogarithmicScheduler, self).__init__(*args, **kwargs)
        self.max_iter = max_iter

    def update(self, curr_iter, max_iter):
        if self.max_iter is not None:
            curr_iter = min(curr_iter, self.max_iter)
            max_iter = min(max_iter, self.max_iter)

        end = self.end_val
        start = self.start_val
        self.val = np.log((np.exp(end) - np.exp(start)) / float(max_iter) * float(curr_iter) + np.exp(start))
        return self.val


class SlackScheduler(LogarithmicScheduler):
    def update(self, curr_iter, max_iter):
        return super(SlackScheduler, self).update(np.log(curr_iter+1), np.log(max_iter+1))


class CosineScheduler(Scheduler):
    def update(self, curr_iter, max_iter):
        self.val = self.end_val + 0.5 * (self.start_val-self.end_val) * (1.0 + np.cos(np.pi*(curr_iter/max_iter)))
        return self.val

    def __repr__(self):
        return 'Cosine(start={}, end={})'.format(self.start_val, self.end_val)


class CosineAnnealingWarmRestartsScheduler(Scheduler):
    def __init__(self, start_val, end_val, interval_freq, name=''):
        super(CosineAnnealingWarmRestartsScheduler, self).__init__(start_val, end_val, name)
        self.int_freq = interval_freq

    def update(self, curr_iter, max_iter):
        curr_iter = curr_iter % self.int_freq
        max_iter = self.int_freq
        self.val = self.end_val + 0.5*(self.start_val-self.end_val) * (1.0 + np.cos(np.pi*(curr_iter/max_iter)))
        return self.val

    def __repr__(self):
        return 'CosineWarmRestarts(start={}, end={}, interval={})'.format(self.start_val, self.end_val, self.int_freq)



class OneCycleScheduler(Scheduler):
    def __init__(self, start_val, end_val, name, cycle_len_pct=0.3, power=1.0):
        super(OneCycleScheduler, self).__init__(start_val, end_val, name)
        self.cycle_len_pct = cycle_len_pct
        self.power = power

    def update(self, curr_iter, max_iter):
        tp = max_iter * self.cycle_len_pct
        if curr_iter >= tp:
            curr_iter -= int(tp)
            max_iter -= int(tp)
            scale_factor = (1 - float(curr_iter) / max_iter) ** self.power
            self.val = self.end_val + (self.start_val-self.end_val) * scale_factor
        else:
            self.val = self.start_val
        return self.val

    def __repr__(self):
        return 'OneCycle(pct_start={}, start={}, end={}, power={})'.format(self.cycle_len_pct, self.start_val, self.end_val, self.power)


class ScheduledModule(nn.Module):
    def __init__(self):
        super(ScheduledModule, self).__init__()
        self.schedulerlist = []

    def __setattr__(self, name, value):
        if isinstance(value, Scheduler):
            self.register_scheduler(value)
        super(ScheduledModule, self).__setattr__(name, value)

    def __getattribute__(self, name):
        value = super(ScheduledModule, self).__getattribute__(name)
        if isinstance(value, Scheduler):
            return value.val
        else:
            return value

    def register_scheduler(self, scheduler):
        assert isinstance(scheduler, Scheduler)
        if isinstance(scheduler, ConstantScheduler):
            print('ScheduledModule INFO :: Request to register scheduler {} ignored as it is of type ConstantScheduler'.format(scheduler))
        else:
            if scheduler not in self.schedulerlist:
                self.schedulerlist.append(scheduler)

    def schedulers(self):
        return self.schedulerlist