import os
import numpy as np

run_tool = 'sbatch tool/train_6h.sh'

with open('./template.yaml', 'r') as f:
	template = f.read()
name_template = '4C3F_cll_eps{eps:.2f}_p{err_q:.0e}_globallipdecay{lambda:.1e}_maxmin_adamw_bs128_kaiming_wd0_01data_lr1e-3_ep800_lobo_seed{seed}'


f_run = open('run_all.sh', 'w')

seeds = range(1,10)

err_q = 3.e-1
eps = 0.15
lambda_ = 1.e-4

for seed in seeds:
	kwargs = {
		'eps': eps,
		'err_q': err_q,
		'lambda': lambda_,
		'seed': seed,
	}
	filename = name_template.format(**kwargs)

	kwargs['dist'] = eps*2
	cfg = template.format(seed=seed)

	assert not os.path.exists(filename)
	with open(filename+'.yaml', 'w') as f:
		f.write(cfg)

	f_run.write('{} config/cifar10_4C3F_cll_random_seeds/{}.yaml\n'.format(run_tool, filename))

f_run.close()
