import os
import numpy as np

run_tool = 'sbatch tool/train_6h.sh'

with open('./template.yaml', 'r') as f:
	template = f.read()
name_template = '6C2F_cll_eps{eps:.2f}_p{err_q:.2e}_globallipdecay1e-6_maxmin_adamw_bs128_kaiming_wd0_01data_lr1e-3_ep800_lobo_seed{seed}'


f_run = open('run_all.sh', 'w')

seeds = range(1,10)
eps = 0.15
err_q = 0.3

for seed in seeds:
	cfg = template.format(dist=eps*2, delta=eps, err_q=err_q, eps=eps, eps_underscore=('{:.2f}'.format(eps)).replace('.', '_'), seed=seed)
	filename = name_template.format(eps=eps, err_q=err_q, seed=seed)

	assert not os.path.exists(filename)
	with open(filename+'.yaml', 'w') as f:
		f.write(cfg)

	f_run.write('{} config/cifar10_6C2F_cll_random_seeds/{}.yaml\n'.format(run_tool, filename))

f_run.close()
