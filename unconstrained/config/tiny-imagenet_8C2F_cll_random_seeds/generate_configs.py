import os
import numpy as np

with open('./template.yaml', 'r') as f:
	template = f.read()
name_template = '8C2F_cll_eps1.0_p2.5e-2_globallipdecay1e-3_maxmin_adamw_bs128_kaiming_wd0_01data_lr2.5e-4_ep800_lobo_seed{seed}'


f_run = open('run_all.sh', 'w')

seeds = range(1,10)
eps = 1.0
err_q = 2.5e-2

for seed in seeds:
	cfg = template.format(seed=seed)
	filename = name_template.format(seed=seed)

	assert not os.path.exists(filename)
	with open(filename+'.yaml', 'w') as f:
		f.write(cfg)

	f_run.write('python -m tool.train --config config/tiny-imagenet_8C2F_cll_random_seeds/{}.yaml\n'.format(filename))

f_run.close()
