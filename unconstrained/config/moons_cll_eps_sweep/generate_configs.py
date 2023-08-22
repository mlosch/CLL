import os
import numpy as np

run_tool = 'python tool/train.py'

with open('./template.yaml', 'r') as f:
	template = f.read()
name_template = '5hid_cll_eps{eps:.2f}_p1e-3to1e-15p50_globallipdecay1e-6pow2_maxmin_adamw_bs100_kaiming_wd0_01data_lr1e-3_ep1000_lrdecay'


f_run = open('run_all.sh', 'w')

#epsilons = np.linspace(0.1, 0.5, 9)
epsilons = [0.1, 0.15, 0.2, 0.25]

for eps in epsilons:
	levels = [-eps, 0, eps]
	cfg = template.format(
		dist=eps*2, 
		delta=eps,
		eps=eps, 
		eps_pretty=('{:.2f}'.format(eps)).replace('.','_'),
		eps2=eps*2,
		eps2_pretty=('{:.2f}'.format(eps*2)).replace('.','_'),
		levels='['+', '.join(['{:.2f}'.format(l) for l in levels])+']'
		)
	filename = name_template.format(eps=eps)

	assert not os.path.exists(filename)
	with open(filename+'.yaml', 'w') as f:
		f.write(cfg)

	f_run.write('{} --config config/moons_cll_eps_sweep/{}.yaml\n'.format(run_tool, filename))

f_run.close()
