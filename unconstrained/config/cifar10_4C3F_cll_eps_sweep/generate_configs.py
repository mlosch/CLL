import os
import numpy as np

run_tool = 'python tool/train.py'

with open('./template.yaml', 'r') as f:
	template = f.read()
name_template = '4C3F_cll_eps{eps:.2f}_p3e-1_globallipdecay1e-4_maxmin_adamw_bs128_kaiming_wd0_01data_lr1e-3_ep800_lobo'


f_run = open('run_all.sh', 'w')

distances = [0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 2.7506] #np.linspace(0.05, 1.0, 20)

for dist in distances:
	eps = dist/2
	cfg = template.format(dist=eps*2, delta=eps, eps=eps, eps_underscore=('{:.2f}'.format(eps)).replace('.', '_'))
	filename = name_template.format(eps=eps)

	assert not os.path.exists(filename)
	with open(filename+'.yaml', 'w') as f:
		f.write(cfg)

	f_run.write('{} --config config/cifar10_4C3F_cll_eps_sweep/{}.yaml\n'.format(run_tool, filename))

f_run.close()
