import os
import numpy as np

run_tool = 'python tool/train.py'

with open('./template.yaml', 'r') as f:
	template = f.read()
name_template = '8C2F_cll_{eps:.2f}_p{err_q:.1e}_globallipdecay{lambda_:.0e}_maxmin_adamw_bs128_kaiming_wd0_01data_lr2.5e-4_ep800_lobo'

f_run = open('run_all.sh', 'w')

probabilities = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
eps = 0.15
lambdas = [1.e-5]

for p_eps in probabilities:
	for lambda_ in lambdas:
		cfg = template.format(lambda_=lambda_, dist=eps*2, delta=eps, err_q=p_eps, eps=eps, eps_underscore=('{:.2f}'.format(eps)).replace('.', '_'))
		filename = name_template.format(eps=eps, err_q=p_eps, lambda_=lambda_)

		assert not os.path.exists(filename)
		with open(filename+'.yaml', 'w') as f:
			f.write(cfg)

		f_run.write('{} --config config/tiny-imagenet_8C2F_cll_slack_sweep/{}.yaml\n'.format(run_tool, filename))

f_run.close()
