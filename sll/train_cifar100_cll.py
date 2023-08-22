import random

scripts = ['python main.py']
cmd = '{script} --ngpus 4 --dataset cifar100 --model-name xlarge --loss cll --last_layer vanilla --data_dir ./data/ --train_dir cll_eps{delta}_p{p}_lam1e-5_xl_model_seed{seed} --clip_p {p} --clip_eps {eps} --clip_delta {delta} --cll_lambda 1.e-5 --seed {seed}'

f_train = open('train_cifar100_cll.sh', 'w')

delta = 1.0
epsilons = [1.0]
ps = [3.e-3]
seeds = range(1,10)

i = 0
for eps in epsilons:
	for p in ps:
		for seed in seeds:
			script = scripts[i]
			i = (i+1)%len(scripts)
			f_train.write(cmd.format(script=script, eps=eps, p=p, delta=delta, seed=seed)+'\n')

f_train.close()
