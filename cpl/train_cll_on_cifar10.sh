# --- XL ---
python main.py --save-dir exp_cifar10/CLP_XL_wCLL_eps0.25_p0.1_lambda1.e-3_trianglr_seed1 --depth 70 --num-channels 120 --depth-linear 15 --n-features 4096 --loss_type CLL --clip_epsilon 0.25 --clip_p 0.1 --clip_delta 0.25 --clip_detach --clip_lambda 1.e-3 --seed 1