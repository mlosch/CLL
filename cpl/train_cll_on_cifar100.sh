# --- XL ---
python main.py --save-dir exp_cifar100/CLP_XL_wCLL_eps1.0_p1.e-3_lambda1.e-3_trianglr_seed1 --dataset c100 --depth 70 --num-channels 120 --depth-linear 15 --n-features 4096 --loss_type CLL --clip_epsilon 1.0 --clip_p 1.e-3 --clip_delta 1.0 --clip_detach --clip_lambda 1.e-3 --seed 1