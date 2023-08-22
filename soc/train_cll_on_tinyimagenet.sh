# LipConv-10
python train_robust.py --conv-layer soc --activation hh1 --dataset tinyimagenet --data-dir data/tiny-imagenet-200 --gamma 0.0 --block-size 2 --loss_type CLL --calib_err_quantile 1.e-1 --calib_epsilon 0.15 --calib_add_epsilon 0.15 --optim AdamW --lr-max 1.e-3 --detach_K --lambda_ 1.e-5 --out-dir tinyimagenet_cll/LipConvNet --seed 1

# LipConv-20
python train_robust.py --conv-layer soc --activation hh1 --dataset tinyimagenet --data-dir data/tiny-imagenet-200 --gamma 0.0 --block-size 4 --loss_type CLL --calib_err_quantile 3.e-1 --calib_epsilon 0.15 --calib_add_epsilon 0.15 --optim AdamW --lr-max 1.e-3 --detach_K --lambda_ 1.e-5 --out-dir tinyimagenet_cll/LipConvNet --seed 1