MYCOMMEND="python3 -u -m torch.distributed.launch --nproc_per_node=2 ssl_perturbation_v2_ddp_pos.py --epochs 1000 --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type samplewise --train_step 10 --min_min_attack_fn eot_v1 --strong_aug --not_shuffle_train_data --eot_size 3 --batch_size 512 --num_workers 8 --gpu_num 2 --dbindex_weight 0 --simclr_weight 1 --pytorch_aug --class_4"
MYCOMMEND2="No_commend2"
MYCOMMEND3="No_commend3"
MYGRES="gpu:v100s:2"
