

TRAIN_STEP='2'

# not_shuffle_train_data


# MY_CMD="python3 -u -m torch.distributed.launch --nproc_per_node=4 ssl_perturbation_v2_ddp.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 12 --step_size 0.8 --attack_type min-min --perturb_type samplewise --train_step 10 --epochs 1000 --min_min_attack_fn eot_v1 --strong_aug --not_shuffle_train_data --eot_size 1 --batch_size 1024 --num_workers 8 --gpu_num 4 --no_eval --local 0,1,2,3 --no_save"

# MY_CMD="python3 perturbation.py --config_path configs/cifar100 --exp_name path/to/your/experiment/cifar100 --version resnet18 --train_data_type CIFAR100 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type samplewise --universal_stop_error 0.01 --local 3"

# MY_CMD="python3 -u ssl_perturbation_v2.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR100 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type samplewise --train_step ${TRAIN_STEP} --min_min_attack_fn eot_v1 --strong_aug --eot_size 1 --shuffle_train_perturb_data --pytorch_aug --simclr_weight 1 --linear_noise_dbindex_weight 1 --use_wholeset_center --local 3 --no_save"

# MY_CMD="python3 -u ssl_perturbation_v2.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR100 --noise_shape 100 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type classwise --train_step 20 --min_min_attack_fn eot_v1 --strong_aug --eot_size 1 --shuffle_train_perturb_data --dbindex_weight 0 --dbindex_label_index 1 --pytorch_aug --local 3 --no_save"

# MY_CMD="python3 -u ssl_perturbation_v2.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type samplewise --train_step ${TRAIN_STEP} --min_min_attack_fn eot_v1 --strong_aug --eot_size 1 --shuffle_train_perturb_data --pytorch_aug --simclr_weight 0 --linear_noise_dbindex_weight 1 --linear_noise_dbindex_index 2 --random_start --not_use_normalized --kmeans_index 1 --kmeans_index2 2 --debug --kmeans_label_file kmeans_cifar10_n_10_100_20 --local 3 --no_save"

# MY_CMD="python3 -u ssl_perturbation_v2.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type samplewise --train_step ${TRAIN_STEP} --min_min_attack_fn eot_v1 --strong_aug --eot_size 1 --shuffle_train_perturb_data --pytorch_aug --simclr_weight 1 --linear_noise_dbindex_weight 1e-15 --local 1 --no_save"

MY_CMD="python3 -u ssl_perturbation_v2.py --epochs 2 --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type samplewise --train_step 2 --min_min_attack_fn eot_v1 --strong_aug --eot_size 1 --shuffle_train_perturb_data --pytorch_aug --simclr_weight 1 --linear_xnoise_dbindex_weight 1e-15 --local 2 --no_save"

# MY_CMD="python3 -u ssl_perturbation_v2.py --piermaro_whole_epoch 42 --epochs 2 --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type samplewise --train_step 20 --min_min_attack_fn eot_v1 --strong_aug --eot_size 1 --shuffle_train_perturb_data --simclr_weight 0 --use_supervised_g --model_g_augment_first --pytorch_aug --local 3 --no_save"

# MY_CMD="python3 -u -m torch.distributed.launch --nproc_per_node=2 ssl_perturbation_v2_ddp.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 12 --step_size 0.8 --attack_type min-min --perturb_type samplewise --train_step ${TRAIN_STEP} --epochs 1000 --min_min_attack_fn eot_v1 --strong_aug --not_shuffle_train_data --eot_size 10 --gpu_num 2 --local 1,2 --no_save"
# use_dbindex_train_model

# MY_CMD="python simclr_transfer.py --batch_size 512 --epochs 1000 --arch resnet18 --perturbation_budget 1 --pre_load_name unlearnable_samplewise_52428805_1_20220505181516_0.5_512_2_checkpoint_perturbation_epoch_20 --train_data_type CIFAR10 --pytorch_aug --samplewise --local 3 --no_save"

# MY_CMD="python simclr_transfer.py --batch_size 512 --epochs 1000 --arch resnet18 --perturbation_budget 1 --pre_load_name unlearnable_samplewise_52802196_1_20220507103138_0.5_512_2_checkpoint_perturbation --save_img_group --train_data_type CIFAR10 --samplewise --local 1"
# MY_CMD="python simclr_transfer.py --batch_size 512 --epochs 1000 --arch resnet18 --perturbation_budget 1 --pre_load_name unlearnable_samplewise_52806124_1_20220507111200_0.5_512_2_checkpoint_perturbation --save_noise_input_space --train_data_type CIFAR10 --samplewise --local 3"
# 52734683 unlearnable_samplewise_52734683_1_20220507001155_0.5_512_2_checkpoint_perturbation
# unlearnable_samplewise_52428866_1_20220505211256_0.5_512_2_checkpoint_perturbation
# unlearnable_samplewise_52806124_1_20220507111200_0.5_512_2_checkpoint_perturbation
# unlearnable_samplewise_52806123_1_20220507111158_0.5_512_2_checkpoint_perturbation 0.08

# 52865694 unlearnable_samplewise_52865694_1_20220507200052_0.5_512_2_checkpoint_perturbation
# 52866132 unlearnable_samplewise_52866132_1_20220507200522_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866270 unlearnable_samplewise_52866270_1_20220507200522_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866443 unlearnable_samplewise_52866443_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866444 unlearnable_samplewise_52866444_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866445 unlearnable_samplewise_52866445_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866446 unlearnable_samplewise_52866446_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866447 unlearnable_samplewise_52866447_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866448 unlearnable_samplewise_52866448_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866449 unlearnable_samplewise_52866449_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866450 unlearnable_samplewise_52866450_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866563 unlearnable_samplewise_52866563_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866589 unlearnable_samplewise_52866589_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# 52866590 unlearnable_samplewise_52866590_1_20220507200823_0.5_512_2_checkpoint_perturbation_epoch_10

# try this 52802261

    # model_zoo = {'VGG19': VGG,
    # 'resnet18': ResNet18,
    # 'PreActResNet18': PreActResNet18,
    # 'GoogLeNet': GoogLeNet,
    # 'DenseNet121': DenseNet121,
    # 'ResNeXt29_2x64d': ResNeXt29_2x64d,
    # 'MobileNet': MobileNet,
    # 'MobileNetV2': MobileNetV2,
    # 'DPN92': DPN92,
    # 'SENet18': SENet18,
    # # 'ShuffleNetV2': ShuffleNetV2(1),
    # 'EfficientNetB0': EfficientNetB0,
    # 'RegNetX_200MF': RegNetX_200MF,
    # 'simpledla': SimpleDLA}

# MY_CMD="python supervised_cifar10.py --train_data_type cifar10 --arch VGG11 --pre_load_name unlearnable_samplewise_52866449_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10 --samplewise --poison_rate 0.5 --local 1 --no_save"
# unlearnable_samplewise_52866449_1_20220507200521_0.5_512_2_checkpoint_perturbation_epoch_10
# unlearnable_samplewise_52428805_1_20220505181516_0.5_512_2_checkpoint_perturbation_epoch_20

# MY_CMD="python supervised_cifar10.py --train_data_type cifar100 --arch resnet18 --pre_load_name unlearnable_classwise_52428199_1_20220505171817_0.5_512_2_checkpoint_perturbation --local 1 --no_save"

# MY_CMD="python3 -u main.py --version resnet18 --exp_name path/to/your/experiment/folder_cifar10_main --config_path configs/cifar10 --train_data_type PoisonCIFAR10 --poison_rate 1.0 --perturb_type samplewise --perturb_tensor_filepath results/cifar10_unlearnable_52378127_1_perturbation.pt --train"

# MY_CMD="python3 perturbation.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type samplewise --universal_stop_error 0.01 --linear_noise_dbindex_weight 1 --simclr_weight 0"

# MY_CMD="python3 -u linear.py --model_path unlearnable_classwise_52015942_1_20220427114727_0.5_512_300_checkpoint_perturbation_budget1.0_class10_retrain_model_checkpoint_model --local 1"

# MY_CMD="python simclr_transfer_plot.py --batch_size 512 --epochs 1000 --arch resnet18 --perturbation_budget 1 --pre_load_name unlearnable_samplewise_51508204_1_20220419160917_0.5_512_300_checkpoint_perturbation --samplewise --load_model --load_model_path unlearnable_samplewise_51508204_1_20220419160917_0.5_512_300_checkpoint_model --kmeans_index 0 --unlearnable_kmeans_label --local 1"

# MY_CMD="python simclr_transfer_plot.py --batch_size 512 --epochs 1000 --arch resnet18 --perturbation_budget 1 --pre_load_name unlearnable_samplewise_51211597_1_20220412000452_0.5_1024_1000_checkpoint_perturbation --samplewise --load_model --load_model_path unlearnable_samplewise_51030219_1_20220409114042_0.5_512_1000_checkpoint_perturbation_budget1.0_class4_retrain_model_model --local 2"


echo $MY_CMD
echo ${MY_CMD}>>local_history.log
$MY_CMD

# MY_CMD="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type theory_model --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --gray_train no --gray_test no --theory_train_data hierarchical_period_dim20_knn4 --theory_test_data hierarchical_period_dim20_test_knn4 --random_drop_feature_num 5 5 1 1 1 --gaussian_aug_std 0 --theory_normalize --thoery_schedule_dim 90 --just_test --load_model --load_model_path unlearnable_theory_44030018_2_20220117210846_0.5_512_1000_final_model --local 2 --no_save"

# echo $MY_CMD
# echo ${MY_CMD}>>local_history.log
# $MY_CMD

# MY_CMD="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type theory_model --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --gray_train no --gray_test no --theory_train_data hierarchical_period_dim20_knn4 --theory_test_data hierarchical_period_dim20_test_knn4 --random_drop_feature_num 5 5 5 1 1 --gaussian_aug_std 0 --theory_normalize --thoery_schedule_dim 90 --just_test --load_model --load_model_path unlearnable_theory_44030018_3_20220117210847_0.5_512_1000_final_model --local 2 --no_save"

# echo $MY_CMD
# echo ${MY_CMD}>>local_history.log
# $MY_CMD

# MY_CMD="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type theory_model --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --gray_train no --gray_test no --theory_train_data hierarchical_period_dim20_knn4 --theory_test_data hierarchical_period_dim20_test_knn4 --random_drop_feature_num 5 5 5 5 1 --gaussian_aug_std 0 --theory_normalize --thoery_schedule_dim 90 --just_test --load_model --load_model_path unlearnable_theory_44030020_1_20220117210918_0.5_512_1000_final_model --local 2 --no_save"

# echo $MY_CMD
# echo ${MY_CMD}>>local_history.log
# $MY_CMD

# MY_CMD="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type theory_model --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --gray_train no --gray_test no --theory_train_data hierarchical_period_dim20_knn4 --theory_test_data hierarchical_period_dim20_test_knn4 --random_drop_feature_num 5 5 5 5 5 --gaussian_aug_std 0 --theory_normalize --thoery_schedule_dim 90 --just_test --load_model --load_model_path unlearnable_theory_44030020_2_20220117210918_0.5_512_1000_final_model --local 2 --no_save"

# echo $MY_CMD
# echo ${MY_CMD}>>local_history.log
# $MY_CMD

# MY_CMD="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type just_test --plot_be_mode single_augmentation --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --gray_train no --gray_test no --augmentation ReCrop_Hflip_Bri --augmentation_prob 1.0 0.5 0.0 0.2 --load_model --load_model_path unlearnable_cleantrain_42873265_1_20211228104344_0.5_512_1000_final_model --not_shuffle_train_data --mix no --local_dev 2 --no_save"

# echo $MY_CMD
# echo ${MY_CMD}>>local_history.log
# $MY_CMD

# TEMP_SAVE_FILE="local_cmd_temp"

# MY_CMD="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type theory_model --epochs 1 --min_min_attack_fn non_eot --strong_aug --class_4 --gray_train no --gray_test no --theory_train_data hierarchical_period_dim20_shuffle_knn4 --theory_test_data hierarchical_period_dim20_shuffle_test_knn4 --random_drop_feature_num 0 0 0 0 0 --gaussian_aug_std 0 --theory_normalize --thoery_schedule_dim 90 --local_dev 2 --just_test_temp_save_file ${TEMP_SAVE_FILE}"

# echo $MY_CMD
# echo ${MY_CMD}>>local_history.log
# $MY_CMD

# echo "${TEMP_SAVE_FILE}" >> ./results_just_test/just_test.txt
# cat ./results_just_test/${TEMP_SAVE_FILE}.txt >> ./results_just_test/just_test.txt

# rm -f ./results_just_test/${TEMP_SAVE_FILE}.txt

# MY_CMD="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type theory_model --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --gray_train no --gray_test no --theory_train_data hierarchical64_16_period_dim30_shuffle_std0.03#0.1_diffmean_knn64 --theory_test_data hierarchical64_16_period_dim30_shuffle_std0.03#0.1_diffmean_test1_knn64 --random_drop_feature_num 9 1 1 --gaussian_aug_std 0 --theory_normalize --thoery_schedule_dim 30 --local 2 --no_save"

# echo $MY_CMD
# echo ${MY_CMD}>>local_history.log
# $MY_CMD

# unlearnable_theory_43990132_1_20220116172552_0.5_512_1000_statistics
# unlearnable_theory_43990132_2_20220116172552_0.5_512_1000_statistics
# unlearnable_theory_43990132_3_20220116172552_0.5_512_1000_statistics
# unlearnable_theory_43990133_1_20220116172556_0.5_512_1000_statistics
# unlearnable_theory_43990133_2_20220116172556_0.5_512_1000_statistics
# unlearnable_theory_43990133_3_20220116172556_0.5_512_1000_statistics
# unlearnable_theory_43990134_1_20220116172605_0.5_512_1000_statistics
# unlearnable_theory_43990134_2_20220116172605_0.5_512_1000_statistics
# unlearnable_theory_43990134_3_20220116172605_0.5_512_1000_statistics
# unlearnable_theory_43990135_1_20220116172633_0.5_512_1000_statistics

# unlearnable_theory_44027147_1_20220117171058_0.5_512_1000_statistics
# unlearnable_theory_44027147_2_20220117171057_0.5_512_1000_statistics
# unlearnable_theory_44027147_3_20220117171058_0.5_512_1000_statistics
# unlearnable_theory_44027148_1_20220117171058_0.5_512_1000_statistics
# unlearnable_theory_44027148_2_20220117171057_0.5_512_1000_statistics
# unlearnable_theory_44027148_3_20220117171058_0.5_512_1000_statistics
# unlearnable_theory_44027149_1_20220117171107_0.5_512_1000_statistics
# unlearnable_theory_44027149_2_20220117171107_0.5_512_1000_statistics
# unlearnable_theory_44027149_3_20220117171107_0.5_512_1000_statistics
# unlearnable_theory_44027150_1_20220117171119_0.5_512_1000_statistics
# python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type theory_model --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --gray_train no --gray_test no --theory_train_data hierarchical_period_dim150_knn4 --theory_test_data hierarchical_period_dim150_test_knn16 --random_drop_feature_num 9 9 9 9 9 --gaussian_aug_std 0 --theory_normalize --thoery_schedule_dim 150

# unlearnable_theory_44027154_1_20220117171416_0.5_512_1000_statistics
# unlearnable_theory_44027154_2_20220117171416_0.5_512_1000_statistics
# unlearnable_theory_44027154_3_20220117171416_0.5_512_1000_statistics
# unlearnable_theory_44027155_1_20220117171416_0.5_512_1000_statistics
# unlearnable_theory_44027155_2_20220117171416_0.5_512_1000_statistics
# unlearnable_theory_44027155_3_20220117171416_0.5_512_1000_statistics
# unlearnable_theory_44027158_1_20220117171416_0.5_512_1000_statistics
# unlearnable_theory_44027158_2_20220117171416_0.5_512_1000_statistics
# unlearnable_theory_44027158_3_20220117171416_0.5_512_1000_statistics
# unlearnable_theory_44027159_1_20220117171416_0.5_512_1000_statistics
# python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type theory_model --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --gray_train no --gray_test no --theory_train_data hierarchical_period_dim20_knn4 --theory_test_data hierarchical_period_dim20_test_knn16 --random_drop_feature_num 0 0 0 0 0 --gaussian_aug_std 0 --theory_normalize --thoery_schedule_dim 90

# python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type theory_model --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --gray_train no --gray_test no --theory_train_data hierarchical_period_dim20_knn4 --theory_test_data hierarchical_period_dim20_test_knn16 --random_drop_feature_num 2 2 2 2 1 --gaussian_aug_std 0 --theory_normalize --thoery_schedule_dim 90
# unlearnable_theory_44027160_1_20220117171437_0.5_512_1000_statistics 2 2 2 2 1
# unlearnable_theory_44027160_2_20220117171437_0.5_512_1000_statistics 2 2 2 2 4
# unlearnable_theory_44027160_3_20220117171437_0.5_512_1000_statistics 2 2 2 2 8
# unlearnable_theory_44027174_1_20220117171549_0.5_512_1000_statistics 2 2 2 2 0

# unlearnable_theory_44027174_2_20220117171549_0.5_512_1000_statistics 2 0 0 0 0
# unlearnable_theory_44027174_3_20220117171549_0.5_512_1000_statistics 2 0 0 0 2

# unlearnable_theory_44027179_1_20220117171740_0.5_512_1000_statistics 0 2 2 0 2
# unlearnable_theory_44027179_2_20220117171740_0.5_512_1000_statistics 1 2 2 1 2
# unlearnable_theory_44027179_3_20220117171740_0.5_512_1000_statistics 2 2 2 2 2
# unlearnable_theory_44027180_1_20220117171739_0.5_512_1000_statistics 3 2 2 3 2
# unlearnable_theory_44027180_2_20220117171739_0.5_512_1000_statistics 4 2 2 4 2
# unlearnable_theory_44027180_3_20220117171739_0.5_512_1000_statistics 5 2 2 5 2
# unlearnable_theory_44027184_1_20220117171740_0.5_512_1000_statistics 6 2 2 6 2
# unlearnable_theory_44027184_2_20220117171740_0.5_512_1000_statistics 7 2 2 7 2
# unlearnable_theory_44027184_3_20220117171740_0.5_512_1000_statistics 8 2 2 8 2
# unlearnable_theory_44027187_1_20220117171837_0.5_512_1000_statistics 9 2 2 9 2

# unlearnable_theory_44027189_1_20220117171841_0.5_512_1000_statistics 0 0 0 0 0
# unlearnable_theory_44027189_2_20220117171841_0.5_512_1000_statistics 1 0 0 0 0
# unlearnable_theory_44027189_3_20220117171841_0.5_512_1000_statistics 1 1 0 0 0
# unlearnable_theory_44027194_1_20220117171939_0.5_512_1000_statistics 1 1 1 0 0
# unlearnable_theory_44027194_2_20220117171939_0.5_512_1000_statistics 1 1 1 1 0
# unlearnable_theory_44027195_1_20220117171939_0.5_512_1000_statistics 1 1 1 1 1

# unlearnable_theory_44027195_2_20220117171939_0.5_512_1000_statistics 3 1 1 1 1
# unlearnable_theory_44027196_1_20220117172126_0.5_512_1000_statistics 3 3 3 1 1
# unlearnable_theory_44027196_2_20220117172126_0.5_512_1000_statistics 3 3 3 3 1

# unlearnable_theory_44030018_1_20220117210847_0.5_512_1000_statistics 5 1 1 1 1
# unlearnable_theory_44030018_2_20220117210846_0.5_512_1000_statistics 5 5 1 1 1
# unlearnable_theory_44030018_3_20220117210847_0.5_512_1000_statistics 5 5 5 1 1
# unlearnable_theory_44030020_1_20220117210918_0.5_512_1000_statistics 5 5 5 5 1
# unlearnable_theory_44030020_2_20220117210918_0.5_512_1000_statistics 5 5 5 5 5

# python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type theory_model --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --gray_train no --gray_test no --theory_train_data hierarchical_period_dim20_knn4 --theory_test_data hierarchical_period_dim20_test_knn16 --random_drop_feature_num 0 0 0 0 0 --gaussian_aug_std 0.01 --theory_normalize --thoery_schedule_dim 90
# unlearnable_theory_44027198_1_20220117172317_0.5_512_1000_statistics 0.01
# unlearnable_theory_44027198_2_20220117172317_0.5_512_1000_statistics 0.02
# unlearnable_theory_44027198_3_20220117172317_0.5_512_1000_statistics 0.03
# unlearnable_theory_44027199_1_20220117172318_0.5_512_1000_statistics 0.04
# unlearnable_theory_44027199_2_20220117172318_0.5_512_1000_statistics 0.05
# unlearnable_theory_44027199_3_20220117172318_0.5_512_1000_statistics 0.06
# unlearnable_theory_44027200_1_20220117172324_0.5_512_1000_statistics 0.07
# unlearnable_theory_44027200_2_20220117172324_0.5_512_1000_statistics 0.1
# unlearnable_theory_44027200_3_20220117172324_0.5_512_1000_statistics 0.15
# unlearnable_theory_44027201_1_20220117172326_0.5_512_1000_statistics 0.2
# unlearnable_theory_44027201_2_20220117172326_0.5_512_1000_statistics 0.5
# unlearnable_theory_44027201_3_20220117172326_0.5_512_1000_statistics 0.75
# unlearnable_theory_44027202_1_20220117172335_0.5_512_1000_statistics 1
# unlearnable_theory_44027202_2_20220117172335_0.5_512_1000_statistics 1.5
# unlearnable_theory_44027202_3_20220117172335_0.5_512_1000_statistics 2
# unlearnable_theory_44027203_1_20220117172639_0.5_512_1000_statistics 3
# unlearnable_theory_44027203_2_20220117172639_0.5_512_1000_statistics 4
# unlearnable_theory_44027203_3_20220117172639_0.5_512_1000_statistics 5

# unlearnable_theory_44027198_1_20220117172317_0.5_512_1000_statistics

# random_samplewise_mnist
# random_samplewise_center_all_18_budget128
# unlearnable_cleantrain_43810886_1_20220115005203_0
# unlearnable_cleantrain_43810887_1_20220115005214_0
# unlearnable_cleantrain_43810887_2_20220115005214_0
# unlearnable_cleantrain_43810886_2_20220115005203_0
# unlearnable_cleantrain_43810884_1_20220115005150_0

# samplewise_mnist
# samplewise_all_center_18_128
# unlearnable_cleantrain_43809527_1_20220115003409_0
# unlearnable_cleantrain_43809528_1_20220115003402_0
# unlearnable_cleantrain_43809528_2_20220115003402_0
# unlearnable_cleantrain_43809527_2_20220115003409_0
# unlearnable_cleantrain_41798562_2_20211210195041_0

# MY_CMD="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 20000 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type clean_train --train_step 100 --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --shuffle_train_perturb_data --gray_train cifar10_20000_triobject --gray_test cifar10_20000_triobject --augmentation Tri --local_dev 2 --no_save"

# echo $MY_CMD
# echo ${MY_CMD}>>local_history.log
# $MY_CMD

# python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type test_find_positive_pair --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --not_shuffle_train_data --augmentation ReCrop_Hflip --batch_size 256 --gray_train mnist_train_4gray --gray_test mnist_train_4gray --load_model --load_model_path unlearnable_cleantrain_43557131_1_20220111124039_0.5_256_1000_final_model --local_dev 2 --no_save

# unlearnable_cleantrain_42483880_1_20211221153535_0.5_512_1000_final_model
# unlearnable_cleantrain_41406147_1_20211203004142_0.5_512_1000_statistics pos
# unlearnable_cleantrain_41459622_2_20211203162721_0.5_512_1000_final_model normal
# unlearnable_cleantrain_41406143_1_20211203004141_0.5_512_1000_statistics pos/neg
# unlearnable_cleantrain_42873264_1_20211228104252_0.5_512_1000_final_model --augmentation_prob 0.0 0.5 0.8 0.2
# unlearnable_cleantrain_42873264_2_20211228104252_0.5_512_1000_final_model --augmentation_prob 1.0 0.0 0.8 0.2
# unlearnable_cleantrain_42873265_1_20211228104344_0.5_512_1000_final_model --augmentation_prob 1.0 0.5 0.0 0.2
# unlearnable_cleantrain_42873265_2_20211228104344_0.5_512_1000_final_model --augmentation_prob 1.0 0.5 0.8 0.0

# grayshiftlarge_font_randomdigit_mnist grayshift_mnist grayshift_font_mnist mnist_train_4gray mnist_4position
# unlearnable_cleantrain_43556970_1_20220111122328_0.5_256_1000_statistics mnist_train_4gray
# unlearnable_cleantrain_43556970_2_20220111122328_0.5_256_1000_statistics CIFAR10 gray
# unlearnable_cleantrain_43557131_1_20220111124039_0.5_256_1000_statistics not_shuffle_train

# unlearnable_cleantrain_42873264_1_20211228104252_0.5_512_1000_statistics
# unlearnable_cleantrain_42873264_2_20211228104252_0.5_512_1000_statistics
# unlearnable_cleantrain_42873265_1_20211228104344_0.5_512_1000_statistics
# unlearnable_cleantrain_42873265_2_20211228104344_0.5_512_1000_statistics

# random_samplewise_mnist
# unlearnable_cleantrain_43810886_1_20220115005203_0
# unlearnable_cleantrain_43810887_1_20220115005214_0
# unlearnable_cleantrain_43810887_2_20220115005214_0
# unlearnable_cleantrain_43810886_2_20220115005203_0
# unlearnable_cleantrain_43810884_1_20220115005150_0

# samplewise_mnist
# unlearnable_cleantrain_43809527_1_20220115003409_0
# unlearnable_cleantrain_43809528_1_20220115003402_0
# unlearnable_cleantrain_43809528_2_20220115003402_0
# unlearnable_cleantrain_43809527_2_20220115003409_0
# unlearnable_cleantrain_41798562_2_20211210195041_0