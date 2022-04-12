# models=(unlearnable_theory_44928179_1_20220201121552_0.5_512_1000_final_model unlearnable_theory_44928179_2_20220201121552_0.5_512_1000_final_model unlearnable_theory_44928179_3_20220201121552_0.5_512_1000_final_model unlearnable_theory_44928180_1_20220201121551_0.5_512_1000_final_model unlearnable_theory_44928180_2_20220201121551_0.5_512_1000_final_model unlearnable_theory_44928180_3_20220201121551_0.5_512_1000_final_model unlearnable_theory_44928181_1_20220201121552_0.5_512_1000_final_model unlearnable_theory_44928181_2_20220201121552_0.5_512_1000_final_model unlearnable_theory_44928181_3_20220201121552_0.5_512_1000_final_model unlearnable_theory_44928182_1_20220201121602_0.5_512_1000_final_model unlearnable_theory_44928182_2_20220201121602_0.5_512_1000_final_model)
# train_datas=(hierarchical_period_dim10_knn4096)
# test_datas=(hierarchical_period_dim10_knn4096)
# # test_datas=(hierarchical_period_dim20_test_knn256 hierarchical_period_dim20_test_knn64 hierarchical_period_dim20_test_knn16 hierarchical_period_dim20_test_knn4)

# for((i=0;i<1;i++));
# do
#     for((j=0;j<11;j++));
#     do
#         MY_CMD="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type theory_model --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --gray_train no --gray_test no --theory_train_data ${train_datas[${i}]} --theory_test_data ${test_datas[${i}]} --random_drop_feature_num $j --gaussian_aug_std 0 --theory_normalize --thoery_schedule_dim 10 --just_test --just_test_plot --load_model --load_model_path ${models[${j}]} --local 2 --no_save"

#         echo $MY_CMD
#         # echo ${MY_CMD}>>local_history.log
#         $MY_CMD
#     done
# done

# unlearnable_theory_44485939_1_20220124103417_0.5_512_1000
# unlearnable_theory_44485939_2_20220124103417_0.5_512_1000
# unlearnable_theory_44485939_3_20220124103417_0.5_512_1000
# unlearnable_theory_44485941_1_20220124103429_0.5_512_1000
# unlearnable_theory_44485941_2_20220124103429_0.5_512_1000
# unlearnable_theory_44485941_3_20220124103429_0.5_512_1000
# unlearnable_theory_44485936_1_20220124103350_0.5_512_1000
# unlearnable_theory_44485936_2_20220124103350_0.5_512_1000
# unlearnable_theory_44485936_3_20220124103350_0.5_512_1000
# ---------------------------------------------------------------
# unlearnable_theory_44486030_1_20220124104410_0.5_512_1000
# unlearnable_theory_44486030_2_20220124104410_0.5_512_1000
# unlearnable_theory_44486030_3_20220124104410_0.5_512_1000
# unlearnable_theory_44486031_1_20220124104416_0.5_512_1000
# unlearnable_theory_44486031_2_20220124104416_0.5_512_1000
# unlearnable_theory_44486031_3_20220124104416_0.5_512_1000
# unlearnable_theory_44486033_1_20220124104423_0.5_512_1000
# unlearnable_theory_44486033_2_20220124104424_0.5_512_1000
# unlearnable_theory_44486033_3_20220124104423_0.5_512_1000

# unlearnable_theory_44027189_1_20220117171841_0.5_512_1000_statistics 0 0 0 0 0
# unlearnable_theory_44027189_2_20220117171841_0.5_512_1000_statistics 1 0 0 0 0
# unlearnable_theory_44027189_3_20220117171841_0.5_512_1000_statistics 1 1 0 0 0
# unlearnable_theory_44027194_1_20220117171939_0.5_512_1000_statistics 1 1 1 0 0
# unlearnable_theory_44027194_2_20220117171939_0.5_512_1000_statistics 1 1 1 1 0
# unlearnable_theory_44027195_1_20220117171939_0.5_512_1000_statistics 1 1 1 1 1

# unlearnable_theory_44030018_1_20220117210847_0.5_512_1000_statistics 5 1 1 1 1
# unlearnable_theory_44030018_2_20220117210846_0.5_512_1000_statistics 5 5 1 1 1
# unlearnable_theory_44030018_3_20220117210847_0.5_512_1000_statistics 5 5 5 1 1
# unlearnable_theory_44030020_1_20220117210918_0.5_512_1000_statistics 5 5 5 5 1
# unlearnable_theory_44030020_2_20220117210918_0.5_512_1000_statistics 5 5 5 5 5


# MY_CMD="python3 -u ssl_perturbation_v2.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 12 --step_size 0.8 --attack_type min-min --perturb_type samplewise --train_step 10 --epochs 1000 --min_min_attack_fn eot_v1 --strong_aug --eot_size 1 --pytorch_aug --not_shuffle_train_data --noise_after_transform --dbindex_weight 0.3 --local 0 --no_save"

MY_CMD="python3 -u -m torch.distributed.launch --nproc_per_node=2 ssl_perturbation_v2_ddp.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 50000 3 32 32 --epsilon 8 --num_steps 12 --step_size 0.8 --attack_type min-min --perturb_type samplewise --train_step 10 --epochs 1000 --min_min_attack_fn eot_v1 --strong_aug --eot_size 1 --pytorch_aug --not_shuffle_train_data --noise_after_transform --gpu_num 2 --dbindex_weight 0.3 --local 0,1 --no_save"

# MY_CMD="python simclr_transfer.py --batch_size 512 --epochs 1000 --arch resnet18 --perturbation_budget 1 --pre_load_name unlearnable_samplewise_51029072_1_20220409111426_0.5_512_1000_checkpoint_perturbation_epoch_3 --samplewise --local 1"

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