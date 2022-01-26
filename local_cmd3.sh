group1=(0247 2489 0568 3569 0789 1248 3459 2789 2456 2679 1239 5679 1359 2578 0389 1578 1237 4789 1279 0478 1348 0239 0238 1269 2368 1357 3568 2345 1467 0157)
group2=(1369 0367 1279 1278 1356 3579 0126 1356 0189 0148 0467 1234 0468 1346 2467 0469 4568 0126 0568 1239 0256 1478 1567 0348 0145 4689 0124 0179 0389 4689)
# test_datas=(hierarchical_period_dim20_test_knn256 hierarchical_period_dim20_test_knn64 hierarchical_period_dim20_test_knn16 hierarchical_period_dim20_test_knn4)


# for((j=0;j<30;j++));
# do
#     MY_CMD="python sample_mnist_2digit.py --digit_group1 ${group1[${i}]} --digit_group2 ${group2[${i}]}"
#     echo $MY_CMD
#     # echo ${MY_CMD}>>local_history.log
#     $MY_CMD
#     MY_CMD="python sample_mnist_2digit.py --digit_group1 ${group2[${i}]} --digit_group2 ${group1[${i}]}"
#     echo $MY_CMD
#     # echo ${MY_CMD}>>local_history.log
#     $MY_CMD
# done


for((i=0;i<30;i++));
do
    echo $i
    MY_CMD="sh job_2digit.sh ${group1[${i}]} ${group2[${i}]}"
    echo $MY_CMD
    # echo ${MY_CMD}>>local_history.log
    $MY_CMD

    sleep 1m

    MY_CMD="sh job_2digit.sh ${group2[${i}]} ${group1[${i}]}"
    echo $MY_CMD
    # echo ${MY_CMD}>>local_history.log
    $MY_CMD

    sleep 5m
done

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

# MY_CMD="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type just_test --plot_be_mode single_augmentation --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --gray_train no --gray_test no --augmentation ReCrop_Hflip_Bri --augmentation_prob 1.0 0.5 0.0 0.2 --load_model --load_model_path unlearnable_cleantrain_42873265_1_20211228104344_0.5_512_1000_final_model --not_shuffle_train_data --mix no --local_dev 2 --no_save"

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

# for i in {4..28..4}
# do
    
#     MY_CMD="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --epsilon 8 --num_steps 20 --step_size 0.8 --attack_type min-min --perturb_type clean_train --train_step 10 --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --shuffle_train_perturb_data --gray_train gray --gray_test freq_cifar10_1024_4class_gray_low_${i} --augmentation ReCrop_Hflip_Bri --load_model --load_model_path unlearnable_cleantrain_42483880_2_20211221153535_0.5_512_1000_final_model --just_test --local_dev 2 --no_save"

#     # echo $MY_CMD
#     # echo ${MY_CMD}>>local_history.log
#     $MY_CMD
# done

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