import numpy as np
import torch


#randomassign_supervised_10class_classwise_32_10_0.8_checkpoint_perturbation_budget1.0_class4_retrain_model_statistics
# randomassign_supervised_10class_classwise_8_10_0.8_budget
#unlearnable_20211025164521_0.5_512_151_checkpoint_perturbation
# supervised_orglabel_4class_perturbation_budget{}.0_class4_orglabel_retrain_model_statistics
# differentiable_20211102231654_0.5_200_512_32.0_30_0.8_perturb_on_random_initial_model_budget32.0_class4_retrain_model_statistics
# random_noise32_perturbation_budget32.0_class4_retrain_model_statistics

# randomassign_supervised_10class_classwise_32_1_0.8
# 32_10_0.8 32_10_1.6 32_1_0.8 8_10_0.8
JOB_INFO="Retrain SimCLR with perturbation from supervised"
MYCOMMEND="python simclr_transfer.py --batch_size 512 --epochs 1000 --arch resnet18 --perturbation_budget 32 --pre_load_name randomassign_supervised_10class_classwise_32_10_0.8_checkpoint_perturbation"

file_path_list = [
# './results/randomassign_supervised_10class_classwise_32_1_0.8_checkpoint_perturbation.pt',
'./results/randomassign_supervised_10class_classwise_32_10_0.8_checkpoint_perturbation.pt',
# './results/randomassign_supervised_10class_classwise_32_10_1.6_checkpoint_perturbation.pt',
# './results/randomassign_supervised_10class_classwise_32_1_0.8_checkpoint_perturbation.pt',
'./results/randomassign_supervised_10class_classwise_8_10_0.8_checkpoint_perturbation.pt',
'./results/unlearnable_20211025164521_0.5_512_151_checkpoint_perturbation.pt',
'./results/differentiable_20211102231654_0.5_200_512_32.0_30_0.8_perturb_on_random_initial_model.pt',
'./results/supervised_orglabel_4class_perturbation.pt',
'./results/random_noise32_perturbation.pt'
]

for file_path in file_path_list:
    perturb = torch.load(file_path).to('cpu').numpy()
    print(file_path, np.mean(np.absolute(perturb)) * 255)