import pandas as pd

#randomassign_supervised_10class_classwise_32_10_0.8_checkpoint_perturbation_budget1.0_class4_retrain_model_statistics
# randomassign_supervised_10class_classwise_8_10_0.8_budget
#unlearnable_20211025164521_0.5_512_151_checkpoint_perturbation
# supervised_orglabel_4class_perturbation_budget{}.0_class4_orglabel_retrain_model_statistics
# differentiable_20211102231654_0.5_200_512_32.0_30_0.8_perturb_on_random_initial_model_budget32.0_class4_retrain_model_statistics
# random_noise32_perturbation_budget32.0_class4_retrain_model_statistics

# budget_list = [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]
budget_list = [1,2,4,8,16,32]

for budget in budget_list:
    print('*{}'.format(budget))

for budget in budget_list:
    csv_data = pd.read_csv('./results/random_noise32_perturbation_budget{}.0_class4_retrain_model_statistics.csv'.format(budget))
    csv_data = csv_data.values
    print(csv_data[999,4])
    # print('*{}'.format(budget))