import torch

sampled_class = {0:0,1:1,2:3,3:7}

perturb_10class = torch.load("my_experiments/class_wise_cifar10_32_10_1.6/perturbation.pt")
perturb_4class_list = []

for i in range(4):
    perturb_4class_list.append(perturb_10class[sampled_class[i]])
    print(perturb_4class_list[i].shape)

perturb_4class = torch.stack(perturb_4class_list, dim=0)
print(perturb_4class.shape)

torch.save(perturb_4class, "my_experiments/class_wise_cifar10_32_10_1.6/4class_perturbation.pt")