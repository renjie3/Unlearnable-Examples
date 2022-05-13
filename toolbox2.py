import numpy as np
import torch
from torch.autograd import Variable
from simclr import test_ssl, train_simclr, train_simclr_noise, train_simclr_noise_return_loss_tensor, train_simclr_noise_return_loss_tensor_eot, train_simclr_noise_return_loss_tensor_target_task, train_simclr_noise_return_loss_tensor_full_gpu, get_dbindex_loss, train_simclr_noise_return_loss_tensor_no_eval, train_simclr_noise_return_loss_tensor_no_eval_pos_only, get_linear_noise_dbindex_loss
from utils import train_diff_transform, train_diff_transform2, train_transform_no_totensor
from byol_utils import train_byol_noise_return_loss_tensor
from simsiam_utils import train_simsiam_noise_return_loss_tensor
from moco import train_moco_noise_return_loss_tensor
from torchvision import transforms

import time

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class PerturbationTool():
    def __init__(self, seed=0, epsilon=0.03137254901, num_steps=20, step_size=0.00784313725):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.seed = seed
        np.random.seed(seed)

    def random_noise(self, noise_shape=[10, 3, 32, 32]):
        # print("device", device)
        random_noise = torch.FloatTensor(*noise_shape).uniform_(-self.epsilon, self.epsilon).to(device)
        return random_noise

    def min_min_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, simclr_weight=1, linear_noise_dbindex_weight=0):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        # perturb_img = Variable(images.data + random_noise, requires_grad=True)

        # perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        perturb = Variable(random_noise, requires_grad=True)
        perturb_img = torch.clamp(images.data + perturb, 0, 1)
        
        eta = random_noise
        for _step in range(self.num_steps):
            # print(_step)
            opt = torch.optim.SGD([perturb], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                if hasattr(model, 'classify'):
                    model.classify = True
                if simclr_weight != 0:
                    logits = model(perturb_img)
                    simclr_loss = criterion(logits, labels)
                else:
                    simclr_loss = 0
                if linear_noise_dbindex_weight != 0:
                    # input('check linear_noise_dbindex_weight')
                    linear_noise_dbindex_loss = get_linear_noise_dbindex_loss(perturb, labels)
                else:
                    linear_noise_dbindex_loss = 0
                loss = simclr_loss * simclr_weight + linear_noise_dbindex_loss * linear_noise_dbindex_weight
                # print(type(loss))
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
            perturb_img.retain_grad()
            loss.backward()
            eta = self.step_size * perturb.grad.data.sign() * (-1)
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)

            perturb = Variable(eta, requires_grad=True)
            perturb_img = torch.clamp(images.data + perturb, 0, 1)

        return perturb_img, eta

    def feature_space_distribution_attack(self, group_model, images, target_feature_space, random_noise):
        # if random_noise is None:
        #     random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)
        # model.eval()
        # for param in model.parameters():
        #     param.requires_grad = False
        criterion = torch.nn.MSELoss(reduce=True, size_average=True)
        group_feature = []

        perturb = Variable(random_noise, requires_grad=True)
        perturb_img = torch.clamp(perturb + images.data, 0, 1)
        for step_idx in range(self.num_steps):
            opt = torch.optim.SGD([perturb], lr=1e-3)
            opt.zero_grad()
            loss = 0

            # Here MSE loss between feature space (1) random_initialized_model(perturbation+images) (2) well_trained_simclr(images)
            for model in group_model:
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                model.zero_grad()
                feature_perturb, _ = model(perturb_img)
                if step_idx == 0:
                    group_feature.append(feature_perturb)
                loss += criterion(feature_perturb, target_feature_space)

            perturb.retain_grad()
            loss.backward()
            eta = self.step_size * perturb.grad.data.sign() * (-1)
            perturb_img = perturb_img.data + eta
            perturb = Variable(torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon), requires_grad=True)
            perturb_img = images.data + perturb
            perturb_img = torch.clamp(perturb_img, 0, 1)
            # print("feature_space_distribution_attack:", loss.item())
        # input()

        return perturb_img, perturb, loss.item(), group_feature

    def min_min_attack_noise_variable(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb = Variable(random_noise, requires_grad=True)
        # perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        perturb_img = torch.clamp(perturb + images.data, 0, 1)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                if hasattr(model, 'classify'):
                    model.classify = True
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
            perturb.retain_grad()
            loss.backward()
            eta = self.step_size * perturb.grad.data.sign() * (-1)
            perturb_img = perturb_img.data + eta
            perturb = Variable(torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon), requires_grad=True)
            perturb_img = images.data + perturb
            perturb_img = torch.clamp(perturb_img, 0, 1)
        #     print("min_min_attack_noise_variable:", loss.item())
        # input()

        return perturb_img, perturb

    def min_min_attack_simclr(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None):
        if random_noise is None:
            random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb = Variable(random_noise, requires_grad=True)
        perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
        perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)
        # perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
        # eta1 = perturb_img1.data - pos_samples_1.data
        # perturb_img2 = torch.clamp(pos_samples_2.data + eta1, 0, 1)
        # eta2 = perturb_img2.data - pos_samples_2.data
        # perturb_img1 = torch.clamp(pos_samples_1.data + eta2, 0, 1)
        # perturb_img2 = torch.clamp(pos_samples_2.data + eta2, 0, 1)

        # perturb_img1 = pos_samples_1.data + perturb
        # perturb_img2 = pos_samples_2.data + perturb
        # perturb_img2 = Variable(torch.clamp(perturb_img2, 0, 1), requires_grad=True)
        eta = random_noise
        train_loss_batch_sum, train_loss_batch_count = 0, 0
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb], lr=1e-3)
            # opt.zero_grad()
            model.zero_grad()
            # if isinstance(criterion, torch.nn.CrossEntropyLoss):
            #     if hasattr(model, 'classify'):
            #         model.classify = True
            #     logits = model(perturb_img)
            #     loss = criterion(logits, labels)
            # else:
            #     logits, loss = criterion(model, perturb_img, labels, optimizer)
            # perturb_img.retain_grad()
            # loss.backward()
            train_loss_batch = train_simclr_noise(model, pos_samples_1, pos_samples_2, perturb, opt, batch_size, temperature)
            train_loss_batch_sum += train_loss_batch * perturb.shape[0]
            train_loss_batch_count += perturb.shape[0]
            # eta = self.step_size * perturb_img.grad.data.sign() * (-1)
            # perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            # eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            # perturb_img = Variable(images.data + eta, requires_grad=True)
            # perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

            eta_step = self.step_size * perturb.grad.data.sign() * (-1) # why here used sign?? renjie3
            sign_print = perturb.grad.data.sign() * (-1)
            # print("+:", np.sum(sign_print.cpu().numpy() == 1))
            # print("-:", np.sum(sign_print.cpu().numpy() == -1))
            # print("0:", np.sum(sign_print.cpu().numpy() == 0))
            perturb_img1 = perturb_img1.data + eta_step
            eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
            perturb_img2 = perturb_img2.data + eta_step
            eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)
            diff_eta = eta1 - eta2
            print(diff_eta.cpu().numpy())
            eta = (eta1 + eta2) / 2
            # print("pos1 and pos2 diff: ", np.sum((eta1 - eta2).cpu().numpy()))
            perturb = Variable(eta, requires_grad=True)
            # perturb_img2 = perturb_img2.data + eta1
            # perturb = Variable(torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon), requires_grad=True)
            perturb_img1 = pos_samples_1.data + perturb
            perturb_img1 = torch.clamp(perturb_img1, 0, 1)
            perturb_img2 = pos_samples_2.data + perturb
            perturb_img2 = torch.clamp(perturb_img2, 0, 1)
            # perturb_img1 = pos_samples_1.data + perturb
            # perturb_img2 = pos_samples_2.data + perturb

            # perturb_img = Variable(images.data + eta, requires_grad=True)
            # perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        # print("eta all")
        # print("+:", np.sum(eta.cpu().numpy() > 0.0313724))
        # print("-:", np.sum(eta.cpu().numpy() < -0.0313724))
        # print(">0:", np.sum(eta.cpu().numpy() > 0))
        # print("<0:", np.sum(eta.cpu().numpy() < 0))
        # print("=0:", np.sum(eta.cpu().numpy() == 0))

        return None, eta, train_loss_batch_sum / float(train_loss_batch_count)

    def min_min_attack_simclr2(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None):
    # after verified that using perturb as variable to train is working 
        if random_noise is None:
            random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb = Variable(random_noise, requires_grad=True)
        perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
        perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)

        eta = random_noise
        train_loss_batch_sum, train_loss_batch_count = 0, 0
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            # if isinstance(criterion, torch.nn.CrossEntropyLoss):
            #     if hasattr(model, 'classify'):
            #         model.classify = True
            #     logits = model(perturb_img)
            #     loss = criterion(logits, labels)
            # else:
            #     logits, loss = criterion(model, perturb_img, labels, optimizer)
            # perturb_img.retain_grad()
            # loss.backward()
            train_loss_batch = train_simclr_noise(model, pos_samples_1, pos_samples_2, perturb, opt, batch_size, temperature)
            train_loss_batch_sum += train_loss_batch * perturb.shape[0]
            train_loss_batch_count += perturb.shape[0]

            eta_step = self.step_size * perturb.grad.data.sign() * (-1) # why here used sign?? renjie3
            sign_print = perturb.grad.data.sign() * (-1)
            # print("+:", np.sum(sign_print.cpu().numpy() == 1))
            # print("-:", np.sum(sign_print.cpu().numpy() == -1))
            # print("0:", np.sum(sign_print.cpu().numpy() == 0))
            perturb_img1 = perturb_img1.data + eta_step
            eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
            perturb_img2 = perturb_img2.data + eta_step
            eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)
            eta = (eta1 + eta2) / 2
            # print("pos1 and pos2 diff: ", np.sum((eta1 - eta2).cpu().numpy()))
            perturb = Variable(eta, requires_grad=True)
            perturb_img1 = pos_samples_1.data + perturb
            perturb_img1 = torch.clamp(perturb_img1, 0, 1)
            perturb_img2 = pos_samples_2.data + perturb
            perturb_img2 = torch.clamp(perturb_img2, 0, 1)
        # print("eta all")
        # print("+:", np.sum(eta.cpu().numpy() > 0.0313724))
        # print("-:", np.sum(eta.cpu().numpy() < -0.0313724))
        # print(">0:", np.sum(eta.cpu().numpy() > 0))
        # print("<0:", np.sum(eta.cpu().numpy() < 0))
        # print("=0:", np.sum(eta.cpu().numpy() == 0))

        return None, eta, train_loss_batch_sum / float(train_loss_batch_count)

    def min_min_attack_simclr_return_loss_tensor(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None, flag_strong_aug=True, target_task="non_eot", noise_after_transform=False, split_transform=False, pytorch_aug=False):
    # after verified that using perturb as variable to train is working 
        if random_noise is None:
            random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb = Variable(random_noise, requires_grad=True)
        perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
        perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)

        eta = random_noise
        train_loss_batch_sum, train_loss_batch_count = 0, 0
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()

            # model(perturb_img1)
            # print('check3')
            # perturb.retain_grad()
            # loss.backward()
            if target_task == "non_eot":
                loss = train_simclr_noise_return_loss_tensor(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug, noise_after_transform=noise_after_transform, split_transform=split_transform, pytorch_aug=pytorch_aug)
            else:
                loss = train_simclr_noise_return_loss_tensor_target_task(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug, target_task)
            perturb.retain_grad()
            loss.backward()
            train_loss_batch = loss.item()/float(perturb.shape[0])
            train_loss_batch_sum += train_loss_batch * perturb.shape[0]
            train_loss_batch_count += perturb.shape[0]

            eta_step = self.step_size * perturb.grad.data.sign() * (-1)
            sign_print = perturb.grad.data.sign() * (-1)
            # print("+:", np.sum(sign_print.cpu().numpy() == 1))
            # print("-:", np.sum(sign_print.cpu().numpy() == -1))
            # print("0:", np.sum(sign_print.cpu().numpy() == 0))
            perturb_img1 = perturb_img1.data + eta_step
            eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
            perturb_img2 = perturb_img2.data + eta_step
            eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)
            # diff_eta = eta1 - eta2
            # print(diff_eta.cpu().numpy())
            eta = (eta1 + eta2) / 2
            print(np.mean(np.absolute(eta.mean(dim=0).to('cpu').numpy())) * 255)
            # print("pos1 and pos2 diff: ", np.sum((eta1 - eta2).cpu().numpy()))
            perturb = Variable(eta, requires_grad=True)
            perturb_img1 = pos_samples_1.data + perturb
            perturb_img1 = torch.clamp(perturb_img1, 0, 1)
            perturb_img2 = pos_samples_2.data + perturb
            perturb_img2 = torch.clamp(perturb_img2, 0, 1)
            print("min_min_attack_simclr_return_loss_tensor:", loss.item())
        # print("eta all")
        # print("+:", np.sum(eta.cpu().numpy() > 0.0313724))
        # print("-:", np.sum(eta.cpu().numpy() < -0.0313724))
        # print(">0:", np.sum(eta.cpu().numpy() > 0))
        # print("<0:", np.sum(eta.cpu().numpy() < 0))
        # print("=0:", np.sum(eta.cpu().numpy() == 0))

        return None, eta, train_loss_batch_sum / float(train_loss_batch_count)
    
    
    def min_min_attack_simclr_return_loss_tensor_model_free(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None, flag_strong_aug=True, target_task="non_eot", noise_after_transform=False):
    # after verified that using perturb as variable to train is working 
        if random_noise is None:
            random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb = Variable(random_noise, requires_grad=True)
        perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
        perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)

        eta = random_noise
        train_loss_batch_sum, train_loss_batch_count = 0, 0
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            # perturb.retain_grad()
            # loss.backward()
            if target_task == "non_eot":
                loss = train_simclr_noise_return_loss_tensor(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug)
            elif target_task in ["pos", "neg"]:
                loss = train_simclr_noise_return_loss_tensor_target_task(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug, target_task)
            else:
                raise("Wrong target_task")
            perturb.retain_grad()
            loss.backward()
            train_loss_batch = loss.item()/float(perturb.shape[0])
            train_loss_batch_sum += train_loss_batch * perturb.shape[0]
            train_loss_batch_count += perturb.shape[0]

            eta_step = self.step_size * perturb.grad.data.sign() * (-1)
            perturb_img1 = perturb_img1.data + eta_step
            eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
            perturb_img2 = perturb_img2.data + eta_step
            eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)
            eta = (eta1 + eta2) / 2
            perturb = Variable(eta, requires_grad=True)
            perturb_img1 = pos_samples_1.data + perturb
            perturb_img1 = torch.clamp(perturb_img1, 0, 1)
            perturb_img2 = pos_samples_2.data + perturb
            perturb_img2 = torch.clamp(perturb_img2, 0, 1)
            print("min_min_attack_simclr_return_loss_tensor:", loss.item())

        return None, eta, train_loss_batch_sum / float(train_loss_batch_count)
    
    def min_min_attack_simclr_return_loss_tensor_model_group(self, pos_samples_1, pos_samples_2, labels, model_group, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None, flag_strong_aug=True, target_task="non_eot", step_size_schedule=8.0 / 255.0):
    # after verified that using perturb as variable to train is working 
        if random_noise is None:
            random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb = Variable(random_noise, requires_grad=True)
        perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
        perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)

        eta = random_noise
        train_loss_batch_sum, train_loss_batch_count = 0, 0
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb], lr=1e-3)
            opt.zero_grad()
            for idx_model, model in enumerate(model_group):
                model.zero_grad()
            # perturb.retain_grad()
            # loss.backward()
            loss = 0
            for model in model_group:
                if target_task == "non_eot":
                    loss += train_simclr_noise_return_loss_tensor(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug)
                else:
                    loss += train_simclr_noise_return_loss_tensor_target_task(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug, target_task)
            perturb.retain_grad()
            loss.backward()
            train_loss_batch = loss.item()/float(perturb.shape[0])
            train_loss_batch_sum += train_loss_batch * perturb.shape[0]
            train_loss_batch_count += perturb.shape[0] * len(model_group)

            print("step_size_schedule", step_size_schedule)
            eta_step = step_size_schedule * perturb.grad.data.sign() * (-1) # why here used sign?? renjie3
            sign_print = perturb.grad.data.sign() * (-1)
            perturb_img1 = perturb_img1.data + eta_step
            eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
            perturb_img2 = perturb_img2.data + eta_step
            eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)
            # diff_eta = eta1 - eta2
            # print(diff_eta.cpu().numpy())
            eta = (eta1 + eta2) / 2
            # print("pos1 and pos2 diff: ", np.sum((eta1 - eta2).cpu().numpy()))
            perturb = Variable(eta, requires_grad=True)
            perturb_img1 = pos_samples_1.data + perturb
            perturb_img1 = torch.clamp(perturb_img1, 0, 1)
            perturb_img2 = pos_samples_2.data + perturb
            perturb_img2 = torch.clamp(perturb_img2, 0, 1)
            print("min_min_attack_simclr_return_loss_tensor:", loss.item())

        return None, eta, train_loss_batch_sum / float(train_loss_batch_count)

    def min_min_attack_byol_return_loss_tensor_eot_v1(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None, flag_strong_aug=True, noise_after_transform=False, eot_size=30, one_gpu_eot_times=1, cross_eot=False, split_transform=False, pytorch_aug=False, dbindex_weight=0, single_noise_after_transform=False, no_eval=False, dbindex_label_index=1, noise_dbindex_weight=0, simclr_weight=1, augmentation_prob=None, clean_weight=0, noise_simclr_weight=0, double_perturb=False, upper_half_linear=False, batch_simclr_mask=None, batch_linear_noise=None, mask_linear_constraint=False, mask1=None, mask2=None, mask_linear_noise_range=[2, 8], use_supervised_g=False, g_net=None, supervised_criterion=None, supervised_weight=0, supervised_transform_train=None, linear_noise_dbindex_weight=0, linear_noise_dbindex_index=1, linear_noise_dbindex_weight2=0, linear_noise_dbindex_index2=2, use_mean_dbindex=True, use_normalized=True, noise_centroids=None, modify_dbindex='', two_stage_PGD=False, model_g_augment_first=False, dbindex_augmentation=False, linear_xnoise_dbindex_weight=0, linear_xnoise_dbindex_index=1):
    # v1 means it can repeat min_min_attack many times serially and average the results.
        if not two_stage_PGD:
            if random_noise is None:
                random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

            perturb = Variable(random_noise, requires_grad=True)

            eta = random_noise
            train_loss_batch_sum, train_loss_batch_count = 0, 0
            for _ in range(self.num_steps):

                start = time.time()

                eot_grad = torch.zeros(perturb.shape, dtype=torch.float).to(device)
                eot_loss = 0

                # perturb_org = torch.clamp(pos_samples_1.data + perturb, 0, 1)

                for i_eot in range(eot_size):
                    time0 = time.time()
                    # if upper_half_linear:
                    #     _perturb = perturb * batch_simclr_mask + batch_linear_noise
                    #     perturb_img1 = torch.clamp(pos_samples_1.data + _perturb, 0, 1)
                    #     perturb_img2 = torch.clamp(pos_samples_2.data + _perturb, 0, 1)
                    # elif mask_linear_constraint:
                    #     _perturb1 = torch.clamp(perturb, -self.epsilon, -self.epsilon*0.25,) * mask1
                    #     _perturb2 = torch.clamp(perturb, self.epsilon*0.25, self.epsilon,) * mask2
                    #     _perturb = _perturb1 + _perturb2
                    #     perturb_img1 = torch.clamp(pos_samples_1.data + _perturb, 0, 1)
                    #     perturb_img2 = torch.clamp(pos_samples_2.data + _perturb, 0, 1)
                    # else:
                    #     perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
                    #     perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)
                    perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
                    perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)
                    opt = torch.optim.SGD([perturb], lr=1e-3)
                    opt.zero_grad()
                    model.zero_grad()

                    # if dbindex_weight != 0:
                    #     dbindex_loss = get_dbindex_loss(model, perturb_img1, labels, [4], True, True, dbindex_label_index, x2=perturb_img2, use_aug=True)
                    # else:
                    #     dbindex_loss = 0

                    # if noise_dbindex_weight != 0:
                    #     perturb1 = torch.clamp(0.5 + perturb, 0, 1)
                    #     perturb2 = torch.clamp(0.5 + perturb, 0, 1)
                    #     noise_dbindex_loss = get_dbindex_loss(model, perturb1, labels, [4], True, True, dbindex_label_index, x2=perturb2, use_aug=True)
                    # else:
                    #     noise_dbindex_loss = 0

                    if simclr_weight != 0:
                        
                            
                        simclr_loss = train_byol_noise_return_loss_tensor(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug, noise_after_transform=noise_after_transform, pytorch_aug=pytorch_aug, single_noise_after_transform=single_noise_after_transform, no_eval=no_eval, augmentation_prob=augmentation_prob, org_pos1=pos_samples_1, org_pos2=pos_samples_2, clean_weight=clean_weight)
                    else:
                        simclr_loss = 0

                    # if noise_simclr_weight != 0:
                    #     perturb1 = torch.clamp(0.5 + perturb, 0, 1)
                    #     perturb2 = torch.clamp(0.5 + perturb, 0, 1)
                    #     noise_simclr_loss = train_simclr_noise_return_loss_tensor(model, perturb1, perturb2, opt, batch_size, temperature, flag_strong_aug, noise_after_transform=noise_after_transform, pytorch_aug=pytorch_aug, single_noise_after_transform=single_noise_after_transform, no_eval=no_eval, augmentation_prob=augmentation_prob, org_pos1=pos_samples_1, org_pos2=pos_samples_2, clean_weight=clean_weight)
                    # else:
                    #     noise_simclr_loss = 0

                    if linear_noise_dbindex_weight != 0:
                        if not dbindex_augmentation:
                            linear_noise_dbindex_loss = get_linear_noise_dbindex_loss(perturb, labels[:, linear_noise_dbindex_index], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids, modify_dbindex=modify_dbindex)
                        else:
                            linear_noise_dbindex_loss = get_linear_noise_dbindex_loss(supervised_transform_train(perturb), labels[:, linear_noise_dbindex_index], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids, modify_dbindex=modify_dbindex)
                    else:
                        linear_noise_dbindex_loss = 0

                    if linear_xnoise_dbindex_weight != 0:
                        linear_xnoise_dbindex_loss = get_linear_noise_dbindex_loss(torch.clamp(pos_samples_1.data + perturb, 0, 1), labels[:, linear_noise_dbindex_index], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids, modify_dbindex=modify_dbindex)
                    else:
                        linear_xnoise_dbindex_loss = 0

                    # if linear_noise_dbindex_weight2 != 0:
                    #     linear_noise_dbindex_loss2 = get_linear_noise_dbindex_loss(perturb, labels[:, linear_noise_dbindex_index2], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids)
                    # else:
                    #     linear_noise_dbindex_loss2 = 0

                    if use_supervised_g != 0:
                        g_net.zero_grad()
                        if not model_g_augment_first:
                            inputs = supervised_transform_train(torch.cat([perturb_img1, perturb_img2], dim=0))
                            feature, outputs = g_net(inputs)
                            supervised_loss = supervised_criterion(outputs, labels[:, 1].repeat((2,)))
                        else:
                            inputs = torch.clamp(supervised_transform_train(torch.clamp(pos_samples_1.data, 0, 1)) + perturb, 0, 1)
                            feature, outputs = g_net(inputs)
                            supervised_loss = supervised_criterion(outputs, labels[:, 1])
                    else:
                        supervised_loss = 0

                    # print(linear_xnoise_dbindex_loss)
                    loss = simclr_loss * simclr_weight + supervised_weight * supervised_loss + linear_noise_dbindex_loss * linear_noise_dbindex_weight + linear_xnoise_dbindex_loss * linear_xnoise_dbindex_weight
                    
                    perturb.retain_grad()
                    loss.backward()
                    
                    eot_grad += perturb.grad.data
                    eot_loss += loss.item()
                
                eot_loss /= eot_size
                eot_grad /= eot_size

                train_loss_batch = loss.item()/float(perturb.shape[0])
                train_loss_batch_sum += train_loss_batch * perturb.shape[0]
                train_loss_batch_count += perturb.shape[0]

                eta_step = self.step_size * eot_grad.sign() * (-1)
                sign_print = perturb.grad.data.sign() * (-1)
                # print("+:", np.sum(sign_print.cpu().numpy() == 1))
                # print("-:", np.sum(sign_print.cpu().numpy() == -1))
                # print("0:", np.sum(sign_print.cpu().numpy() == 0))
                perturb_img1 = perturb_img1.data + eta_step
                eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
                perturb_img2 = perturb_img2.data + eta_step
                eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)
                # diff_eta = eta1 - eta2
                # print(diff_eta.cpu().numpy())
                eta = (eta1 + eta2) / 2
                # if mask_linear_constraint:
                #     if mask_linear_noise_range[0] >= mask_linear_noise_range[1]:
                #         raise('wrong parameter mask_linear_noise_range')
                #     _eta1 = torch.clamp(eta, -mask_linear_noise_range[1] / 255.0, -mask_linear_noise_range[0] / 255.0,) * mask1
                #     _eta2 = torch.clamp(eta, mask_linear_noise_range[0] / 255.0, mask_linear_noise_range[1] / 255.0,) * mask2
                #     eta = _eta1 + _eta2
                # print("pos1 and pos2 diff: ", np.sum((eta1 - eta2).cpu().numpy()))
                perturb = Variable(eta, requires_grad=True)
                # perturb_img1 = pos_samples_1.data + perturb
                # perturb_img1 = torch.clamp(perturb_img1, 0, 1)
                # perturb_img2 = pos_samples_2.data + perturb
                # perturb_img2 = torch.clamp(perturb_img2, 0, 1)
                print("min_min_attack_byol_return_loss_tensor_eot_v1:", eot_loss)
                # input()

                end = time.time()

                print("time: ", end - start)
            # print("eta all")
            # print("+:", np.sum(eta.cpu().numpy() > 0.0313724))
            # print("-:", np.sum(eta.cpu().numpy() < -0.0313724))
            # print(">0:", np.sum(eta.cpu().numpy() > 0))
            # print("<0:", np.sum(eta.cpu().numpy() < 0))
            # print("=0:", np.sum(eta.cpu().numpy() == 0))

            return None, eta, train_loss_batch_sum / float(train_loss_batch_count)
        else:

            raise('require development')
            
            if random_noise is None:
                random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

            perturb = Variable(random_noise, requires_grad=True)

            eta = random_noise
            train_loss_batch_sum, train_loss_batch_count = 0, 0
            for _ in range(self.num_steps):

                start = time.time()

                eot_grad = torch.zeros(perturb.shape, dtype=torch.float).to(device)
                eot_linear_noise_grad = torch.zeros(perturb.shape, dtype=torch.float).to(device)
                eot_loss = 0

                for i_eot in range(eot_size):
                    opt = torch.optim.SGD([perturb], lr=1e-3)
                    opt.zero_grad()
                    model.zero_grad()

                    perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
                    perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)
                            
                    simclr_loss = train_simclr_noise_return_loss_tensor(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug, noise_after_transform=False, pytorch_aug=pytorch_aug, single_noise_after_transform=single_noise_after_transform, no_eval=no_eval, augmentation_prob=augmentation_prob, org_pos1=pos_samples_1, org_pos2=pos_samples_2, clean_weight=clean_weight)

                    loss = simclr_loss
                    
                    perturb.retain_grad()
                    loss.backward()
                    
                    eot_grad += perturb.grad.data
                    eot_loss += loss.item()

                    opt = torch.optim.SGD([perturb], lr=1e-3)
                    opt.zero_grad()
                    model.zero_grad()

                    if not dbindex_augmentation:
                        linear_noise_dbindex_loss = get_linear_noise_dbindex_loss(perturb, labels[:, linear_noise_dbindex_index], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids, modify_dbindex=modify_dbindex)
                    else:
                        linear_noise_dbindex_loss = get_linear_noise_dbindex_loss(supervised_transform_train(perturb), labels[:, linear_noise_dbindex_index], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids, modify_dbindex=modify_dbindex)

                    loss = linear_noise_dbindex_loss
                    
                    perturb.retain_grad()
                    loss.backward()
                    
                    eot_linear_noise_grad += perturb.grad.data
                
                eot_loss /= eot_size
                eot_grad /= eot_size

                train_loss_batch = loss.item()/float(perturb.shape[0])
                train_loss_batch_sum += train_loss_batch * perturb.shape[0]
                train_loss_batch_count += perturb.shape[0]

                eta_step = self.step_size * eot_grad.sign() * (-1) + linear_noise_dbindex_weight * self.step_size * eot_linear_noise_grad.sign() * (-1)
                sign_print = perturb.grad.data.sign() * (-1)

                perturb_img1 = perturb_img1.data + eta_step
                eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
                perturb_img2 = perturb_img2.data + eta_step
                eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)

                eta = (eta1 + eta2) / 2

                # print("pos1 and pos2 diff: ", np.sum((eta1 - eta2).cpu().numpy()))
                perturb = Variable(eta, requires_grad=True)

                print("min_min_attack_byol_return_loss_tensor_eot_v1:", eot_loss)

                end = time.time()
                print("time: ", end - start)
            # print("eta all")
            # print("+:", np.sum(eta.cpu().numpy() > 0.0313724))
            # print("-:", np.sum(eta.cpu().numpy() < -0.0313724))
            # print(">0:", np.sum(eta.cpu().numpy() > 0))
            # print("<0:", np.sum(eta.cpu().numpy() < 0))
            # print("=0:", np.sum(eta.cpu().numpy() == 0))

            return None, eta, train_loss_batch_sum / float(train_loss_batch_count)

    def min_min_attack_simsiam_return_loss_tensor_eot_v1(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None, flag_strong_aug=True, noise_after_transform=False, eot_size=30, one_gpu_eot_times=1, cross_eot=False, split_transform=False, pytorch_aug=False, dbindex_weight=0, single_noise_after_transform=False, no_eval=False, dbindex_label_index=1, noise_dbindex_weight=0, simclr_weight=1, augmentation_prob=None, clean_weight=0, noise_simclr_weight=0, double_perturb=False, upper_half_linear=False, batch_simclr_mask=None, batch_linear_noise=None, mask_linear_constraint=False, mask1=None, mask2=None, mask_linear_noise_range=[2, 8], use_supervised_g=False, g_net=None, supervised_criterion=None, supervised_weight=0, supervised_transform_train=None, linear_noise_dbindex_weight=0, linear_noise_dbindex_index=1, linear_noise_dbindex_weight2=0, linear_noise_dbindex_index2=2, use_mean_dbindex=True, use_normalized=True, noise_centroids=None, modify_dbindex='', two_stage_PGD=False, model_g_augment_first=False, dbindex_augmentation=False, linear_xnoise_dbindex_weight=0, linear_xnoise_dbindex_index=1):
    # v1 means it can repeat min_min_attack many times serially and average the results.
        if not two_stage_PGD:
            if random_noise is None:
                random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

            perturb = Variable(random_noise, requires_grad=True)

            eta = random_noise
            train_loss_batch_sum, train_loss_batch_count = 0, 0
            for _ in range(self.num_steps):

                start = time.time()

                eot_grad = torch.zeros(perturb.shape, dtype=torch.float).to(device)
                eot_loss = 0

                # perturb_org = torch.clamp(pos_samples_1.data + perturb, 0, 1)

                for i_eot in range(eot_size):
                    time0 = time.time()
                    
                    perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
                    perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)
                    opt = torch.optim.SGD([perturb], lr=1e-3)
                    opt.zero_grad()
                    model.zero_grad()

                    if simclr_weight != 0:
                            
                        simclr_loss = train_simsiam_noise_return_loss_tensor(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug, noise_after_transform=noise_after_transform, pytorch_aug=pytorch_aug, single_noise_after_transform=single_noise_after_transform, no_eval=no_eval, augmentation_prob=augmentation_prob, org_pos1=pos_samples_1, org_pos2=pos_samples_2, clean_weight=clean_weight)
                    else:
                        simclr_loss = 0

                    if linear_noise_dbindex_weight != 0:
                        if not dbindex_augmentation:
                            linear_noise_dbindex_loss = get_linear_noise_dbindex_loss(perturb, labels[:, linear_noise_dbindex_index], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids, modify_dbindex=modify_dbindex)
                        else:
                            linear_noise_dbindex_loss = get_linear_noise_dbindex_loss(supervised_transform_train(perturb), labels[:, linear_noise_dbindex_index], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids, modify_dbindex=modify_dbindex)
                    else:
                        linear_noise_dbindex_loss = 0

                    if linear_xnoise_dbindex_weight != 0:
                        linear_xnoise_dbindex_loss = get_linear_noise_dbindex_loss(torch.clamp(pos_samples_1.data + perturb, 0, 1), labels[:, linear_noise_dbindex_index], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids, modify_dbindex=modify_dbindex)
                    else:
                        linear_xnoise_dbindex_loss = 0

                    if use_supervised_g != 0:
                        g_net.zero_grad()
                        if not model_g_augment_first:
                            inputs = supervised_transform_train(torch.cat([perturb_img1, perturb_img2], dim=0))
                            feature, outputs = g_net(inputs)
                            supervised_loss = supervised_criterion(outputs, labels[:, 1].repeat((2,)))
                        else:
                            inputs = torch.clamp(supervised_transform_train(torch.clamp(pos_samples_1.data, 0, 1)) + perturb, 0, 1)
                            feature, outputs = g_net(inputs)
                            supervised_loss = supervised_criterion(outputs, labels[:, 1])
                    else:
                        supervised_loss = 0

                    # print(linear_xnoise_dbindex_loss)
                    loss = simclr_loss * simclr_weight + supervised_weight * supervised_loss + linear_noise_dbindex_loss * linear_noise_dbindex_weight + linear_xnoise_dbindex_loss * linear_xnoise_dbindex_weight
                    
                    perturb.retain_grad()
                    loss.backward()
                    
                    eot_grad += perturb.grad.data
                    eot_loss += loss.item()
                
                eot_loss /= eot_size
                eot_grad /= eot_size

                train_loss_batch = loss.item()/float(perturb.shape[0])
                train_loss_batch_sum += train_loss_batch * perturb.shape[0]
                train_loss_batch_count += perturb.shape[0]

                eta_step = self.step_size * eot_grad.sign() * (-1)
                sign_print = perturb.grad.data.sign() * (-1)
                # print("+:", np.sum(sign_print.cpu().numpy() == 1))
                # print("-:", np.sum(sign_print.cpu().numpy() == -1))
                # print("0:", np.sum(sign_print.cpu().numpy() == 0))
                perturb_img1 = perturb_img1.data + eta_step
                eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
                perturb_img2 = perturb_img2.data + eta_step
                eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)
                # diff_eta = eta1 - eta2
                # print(diff_eta.cpu().numpy())
                eta = (eta1 + eta2) / 2
                
                perturb = Variable(eta, requires_grad=True)
                
                print("min_min_attack_byol_return_loss_tensor_eot_v1:", eot_loss)
                # input()

                end = time.time()

                print("time: ", end - start)
            

            return None, eta, train_loss_batch_sum / float(train_loss_batch_count)
        else:
            
            if random_noise is None:
                random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

            perturb = Variable(random_noise, requires_grad=True)

            eta = random_noise
            train_loss_batch_sum, train_loss_batch_count = 0, 0
            for _ in range(self.num_steps):

                start = time.time()

                eot_grad = torch.zeros(perturb.shape, dtype=torch.float).to(device)
                eot_linear_noise_grad = torch.zeros(perturb.shape, dtype=torch.float).to(device)
                eot_loss = 0

                for i_eot in range(eot_size):
                    opt = torch.optim.SGD([perturb], lr=1e-3)
                    opt.zero_grad()
                    model.zero_grad()

                    perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
                    perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)
                            
                    simclr_loss = train_simclr_noise_return_loss_tensor(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug, noise_after_transform=False, pytorch_aug=pytorch_aug, single_noise_after_transform=single_noise_after_transform, no_eval=no_eval, augmentation_prob=augmentation_prob, org_pos1=pos_samples_1, org_pos2=pos_samples_2, clean_weight=clean_weight)

                    loss = simclr_loss
                    
                    perturb.retain_grad()
                    loss.backward()
                    
                    eot_grad += perturb.grad.data
                    eot_loss += loss.item()

                    opt = torch.optim.SGD([perturb], lr=1e-3)
                    opt.zero_grad()
                    model.zero_grad()

                    if not dbindex_augmentation:
                        linear_noise_dbindex_loss = get_linear_noise_dbindex_loss(perturb, labels[:, linear_noise_dbindex_index], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids, modify_dbindex=modify_dbindex)
                    else:
                        linear_noise_dbindex_loss = get_linear_noise_dbindex_loss(supervised_transform_train(perturb), labels[:, linear_noise_dbindex_index], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids, modify_dbindex=modify_dbindex)

                    loss = linear_noise_dbindex_loss
                    
                    perturb.retain_grad()
                    loss.backward()
                    
                    eot_linear_noise_grad += perturb.grad.data
                
                eot_loss /= eot_size
                eot_grad /= eot_size

                train_loss_batch = loss.item()/float(perturb.shape[0])
                train_loss_batch_sum += train_loss_batch * perturb.shape[0]
                train_loss_batch_count += perturb.shape[0]

                eta_step = self.step_size * eot_grad.sign() * (-1) + linear_noise_dbindex_weight * self.step_size * eot_linear_noise_grad.sign() * (-1)
                sign_print = perturb.grad.data.sign() * (-1)

                perturb_img1 = perturb_img1.data + eta_step
                eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
                perturb_img2 = perturb_img2.data + eta_step
                eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)

                eta = (eta1 + eta2) / 2

                # print("pos1 and pos2 diff: ", np.sum((eta1 - eta2).cpu().numpy()))
                perturb = Variable(eta, requires_grad=True)

                print("min_min_attack_simclr_return_loss_tensor_eot_v1:", eot_loss)

                end = time.time()
                print("time: ", end - start)
            

            return None, eta, train_loss_batch_sum / float(train_loss_batch_count)

    def min_min_attack_moco_return_loss_tensor_eot_v1(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None, flag_strong_aug=True, noise_after_transform=False, eot_size=30, one_gpu_eot_times=1, cross_eot=False, split_transform=False, pytorch_aug=False, dbindex_weight=0, single_noise_after_transform=False, no_eval=False, dbindex_label_index=1, noise_dbindex_weight=0, simclr_weight=1, augmentation_prob=None, clean_weight=0, noise_simclr_weight=0, double_perturb=False, upper_half_linear=False, batch_simclr_mask=None, batch_linear_noise=None, mask_linear_constraint=False, mask1=None, mask2=None, mask_linear_noise_range=[2, 8], use_supervised_g=False, g_net=None, supervised_criterion=None, supervised_weight=0, supervised_transform_train=None, linear_noise_dbindex_weight=0, linear_noise_dbindex_index=1, linear_noise_dbindex_weight2=0, linear_noise_dbindex_index2=2, use_mean_dbindex=True, use_normalized=True, noise_centroids=None, modify_dbindex='', two_stage_PGD=False, model_g_augment_first=False, dbindex_augmentation=False, linear_xnoise_dbindex_weight=0, linear_xnoise_dbindex_index=1, k_grad=False):
    # v1 means it can repeat min_min_attack many times serially and average the results.
        if not two_stage_PGD:
            if random_noise is None:
                random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

            perturb = Variable(random_noise, requires_grad=True)

            eta = random_noise
            train_loss_batch_sum, train_loss_batch_count = 0, 0
            for _ in range(self.num_steps):

                start = time.time()

                eot_grad = torch.zeros(perturb.shape, dtype=torch.float).to(device)
                eot_loss = 0

                # perturb_org = torch.clamp(pos_samples_1.data + perturb, 0, 1)

                for i_eot in range(eot_size):
                    time0 = time.time()
                    
                    perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
                    perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)
                    opt = torch.optim.SGD([perturb], lr=1e-3)
                    opt.zero_grad()
                    model.zero_grad()

                    if simclr_weight != 0:
                        simclr_loss = train_moco_noise_return_loss_tensor(model, perturb_img1, perturb_img2, k_grad=k_grad)
                    else:
                        simclr_loss = 0

                    if linear_noise_dbindex_weight != 0:
                        if not dbindex_augmentation:
                            linear_noise_dbindex_loss = get_linear_noise_dbindex_loss(perturb, labels[:, linear_noise_dbindex_index], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids, modify_dbindex=modify_dbindex)
                        else:
                            linear_noise_dbindex_loss = get_linear_noise_dbindex_loss(supervised_transform_train(perturb), labels[:, linear_noise_dbindex_index], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids, modify_dbindex=modify_dbindex)
                    else:
                        linear_noise_dbindex_loss = 0

                    if linear_xnoise_dbindex_weight != 0:
                        linear_xnoise_dbindex_loss = get_linear_noise_dbindex_loss(torch.clamp(pos_samples_1.data + perturb, 0, 1), labels[:, linear_noise_dbindex_index], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids, modify_dbindex=modify_dbindex)
                    else:
                        linear_xnoise_dbindex_loss = 0

                    if use_supervised_g != 0:
                        g_net.zero_grad()
                        if not model_g_augment_first:
                            inputs = supervised_transform_train(torch.cat([perturb_img1, perturb_img2], dim=0))
                            feature, outputs = g_net(inputs)
                            supervised_loss = supervised_criterion(outputs, labels[:, 1].repeat((2,)))
                        else:
                            inputs = torch.clamp(supervised_transform_train(torch.clamp(pos_samples_1.data, 0, 1)) + perturb, 0, 1)
                            feature, outputs = g_net(inputs)
                            supervised_loss = supervised_criterion(outputs, labels[:, 1])
                    else:
                        supervised_loss = 0

                    # print(linear_xnoise_dbindex_loss)
                    loss = simclr_loss * simclr_weight + supervised_weight * supervised_loss + linear_noise_dbindex_loss * linear_noise_dbindex_weight + linear_xnoise_dbindex_loss * linear_xnoise_dbindex_weight
                    
                    perturb.retain_grad()
                    loss.backward()
                    
                    eot_grad += perturb.grad.data
                    eot_loss += loss.item()
                
                eot_loss /= eot_size
                eot_grad /= eot_size

                train_loss_batch = loss.item()/float(perturb.shape[0])
                train_loss_batch_sum += train_loss_batch * perturb.shape[0]
                train_loss_batch_count += perturb.shape[0]

                eta_step = self.step_size * eot_grad.sign() * (-1)
                sign_print = perturb.grad.data.sign() * (-1)
                # print("+:", np.sum(sign_print.cpu().numpy() == 1))
                # print("-:", np.sum(sign_print.cpu().numpy() == -1))
                # print("0:", np.sum(sign_print.cpu().numpy() == 0))
                perturb_img1 = perturb_img1.data + eta_step
                eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
                perturb_img2 = perturb_img2.data + eta_step
                eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)
                # diff_eta = eta1 - eta2
                # print(diff_eta.cpu().numpy())
                eta = (eta1 + eta2) / 2
                
                perturb = Variable(eta, requires_grad=True)
                
                print("min_min_attack_moco_return_loss_tensor_eot_v1:", eot_loss)
                # input()

                end = time.time()

                print("time: ", end - start)
            

            return None, eta, train_loss_batch_sum / float(train_loss_batch_count)
        else:
            
            if random_noise is None:
                random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

            perturb = Variable(random_noise, requires_grad=True)

            eta = random_noise
            train_loss_batch_sum, train_loss_batch_count = 0, 0
            for _ in range(self.num_steps):

                start = time.time()

                eot_grad = torch.zeros(perturb.shape, dtype=torch.float).to(device)
                eot_linear_noise_grad = torch.zeros(perturb.shape, dtype=torch.float).to(device)
                eot_loss = 0

                for i_eot in range(eot_size):
                    opt = torch.optim.SGD([perturb], lr=1e-3)
                    opt.zero_grad()
                    model.zero_grad()

                    perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
                    perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)
                            
                    simclr_loss = train_simclr_noise_return_loss_tensor(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug, noise_after_transform=False, pytorch_aug=pytorch_aug, single_noise_after_transform=single_noise_after_transform, no_eval=no_eval, augmentation_prob=augmentation_prob, org_pos1=pos_samples_1, org_pos2=pos_samples_2, clean_weight=clean_weight)

                    loss = simclr_loss
                    
                    perturb.retain_grad()
                    loss.backward()
                    
                    eot_grad += perturb.grad.data
                    eot_loss += loss.item()

                    opt = torch.optim.SGD([perturb], lr=1e-3)
                    opt.zero_grad()
                    model.zero_grad()

                    if not dbindex_augmentation:
                        linear_noise_dbindex_loss = get_linear_noise_dbindex_loss(perturb, labels[:, linear_noise_dbindex_index], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids, modify_dbindex=modify_dbindex)
                    else:
                        linear_noise_dbindex_loss = get_linear_noise_dbindex_loss(supervised_transform_train(perturb), labels[:, linear_noise_dbindex_index], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids, modify_dbindex=modify_dbindex)

                    loss = linear_noise_dbindex_loss
                    
                    perturb.retain_grad()
                    loss.backward()
                    
                    eot_linear_noise_grad += perturb.grad.data
                
                eot_loss /= eot_size
                eot_grad /= eot_size

                train_loss_batch = loss.item()/float(perturb.shape[0])
                train_loss_batch_sum += train_loss_batch * perturb.shape[0]
                train_loss_batch_count += perturb.shape[0]

                eta_step = self.step_size * eot_grad.sign() * (-1) + linear_noise_dbindex_weight * self.step_size * eot_linear_noise_grad.sign() * (-1)
                sign_print = perturb.grad.data.sign() * (-1)

                perturb_img1 = perturb_img1.data + eta_step
                eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
                perturb_img2 = perturb_img2.data + eta_step
                eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)

                eta = (eta1 + eta2) / 2

                # print("pos1 and pos2 diff: ", np.sum((eta1 - eta2).cpu().numpy()))
                perturb = Variable(eta, requires_grad=True)

                print("min_min_attack_moco_return_loss_tensor_eot_v1:", eot_loss)

                end = time.time()
                print("time: ", end - start)
            

            return None, eta, train_loss_batch_sum / float(train_loss_batch_count)

    def min_min_attack_simclr_return_loss_tensor_eot_v1(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None, flag_strong_aug=True, noise_after_transform=False, eot_size=30, one_gpu_eot_times=1, cross_eot=False, split_transform=False, pytorch_aug=False, dbindex_weight=0, single_noise_after_transform=False, no_eval=False, dbindex_label_index=1, noise_dbindex_weight=0, simclr_weight=1, augmentation_prob=None, clean_weight=0, noise_simclr_weight=0, double_perturb=False, upper_half_linear=False, batch_simclr_mask=None, batch_linear_noise=None, mask_linear_constraint=False, mask1=None, mask2=None, mask_linear_noise_range=[2, 8], use_supervised_g=False, g_net=None, supervised_criterion=None, supervised_weight=0, supervised_transform_train=None, linear_noise_dbindex_weight=0, linear_noise_dbindex_index=1, linear_noise_dbindex_weight2=0, linear_noise_dbindex_index2=2, use_mean_dbindex=True, use_normalized=True, noise_centroids=None, modify_dbindex='', two_stage_PGD=False, model_g_augment_first=False, dbindex_augmentation=False, linear_xnoise_dbindex_weight=0, linear_xnoise_dbindex_index=1, reverse_code=False, crop_linear_noise=False, linear_noise_filter=False):
    # v1 means it can repeat min_min_attack many times serially and average the results.
        if not two_stage_PGD:
            if random_noise is None:
                random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

            perturb = Variable(random_noise, requires_grad=True)

            eta = random_noise
            train_loss_batch_sum, train_loss_batch_count = 0, 0
            for _ in range(self.num_steps):

                start = time.time()

                eot_grad = torch.zeros(perturb.shape, dtype=torch.float).to(device)
                eot_loss = 0

                # perturb_org = torch.clamp(pos_samples_1.data + perturb, 0, 1)

                for i_eot in range(eot_size):
                    time0 = time.time()
                    if upper_half_linear:
                        _perturb = perturb * batch_simclr_mask + batch_linear_noise
                        perturb_img1 = torch.clamp(pos_samples_1.data + _perturb, 0, 1)
                        perturb_img2 = torch.clamp(pos_samples_2.data + _perturb, 0, 1)
                    elif mask_linear_constraint:
                        _perturb1 = torch.clamp(perturb, -self.epsilon, -self.epsilon*0.25,) * mask1
                        _perturb2 = torch.clamp(perturb, self.epsilon*0.25, self.epsilon,) * mask2
                        _perturb = _perturb1 + _perturb2
                        perturb_img1 = torch.clamp(pos_samples_1.data + _perturb, 0, 1)
                        perturb_img2 = torch.clamp(pos_samples_2.data + _perturb, 0, 1)
                    else:
                        perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
                        perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)
                    opt = torch.optim.SGD([perturb], lr=1e-3)
                    opt.zero_grad()
                    model.zero_grad()

                    if dbindex_weight != 0:
                        dbindex_loss = get_dbindex_loss(model, perturb_img1, labels, [4], True, True, dbindex_label_index, x2=perturb_img2, use_aug=True)
                    else:
                        dbindex_loss = 0

                    if noise_dbindex_weight != 0:
                        perturb1 = torch.clamp(0.5 + perturb, 0, 1)
                        perturb2 = torch.clamp(0.5 + perturb, 0, 1)
                        noise_dbindex_loss = get_dbindex_loss(model, perturb1, labels, [4], True, True, dbindex_label_index, x2=perturb2, use_aug=True)
                    else:
                        noise_dbindex_loss = 0

                    if simclr_weight != 0:
                        if double_perturb:
                            perturb_img1 = train_diff_transform(perturb_img1)
                            perturb_img2 = train_diff_transform(perturb_img2)
                            perturb_img1 = torch.clamp(perturb_img1 + perturb, 0, 1)
                            perturb_img2 = torch.clamp(perturb_img2 + perturb, 0, 1)
                            _noise_after_transform = False
                            # input('check double')
                        else:
                            _noise_after_transform = noise_after_transform
                            
                        simclr_loss = train_simclr_noise_return_loss_tensor(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug, noise_after_transform=_noise_after_transform, pytorch_aug=pytorch_aug, single_noise_after_transform=single_noise_after_transform, no_eval=no_eval, augmentation_prob=augmentation_prob, org_pos1=pos_samples_1, org_pos2=pos_samples_2, clean_weight=clean_weight)
                    else:
                        simclr_loss = 0

                    if noise_simclr_weight != 0:
                        perturb1 = torch.clamp(0.5 + perturb, 0, 1)
                        perturb2 = torch.clamp(0.5 + perturb, 0, 1)
                        noise_simclr_loss = train_simclr_noise_return_loss_tensor(model, perturb1, perturb2, opt, batch_size, temperature, flag_strong_aug, noise_after_transform=noise_after_transform, pytorch_aug=pytorch_aug, single_noise_after_transform=single_noise_after_transform, no_eval=no_eval, augmentation_prob=augmentation_prob, org_pos1=pos_samples_1, org_pos2=pos_samples_2, clean_weight=clean_weight)
                    else:
                        noise_simclr_loss = 0

                    if linear_noise_dbindex_weight != 0:
                        if crop_linear_noise:
                            crop_aug = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                ])
                            croped_perturb = crop_aug(perturb)
                            linear_noise_dbindex_loss = get_linear_noise_dbindex_loss(croped_perturb, labels[:, linear_noise_dbindex_index], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids, modify_dbindex=modify_dbindex, reverse_code=reverse_code)
                        else:
                            # input('check no here')
                            linear_noise_dbindex_loss = get_linear_noise_dbindex_loss(perturb, labels[:, linear_noise_dbindex_index], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids, modify_dbindex=modify_dbindex, reverse_code=reverse_code)
                    else:
                        linear_noise_dbindex_loss = 0

                    if linear_xnoise_dbindex_weight != 0:
                        linear_xnoise_dbindex_loss = get_linear_noise_dbindex_loss(torch.clamp(pos_samples_1.data + perturb, 0, 1), labels[:, linear_noise_dbindex_index], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids, modify_dbindex=modify_dbindex)
                    else:
                        linear_xnoise_dbindex_loss = 0

                    if linear_noise_dbindex_weight2 != 0:
                        linear_noise_dbindex_loss2 = get_linear_noise_dbindex_loss(perturb, labels[:, linear_noise_dbindex_index2], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids)
                    else:
                        linear_noise_dbindex_loss2 = 0

                    if use_supervised_g != 0:
                        g_net.zero_grad()
                        if not model_g_augment_first:
                            inputs = supervised_transform_train(torch.cat([perturb_img1, perturb_img2], dim=0))
                            feature, outputs = g_net(inputs)
                            supervised_loss = supervised_criterion(outputs, labels[:, 1].repeat((2,)))
                        else:
                            inputs = torch.clamp(supervised_transform_train(torch.clamp(pos_samples_1.data, 0, 1)) + perturb, 0, 1)
                            feature, outputs = g_net(inputs)
                            supervised_loss = supervised_criterion(outputs, labels[:, 1])
                    else:
                        supervised_loss = 0

                    print(linear_noise_dbindex_loss)
                    loss = dbindex_loss * dbindex_weight + simclr_loss * simclr_weight + noise_dbindex_loss * noise_dbindex_weight + noise_simclr_loss * noise_simclr_weight + supervised_weight * supervised_loss + linear_noise_dbindex_loss * linear_noise_dbindex_weight + linear_noise_dbindex_weight2 * linear_noise_dbindex_loss2 + linear_xnoise_dbindex_loss * linear_xnoise_dbindex_weight
                    
                    perturb.retain_grad()
                    loss.backward()

                    print(loss.dtype)
                    
                    eot_grad += perturb.grad.data
                    eot_loss += loss.item()
                
                eot_loss /= eot_size
                eot_grad /= eot_size

                if linear_noise_filter:
                    filter_mask = torch.abs(eot_grad) > (linear_noise_dbindex_weight * 0.1)
                    eot_grad = filter_mask * eot_grad
                    print(eot_grad.dtype)
                    # print(loss)
                    input()

                train_loss_batch = loss.item()/float(perturb.shape[0])
                train_loss_batch_sum += train_loss_batch * perturb.shape[0]
                train_loss_batch_count += perturb.shape[0]

                eta_step = self.step_size * eot_grad.sign() * (-1)
                sign_print = perturb.grad.data.sign() * (-1)
                # print("+:", np.sum(sign_print.cpu().numpy() == 1))
                # print("-:", np.sum(sign_print.cpu().numpy() == -1))
                # print("0:", np.sum(sign_print.cpu().numpy() == 0))
                perturb_img1 = perturb_img1.data + eta_step
                eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
                perturb_img2 = perturb_img2.data + eta_step
                eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)
                # diff_eta = eta1 - eta2
                # print(diff_eta.cpu().numpy())
                eta = (eta1 + eta2) / 2
                if mask_linear_constraint:
                    if mask_linear_noise_range[0] >= mask_linear_noise_range[1]:
                        raise('wrong parameter mask_linear_noise_range')
                    _eta1 = torch.clamp(eta, -mask_linear_noise_range[1] / 255.0, -mask_linear_noise_range[0] / 255.0,) * mask1
                    _eta2 = torch.clamp(eta, mask_linear_noise_range[0] / 255.0, mask_linear_noise_range[1] / 255.0,) * mask2
                    eta = _eta1 + _eta2
                # print("pos1 and pos2 diff: ", np.sum((eta1 - eta2).cpu().numpy()))
                perturb = Variable(eta, requires_grad=True)
                # perturb_img1 = pos_samples_1.data + perturb
                # perturb_img1 = torch.clamp(perturb_img1, 0, 1)
                # perturb_img2 = pos_samples_2.data + perturb
                # perturb_img2 = torch.clamp(perturb_img2, 0, 1)
                print("min_min_attack_simclr_return_loss_tensor_eot_v1:", eot_loss)

                end = time.time()

                print("time: ", end - start)
            # print("eta all")
            # print("+:", np.sum(eta.cpu().numpy() > 0.0313724))
            # print("-:", np.sum(eta.cpu().numpy() < -0.0313724))
            # print(">0:", np.sum(eta.cpu().numpy() > 0))
            # print("<0:", np.sum(eta.cpu().numpy() < 0))
            # print("=0:", np.sum(eta.cpu().numpy() == 0))

            return None, eta, train_loss_batch_sum / float(train_loss_batch_count)
        else:
            
            if random_noise is None:
                random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

            perturb = Variable(random_noise, requires_grad=True)

            eta = random_noise
            train_loss_batch_sum, train_loss_batch_count = 0, 0
            for _ in range(self.num_steps):

                start = time.time()

                eot_grad = torch.zeros(perturb.shape, dtype=torch.float).to(device)
                eot_linear_noise_grad = torch.zeros(perturb.shape, dtype=torch.float).to(device)
                eot_loss = 0

                for i_eot in range(eot_size):
                    opt = torch.optim.SGD([perturb], lr=1e-3)
                    opt.zero_grad()
                    model.zero_grad()

                    perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
                    perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)
                            
                    simclr_loss = train_simclr_noise_return_loss_tensor(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug, noise_after_transform=False, pytorch_aug=pytorch_aug, single_noise_after_transform=single_noise_after_transform, no_eval=no_eval, augmentation_prob=augmentation_prob, org_pos1=pos_samples_1, org_pos2=pos_samples_2, clean_weight=clean_weight)

                    loss = simclr_loss
                    
                    perturb.retain_grad()
                    loss.backward()
                    
                    eot_grad += perturb.grad.data
                    eot_loss += loss.item()

                    opt = torch.optim.SGD([perturb], lr=1e-3)
                    opt.zero_grad()
                    model.zero_grad()

                    linear_noise_dbindex_loss = get_linear_noise_dbindex_loss(perturb, labels[:, linear_noise_dbindex_index], use_mean_dbindex=use_mean_dbindex, use_normalized=use_normalized, noise_centroids=noise_centroids, modify_dbindex=modify_dbindex)
                    
                    loss = linear_noise_dbindex_loss
                    
                    perturb.retain_grad()
                    loss.backward()
                    
                    eot_linear_noise_grad += perturb.grad.data
                
                eot_loss /= eot_size
                eot_grad /= eot_size

                train_loss_batch = loss.item()/float(perturb.shape[0])
                train_loss_batch_sum += train_loss_batch * perturb.shape[0]
                train_loss_batch_count += perturb.shape[0]

                eta_step = self.step_size * eot_grad.sign() * (-1) + linear_noise_dbindex_weight * self.step_size * eot_linear_noise_grad.sign() * (-1)
                sign_print = perturb.grad.data.sign() * (-1)

                perturb_img1 = perturb_img1.data + eta_step
                eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
                perturb_img2 = perturb_img2.data + eta_step
                eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)

                eta = (eta1 + eta2) / 2

                # print("pos1 and pos2 diff: ", np.sum((eta1 - eta2).cpu().numpy()))
                perturb = Variable(eta, requires_grad=True)

                print("min_min_attack_simclr_return_loss_tensor_eot_v1:", eot_loss)

                end = time.time()
                print("time: ", end - start)
            # print("eta all")
            # print("+:", np.sum(eta.cpu().numpy() > 0.0313724))
            # print("-:", np.sum(eta.cpu().numpy() < -0.0313724))
            # print(">0:", np.sum(eta.cpu().numpy() > 0))
            # print("<0:", np.sum(eta.cpu().numpy() < 0))
            # print("=0:", np.sum(eta.cpu().numpy() == 0))

            return None, eta, train_loss_batch_sum / float(train_loss_batch_count)

    def min_min_attack_simclr_return_loss_tensor_eot_v1_no_eval(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None, flag_strong_aug=True, noise_after_transform=False, eot_size=30, one_gpu_eot_times=1, cross_eot=False, split_transform=False, pytorch_aug=False, dbindex_weight=0, simclr_weight=1):
    # v1 means it can repeat min_min_attack many times serially and average the results.
        if random_noise is None:
            random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb = Variable(random_noise, requires_grad=True)

        eta = random_noise
        train_loss_batch_sum, train_loss_batch_count = 0, 0
        for _ in range(self.num_steps):

            start = time.time()

            eot_grad = torch.zeros(perturb.shape, dtype=torch.float).to(device)
            eot_loss = 0

            # perturb_org = torch.clamp(pos_samples_1.data + perturb, 0, 1)

            for i_eot in range(eot_size):
                perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
                perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)
                opt = torch.optim.SGD([perturb], lr=1e-3)
                opt.zero_grad()
                model.zero_grad()

                if dbindex_weight != 0:
                    perturb_img_db = []
                    input_dbindex_num = 1
                    for i in range(input_dbindex_num):
                        if not noise_after_transform:
                            if pytorch_aug:
                                perturb_img_db1 = train_transform_no_totensor(torch.clamp(pos_samples_1.data + perturb, 0, 1))
                                perturb_img_db2 = train_transform_no_totensor(torch.clamp(pos_samples_2.data + perturb, 0, 1))
                            else:
                                perturb_img_db1 = train_diff_transform(torch.clamp(pos_samples_1.data + perturb, 0, 1))
                                perturb_img_db2 = train_diff_transform(torch.clamp(pos_samples_2.data + perturb, 0, 1))
                        perturb_img_db.append(perturb_img_db1)
                        perturb_img_db.append(perturb_img_db2)
                    perturb_img_db = torch.cat(perturb_img_db, dim=0)
                    db_labels = labels.repeat((input_dbindex_num * 2,1))

                    dbindex_loss = get_dbindex_loss(model, perturb_img_db, db_labels, [10], True, True)
                else:
                    dbindex_loss = 0

                if one_gpu_eot_times == 1:
                    if simclr_weight != 0:
                        simclr_loss = train_simclr_noise_return_loss_tensor_no_eval(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug, noise_after_transform=noise_after_transform, pytorch_aug=pytorch_aug)
                    else:
                        simclr_loss = 0
                    loss = dbindex_loss * dbindex_weight + simclr_loss * simclr_weight
                else:
                    loss = train_simclr_noise_return_loss_tensor_full_gpu(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug, noise_after_transform=noise_after_transform, gpu_times=one_gpu_eot_times, cross_eot=cross_eot)
                perturb.retain_grad()
                loss.backward()
                
                eot_grad += perturb.grad.data
                eot_loss += loss.item()
            
            eot_loss /= eot_size
            eot_grad /= eot_size

            train_loss_batch = loss.item()/float(perturb.shape[0])
            train_loss_batch_sum += train_loss_batch * perturb.shape[0]
            train_loss_batch_count += perturb.shape[0]

            eta_step = self.step_size * eot_grad.sign() * (-1)
            sign_print = perturb.grad.data.sign() * (-1)
            # print("+:", np.sum(sign_print.cpu().numpy() == 1))
            # print("-:", np.sum(sign_print.cpu().numpy() == -1))
            # print("0:", np.sum(sign_print.cpu().numpy() == 0))
            perturb_img1 = perturb_img1.data + eta_step
            eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
            perturb_img2 = perturb_img2.data + eta_step
            eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)
            # diff_eta = eta1 - eta2
            # print(diff_eta.cpu().numpy())
            eta = (eta1 + eta2) / 2
            # print("pos1 and pos2 diff: ", np.sum((eta1 - eta2).cpu().numpy()))
            perturb = Variable(eta, requires_grad=True)
            # perturb_img1 = pos_samples_1.data + perturb
            # perturb_img1 = torch.clamp(perturb_img1, 0, 1)
            # perturb_img2 = pos_samples_2.data + perturb
            # perturb_img2 = torch.clamp(perturb_img2, 0, 1)
            # print("min_min_attack_simclr_return_loss_tensor_eot_v1:", eot_loss)

            end = time.time()

            # print("time: ", end - start)
        # print("eta all")
        # print("+:", np.sum(eta.cpu().numpy() > 0.0313724))
        # print("-:", np.sum(eta.cpu().numpy() < -0.0313724))
        # print(">0:", np.sum(eta.cpu().numpy() > 0))
        # print("<0:", np.sum(eta.cpu().numpy() < 0))
        # print("=0:", np.sum(eta.cpu().numpy() == 0))

        return None, eta, train_loss_batch_sum / float(train_loss_batch_count)

    def min_min_attack_simclr_return_loss_tensor_eot_v1_no_eval_pos_only(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None, flag_strong_aug=True, noise_after_transform=False, eot_size=30, one_gpu_eot_times=1, cross_eot=False, split_transform=False, pytorch_aug=False, dbindex_weight=0, simclr_weight=1):
    # v1 means it can repeat min_min_attack many times serially and average the results.
        if random_noise is None:
            random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb = Variable(random_noise, requires_grad=True)

        eta = random_noise
        train_loss_batch_sum, train_loss_batch_count = 0, 0
        for _ in range(self.num_steps):

            start = time.time()

            eot_grad = torch.zeros(perturb.shape, dtype=torch.float).to(device)
            eot_loss = 0

            # perturb_org = torch.clamp(pos_samples_1.data + perturb, 0, 1)

            for i_eot in range(eot_size):
                perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
                perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)
                opt = torch.optim.SGD([perturb], lr=1e-3)
                opt.zero_grad()
                model.zero_grad()

                if dbindex_weight != 0:
                    perturb_img_db = []
                    input_dbindex_num = 1
                    for i in range(input_dbindex_num):
                        if not noise_after_transform:
                            if pytorch_aug:
                                perturb_img_db1 = train_transform_no_totensor(torch.clamp(pos_samples_1.data + perturb, 0, 1))
                                perturb_img_db2 = train_transform_no_totensor(torch.clamp(pos_samples_2.data + perturb, 0, 1))
                            else:
                                perturb_img_db1 = train_diff_transform(torch.clamp(pos_samples_1.data + perturb, 0, 1))
                                perturb_img_db2 = train_diff_transform(torch.clamp(pos_samples_2.data + perturb, 0, 1))
                        perturb_img_db.append(perturb_img_db1)
                        perturb_img_db.append(perturb_img_db2)
                    perturb_img_db = torch.cat(perturb_img_db, dim=0)
                    db_labels = labels.repeat((input_dbindex_num * 2,1))

                    dbindex_loss = get_dbindex_loss(model, perturb_img_db, db_labels, [10], True, True)
                else:
                    dbindex_loss = 0

                if one_gpu_eot_times == 1:
                    if simclr_weight != 0:
                        simclr_loss = train_simclr_noise_return_loss_tensor_no_eval_pos_only(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug, noise_after_transform=noise_after_transform, pytorch_aug=pytorch_aug)
                    else:
                        simclr_loss = 0
                    loss = dbindex_loss * dbindex_weight + simclr_loss * simclr_weight
                else:
                    loss = train_simclr_noise_return_loss_tensor_full_gpu(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug, noise_after_transform=noise_after_transform, gpu_times=one_gpu_eot_times, cross_eot=cross_eot)
                perturb.retain_grad()
                loss.backward()
                
                eot_grad += perturb.grad.data
                eot_loss += loss.item()
            
            eot_loss /= eot_size
            eot_grad /= eot_size

            train_loss_batch = loss.item()/float(perturb.shape[0])
            train_loss_batch_sum += train_loss_batch * perturb.shape[0]
            train_loss_batch_count += perturb.shape[0]

            eta_step = self.step_size * eot_grad.sign() * (-1)
            sign_print = perturb.grad.data.sign() * (-1)
            # print("+:", np.sum(sign_print.cpu().numpy() == 1))
            # print("-:", np.sum(sign_print.cpu().numpy() == -1))
            # print("0:", np.sum(sign_print.cpu().numpy() == 0))
            perturb_img1 = perturb_img1.data + eta_step
            eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
            perturb_img2 = perturb_img2.data + eta_step
            eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)
            # diff_eta = eta1 - eta2
            # print(diff_eta.cpu().numpy())
            eta = (eta1 + eta2) / 2
            # print("pos1 and pos2 diff: ", np.sum((eta1 - eta2).cpu().numpy()))
            perturb = Variable(eta, requires_grad=True)
            # perturb_img1 = pos_samples_1.data + perturb
            # perturb_img1 = torch.clamp(perturb_img1, 0, 1)
            # perturb_img2 = pos_samples_2.data + perturb
            # perturb_img2 = torch.clamp(perturb_img2, 0, 1)
            # print("min_min_attack_simclr_return_loss_tensor_eot_v1:", eot_loss)

            end = time.time()

            # print("time: ", end - start)
        # print("eta all")
        # print("+:", np.sum(eta.cpu().numpy() > 0.0313724))
        # print("-:", np.sum(eta.cpu().numpy() < -0.0313724))
        # print(">0:", np.sum(eta.cpu().numpy() > 0))
        # print("<0:", np.sum(eta.cpu().numpy() < 0))
        # print("=0:", np.sum(eta.cpu().numpy() == 0))

        return None, eta, train_loss_batch_sum / float(train_loss_batch_count)
    
    def min_min_attack_simclr_return_loss_tensor_eot_v1_model_free(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None, flag_strong_aug=True, target_task="non_eot", ):
    # v1 means it can repeat min_min_attack many times serially and average the results.
        if random_noise is None:
            random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb = Variable(random_noise, requires_grad=True)
        eot_size = 30

        eta = random_noise
        train_loss_batch_sum, train_loss_batch_count = 0, 0
        for _ in range(self.num_steps):

            eot_grad = torch.zeros(perturb.shape, dtype=torch.float).to(device)
            eot_loss = 0

            for i_eot in range(eot_size):
                perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
                perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)
                opt = torch.optim.SGD([perturb], lr=1e-3)
                opt.zero_grad()
                model.zero_grad()
                if target_task == "eot_v1":
                    loss = train_simclr_noise_return_loss_tensor_model_free(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug)
                elif target_task == "eot_v1_pos":
                    loss = train_simclr_noise_return_loss_tensor_target_task_model_free(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug, "pos")
                elif target_task == "eot_v1_neg":
                    loss = train_simclr_noise_return_loss_tensor_target_task_model_free(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug, "pos")
                else:
                    raise("Wrong target_task")
                perturb.retain_grad()
                loss.backward()
                eot_grad += perturb.grad.data
                eot_loss += loss.item()
            
            eot_loss /= eot_size
            eot_grad /= eot_size

            train_loss_batch = loss.item()/float(perturb.shape[0])
            train_loss_batch_sum += train_loss_batch * perturb.shape[0]
            train_loss_batch_count += perturb.shape[0]

            eta_step = self.step_size * eot_grad.sign() * (-1)
            sign_print = perturb.grad.data.sign() * (-1)
            perturb_img1 = perturb_img1.data + eta_step
            eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
            perturb_img2 = perturb_img2.data + eta_step
            eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)
            eta = (eta1 + eta2) / 2
            perturb = Variable(eta, requires_grad=True)
            print("min_min_attack_simclr_return_loss_tensor_eot_v1:", eot_loss)

        return None, eta, train_loss_batch_sum / float(train_loss_batch_count)

    def min_min_attack_simclr_return_loss_tensor_eot_v2(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None, flag_strong_aug=True):
    # v1 means parallel 
        if random_noise is None:
            random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        eot_size = 5
        perturb = Variable(random_noise, requires_grad=True)
        eot_perturb = perturb.repeat(eot_size,1,1,1)
        perturb_chunks = torch.chunk(eot_perturb, eot_size, dim=0)

        train_loss_batch_sum, train_loss_batch_count = 0, 0
        for _ in range(self.num_steps):

            eot_grad = torch.zeros(perturb.shape, dtype=torch.float).to(device)
            eot_loss = 0
            opt = torch.optim.SGD([perturb], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()

            for i_eot in range(eot_size):
                perturb_img1 = [torch.clamp(pos_samples_1.data + perturb_chunks[i_perturb], 0, 1) for i_perturb in range(eot_size)]
                perturb_img2 = [torch.clamp(pos_samples_2.data + perturb_chunks[i_perturb], 0, 1) for i_perturb in range(eot_size)]

            eot_loss = train_simclr_noise_return_loss_tensor_eot(model, perturb_img1, perturb_img2, opt, batch_size, temperature, eot_size, flag_strong_aug)
            for i_eot in range(eot_size):
                perturb_chunks[i_eot].retain_grad()
            eot_perturb.retain_grad()
            perturb.retain_grad()
            eot_loss.backward()

            check_eot = 0

            # # checked via following that repeat is accumulating the gradients. chunk works like this chunk.grad = [subchunk[0], subchunk[1], subchunk[2]]
            # for i_eot in range(eot_size):
            #     eot_grad += perturb_chunks[i_eot].grad.data
            #     check_eot += perturb_chunks[i_eot].grad.data.mean()
            # print("chunk:", check_eot / 10)
            # print("eot_perturb:", eot_perturb.grad.data.mean())
            # print("perturb:", perturb.grad.data.mean() / 10)

            eta_step = self.step_size * perturb.grad.data.sign() * (-1)

            perturb_img1[0] = perturb_img1[0].data + eta_step
            eta1 = torch.clamp(perturb_img1[0].data - pos_samples_1.data, -self.epsilon, self.epsilon)
            perturb_img2[0] = perturb_img2[0].data + eta_step
            eta2 = torch.clamp(perturb_img2[0].data - pos_samples_2.data, -self.epsilon, self.epsilon)
            eta = (eta1 + eta2) / 2

            perturb = Variable(eta, requires_grad=True)
            eot_perturb = perturb.repeat(eot_size,1,1,1)
            # eot_perturb = Variable(perturb.repeat(eot_size,1,1,1), requires_grad=True)
            perturb_chunks = torch.chunk(eot_perturb, eot_size, dim=0)
            print("min_min_attack_simclr_return_loss_tensor_eot_v2:", eot_loss.item())

        return None, eta, None


    def min_min_attack_simclr_return_loss_tensor_eot_v3(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None, flag_strong_aug=True):
    # v1 combine v1 and v2
        if random_noise is None:
            random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        eot_size_v1 = 15
        eot_size_v2 = 5

        eta = random_noise

        train_loss_batch_sum, train_loss_batch_count = 0, 0
        for _ in range(self.num_steps):

            eot_grad = torch.zeros(eta.shape, dtype=torch.float).to(device)
            eot_loss = 0

            for i_eot_v1 in range(eot_size_v1):

                perturb = Variable(eta, requires_grad=True)

                opt = torch.optim.SGD([perturb], lr=1e-3)
                opt.zero_grad()
                model.zero_grad()

                eot_perturb = perturb.repeat(eot_size_v2,1,1,1)
                perturb_chunks = torch.chunk(eot_perturb, eot_size_v2, dim=0)

                for i_eot in range(eot_size_v2):
                    perturb_img1 = [torch.clamp(pos_samples_1.data + perturb_chunks[i_perturb], 0, 1) for i_perturb in range(eot_size_v2)]
                    perturb_img2 = [torch.clamp(pos_samples_2.data + perturb_chunks[i_perturb], 0, 1) for i_perturb in range(eot_size_v2)]

                eot_loss_v2 = train_simclr_noise_return_loss_tensor_eot(model, perturb_img1, perturb_img2, opt, batch_size, temperature, eot_size_v2, flag_strong_aug)
                for i_eot in range(eot_size_v2):
                    perturb_chunks[i_eot].retain_grad()
                eot_perturb.retain_grad()
                perturb.retain_grad()
                eot_loss_v2.backward()

                eot_grad += perturb.grad.data
                eot_loss += eot_loss_v2.item()

            eot_loss /= eot_size_v1
            eot_grad /= eot_size_v1

            eta_step = self.step_size * eot_grad.sign() * (-1)

            perturb_img1[0] = perturb_img1[0].data + eta_step
            eta1 = torch.clamp(perturb_img1[0].data - pos_samples_1.data, -self.epsilon, self.epsilon)
            perturb_img2[0] = perturb_img2[0].data + eta_step
            eta2 = torch.clamp(perturb_img2[0].data - pos_samples_2.data, -self.epsilon, self.epsilon)
            eta = (eta1 + eta2) / 2

            # perturb = Variable(eta, requires_grad=True)
            # eot_perturb = perturb.repeat(eot_size_v2,1,1,1)
            # perturb_chunks = torch.chunk(eot_perturb, eot_size_v2, dim=0)
            print("min_min_attack_simclr_return_loss_tensor_eot_v3:", eot_loss)

        return None, eta, None

    def min_min_attack_simclr_large_noise(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None):
    # after verified that using perturb as variable to train is working 
        if random_noise is None:
            random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb = Variable(random_noise, requires_grad=True)
        perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
        perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)

        eta = random_noise
        train_loss_batch_sum, train_loss_batch_count = 0, 0
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            # perturb.retain_grad()
            # loss.backward()
            loss = train_simclr_noise_return_loss_tensor(model, perturb_img1, perturb_img2, opt, batch_size, temperature)
            perturb.retain_grad()
            loss.backward()
            train_loss_batch = loss.item()/float(perturb.shape[0])
            train_loss_batch_sum += train_loss_batch * perturb.shape[0]
            train_loss_batch_count += perturb.shape[0]

            eta_step = self.step_size * perturb.grad.data.sign() * (-1) # why here used sign?? renjie3
            sign_print = perturb.grad.data.sign() * (-1)
            # print("+:", np.sum(sign_print.cpu().numpy() == 1))
            # print("-:", np.sum(sign_print.cpu().numpy() == -1))
            # print("0:", np.sum(sign_print.cpu().numpy() == 0))
            perturb_img1 = perturb_img1.data + eta_step
            eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
            perturb_img2 = perturb_img2.data + eta_step
            eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)
            # diff_eta = eta1 - eta2
            # print(diff_eta.cpu().numpy())
            eta = (eta1 + eta2) / 2
            # print("pos1 and pos2 diff: ", np.sum((eta1 - eta2).cpu().numpy()))
            perturb = Variable(eta, requires_grad=True)
            perturb_img1 = pos_samples_1.data + perturb
            perturb_img1 = torch.clamp(perturb_img1, 0, 1)

            # diff_img1 = perturb_img1_noclamp - perturb_img1
            # print(perturb_img1_noclamp.shape)
            # print(perturb_img1_noclamp.cpu().detach().numpy()[0][0][0])
            # print(perturb_img1.cpu().detach().numpy()[0][0][0])
            # print(pos_samples_1.data[0][0])
            # print("eta: ", eta.cpu().detach().numpy()[0][0][0])
            # print("diff_img1: ", diff_img1.cpu().detach().numpy()[0][0][0])
            perturb_img2 = pos_samples_2.data + perturb
            perturb_img2 = torch.clamp(perturb_img2, 0, 1)

            print("min_min_attack_simclr_large_noise:", loss.item())

        #     print(loss.item())
        
        # input()
        # print("eta all")
        # print(eta.cpu().numpy()[0][0])
        # print("+:", np.sum(eta.cpu().numpy() > 63 / 255.0))
        # print("-:", np.sum(eta.cpu().numpy() < -63 / 255.0))
        # print(">0:", np.sum(eta.cpu().numpy() > 0))
        # print("<0:", np.sum(eta.cpu().numpy() < 0))
        # print("=0:", np.sum(eta.cpu().numpy() == 0))

        return None, eta, train_loss_batch_sum / float(train_loss_batch_count)

    def min_min_attack_simclr_return_loss_tensor_print(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None):
    # after verified that using perturb as variable to train is working 
        if random_noise is None:
            random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb = Variable(random_noise, requires_grad=True)
        perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
        perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)

        eta = random_noise
        train_loss_batch_sum, train_loss_batch_count = 0, 0
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            # perturb.retain_grad()
            # loss.backward()
            loss = train_simclr_noise_return_loss_tensor(model, perturb_img1, perturb_img2, opt, batch_size, temperature)
            perturb.retain_grad()
            loss.backward()
            train_loss_batch = loss.item()/float(perturb.shape[0])
            train_loss_batch_sum += train_loss_batch * perturb.shape[0]
            train_loss_batch_count += perturb.shape[0]

            eta_step = self.step_size * perturb.grad.data.sign() * (-1) # why here used sign?? renjie3
            sign_print = perturb.grad.data.sign() * (-1)
            # print("+:", np.sum(sign_print.cpu().numpy() == 1))
            # print("-:", np.sum(sign_print.cpu().numpy() == -1))
            # print("0:", np.sum(sign_print.cpu().numpy() == 0))
            perturb_img1 = perturb_img1.data + eta_step
            eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
            perturb_img2 = perturb_img2.data + eta_step
            eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)
            eta = (eta1 + eta2) / 2
            # print("pos1 and pos2 diff: ", np.sum((eta1 - eta2).cpu().numpy()))
            perturb = Variable(eta, requires_grad=True)
            perturb_img1 = pos_samples_1.data + perturb
            perturb_img1 = torch.clamp(perturb_img1, 0, 1)
            perturb_img2 = pos_samples_2.data + perturb
            perturb_img2 = torch.clamp(perturb_img2, 0, 1)
        print("eta all")
        print("+:", np.sum(eta.cpu().numpy() > 0.0313724))
        print("-:", np.sum(eta.cpu().numpy() < -0.0313724))
        print(">0:", np.sum(eta.cpu().numpy() > 0))
        print("<0:", np.sum(eta.cpu().numpy() < 0))
        print("=0:", np.sum(eta.cpu().numpy() == 0))

        return None, eta, train_loss_batch_sum / float(train_loss_batch_count)

    def min_min_attack_pos1_pertub(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None):
        # just train the noise on image 1
        if random_noise is None:
            random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img1 = Variable(pos_samples_1.data + random_noise, requires_grad=True)
        perturb_img1 = Variable(torch.clamp(perturb_img1, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img1], lr=1e-3)
            # opt.zero_grad()
            model.zero_grad()
            # if isinstance(criterion, torch.nn.CrossEntropyLoss):
            #     if hasattr(model, 'classify'):
            #         model.classify = True
            #     logits = model(perturb_img)
            #     loss = criterion(logits, labels)
            # else:
            #     logits, loss = criterion(model, perturb_img, labels, optimizer)
            train_simclr_noise_pos1_pertub(model, perturb_img1, torch.clamp(pos_samples_2.data + eta, 0, 1), opt, batch_size, temperature)
            # perturb_img.retain_grad()
            # loss.backward()
            eta = self.step_size * perturb_img1.grad.data.sign() * (-1)
            # sign_print = perturb_img1.grad.data.sign() * (-1)
            # print("+:", np.sum(sign_print.cpu().numpy()[0] == 1))
            # print("-:", np.sum(sign_print.cpu().numpy()[0] == -1))
            # print("0:", np.sum(sign_print.cpu().numpy()[0] == 0))
            perturb_img1 = Variable(perturb_img1.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
            perturb_img1 = Variable(pos_samples_1.data + eta, requires_grad=True)
            perturb_img1 = Variable(torch.clamp(perturb_img1, 0, 1), requires_grad=True)
        # print(eta.cpu().numpy()[0])
        # print(eta.shape)
        # sign_print = perturb_img1.grad.data.sign() * (-1)
        # print("+:", np.sum(eta.cpu().numpy() > 0.0313724))
        # print("-:", np.sum(eta.cpu().numpy() < -0.0313724))
        # print(">0:", np.sum(eta.cpu().numpy() > 0))
        # print("<0:", np.sum(eta.cpu().numpy() < 0))

        return perturb_img1, eta

    def min_min_attack_pos2_pertub(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None):
        # just train the noise on image 1
        if random_noise is None:
            random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img1 = Variable(pos_samples_2.data + random_noise, requires_grad=True)
        perturb_img1 = Variable(torch.clamp(perturb_img1, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img1], lr=1e-3)
            # opt.zero_grad()
            model.zero_grad()
            # if isinstance(criterion, torch.nn.CrossEntropyLoss):
            #     if hasattr(model, 'classify'):
            #         model.classify = True
            #     logits = model(perturb_img)
            #     loss = criterion(logits, labels)
            # else:
            #     logits, loss = criterion(model, perturb_img, labels, optimizer)
            train_simclr_noise_pos1_pertub(model, perturb_img1, torch.clamp(pos_samples_1.data + eta, 0, 1), opt, batch_size, temperature)
            # perturb_img.retain_grad()
            # loss.backward()
            eta = self.step_size * perturb_img1.grad.data.sign() * (-1)
            # sign_print = perturb_img1.grad.data.sign() * (-1)
            # print("+:", np.sum(sign_print.cpu().numpy()[0] == 1))
            # print("-:", np.sum(sign_print.cpu().numpy()[0] == -1))
            # print("0:", np.sum(sign_print.cpu().numpy()[0] == 0))
            perturb_img1 = Variable(perturb_img1.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img1.data - pos_samples_2.data, -self.epsilon, self.epsilon)
            perturb_img1 = Variable(pos_samples_2.data + eta, requires_grad=True)
            perturb_img1 = Variable(torch.clamp(perturb_img1, 0, 1), requires_grad=True)
        # print(eta.cpu().numpy()[0])
        # print(eta.shape)
        # sign_print = perturb_img1.grad.data.sign() * (-1)
        print("+:", np.sum(eta.cpu().numpy() > 0.0313724))
        print("-:", np.sum(eta.cpu().numpy() < -0.0313724))
        print(">0:", np.sum(eta.cpu().numpy() > 0))
        print("<0:", np.sum(eta.cpu().numpy() < 0))

        return perturb_img1, eta

    def min_max_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
            loss.backward()

            eta = self.step_size * perturb_img.grad.data.sign()
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def _patch_noise_extend_to_img(self, noise, image_size=[3, 32, 32], patch_location='center'):
        c, h, w = image_size[0], image_size[1], image_size[2]
        mask = np.zeros((c, h, w), np.float32)
        x_len, y_len = noise.shape[1], noise.shape[1]

        if patch_location == 'center' or (h == w == x_len == y_len):
            x = h // 2
            y = w // 2
        elif patch_location == 'random':
            x = np.random.randint(x_len // 2, w - x_len // 2)
            y = np.random.randint(y_len // 2, h - y_len // 2)
        else:
            raise('Invalid patch location')

        x1 = np.clip(x - x_len // 2, 0, h)
        x2 = np.clip(x + x_len // 2, 0, h)
        y1 = np.clip(y - y_len // 2, 0, w)
        y2 = np.clip(y + y_len // 2, 0, w)
        if type(noise) is np.ndarray:
            pass
        else:
            mask[:, x1: x2, y1: y2] = noise.cpu().numpy()
        return ((x1, x2, y1, y2), torch.from_numpy(mask).to(device))
