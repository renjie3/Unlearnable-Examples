import numpy as np
import torch
from torch.autograd import Variable
from simclr import test_ssl, train_simclr, train_simclr_noise, train_simclr_noise_return_loss_tensor, train_simclr_noise_return_loss_tensor_eot, train_simclr_noise_return_loss_tensor_target_task
from utils import train_diff_transform, train_diff_transform2

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

    def min_min_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            aug_perturb_img = train_diff_transform(perturb_img)
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                if hasattr(model, 'classify'):
                    model.classify = True
                logits = model(aug_perturb_img)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, aug_perturb_img, labels, optimizer)
            perturb_img.retain_grad()
            loss.backward()
            eta = self.step_size * perturb_img.grad.data.sign() * (-1)
            # sign_print = perturb_img.grad.data.sign() * (-1)
            # print("+:", np.sum(sign_print.cpu().numpy()[0] == 1))
            # print("-:", np.sum(sign_print.cpu().numpy()[0] == -1))
            # print("0:", np.sum(sign_print.cpu().numpy()[0] == 0))
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        #     print(loss.item())
        # input()
        # print(eta.cpu().numpy()[0])
        # print(eta.shape)
        # print("+:", np.sum(eta.cpu().numpy() > 0.0313724))
        # print("-:", np.sum(eta.cpu().numpy() < -0.0313724))
        # print(">0:", np.sum(eta.cpu().numpy() > 0))
        # print("<0:", np.sum(eta.cpu().numpy() < 0))

        return perturb_img, eta, loss.item()

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

    def min_min_attack_simclr_return_loss_tensor(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None, flag_strong_aug=True, target_task="non_eot"):
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
            else:
                loss = train_simclr_noise_return_loss_tensor_target_task(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug, target_task)
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

    def min_min_attack_simclr_return_loss_tensor_eot_v1(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None, flag_strong_aug=True):
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
                loss = train_simclr_noise_return_loss_tensor(model, perturb_img1, perturb_img2, opt, batch_size, temperature, flag_strong_aug)
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
            print("min_min_attack_simclr_return_loss_tensor_eot_v1:", eot_loss)
        # print("eta all")
        # print("+:", np.sum(eta.cpu().numpy() > 0.0313724))
        # print("-:", np.sum(eta.cpu().numpy() < -0.0313724))
        # print(">0:", np.sum(eta.cpu().numpy() > 0))
        # print("<0:", np.sum(eta.cpu().numpy() < 0))
        # print("=0:", np.sum(eta.cpu().numpy() == 0))

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
