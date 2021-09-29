import numpy as np
import torch
from torch.autograd import Variable
from simclr import test_ssl, train_simclr, train_simclr_noise, train_simclr_noise_pos1_pertub

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
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                if hasattr(model, 'classify'):
                    model.classify = True
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
            perturb_img.retain_grad()
            loss.backward()
            eta = self.step_size * perturb_img.grad.data.sign() * (-1)
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

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

        return perturb_img, eta

    def min_min_attack_simclr(self, pos_samples_1, pos_samples_2, labels, model, optimizer, criterion, random_noise=None, sample_wise=False, batch_size=512, temperature=None):
        if random_noise is None:
            random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb = Variable(random_noise, requires_grad=True)
        perturb_img1 = Variable(torch.clamp(pos_samples_1.data + perturb, 0, 1), requires_grad=True)
        perturb_img2 = Variable(torch.clamp(pos_samples_2.data + perturb, 0, 1), requires_grad=True)
        # perturb_img2 = Variable(torch.clamp(perturb_img2, 0, 1), requires_grad=True)
        eta = random_noise
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
            train_simclr_noise(model, perturb_img1, perturb_img2, perturb, opt, batch_size, temperature)

            # eta = self.step_size * perturb_img.grad.data.sign() * (-1)
            # perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            # eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            # perturb_img = Variable(images.data + eta, requires_grad=True)
            # perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

            eta = self.step_size * perturb.grad.data.sign() * (-1) # why here used sign?? renjie3
            perturb_img1 = perturb_img1.data + eta
            eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
            perturb_img2 = perturb_img2.data + eta1
            perturb = Variable(torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon), requires_grad=True)
            perturb_img1 = pos_samples_1.data + eta
            perturb_img1 = torch.clamp(perturb_img1, 0, 1)
            perturb_img2 = pos_samples_2.data + eta
            perturb_img2 = torch.clamp(perturb_img2, 0, 1)

            # perturb_img = Variable(images.data + eta, requires_grad=True)
            # perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        print(eta.cpu().numpy()[0])
        print(eta.shape)

        return None, eta

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
            perturb_img1 = Variable(perturb_img1.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
            perturb_img1 = Variable(pos_samples_1.data + eta, requires_grad=True)
            perturb_img1 = Variable(torch.clamp(perturb_img1, 0, 1), requires_grad=True)

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
