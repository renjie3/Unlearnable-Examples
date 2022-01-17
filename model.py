import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, resnet18


class Model(nn.Module):
    def __init__(self, feature_dim=128, cifar_head=True, arch='resnet18', train_mode='clean_train', f_logits_dim=1024):
        super(Model, self).__init__()

        self.f = []
        self.train_mode = train_mode
        if arch == 'resnet18':
            backbone = resnet18()
            encoder_dim = 512
        elif arch == 'resnet50':
            backbone = resnet50()
            encoder_dim = 2048
        else:
            raise NotImplementedError

        for name, module in backbone.named_children():
            if name == 'conv1' and cifar_head == True:
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        
        # encoder
        self.f = nn.Sequential(*self.f)
        # logts for clean_train_softmax
        if self.train_mode == 'clean_train_softmax':
            self.f_logits = nn.Sequential(nn.Linear(encoder_dim, f_logits_dim, bias=True))
        # projection head
        self.g = nn.Sequential(nn.Linear(encoder_dim, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        if self.train_mode == 'clean_train_softmax':
            logits = self.f_logits(feature)
            return F.normalize(feature, dim=-1), F.normalize(logits, dim=-1), F.normalize(out, dim=-1)
        else:
            return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
        # return feature, F.normalize(out, dim=-1)
        

class LooC(nn.Module):
    def __init__(self, feature_dim=128, cifar_head=True, arch='resnet18', train_mode='clean_train', n_zspace=3):
        super(LooC, self).__init__()

        self.f = []
        self.train_mode = train_mode
        self.n_zspace = n_zspace
        if arch == 'resnet18':
            backbone = resnet18()
            encoder_dim = 512
        elif arch == 'resnet50':
            backbone = resnet50()
            encoder_dim = 2048
        else:
            raise NotImplementedError

        for name, module in backbone.named_children():
            if name == 'conv1' and cifar_head == True:
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g0 = nn.Sequential(nn.Linear(encoder_dim, 512, bias=False), nn.BatchNorm1d(512),nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        self.g1 = nn.Sequential(nn.Linear(encoder_dim, 512, bias=False), nn.BatchNorm1d(512),nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        self.g2 = nn.Sequential(nn.Linear(encoder_dim, 512, bias=False), nn.BatchNorm1d(512),nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        if n_zspace == 3:
            self.g3 = nn.Sequential(nn.Linear(encoder_dim, 512, bias=False), nn.BatchNorm1d(512),nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, views):
        features = []
        outs = []
        for x in views:
            feature = self.f(x)
            feature = torch.flatten(feature, start_dim=1)
            features.append(F.normalize(feature, dim=-1))
        for i in range(self.n_zspace + 1):
            out = []
            for feature in features:
                if i == 0:
                    z_space = F.normalize(self.g0(feature), dim=-1)
                elif i == 1:
                    z_space = F.normalize(self.g1(feature), dim=-1)
                elif i == 2:
                    z_space = F.normalize(self.g2(feature), dim=-1)
                elif i == 3:
                    z_space = F.normalize(self.g3(feature), dim=-1)
                out.append(z_space)
            outs.append(out)
        return features, outs
        # return feature, F.normalize(out, dim=-1)
        
class MICL(nn.Module):
    def __init__(self, feature_dim=128, cifar_head=True, arch='resnet18', train_mode='clean_train', n_zspace=3):
        super(MICL, self).__init__()

        f = []
        self.train_mode = train_mode
        self.n_zspace = n_zspace
        if arch == 'resnet18':
            backbone = resnet18()
            encoder_dim = 512
        elif arch == 'resnet50':
            backbone = resnet50()
            encoder_dim = 2048
        else:
            raise NotImplementedError

        for name, module in backbone.named_children():
            if name == 'conv1' and cifar_head == True:
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                f.append(module)
        
        # encoder
        self.f0 = nn.Sequential(*f)
        self.f1 = nn.Sequential(*f)
        if n_zspace == 3:
            self.f2 = nn.Sequential(*f)
        # projection head
        self.g0 = nn.Sequential(nn.Linear(encoder_dim, 512, bias=False), nn.BatchNorm1d(512),nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        self.g1 = nn.Sequential(nn.Linear(encoder_dim, 512, bias=False), nn.BatchNorm1d(512),nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        if n_zspace == 3:
            self.g2 = nn.Sequential(nn.Linear(encoder_dim, 512, bias=False), nn.BatchNorm1d(512),nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, views):
        features = []
        outs = []
        for i in range(self.n_zspace):
            if i == 0:
                feature = self.f0(views[i])
                feature = torch.flatten(feature, start_dim=1)
                z_space = F.normalize(self.g0(feature), dim=-1)
            elif i == 1:
                feature = self.f1(views[i])
                feature = torch.flatten(feature, start_dim=1)
                z_space = F.normalize(self.g1(feature), dim=-1)
            elif i == 2:
                feature = self.f2(views[i])
                feature = torch.flatten(feature, start_dim=1)
                z_space = F.normalize(self.g2(feature), dim=-1)
            features.append(F.normalize(feature, dim=-1))
            outs.append(z_space)
        return features, outs
        # return feature, F.normalize(out, dim=-1)

# class TheoryModel(nn.Module):
#     def __init__(self, train_mode='theory_model', normalize=False):
#         super(TheoryModel, self).__init__()
#         self.normalize = normalize
#         # encoder
#         self.f = nn.Linear(2, 3, bias=False)

#     def forward(self, x):
#         feature = self.f(x)
#         if self.normalize:
#             return F.normalize(feature, dim=-1)
#         else:
#             return feature

class TheoryModel(nn.Module):
    def __init__(self, train_mode='theory_model', normalize=False):
        super(TheoryModel, self).__init__()
        self.normalize = normalize
        # encoder
        self.f = nn.Sequential(nn.Linear(90, 128, bias=False), nn.BatchNorm1d(128),nn.ReLU(inplace=True), nn.Linear(128, 16, bias=True))

    def forward(self, x):
        feature = self.f(x)
        if self.normalize:
            return F.normalize(feature, dim=-1)
        else:
            return feature