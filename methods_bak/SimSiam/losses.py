import torch.nn.functional as F 

def negcos(p1, p2, z1, z2, k_grad=False):
    p1 = F.normalize(p1, dim=1); p2 = F.normalize(p2, dim=1)
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)

    if k_grad:
        return - 0.5 * ((p1*z2).sum(dim=1).mean() + (p2*z1).sum(dim=1).mean())
    else:
        return - 0.5 * ((p1*z2.detach()).sum(dim=1).mean() + (p2*z1.detach()).sum(dim=1).mean())