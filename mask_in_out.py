import torch


from torchvision import utils as vutils

data = torch.load('./results/unlearnable_samplewise_10000_20220518091221_0.5_512_300_checkpoint_perturbation.pt').cuda()

mask = torch.ones(data.shape[1:]).cuda()

print(mask.shape)

i = 6
mask[:, :i, :] = 0
mask[:, 32-i:32, :] = 0
mask[:, :, 32-i:32] = 0
mask[:, :, :i] = 0

mask = mask.unsqueeze(0).repeat(data.shape[0], 1, 1, 1)

# print(mask.shape)

# print(torch.sum(data[:1] != 0))
# img = data[:1]
# img = img - torch.min(img)
# img = img / torch.max(img)

# print(torch.min(img))
# print(torch.max(img))

# vutils.save_image(mask[:1], 'test.png')

data_in = data * mask
data_out = data * (1-mask)

torch.save(data_in, './results/unlearnable_samplewise_10000_20220518091221_0.5_512_300_checkpoint_perturbation_epoch5_{}_in.pt'.format(i))
torch.save(data_out, './results/unlearnable_samplewise_10000_20220518091221_0.5_512_300_checkpoint_perturbation_epoch5_{}_out.pt'.format(i))
