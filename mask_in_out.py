import torch

data = torch.load(perturb_tensor_filepath).to('cpu').numpy()

mask = torch.zeros(data.shape[1:])

print(mask)
