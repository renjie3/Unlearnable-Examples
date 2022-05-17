from .SimCLR.simclr import SimCLRModel
from .MoCo.moco import MoCoModel
from .SimSiam.simsiam import SimSiamModel


def set_model(method, arch, dataset):
    if method == 'simclr':
        return SimCLRModel(method, arch, dataset)
    elif method == 'moco':
        return MoCoModel(method, arch, dataset)
    elif method == 'simsiam':
        return SimSiamModel(method, arch, dataset) 

