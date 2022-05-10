from .SimSiam.simsiam import SimSiamModel


def set_model(arch, dataset):
    return SimSiamModel(arch, dataset) 

