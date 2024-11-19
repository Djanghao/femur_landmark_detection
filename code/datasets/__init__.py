from .femur import *

def select_dataset_voting(dataset):
    if dataset == 'femur':
        return Femur_TPL_Voting