import os
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.manual_seed(seed)


def get_files(directory, extension):
    for root, _, files in os.walk(directory):
        return [os.path.join(root, file) for file in files
                if os.path.splitext(file)[1] == extension]
