import random
import numpy as np
import torch

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# slow & deterministic
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# # fast & non-deterministic
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = True
# torch.use_deterministic_algorithms(True)

class Configs:
    def __init__(self):
        self.root_dir = '{need_to_set}/Modularizing_while_Training_DNN'
        self.data_dir = f'{self.root_dir}/data'
        self.dataset_dir = f'{self.data_dir}/dataset'
        self.tensorboard_dir = f'{self.data_dir}/tensorboard_log'
