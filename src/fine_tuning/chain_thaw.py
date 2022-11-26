from transformers import GPT2LMHeadModel
import torch.nn as nn
from fine_tuning.utils import freeze_model, chain_thaw

class Model_with_Chain_Thaw():

    # model_type options: '', '-medium', '-large'
    def __init__(self, base_model = 'GPT2', model_size = '', unfreeze_freq = 100):
        self.base_model = base_model
        self.model_size = model_size
        self.technique = 'Chain Thaw'
        self.unfreeze_freq = unfreeze_freq
        if base_model == 'GPT2':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2' + model_size, return_dict=False)
        else:
            raise Exception("Error: this library supports only GPT2 as the base model for now.")

        self.model = freeze_model(self.model)
        self.model = chain_thaw(self.model, 1)
