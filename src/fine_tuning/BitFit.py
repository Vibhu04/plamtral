from transformers import GPT2LMHeadModel
import torch.nn as nn
from utils import freeze_except_bias

class Model_with_BitFit():

    # model_type options: '', '-medium', '-large'
    def __init__(self, base_model = 'GPT2', model_size = '', freeze_initial_layers = 0, freeze_layer = None):
        self.base_model = base_model
        self.model_size = model_size
        self.technique = 'BitFit'
        if base_model == 'GPT2':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2' + model_size, return_dict=False)
        else:
            raise Exception("Error: this library supports only GPT2 as the base model for now.")
            
        self.model = freeze_except_bias(self.model)
