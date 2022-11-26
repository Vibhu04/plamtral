from transformers import GPT2LMHeadModel
import torch.nn as nn
from fine_tuning.utils import freeze_first_n_layers, freeze_nth_layer

class Model_with_vanilla_finetuning():

    # model_type options: '', '-medium', '-large'
    def __init__(self, base_model = 'GPT2', model_size = '', freeze_initial_layers = 0, freeze_layer = None):
      
        self.base_model = base_model
        self.model_size = model_size
        self.technique = 'Vanilla Fine Tuning'
        if base_model == 'GPT2':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2' + model_size, return_dict=False)
        else:
            raise Exception("Error: this library supports only GPT2 as the base model for now.")
            
        if freeze_initial_layers > 0:
            freeze_first_n_layers(self.model, freeze_initial_layers)
        if freeze_layer is not None and freeze_layer > freeze_initial_layers:
            freeze_nth_layer(self.model, freeze_layer)
