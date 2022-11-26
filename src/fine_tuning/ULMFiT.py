from transformers import GPT2LMHeadModel
import torch.nn as nn
from fine_tuning.utils import freeze_model, gradually_unfreeze

class Model_with_gradual_unfreezing():

    # model_type options: '', '-medium', '-large'
    def __init__(self, base_model = 'GPT2', model_size = '', unfreeze_freq = 100):
        self.base_model = base_model
        self.model_size = model_size
        self.technique = 'Gradual Unfreezing'
        self.unfreeze_freq = unfreeze_freq
        if base_model == 'GPT2':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2' + model_size, return_dict=False)
        else:
            raise Exception("Error: this library supports only GPT2 as the base model for now.")

        self.model = freeze_model(self.model)
        self.model = gradually_unfreeze(self.model, 1)


class Model_with_ULMFiT(Model_with_gradual_unfreezing):

    def __init__(self, base_model = 'GPT2', model_size = '', unfreeze_freq = 100, dlr_factor = 0.384615):
        super(Model_with_ULMFiT, self).__init__(base_model, model_size, unfreeze_freq)
        self.technique = 'ULMFiT'
        self.dlr_factor = dlr_factor