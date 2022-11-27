from transformers import GPT2LMHeadModel
import torch.nn as nn
from adapter import Adaptered


class Model_with_adapterdrop_spec():
    """
    Specialized AdapterDrop: Removing adapters from the first n transformer 
    layers, where n is fixed during training. This yields separate models for
    each possible n.
    For more information, please refer to the following paper:
    https://arxiv.org/pdf/2010.11918.pdf
    """
    # model_type options: '', '-medium', '-large'
    def __init__(self, base_model = 'GPT2', model_size = '', num_drop = 0):
        self.base_model = base_model
        self.model_size = model_size
        self.technique = 'AdapterDrop Specialised'
        if base_model == 'GPT2':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2' + model_size, return_dict=False)
        else:
            raise Exception("Error: this library supports only GPT2 as the base model for now.")        
        self.num_drop = num_drop
        for params in self.model.parameters():
            params.requires_grad = False
        for i in range(self.num_drop, 12):
            self.model.transformer.h[i].attn.c_proj = Adaptered(self.model.transformer.h[i].attn.c_proj)
            self.model.transformer.h[i].mlp.c_proj = Adaptered(self.model.transformer.h[i].mlp.c_proj)


    def get_model(self):

        return self.model

    def get_num_drop(self):

        return self.num_drop


class RobustAdaptered(Adaptered):

    def __init__(self, orig_layer, drop=False):

        super(RobustAdaptered, self).__init__(orig_layer)
        self.drop = drop

    def forward(self, *x):

        orig_out = self.orig_layer(*x)
        if self.drop:
            return orig_out 
        output = (self.adapter.forward(orig_out[0].unsqueeze(0))[0],)[0]

        return output

    

class Model_with_adapterdrop_rob():
    """
    Robust AdapterDrop: Drawing the integer n randomly from [0, 11] for each 
    training batch. This yields one robust model that is applicable to a varying 
    number of dropped layers.
    For more information, please refer to the following paper:
    https://arxiv.org/pdf/2010.11918.pdf
    """
    def __init__(self, base_model = 'GPT2', model_size = ''):
        self.base_model = base_model
        self.model_size = model_size
        self.technique = 'AdapterDrop Robust'
        if base_model == 'GPT2':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2' + model_size, return_dict=False)
        else:
            raise Exception("Error: this library supports only GPT2 as the base model for now.")        
        for params in self.model.parameters():
            params.requires_grad = False
        for i in range(12):
            self.model.transformer.h[i].attn.c_proj = RobustAdaptered(self.model.transformer.h[i].attn.c_proj)
            self.model.transformer.h[i].mlp.c_proj = RobustAdaptered(self.model.transformer.h[i].mlp.c_proj)


    def get_model(self):

        return self.model

    def drop_adapters(self, n = 0):

        for i in range(n):
            self.model.transformer.h[i].attn.c_proj.drop = True
            self.model.transformer.h[i].mlp.c_proj.drop = True

    def reset_adapters(self):

        for i in range(12):
            self.model.transformer.h[i].attn.c_proj.drop = False
            self.model.transformer.h[i].mlp.c_proj.drop = False
