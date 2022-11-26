from transformers import GPT2LMHeadModel
import torch.nn as nn
from parameter_efficient.parallel_adapter import LayerAdaptered, Sum
from parameter_efficient.prefix_tuning import PrefixTuned



class PrefixRemoved(nn.Module):

    def __init__(self, layer, prefix_len):
        super().__init__()
        self.layer = layer
        self.prefix_len = prefix_len

    def forward(self, *x):

        out = self.layer(*x)
        out = out[:, self.prefix_len:, :]

        return out



class Model_with_mam_adapter():

    """
    An adapter variant which is designed by integrating favorable design 
    elements from other popular parametre efficient approaches like adapters, 
    prefix tuning and LoRA.
    For more information, please refer to the following paper:
    https://arxiv.org/pdf/2110.04366.pdf
    """
    # model_type options: '', '-medium', '-large'
    def __init__(self, base_model = 'GPT2', model_size = '', prefix_len = 10, k = 42, r = 512, scale = 4):
        self.base_model = base_model
        self.model_size = model_size
        self.technique = "MAM Adapter"
        if base_model == 'GPT2':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2' + model_size, return_dict=False)
        else:
            raise Exception("Error: this library supports only GPT2 as the base model for now.")
        for params in self.model.parameters():
            params.requires_grad = False
        for i in range(12):
            self.model.transformer.h[i].attn.c_attn = PrefixTuned(self.model.transformer.h[i].attn.c_attn, prefix_len, k, False, False)
            self.model.transformer.h[i].attn.resid_dropout = PrefixRemoved(self.model.transformer.h[i].attn.resid_dropout, prefix_len)
            self.model.transformer.h[i].mlp.c_fc = LayerAdaptered(self.model.transformer.h[i].mlp.c_fc, r, scale)
            self.model.transformer.h[i].mlp.dropout = Sum(self.model.transformer.h[i].mlp.dropout, self.model.transformer.h[i].mlp.c_fc)

    def get_model(self):

        return self.model

