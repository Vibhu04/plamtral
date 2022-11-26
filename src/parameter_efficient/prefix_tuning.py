from transformers import GPT2LMHeadModel
import torch.nn as nn
import torch


# To figure out how to make this local
to_add = None

class PrefixTuned(nn.Module):

    def __init__(self, layer, prefix_len, k, replace_prefix, calc_dif):
        super().__init__()
        self.layer = layer
        self.prefix_len = prefix_len
        self.calc_dif = calc_dif
        self.replace_prefix = replace_prefix
        self.prefix_mlp = nn.Linear(k, 768)
        self.prefix_prime = nn.parameter.Parameter(torch.rand(prefix_len, k))
        self.prefix = None
        self.old_inp = None
        self.new_inp = None
        
        
    def forward(self, *x):

        global to_add
        # Parameterisation of the prefix
        self.prefix = self.prefix_mlp(self.prefix_prime).unsqueeze(0)
        self.old_inp = x
        num_inputs = x[0].size(dim=1)
        x = list(x)
        if self.replace_prefix:
            x[0] = torch.cat((self.prefix, x[0][:, self.prefix_len:]), dim=1)
        else:
            x[0] = torch.cat((self.prefix, x[0]), dim=1)
        x = tuple(x)
        self.prefixed = True
        out = self.layer(x[0])
        self.new_inp = x
        if self.calc_dif:
            to_add = torch.sub(self.new_inp[0], self.old_inp[0])

        return out



class SumOutput(nn.Module):

      def __init__(self, layer):
          super().__init__()
          self.layer = layer

      def forward(self, *x):

          out = self.layer(*x)
          out += to_add

          return out




class Model_with_prefix_tuning():
    """
    Prefix Tuning keeps language model parameters frozen, but optimizes a small 
    continuous task-specific vector (called the prefix).
    For more information, please refer to the following paper:
    https://arxiv.org/pdf/2101.00190.pdf
    """
    # model_type options: '', '-medium', '-large'
    def __init__(self, base_model = 'GPT2', model_size = '', prefix_len = 10, k = 42):
        self.base_model = base_model
        self.model_size = model_size
        self.technique = 'Prefix Tuning'
        if base_model == 'GPT2':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2' + model_size, return_dict=False)
        else:
            raise Exception("Error: this library supports only GPT2 as the base model for now.")
        for params in self.model.parameters():
            params.requires_grad = False
        for i in range(1, 12):
              self.model.transformer.h[i].ln_1 = PrefixTuned(self.model.transformer.h[i].ln_1, prefix_len, k, True, True)
              self.model.transformer.h[i].attn.resid_dropout = SumOutput(self.model.transformer.h[i].attn.resid_dropout)
              self.model.transformer.h[i].mlp.dropout = SumOutput(self.model.transformer.h[i].mlp.dropout)
        self.model.transformer.ln_f = PrefixTuned(self.model.transformer.ln_f, prefix_len, k, True, True)


    def get_model(self):

        return self.model