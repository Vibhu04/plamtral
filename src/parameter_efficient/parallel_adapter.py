from transformers import GPT2LMHeadModel
import torch.nn as nn
from adapter import Adapter



class LayerAdaptered(nn.Module):
    """
    Inserts an adapter parallel to orig_layer
    """
    def __init__(self, orig_layer, adap_size, scale_factor):
        super().__init__()
        self.orig_layer = orig_layer
        self.adapter = Adapter(size=adap_size, add_res=False, scale_factor=scale_factor)
        self.adapter_out = 0

    def forward(self, *x):

        orig_out = self.orig_layer(*x)
        self.adapter_out = self.adapter.forward(*x)

        return orig_out

    def get_adap_out(self):

        return self.adapter_out




class Sum(nn.Module):

    def __init__(self, orig_layer, adap_obj):
        super().__init__()
        self.orig_layer = orig_layer
        self.adap_obj = adap_obj

    def forward(self, *x):
        
        orig_out = self.orig_layer(*x)
        out = orig_out + self.adap_obj.adapter_out

        return out
	  




class EmbeddingAdaptered(nn.Module):
  
    def __init__(self, embed_layer):
        super().__init__()
        self.embed_layer = embed_layer
        self.adapter = Adapter()

    def forward(self, *x):

        embed_out = self.embed_layer(*x)
        adapter_out = (self.adapter.forward(embed_out[0].unsqueeze(0))[0],)[0]
        out = embed_out + adapter_out

        return out




class Model_with_parallel_adapter():
    """
    An adapter variant that arranges adapters in parallel instead of in series.
    For more information, please refer to the following paper:
    https://arxiv.org/pdf/2104.08154v1.pdf
    """
    # model_type options: '', '-medium', '-large'
    def __init__(self, base_model = 'GPT2', model_size = '', adap_size = 100):
        self.base_model = base_model
        self.model_size = model_size
        self.technique = 'Parallel Adapter'
        if base_model == 'GPT2':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2' + model_size, return_dict=False)
        else:
            raise Exception("Error: this library supports only GPT2 as the base model for now.")
        for params in self.model.parameters():
            params.requires_grad = False
        self.model.transformer.wte = EmbeddingAdaptered(self.model.transformer.wte)
        for i in range(12):
            self.model.transformer.h[i].attn.c_attn = LayerAdaptered(self.model.transformer.h[i].attn.c_attn, adap_size, None)
            self.model.transformer.h[i].attn.resid_dropout = Sum(self.model.transformer.h[i].attn.resid_dropout, self.model.transformer.h[i].attn.c_attn)
            self.model.transformer.h[i].mlp.c_fc = LayerAdaptered(self.model.transformer.h[i].mlp.c_fc, adap_size, None)
            self.model.transformer.h[i].mlp.dropout = Sum(self.model.transformer.h[i].mlp.dropout, self.model.transformer.h[i].mlp.c_fc)

    def get_model(self):

        return self.model