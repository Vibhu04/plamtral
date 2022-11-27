from transformers import GPT2LMHeadModel
import torch.nn as nn
from adapter import Adapter

    

class BapnaAdapter(Adapter):
    """
    A variant of the Houlsby Adapter. 
    For more information, please refer to the following paper:
    https://aclanthology.org/D19-1165.pdf
    """
    def __init__(self, size = 6, model_dim = 768):
        super(BapnaAdapter, self).__init__(size=size, model_dim=model_dim)

        self.adapter_block = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, size),
            nn.ReLU(),
            nn.Linear(size, model_dim)
        )


class BapnaAdaptered(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.adapter = BapnaAdapter()
        self.layer = layer

    def forward(self, *x):

        adap_out = self.adapter.forward(*x)
        out = self.layer(adap_out[0].unsqueeze(0))

        return out




class Model_with_adapter_bapna():

    # model_type options: '', '-medium', '-large'
    def __init__(self, base_model = 'GPT2', model_size = ''):
        self.base_model = base_model
        self.model_size = model_size
        self.technique = 'Bapna Adapter'
        if base_model == 'GPT2':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2' + model_size, return_dict=False)
        else:
            raise Exception("Error: this library supports only GPT2 as the base model for now.")

        for name, params in self.model.named_parameters():
            # Freeze model except the layer norm layers
            if 'ln' not in name:
                params.requires_grad = False
        for i in range(1, 12):
            # Insert the Bapna adapter at the end of each transformer block
            # (applying it to ln_1 layer means applying it at the end of the previous transformer block)
            self.model.transformer.h[i].ln_1 = BapnaAdaptered(self.model.transformer.h[i].ln_1)
        self.model.transformer.ln_f = BapnaAdaptered(self.model.transformer.ln_f)


    def get_model(self):

        return self.model