from transformers import GPT2LMHeadModel
import torch.nn as nn

class Adapter(nn.Module):
    """
    The adapter first projects the original d-dimensional features into a 
    smaller dimension m, applies a nonlinearity, then projects back to d dimensions.
    For more information please refer to the following paper:
    https://arxiv.org/pdf/1902.00751.pdf
    """
    def __init__(self, size = 6, model_dim = 768, scale_factor = None, add_res = True):
        super().__init__()

        self.adapter_block = nn.Sequential(
            nn.Linear(model_dim, size),
            nn.ReLU(),
            nn.Linear(size, model_dim)
        )

        self.scale_factor = scale_factor
        self.add_res = add_res

    def forward(self, x):

        out = self.adapter_block(x)
        # Skip connection
        if self.add_res:
            out += x
        if self.scale_factor is not None:
            out *= self.scale_factor
        

        return out


class Adaptered(nn.Module):
    def __init__(self, orig_layer):
        super().__init__()
        self.orig_layer = orig_layer
        self.adapter = Adapter()

    def forward(self, *x):

        orig_out = self.orig_layer(*x)
        output = (self.adapter.forward(orig_out[0].unsqueeze(0))[0],)[0]

        return output




class Model_with_adapter():

    # model_type options: '', '-medium', '-large'
    def __init__(self, base_model = 'GPT2', model_size = ''):
        self.base_model = base_model
        self.model_size = model_size
        self.technique = 'Houlsby Adapter'
        if base_model == 'GPT2':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2' + model_size, return_dict=False)
        else:
            raise Exception("Error: this library supports only GPT2 as the base model for now.")
        for name, params in self.model.named_parameters():
            # Freeze model except the layer norm layers
            if 'ln' not in name:
                params.requires_grad = False
        # Insert adapters at the end of the attention and mlp sublayers
        for i in range(12):
            self.model.transformer.h[i].attn.c_proj = Adaptered(self.model.transformer.h[i].attn.c_proj)
            self.model.transformer.h[i].mlp.c_proj = Adaptered(self.model.transformer.h[i].mlp.c_proj)


    def get_model(self):

        return self.model