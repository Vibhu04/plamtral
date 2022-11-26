from transformers import GPT2LMHeadModel
import torch.nn as nn


class LoRABlock(nn.Module):

    def __init__(self, r, in_dim, out_dim, alpha):

        self.r = r
        self.alpha = alpha
        super().__init__()
        self.lora_A = nn.Linear(in_dim, r)
        self.lora_B = nn.Linear(r, out_dim)
        self.lora_B.weight.data.zero_()
        self.lora_B.bias.data.zero_()

    def forward(self, x):

        out_A = self.lora_A(x)
        out = self.lora_B(out_A)

        return out

    def get_r():

        return self.r

    def get_alpha():

        return self.alpha


class LoRALayer(nn.Module):

    def __init__(self, orig_layer):
        super().__init__()
        self.orig_layer = orig_layer
        self.lora = LoRABlock(2, 768, 768, 1)

    def forward(self, *x):

        orig_out = self.orig_layer(*x)
        lora_out = self.lora.forward(*x)
        lora_out *= self.lora.get_r() / self.lora.get_alpha()

        out = orig_out + lora_out

        return out



class Model_with_LoRA():
    """
    LoRA approach freezes the pretrained model weights and injects trainable 
    rank decomposition matrices into each layer of the Transformer architecture.
    For more information, please refer to the following paper:
    https://arxiv.org/pdf/2106.09685.pdf
    """
    def __init__(self, base_model = 'GPT2', model_size = ''):
        self.base_model = base_model
        self.model_size = model_size
        self.technique = 'LoRA'
        if base_model == 'GPT2':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2' + model_size, return_dict=False)
        else:
            raise Exception("Error: this library supports only GPT2 as the base model for now.")
        for params in self.model.parameters():
            params.requires_grad = False
        for i in range(12):
            # Insert LoRA modules into the attention sublayers
            self.model.transformer.h[i].attn.c_attn = LoRALayer(self.model.transformer.h[i].attn.c_attn)
            self.model.transformer.h[i].attn.c_proj = LoRALayer(self.model.transformer.h[i].attn.c_proj)

        
    def get_model(self):

        return self.model
