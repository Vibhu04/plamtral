from transformers import GPT2LMHeadModel
import torch.nn as nn
import torch



class PromptTuned(nn.Module):

    def __init__(self, inp_layer, prompt_len):
        super().__init__()
        self.inp_layer = inp_layer
        self.prompt_len = prompt_len
        self.prompt = torch.nn.parameter.Parameter(data=torch.rand(1, prompt_len, 768))
        
        
    def forward(self, *x):

        num_inputs = x[0].size(dim=1)
        x = list(x)
        x[0] = (x[0][:, self.prompt_len:])
        x = tuple(x)
        inp_out = self.inp_layer(*x)
        cat_out = torch.cat((self.prompt, inp_out), dim=1)

        return cat_out



class Model_with_prompt_tuning():
    """
    Prompt tuning prepends a parameterised prompt to the input of the model.
    For more information, please refer to the following paper:
    https://aclanthology.org/2021.emnlp-main.243.pdf
    """
    # model_type options: '', '-medium', '-large'
    def __init__(self, base_model = 'GPT2', model_size = '', prompt_len=2):
        self.base_model = base_model
        self.model_size = model_size
        self.technique = 'Prompt Tuning'
        if base_model == 'GPT2':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2' + model_size, return_dict=False)
        else:
            raise Exception("Error: this library supports only GPT2 as the base model for now.")
        for params in self.model.parameters():
            params.requires_grad = False

        self.model.transformer.wte = PromptTuned(self.model.transformer.wte, prompt_len)
        


    def get_model(self):

        return self.model