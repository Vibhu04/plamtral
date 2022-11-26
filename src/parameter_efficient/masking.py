from transformers import GPT2LMHeadModel
import torch.nn as nn
import torch.autograd



class STEFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return nn.functional.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self, threshold):
        super(StraightThroughEstimator, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class Masked(nn.Module):

    def __init__(self, orig_layer, threshold):
        super().__init__()
        self.orig_layer = orig_layer
        self.threshold = threshold
        print(self.orig_layer.weight.size())
        #input()
        self.pre_mask = nn.Parameter(torch.rand(self.orig_layer.weight.size(), requires_grad=True)).to('cuda')
        self.masking_layer = StraightThroughEstimator(self.threshold)

    def forward(self, *x):

        #orig_out = self.orig_layer(*x)
        print(self.pre_mask)
        input("Pre mask")
        mask = nn.Parameter(self.masking_layer.forward(self.pre_mask)).to('cuda')
        masked_layer = self.orig_layer
        print(self.orig_layer.weight.is_cuda)
        print(masked_layer.weight.is_cuda)
        print(mask.is_cuda)
        masked_layer.weight = nn.Parameter(self.orig_layer.weight * mask).to('cuda')
        # print("Mask:", mask)
        # input()
        output = masked_layer(*x)
        #output = (self.adapter.forward(orig_out[0].unsqueeze(0))[0],)
        return output



class Model_with_mask(nn.Module):

    # model_type options: '', '-medium', '-large'
    def __init__(self, base_model = 'GPT2', model_type = ''):
        super().__init__()
        self.threshold = 0.5
        if base_model == 'GPT2':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2' + model_type, return_dict=False)
        else:
            raise Exception("Error: this library supports only GPT2 as the base model for now.")
        for i in range(12):
            self.model.transformer.h[i].attn.c_attn = Masked(self.model.transformer.h[i].attn.c_attn, self.threshold)
            #self.model.transformer.h[i].attn.c_proj = Masked(self.model.transformer.h[i].attn.c_proj, self.threshold)
            #self.model.transformer.h[i].mlp.c_fc = Masked(self.model.transformer.h[i].mlp.c_fc, self.threshold)
            #self.model.transformer.h[i].mlp.c_proj = Masked(self.model.transformer.h[i].mlp.c_proj, self.threshold)


    def get_model(self):

        return self.model





