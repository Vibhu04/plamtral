import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import random
from transformers import get_linear_schedule_with_warmup
from transformers import GPT2LMHeadModel, AutoConfig
from fine_tuning.utils import *
from fine_tuning.stlr import STLR
from data.dataset import TL_Dataset



def load_dataloaders(base_model = 'GPT2', model_size = '', dataset_path = None, block_size = 1000, train_split = 0.95, test_split = 0.25, batch_size = 1, num_workers = 0):

    full_dataset = TL_Dataset(base_model, model_size, dataset_path, block_size)
    train_size = int(train_split * len(full_dataset))
    valtest_size = len(full_dataset) - train_size
    train_dataset, valtest_dataset = torch.utils.data.random_split(full_dataset, [train_size, valtest_size])
    test_size = int(test_split * valtest_size)
    val_size = valtest_size - test_size
    val_dataset, test_dataset = torch.utils.data.random_split(valtest_dataset, [val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader



def get_optimizer(model, model_obj, base_lr, weight_decay):

    if model_obj.technique == 'ULMFiT':
        model_params = apply_discriminative_lr(model, base_lr, model_obj.dlr_factor)
        optimizer = AdamW(model_params, weight_decay=weight_decay)
    else:
        optimizer = AdamW(model.parameters(), weight_decay=weight_decay, lr=base_lr) 

    return optimizer



def print_model(model, print_params=False, print_sum_weights=True, check=False):
    for idx, (name, param) in enumerate(model.named_parameters()):
        print(f'{idx}: {name}')
        if print_params:
            print(param)
        print("Requires grad:", param.requires_grad)
        print("Shape:", param.size())
        print("Sum of weights:", param.abs().sum())
        if check:
            input()
    input()


def print_lr(optimizer):
    print('Learning rates:')
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    input()


def prepend_tokens(tokens, num_tokens, token, device): 

    to_prepend = torch.full((1, num_tokens), token).to(device)
    tokens = torch.cat((to_prepend, tokens), dim=1).type(torch.long)

    return tokens


# def adjust_model(model, model_obj, seq_count):
#     """
#     Checks args and makes modifications to model as necessary.
#     This function is called before the training loop.
#     """
#     technique = model_obj.technique
#     elif technique == 'Chain Thaw':
#         model = freeze_model(model) 
#         model = chain_thaw(model, 1)

#     return model


def batch_routine(model, model_obj):

    """
    Checks args and makes modifications to model as necessary.
    This function is called in the training loop at the end of each batch.
    """
    if model_obj.technique == 'Adapter Drop Robust':
        model_obj.model = model
        model_obj.reset_adapters()
        num_drop = random.randint(0, 11)
        model_obj.drop_adapters(num_drop)
        model = model_obj.model

    return model


def process_tokens(tokens, device, technique):

    if technique == 'Prefix Tuning':
        tokens = prepend_tokens(tokens, 2, 0, device)
    if technique == 'Prompt Tuning':
        tokens = prepend_tokens(tokens, 2, 0, device)

    return tokens



def modify(model, model_obj, seq_count):

    """
    Checks args and makes modifications to model as necessary.
    This function is called in the training loop at the end of each iteration.
    """
    technique = model_obj.technique
    if technique == 'ULMFiT' or technique == 'Gradual Unfreezing':
        if seq_count % model_obj.unfreeze_freq == 0:
            model = gradually_unfreeze(model, 1)
    elif technique == 'Chain Thaw':
        if seq_count % args.unfreeze_freq == 0:
            model = chain_thaw(model, 1)

    return model



def select_scheduler(scheduler, technique, optimizer, train_size, epochs, actual_batch_size, warmup_steps):

    if technique == 'ULMFiT' or scheduler == 'STLR':
        num_iters = epochs * train_size / actual_batch_size
        scheduler = STLR(optimizer, num_iters=num_iters, cut_frac=0.1, ratio=32)
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
        )

    return scheduler



def get_GPT2LMH(model_size, dropout):

    config = AutoConfig.from_pretrained('gpt2' + model_size)
    if dropout is not None:
        config.resid_pdrop = dropout
        config.embd_pdrop = dropout
        config.attn_pdrop = dropout
    model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path = 'gpt2' + model_size, config = config)

    return model



def verify_args(args):

    """
    Verify whether the args provided are mutually consistent or not.
    """
    if args.gradual_unfreezing and args.chain_thaw:
        raise Exception("Gradual unfreezing and chain thaw cannot both be enabled together")


