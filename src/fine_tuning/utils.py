def apply_discriminative_lr(model, base_lr, dlr_factor):
    """
    Applies learning rate to parameter groups in an exponentially decreasing 
    manner. Refer to the following paper for more information:
    https://arxiv.org/pdf/1801.06146.pdf
    """

    lr = base_lr
    parameters = []
    prev_group_name = ''

    for name, param in reversed(list(model.named_parameters())):
        cur_group_name = get_group_name(name)
        if cur_group_name != prev_group_name:
            if prev_group_name != '':
                lr *= dlr_factor
            prev_group_name = cur_group_name

        parameters += [{'params': param, 'lr': lr}]

    parameters.reverse()

    return parameters



def freeze_model(model):

    for p in model.parameters():
        p.requires_grad = False

    return model


def freeze_except_bias(model):

    for name, params in model.named_parameters():
        if 'bias' in name:
            params.requires_grad = True
        else:
            params.requires_grad = False

    return model


def get_group_name(name):
    """
    Returns the group name given a model layer name.
    Note: this is specific to gpt-2 for the moment.
    """
    tags = name.split('.')
    if len(tags) == 3:
        return ".".join(tags[:2])
    else:
        return ".".join(tags[:4])


def last_unfrozen_group(model):
    """
    Returns the name of the last unfrozen group and of the previous-to-last.
    """
    prev_group_name = ''
    for name, param in model.named_parameters():
        group_name = get_group_name(name)
        if not param.requires_grad:
            if group_name != prev_group_name:
                prev_group_name = group_name
        else:
            break

    return group_name, prev_group_name


def get_prev_group_name(model, group_name):
    """
    Given a group name, return the name of the previous group.
    """
    prev_group_name = ''
    for name, param in model.named_parameters():
        cur_group_name = get_group_name(name)
        if cur_group_name == group_name:
            return prev_group_name
        else:
            if cur_group_name != prev_group_name:
                prev_group_name = cur_group_name


def get_next_group_name(model, group_name):
    """
    Given a group name, return the name of the next group.
    """
    next_group_name = ''
    for name, param in reversed(list(model.named_parameters())):
        cur_group_name = get_group_name(name)
        if cur_group_name == group_name:
            return next_group_name
        else:
            if cur_group_name != next_group_name:
                next_group_name = cur_group_name


def gradually_unfreeze(model, num_groups):
    """
    Unfreezes the model layer by layer from the top according to 
    args.unfreeze_freq. Refer to the following paper for more information:
    https://arxiv.org/pdf/1801.06146.pdf
    """
  
    groups_to_unfreeze = []

    last_unfrozen, to_unfreeze = last_unfrozen_group(model)
    groups_to_unfreeze.append(to_unfreeze)
    for i in range(num_groups - 1):
        group_name = get_prev_group_name(groups_to_unfreeze[-1])
        groups_to_unfreeze.append(group_name)

    for name, param in model.named_parameters():
        for group in groups_to_unfreeze:
            if group in name:
                param.requires_grad = True
                break

    return model


def chain_thaw(model, num_groups):
    """
    Unfreeze layers from the top one by one and one at a time.
    """

    groups_to_freeze = []
    groups_to_unfreeze = []
    last_unfrozen, to_unfreeze = last_unfrozen_group(model)
    groups_to_freeze.append(last_unfrozen)
    groups_to_unfreeze.append(to_unfreeze)

    for i in range(num_groups - 1):
        groups_to_freeze.append(get_next_group_name(model, groups_to_freeze[-1]))
        groups_to_unfreeze.append(get_prev_group_name(model, groups_to_unfreeze[-1]))

    for name, param in model.named_parameters():
        for group in groups_to_unfreeze:
            if group in name:
                param.requires_grad = True
                break
        else:
            for group in groups_to_freeze:
                if group in name:
                    param.requires_grad = False
                    break

    return model



def unfreeze_last_n_layers(model, n):
    """
    Unfreeze last n layers (groups) of the model.
    """
    prev_group_name = ''
    unfrozen = 0
    for name, param in reversed(list(model.named_parameters())):
        group_name = get_group_name(name)
        if group_name != prev_group_name:
            prev_group_name = group_name
            unfrozen += 1
            if unfrozen > n:
                break
        param.requires_grad = True

    return model


def freeze_first_n_layers(model, n):
    """
    Freeze first n layers (groups) of the model
    """
    prev_group_name = ''
    frozen = 0
    for name, param in model.named_parameters():
        group_name = get_group_name(name)
        if group_name != prev_group_name:
            prev_group_name = group_name
            frozen += 1
            if frozen > n:
                break
        param.requires_grad = False

    return model



def freeze_nth_layer(model, n):
    """
    Freeze nth layer (group) of the model
    """
    prev_group_name = ''
    layer = 0
    for name, param in model.named_parameters():
        group_name = get_group_name(name)
        if group_name != prev_group_name:
            prev_group_name = group_name
            layer += 1
            if layer > n:
                break
        if layer == n:
            param.requires_grad = False

    return model


