# -*- coding:utf-8 -*-
# author: Xinge
# @file: load_save_util.py 

import torch


def load_checkpoint(model_load_path, model):
    my_model_dict = model.state_dict()
    checkpoint = torch.load(model_load_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # Full checkpoint format with model_state_dict, optimizer, etc.
            pre_weight = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            # Checkpoint format with state_dict key
            pre_weight = checkpoint['state_dict']
        else:
            # Assume it's a direct state_dict (OrderedDict)
            pre_weight = checkpoint
    else:
        pre_weight = checkpoint

    part_load = {}
    match_size = 0
    nomatch_size = 0
    for k in pre_weight.keys():
        value = pre_weight[k]
        # Skip non-tensor values
        if not isinstance(value, torch.Tensor):
            continue
        
        # Strip 'module.' prefix if present (from DataParallel/DistributedDataParallel)
        key = k
        if k.startswith('module.'):
            key = k[7:]  # Remove 'module.' prefix
        
        if key in my_model_dict and my_model_dict[key].shape == value.shape:
            # print("loading ", k)
            match_size += 1
            part_load[key] = value
        else:
            nomatch_size += 1

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)

    return model


def load_checkpoint_1b1(model_load_path, model):
    my_model_dict = model.state_dict()
    pre_weight = torch.load(model_load_path)

    part_load = {}
    match_size = 0
    nomatch_size = 0

    pre_weight_list = [*pre_weight]
    my_model_dict_list = [*my_model_dict]

    for idx in range(len(pre_weight_list)):
        key_ = pre_weight_list[idx]
        key_2 = my_model_dict_list[idx]
        value_ = pre_weight[key_]
        if my_model_dict[key_2].shape == pre_weight[key_].shape:
            # print("loading ", k)
            match_size += 1
            part_load[key_2] = value_
        else:
            print(key_)
            print(key_2)
            nomatch_size += 1

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)

    return model
