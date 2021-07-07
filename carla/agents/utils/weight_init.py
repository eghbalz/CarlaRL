"""
Created by Hamid Eghbal-zadeh at 19.11.20
Johannes Kepler University of Linz
"""
import torch.nn as nn
import math
import torch.nn.init as init
import torch


def weights_init_kaiming_normal(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out')
            init.constant_(m.bias, 0)
        elif type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    init.kaiming_normal_(param.data, mode='fan_out')
                elif 'weight_hh' in name:
                    init.kaiming_normal_(param.data, mode='fan_out')
                elif 'bias' in name:
                    param.data.fill_(0)


def weights_init_kaiming_uniform(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_uniform_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, mode='fan_out')
            init.constant_(m.bias, 0)
        elif type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    init.kaiming_uniform_(param.data, mode='fan_out')
                elif 'weight_hh' in name:
                    init.kaiming_uniform_(param.data, mode='fan_out')
                elif 'bias' in name:
                    param.data.fill_(0)


def weights_init_xavier_with_nonlin(net,nonlin='relu'):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(nonlin))
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(nonlin))
            init.constant_(m.bias, 0)


def weights_init_xavier(module):
    if isinstance(module, nn.Conv2d):
        init.xavier_uniform_(module.weight.data,1.)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


def lecun_normal_(tensor):
    mode = 'fan_in'
    gain = 1.
    fan = init._calculate_correct_fan(tensor, mode)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)


def weights_init_lecun(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            lecun_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            lecun_normal_(m.weight)
            init.constant_(m.bias, 0)
        elif type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    lecun_normal_(param.data)
                elif 'weight_hh' in name:
                    lecun_normal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)


def he_normal_(tensor):
    mode = 'fan_in'
    gain = 2.
    fan = init._calculate_correct_fan(tensor, mode)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)


def weights_init_he(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
