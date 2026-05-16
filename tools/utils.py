'''
* @name: tools.py
* @description: Other functions.
'''
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class AverageMeter(object):
    def __init__(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def update(self, value, count):
        self.value = value
        self.value_sum += value * count
        self.count += count
        self.value_avg = self.value_sum / self.count


class Contrastive_loss(nn.Module):
    def __init__(self, tau):
        super(Contrastive_loss, self).__init__()
        self.tau = tau

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z2))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True):
        l1 = self.semi_loss(z1, z2)
        l2 = self.semi_loss(z2, z1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret


def totolloss(mu, logvar, z, tgt):

    kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2
    kl_loss = kl_loss.sum(dim=1).mean()
    IB_loss = F.mse_loss(z, tgt)

    return (kl_loss + IB_loss) * 1e-3


def KL_regular(mu_1, logvar_1, mu_2, logvar_2, mu_3, logvar_3):
    var_1 = torch.exp(logvar_1)
    var_2 = torch.exp(logvar_2)
    var_3 = torch.exp(logvar_3)
    KL_loss1 = logvar_2 - logvar_1 + ((var_1.pow(2) + (mu_1 - mu_2).pow(2)) / (2 * var_2.pow(2))) - 0.5
    KL_loss2 = logvar_3 - logvar_1 + ((var_1.pow(2) + (mu_1 - mu_3).pow(2)) / (2 * var_3.pow(2))) - 0.5
    KL_loss3 = logvar_2 - logvar_3 + ((var_3.pow(2) + (mu_3 - mu_2).pow(2)) / (2 * var_2.pow(2))) - 0.5
    KL_loss = (KL_loss1 + KL_loss2 + KL_loss3).sum(dim=1).mean()
    return KL_loss


def reparameterise(mu, std):
    """
    mu : [batch_size,z_dim]
    std : [batch_size,z_dim]
    """
    # get epsilon from standard normal
    eps = torch.randn_like(std)
    return mu + std * eps


def con_loss(txt_mu, img_mu, aud_mu):
    Conloss = Contrastive_loss(0.5)
    while True:
        t_z1 = txt_mu.view(txt_mu.size(0), -1)
        t_z2 = txt_mu.view(txt_mu.size(0), -1)
        if not np.array_equal(t_z1, t_z2):
            break
    while True:
        i_z1 = img_mu.view(txt_mu.size(0), -1)
        i_z2 = img_mu.view(txt_mu.size(0), -1)

        if not np.array_equal(i_z1, i_z2):
            break
    while True:
        a_z1 = aud_mu.view(txt_mu.size(0), -1)
        a_z2 = aud_mu.view(txt_mu.size(0), -1)

        if not np.array_equal(a_z1, a_z2):
            break
    loss_t = Conloss(t_z1, t_z2)
    loss_i = Conloss(i_z1, i_z2)
    loss_a = Conloss(a_z1, a_z2)
    return loss_t + loss_i + loss_a