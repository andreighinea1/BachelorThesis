from operator import itemgetter

import torch


def sort_dict_by_keys(d: dict, reverse=False, top=None):
    r = {k: v for k, v in sorted(d.items(), key=itemgetter(0), reverse=reverse)}
    if top:
        return dict(list(r.items())[:top])
    return r


def sort_dict_by_values(d: dict, reverse=False, top=None):
    r = {k: v for k, v in sorted(d.items(), key=itemgetter(1), reverse=reverse)}
    if top:
        return dict(list(r.items())[:top])
    return r


def custom_logsumexp(x):
    # Ref: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    max_x = torch.max(x, dim=1, keepdim=True)[0]
    exp_negative = torch.exp(x - max_x)
    sum_exp_negative = torch.sum(exp_negative, dim=1, keepdim=True)
    return torch.log(sum_exp_negative + 1e-6) + max_x.squeeze()
