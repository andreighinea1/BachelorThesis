from operator import itemgetter


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
