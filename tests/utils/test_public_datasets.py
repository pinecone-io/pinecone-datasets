import numpy as np


def is_dicts_equal(d1, d2):
    return d1.keys() == d2.keys() and recursive_dict_compare(d1, d2)


def deep_list_cmp(l1, l2):
    same = True
    for left, right in zip(l1, l2):
        same = same and left == right
    return same


def approx_deep_list_cmp(l1, l2):
    same = True
    for left, right in zip(l1, l2):
        same = same and np.isclose(left, right)
    return same


def recursive_dict_compare(d1, d2):
    for k, v in d1.items():
        if isinstance(v, dict):
            return recursive_dict_compare(v, d2[k])
        elif isinstance(v, (list, np.ndarray)):
            return deep_list_cmp(v, d2[k])
        return v == d2[k]
