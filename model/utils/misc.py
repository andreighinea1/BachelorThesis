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


def get_pred_str(all_predictions, verdict_to_str):
    total_predictions = len(all_predictions)
    class_counts = {i: (all_predictions == i + 1).sum() for i in [-1, 0, 1]}
    percentages = {verdict_to_str[k]: f"{v / total_predictions:.2%}" for k, v in class_counts.items()}
    return str(percentages).replace("'", "")


def _get_accuracy_for_section(all_predictions, verdict_to_str, start_p=0.0, end_p=1.0):
    start = int(start_p * all_predictions.shape[0])
    end = int(end_p * all_predictions.shape[0])
    pred = all_predictions[start:end]
    return get_pred_str(pred, verdict_to_str)


def accuracy_bar_section(end_str, percentage, first=True, total_bar_length=20):
    filled_length = int(total_bar_length * percentage)
    empty_length = total_bar_length - filled_length
    if first:
        bar = '█' * filled_length + '-' * empty_length
    else:
        bar = '-' * empty_length + '█' * filled_length

    start_label = "First" if first else "Last "
    return f"{start_label} {int(percentage * 100)}% [{bar}] -> {end_str}"


def print_accuracy_progress_bars(predictions_ok_tensor, verdict_to_str, total_bar_length=20):
    first_20 = _get_accuracy_for_section(predictions_ok_tensor, verdict_to_str, start_p=0, end_p=0.2)
    first_50 = _get_accuracy_for_section(predictions_ok_tensor, verdict_to_str, start_p=0, end_p=0.5)
    first_80 = _get_accuracy_for_section(predictions_ok_tensor, verdict_to_str, start_p=0, end_p=0.8)
    last_80 = _get_accuracy_for_section(predictions_ok_tensor, verdict_to_str, start_p=0.2, end_p=1.0)
    last_50 = _get_accuracy_for_section(predictions_ok_tensor, verdict_to_str, start_p=0.5, end_p=1.0)
    last_20 = _get_accuracy_for_section(predictions_ok_tensor, verdict_to_str, start_p=0.8, end_p=1.0)

    print(accuracy_bar_section(first_20, 0.2, first=True, total_bar_length=total_bar_length))
    print(accuracy_bar_section(first_50, 0.5, first=True, total_bar_length=total_bar_length))
    print(accuracy_bar_section(first_80, 0.8, first=True, total_bar_length=total_bar_length))
    print(accuracy_bar_section(last_80, 0.8, first=False, total_bar_length=total_bar_length))
    print(accuracy_bar_section(last_50, 0.5, first=False, total_bar_length=total_bar_length))
    print(accuracy_bar_section(last_20, 0.2, first=False, total_bar_length=total_bar_length))
    return
