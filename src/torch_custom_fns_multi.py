from torch_custom_fns import _mag_sort, two_sum_reduce

from multiprocessing import Pool

import torch


def _one_pass_mean_std_multi(batch_iter):

    total_iters = 0

    acc = []
    acc2 = []
    for batch in batch_iter:

        batch_size = len(batch)

        total_iters += batch_size

        batch = batch.to(torch.float64)

        batch2 = batch * batch

        a = two_sum_reduce(batch)
        acc += a

        a2 = two_sum_reduce(batch2)
        acc2 += a2

    acc = two_sum_reduce(acc, minmum=True)
    acc2 = two_sum_reduce(acc2, minmum=True)

    return {"acc": acc, "acc2": acc2, "total_iters": total_iters}


def one_pass_mean_std_multi(batch_iters, num_procs):

    with Pool(num_procs) as p:
        all_acc_data = p.map(_one_pass_mean_std_multi, batch_iters)

    # collapse all results
    acc = []
    acc2 = []
    total_iters = 0
    for acc_data in all_acc_data:

        acc += acc_data["acc"]
        acc2 += acc_data["acc2"]
        total_iters += acc_data["total_iters"]

    acc = two_sum_reduce(acc, minmum=True)
    acc2 = two_sum_reduce(acc2, minmum=True)

    mean_acc = [a / total_iters for a in acc]
    mean_acc = two_sum_reduce(acc=mean_acc)
    mean_acc_t = torch.stack(mean_acc)
    mean_acc_t = _mag_sort(mean_acc_t)
    mean = mean_acc_t[-1]

    mean_acc_sqr = [a1 * a2 for a1 in mean_acc for a2 in mean_acc]
    mean_acc_sqr = two_sum_reduce(acc=mean_acc_sqr)
    neg_mean_acc_sqr = [-a for a in mean_acc_sqr]

    mean_acc2 = [a / total_iters for a in acc2]
    mean_acc2 = two_sum_reduce(acc=mean_acc2)

    std_2_acc = two_sum_reduce(acc=mean_acc2 + neg_mean_acc_sqr)

    std_2_acc_t = torch.stack(std_2_acc)
    std_2_acc_t = _mag_sort(std_2_acc_t)
    std_2 = std_2_acc_t[-1]
    std = std_2.sqrt()

    return mean, std, total_iters
