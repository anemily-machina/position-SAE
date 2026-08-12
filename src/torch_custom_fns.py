from utils import set_random_seeds


import math
from random import sample
import time

import torch
from tqdm import tqdm


def tensor_two_sum(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    https://uwplse.org/2025/08/04/two-sum.html

    taken directly without the wrapping of the custom pip package

    Technically this should work for any inputs that have properly defined + and - operations
    not just tensors
    """

    s = x + y
    xx = s - y
    yy = s - xx
    ex = x - xx
    ey = y - yy
    e = ex + ey
    return s, e


def _mag_sort(t: torch.Tensor) -> torch.Tensor:

    indices = torch.argsort(torch.abs(t), dim=0, descending=False)

    sorted_t = torch.gather(t, dim=0, index=indices)

    return sorted_t


def two_sum_reduce(acc: list[torch.Tensor], minmum=False):

    acc = iter_two_sum(vecs=acc, acc=None)

    prev_size = None
    next_size = len(acc)

    # until there is no more compacting
    while minmum and prev_size != next_size:

        acc = iter_two_sum(vecs=acc, acc=None)

        prev_size = next_size
        next_size = len(acc)

    return acc


def iter_two_sum(vecs: list[torch.Tensor], acc: list[torch.Tensor]) -> torch.Tensor:
    """

    vecs will be converted to the accumilator dtype and device

    add each vector in vecs to acc (technically works for anydimensional tensor as long as they are all the same dimension)

    vecs - indexable list of embddings of size d (probably a 2D tensor)

    acc - list of accumilators, if None uses the first element of vecs
    acc[0] = running sum, acc[k] = running sum of errors for computations of depth k (added as needed)

    you shoud run this algorithm again on the with vecs=acc and acc = 0s until vecs = accs
    """

    if acc is None:

        acc = [vecs[0].to(torch.float64)]

        vecs = vecs[1:]

    for v in vecs:

        v = v.to(torch.float64)

        curr_acc = acc[0]

        s, e = tensor_two_sum(v, curr_acc)

        acc[0] = s

        e_i = 1
        while e.any():

            e = e.squeeze()

            # if this is a new level of error
            if e_i == len(acc):

                acc.append(e)
                break

            else:

                curr_acc = acc[e_i]

                s, e = tensor_two_sum(e, curr_acc)

                acc[e_i] = s

            e_i += 1

    return acc


def _one_test_sum(all_values):

    acc = None
    for values in tqdm(all_values, total=len(all_values), ncols=60):

        acc = iter_two_sum(values, acc)

        if len(acc) > 10:
            acc = two_sum_reduce(acc=acc)

    acc = two_sum_reduce(acc=acc)

    return acc


def _test_tensor_two_sum():

    display_iters = 1000
    total_iters = 30000

    dt = torch.float16  # low so we can see errors quicly

    ti = total_iters
    all_values = []

    while ti > 0:

        di = min(ti, display_iters)

        values = torch.rand([di, 3], dtype=dt)

        all_values.append(values)

        ti -= di

    acc = _one_test_sum(all_values)

    rand_values = sample(all_values, k=len(all_values))

    rand_acc = _one_test_sum(rand_values)

    neg_acc = [-a for a in acc]

    comb_acc = [rand_acc + neg_acc]

    _one_test_sum(comb_acc)

    exit()


def one_pass_mean_std(batch_iter, num_batches=None):

    total_iters = 0

    acc = []
    acc2 = []
    for batch in tqdm(batch_iter, total=num_batches, ncols=60):

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


def two_pass_mean_std(batch_iter, num_batches=None):

    total_iters = 0

    acc = []
    for batch in tqdm(batch_iter, total=num_batches, ncols=60):

        batch_size = len(batch)
        total_iters += batch_size

        batch = batch.to(torch.float64)

        a = two_sum_reduce(batch)
        acc += a

    acc = two_sum_reduce(acc, minmum=True)

    mean_acc = [a / total_iters for a in acc]
    mean_acc = two_sum_reduce(acc=mean_acc)
    mean_acc_t = torch.stack(mean_acc)
    mean_acc_t = _mag_sort(mean_acc_t)

    mean = mean_acc_t[-1]

    acc = []
    for batch in tqdm(batch_iter, total=num_batches, ncols=60):

        batch = batch - mean

        batch = batch * batch

        a = two_sum_reduce(batch)
        acc += a

    acc = two_sum_reduce(acc, minmum=True)

    std_2_acc = [a / total_iters for a in acc]
    std_2_acc = two_sum_reduce(acc=std_2_acc)
    std_2_acc_t = torch.stack(std_2_acc)
    std_2_acc_t = _mag_sort(std_2_acc_t)
    std_2 = std_2_acc_t[-1]
    std = std_2.sqrt()

    return mean, std


def _test_two_pass_mean_std(all_values):

    mean, std = two_pass_mean_std(all_values)

    print()
    print(mean)
    print(std)
    print()

    inv_std = std.reciprocal()

    norm_values = []

    for values in all_values:

        v = values - mean

        v = v * inv_std

        norm_values.append(v)

    mean, std = two_pass_mean_std(norm_values)

    print()
    print(mean)
    print(std)
    print()


def _test_one_pass_mean_std(all_values):

    mean, std, total_iters = one_pass_mean_std(all_values)

    print()
    print(mean)
    print(std)
    print()

    inv_std = std.reciprocal()

    norm_values = []

    for values in all_values:

        v = values - mean

        v = v * inv_std

        norm_values.append(v)

    mean, std, total_iters = one_pass_mean_std(norm_values)

    print()
    print(mean)
    print(std)
    print()


def _test_mean_std():

    d = 4

    display_iters = 1000
    total_iters = 3000000

    ti = total_iters
    all_values = []

    while ti > 0:

        di = min(ti, display_iters)

        half_d = d // 2

        v1 = torch.rand([di, half_d], dtype=torch.float16)
        v2 = torch.rand([di, half_d], dtype=torch.float16)

        v1 = v1 - 0.5
        v2 = v2 + 1.0

        values = torch.cat([v1, v2], dim=1)

        values = values.to(torch.float64)

        all_values.append(values)

        ti -= di

    print()
    print("-------------- Two Pass ---------------")
    print()

    _test_two_pass_mean_std(all_values)

    print()
    print("-------------- One Pass ---------------")
    print()

    _test_one_pass_mean_std(all_values)


def main():
    set_random_seeds(4321)
    # _test_tensor_two_sum()
    _test_mean_std()


if __name__ == "__main__":
    main()
