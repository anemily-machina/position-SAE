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


def two_sum_reduce(
    acc: list[torch.Tensor] = None, acc_t: torch.Tensor = None
) -> torch.Tensor:
    """
    reduces an accumilator list as best as possible

    """
    assert acc is None or acc_t is None

    if acc_t is None:

        d = len(acc[0])

        acc_t = torch.stack(acc)

    else:

        d = acc_t.size(1)

    acc_t = _mag_sort(acc_t)

    # row is the set of accumilators for each dimension now
    acc_t = acc_t.t()

    reduced = []

    for v in acc_t:

        # reduce
        v_r = iter_two_sum(v, None)

        reduced.append(v_r)

    max_r_size = max([len(r) for r in reduced])

    acc_r = [
        torch.zeros([d], device=acc_t.device, dtype=acc_t.dtype)
        for _ in range(max_r_size)
    ]

    for i, v_r in enumerate(reduced):

        for j, value in enumerate(v_r):

            acc_r[j][i] = value

    return acc_r


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

    acc_cap = 1
    acc_cap2 = 1

    total_iters = 0

    compact_iters = 10000
    ci = compact_iters

    acc = None
    acc2 = None
    for batch in tqdm(batch_iter, total=num_batches, ncols=60):

        batch_size = len(batch)

        total_iters += batch_size
        ci -= batch_size

        batch = batch.to(torch.float64)

        batch2 = batch * batch

        acc = iter_two_sum(batch, acc)

        acc2 = iter_two_sum(batch2, acc2)

        if len(acc) > acc_cap:
            print()
            print(f"accumilator 1 too large {len(acc)} > {acc_cap}")
            print()

            acc = two_sum_reduce(acc=acc)
            reduced = True

            if len(acc) > acc_cap:

                print()
                print(f"accumilator 1 too large after reduction size > {acc_cap}")

                acc_cap = math.ceil(acc_cap * 1.1)

                print(f"expanding acc cap 1 to {acc_cap}")
                print()

        if len(acc2) > acc_cap2:

            print()
            print(f"accumilator 2 too large {len(acc2)} > {acc_cap2}")
            print()

            acc2 = two_sum_reduce(acc=acc2)

            if len(acc2) > acc_cap2:

                print()
                print(f"accumilator 2 too large after reduction size > {acc_cap2}")

                acc_cap2 = math.ceil(acc_cap2 * 1.1)

                print(f"expanding acc cap 2 to {acc_cap2}")
                print()

        if ci <= 0:
            ci = compact_iters

            acc = two_sum_reduce(acc=acc)
            acc2 = two_sum_reduce(acc=acc2)

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

    return mean, std


def two_pass_mean_std(batch_iter, num_batches=None):

    acc_cap = 1

    compact_iters = 10000
    ci = compact_iters

    total_iters = 0

    acc = None
    for batch in tqdm(batch_iter, total=num_batches, ncols=60):

        batch_size = len(batch)
        total_iters += batch_size

        ci -= batch_size

        batch = batch.to(torch.float64)

        acc = iter_two_sum(batch, acc)

        if len(acc) > acc_cap:

            print()
            print(f"accumilator too large {len(acc)} > {acc_cap}")
            print()

            acc = two_sum_reduce(acc=acc)

            if len(acc) > acc_cap:

                print()
                print(f"accumilator too large after reduction size > {acc_cap}")

                acc_cap = math.ceil(acc_cap * 1.1)

                print(f"expanding acc cap to {acc_cap}")
                print()

        if ci <= 0:
            ci = compact_iters

            acc = two_sum_reduce(acc=acc)

    mean_acc = [a / total_iters for a in acc]
    mean_acc = two_sum_reduce(acc=mean_acc)
    mean_acc_t = torch.stack(mean_acc)
    mean_acc_t = _mag_sort(mean_acc_t)

    mean = mean_acc_t[-1]

    acc_cap = 1
    ci = compact_iters
    acc = None
    for batch in tqdm(batch_iter, total=num_batches, ncols=60):

        ci -= len(batch)

        batch = batch - mean

        batch = batch * batch

        acc = iter_two_sum(batch, acc)

        if len(acc) > acc_cap:

            print()
            print(f"accumilator too large {len(acc)} > {acc_cap}")
            print()

            acc = two_sum_reduce(acc=acc)

            if len(acc) > acc_cap:

                print()
                print(f"accumilator too large after reduction size > {acc_cap}")

                acc_cap = math.ceil(acc_cap * 1.1)

                print(f"expanding acc cap to {acc_cap}")
                print()

        if ci <= 0:
            ci = compact_iters

            acc = two_sum_reduce(acc=acc)

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

    mean, std = one_pass_mean_std(all_values)

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

    mean, std = one_pass_mean_std(norm_values)

    print()
    print(mean)
    print(std)
    print()


def _test_mean_std():

    display_iters = 1000
    total_iters = 300000  # 0

    ti = total_iters
    all_values = []

    while ti > 0:

        di = min(ti, display_iters)

        values = torch.rand([di, 3], dtype=torch.float16)
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
