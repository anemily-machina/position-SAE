from torch_custom_fns import _mag_sort, two_sum_reduce


from multiprocessing import Pool
from time import time, sleep

import torch
from tqdm import tqdm


def _one_pass_mean_std_multi(args):

    bi, batch_iter_fn, i = args

    # so they don't block each other loading stuff
    sleep(i * 10)
    batch_iter = batch_iter_fn(bi)

    pbar = None
    if i == 0:
        pbar = tqdm(desc="worker 0")
        # batch_iter = tqdm(batch_iter, "worker 0", ncols=60)

    total_iters = 0

    mini_batch_size = 100000

    acc = []
    acc2 = []
    for batch in batch_iter:

        batch_size = len(batch)

        total_iters += batch_size

        mb_i = 0
        while mb_i < len(batch):
            next_mb_i = mb_i + mini_batch_size

            mini_batch = batch[mb_i:next_mb_i]

            mini_batch = mini_batch.to(torch.float64)
            mini_batch2 = mini_batch * mini_batch

            a = two_sum_reduce(mini_batch)
            acc += a

            a2 = two_sum_reduce(mini_batch2)
            acc2 += a2

            mb_i = next_mb_i

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    acc = two_sum_reduce(acc, minmum=True)
    acc2 = two_sum_reduce(acc2, minmum=True)

    return {"acc": acc, "acc2": acc2, "total_iters": total_iters}


def one_pass_mean_std_multi(batch_iter_inputs, batch_iter_clas, num_procs):

    print()
    print("multiprocessing mean/std")
    print()

    start_time = time()

    proc_input = [(bi, batch_iter_clas, i) for i, bi in enumerate(batch_iter_inputs)]

    with Pool(num_procs) as p:
        all_acc_data = p.map(_one_pass_mean_std_multi, proc_input)

    print()
    print("combining each child results ")
    print()

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

    total_time = (time() - start_time) / 60

    print()
    print(f"multiprocessing mean/std took {total_time} minutes")
    print()

    return mean, std, total_iters
