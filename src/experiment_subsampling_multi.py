"""
python src/experiment_subsampling_multi.py -o "../data/positional-SAE/experiments_subsampling" --device "cpu" -p 6
python src/experiment_subsampling_multi.py -o "../data/positional-SAE/experiments_subsampling" --device "cuda" -p 6

python src\\experiment_subsampling.py -o "../data/positional-SAE/experiments_subsampling"
python src\\experiment_subsampling.py -o "../data/positional-SAE/experiments_subsampling"

"""

from ai_models import load_model, load_tokenizer, get_emb_fn
from utils import (
    make_folder,
    load_json,
    load_pickle,
    load_torch,
    save_json,
    save_pickle,
    save_torch,
    set_random_seeds,
)
from torch_custom_fns_multi import one_pass_mean_std_multi

from argparse import ArgumentParser
import json
import math
import os
from random import sample
from time import time, sleep


from datasets import load_dataset
from sklearn.cluster import MiniBatchKMeans
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

chunk_size = None
device = None
layer = None
max_sent_length = None
output_folder = None
total_sents = None
tracking_json_fname = None
num_procs = None


def parse_args():

    parser = ArgumentParser()

    parser.add_argument("-o", "--output-folder", required=True)

    parser.add_argument("-d", "--device", required=False, default="cpu")
    parser.add_argument("-p", "--num_procs", required=False, default=16, type=int)
    parser.add_argument(
        "-ai", "--ai-config", required=False, default="configs/ai/pythia-70m.json"
    )
    parser.add_argument(
        "-ds",
        "--dataset-config",
        required=False,
        default="configs/datasets/the_pile.json",
    )

    parser.add_argument("-seed", "--rng-seed", required=False, default=4321, type=int)

    args = parser.parse_args()

    return args


def get_tracking_json():

    if os.path.isfile(tracking_json_fname):
        tracking_json = load_json(tracking_json_fname)
    else:
        tracking_json = {}

    return tracking_json


def save_tracking_json(tracking_json):

    save_json(tracking_json, tracking_json_fname, indent=2)


def make_embeddings(ai_config, dataset_config, batch_size):

    tracking_json = get_tracking_json()
    embeddings_made = tracking_json.get("embeddings_made", False)

    if embeddings_made:
        return

    emb_cache_folder = os.path.join(output_folder, "emb_cache")
    make_folder(emb_cache_folder)

    chunk_sizes = []
    chunk_i = 0
    while chunk_i < total_sents:
        next_chunk_i = chunk_i + chunk_size
        next_chunk_i = min(next_chunk_i, total_sents)

        chunk_sizes.append(next_chunk_i - chunk_i)

        chunk_i = next_chunk_i

    tokenizer = load_tokenizer(ai_config)
    ai_model = load_model(ai_config)
    ai_model.to(device)

    tokenizer_kwargs = {"max_length": max_sent_length}

    emb_fn = get_emb_fn(
        tokenizer=tokenizer,
        ai_model=ai_model,
        config=ai_config,
        layers=[layer],
        input_device=device,
        output_device="cpu",
        tokenizer_kwargs=tokenizer_kwargs,
    )

    data = load_dataset(**dataset_config)

    dataloader = DataLoader(data, batch_size=batch_size)

    sent_buffer = []
    cs_i = 0

    data_iter = iter(dataloader)

    for cs_i in tqdm(range(len(chunk_sizes)), total=len(chunk_sizes), ncols=50):

        curr_cs = chunk_sizes[cs_i]
        cs_fname = os.path.join(emb_cache_folder, f"{cs_i}.pt")

        # fill sent buffer if needed
        while len(sent_buffer) < curr_cs:

            # get a new batch
            batch = None
            while batch is None:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # it doesn't make a lot of sense for this to happen for this experiment
                    print()
                    print("End of input reached, restaring dataloader")
                    print(
                        "it doesn't make a lot of sense for this to happen for this experiment"
                    )
                    print()
                    data_iter = iter(dataloader)

            batch_sents = batch["text"]

            sent_buffer += batch_sents

        chunk_sents = sent_buffer[:curr_cs]
        sent_buffer = sent_buffer[curr_cs:]

        # if the chunk file exists the chunk has been saved. no embedding needed (cache check)
        if os.path.isfile(cs_fname):
            continue

        # embed the chunk sentences
        chunk_data = DataLoader(chunk_sents, batch_size=batch_size)
        emb_buffer = []
        for batch_sents in chunk_data:

            emb_dict = emb_fn(batch_sents)
            embs = emb_dict[layer]

            emb_buffer += embs

        # save file safely so that the cache check works
        save_torch(emb_buffer, cs_fname)

    tracking_json["embeddings_made"] = True

    save_tracking_json(tracking_json)


class EmbIter:

    def __init__(self, args_dict):

        files = args_dict["files"]
        sub_rate = args_dict.get("sub_rate", 1.0)
        worker_i = args_dict.get("worker_i", None)
        stan_mean = args_dict.get("stan_mean", None)
        stan_std = args_dict.get("stan_std", None)
        batch_size = args_dict.get("batch_size", 1000000)

        if stan_std is not None:
            stan_std[stan_std == 0] = 1
            inv_stan_std = stan_std.reciprocal()
        else:
            inv_stan_std = None

        assert 0 < sub_rate <= 1.0

        # can't divide by std if mean is not provided
        assert inv_stan_std is None or stan_mean is not None

        self.files = files
        self.sub_rate = sub_rate
        self.worker_i = worker_i
        self.mean = stan_mean
        self.inv_std = inv_stan_std
        self.batch_size = batch_size

        self.file_i = 0
        self.emb_buffer = []

    def __iter__(self):
        return self

    def _load_next_embs(self):

        if self.file_i >= len(self.files):
            return

        fname = self.files[self.file_i]

        file_embs = load_torch(fname, map_location="cpu")

        subsample_embs = []
        for embs in file_embs:

            if self.sub_rate == 1.0:
                sub_embs = embs

            else:
                num_embs = len(embs)
                subsample_num = math.ceil(num_embs * self.sub_rate)
                keep_idx = sample(range(num_embs), k=subsample_num)
                sub_embs = embs[keep_idx]

            if self.mean is not None:
                sub_embs = sub_embs - self.mean

                if self.inv_std is not None:
                    sub_embs = sub_embs * self.inv_std

            subsample_embs.append(sub_embs)

        self.file_i += 1

        self.emb_buffer += subsample_embs

    def __next__(self):

        if len(self.emb_buffer) == 0:
            self._load_next_embs()
            # if the buffer is ever empty after an expansion check we are done
            if len(self.emb_buffer) == 0:
                raise StopIteration

        batch_buffer = []
        cur_batch_size = 0

        # until the batch is big enough or we've run out of embeddings
        while cur_batch_size < self.batch_size and len(self.emb_buffer) > 0:

            # find the minimal eb_i that will fill up the batch or we run out of embeddings
            add_batch_size = 0
            eb_i = 0
            while add_batch_size + cur_batch_size < self.batch_size and eb_i < len(
                self.emb_buffer
            ):
                add_batch_size += len(self.emb_buffer[eb_i])

                eb_i += 1

            # update the current batch
            cur_batch_size += add_batch_size
            batch_buffer += self.emb_buffer[:eb_i]

            # update the embeddings buffer

            self.emb_buffer = self.emb_buffer[eb_i:]

            if len(self.emb_buffer) == 0:
                self._load_next_embs()

        # make the batch
        batch_embs = torch.cat(batch_buffer, dim=0)

        if len(batch_embs) > self.batch_size:
            remaining = batch_embs[self.batch_size :]
            batch_embs = batch_embs[: self.batch_size]

            self.emb_buffer = [remaining] + self.emb_buffer

        return batch_embs


def subsample_mean_std(sub_rate=1.0, standardization_p=None):

    assert 0 < sub_rate <= 1.0

    emb_cache_folder = os.path.join(output_folder, "emb_cache")
    emb_files = os.listdir(emb_cache_folder)
    emb_files = sorted(emb_files, key=lambda x: int(x.split(".")[0]))
    emb_files = [os.path.join(emb_cache_folder, f) for f in emb_files]

    data_size = 1 + len(emb_files) // num_procs
    d_i = 0
    fname_chunks = []
    while d_i < len(emb_files):
        next_d_i = d_i + data_size
        fc = emb_files[d_i:next_d_i]
        fname_chunks.append(fc)
        d_i = next_d_i

    if standardization_p is not None:
        stan_mean = standardization_p["mean"]
        stan_std = standardization_p["std"]
    else:
        stan_mean = None
        stan_std = None

    data_iter_input = [
        {
            "files": fc,
            "sub_rate": sub_rate,
            "worker_i": i,
            "stan_mean": stan_mean,
            "stan_std": stan_std,
        }
        for i, fc in enumerate(fname_chunks)
    ]

    mean, std, total_iters = one_pass_mean_std_multi(
        data_iter_input, EmbIter, num_procs=num_procs
    )

    result = {"mean": mean, "std": std, "total_iters": total_iters}

    return result


def _confirm_baseline(baseline_result):

    print()
    print("Confirming baseline result")
    print()

    result = subsample_mean_std(sub_rate=1.0, standardization_p=baseline_result)

    print()
    print()
    print("mean")
    print(result["mean"])
    print("std")
    print(result["std"])
    print()


def mean_std_experiments():

    exp_key = "mean_std_exp_multi"
    exp_folder = os.path.join(output_folder, "mean_std_exp_multi")
    make_folder(exp_folder)

    tracking_json = get_tracking_json()

    if exp_key not in tracking_json:
        tracking_json[exp_key] = {}

    subsample_rates = [(20 - k) / 20 for k in range(0, 20)]
    number_of_trials = [1] + [20] * (len(subsample_rates) - 1)

    baseline_result = None
    smallest_result = None

    for sub_rate, num_trials in zip(subsample_rates, number_of_trials):

        sub_results = []

        for nt_i in range(num_trials):

            print()
            print(f"mean std experiement with subsample rate: {sub_rate}")
            print(f"Trial number: {nt_i}")
            print()

            exp_fname = os.path.join(exp_folder, f"{sub_rate}_{nt_i}.pt")

            if os.path.isfile(exp_fname):

                print()
                print(f"experiment already completed.. skipping...")
                print()

                result = load_torch(exp_fname)

            else:

                result = subsample_mean_std(sub_rate=sub_rate)

                save_torch(result, exp_fname)

            if baseline_result is None:
                baseline_result = result

            if sub_rate == 0.05 and smallest_result is None:
                smallest_result = result

            sub_results.append(result)

        mean_errs = []
        std_errs = []
        total_iters = []

        b_mean = baseline_result["mean"]
        b_std = baseline_result["std"]

        for result in sub_results:

            t = torch.Tensor([123])

            t.abs().mean()

            r_mean = result["mean"]
            r_std = result["std"]
            r_iters = result["total_iters"]

            err_mean = float((r_mean - b_mean).abs().mean())
            mean_errs.append(err_mean)

            err_std = float((r_std - b_std).abs().mean())
            std_errs.append(err_std)

            total_iters.append(r_iters)

        stat_keys = ["mean_stats", "std_stats", "iter_stats"]
        stat_vecs = [mean_errs, std_errs, total_iters]
        stat_iter = zip(stat_keys, stat_vecs)

        tracking_json[exp_key][sub_rate] = {}
        for stat_key, stat_vec in stat_iter:

            m = sum(stat_vec) / len(stat_vec)

            std = sum([(v - m) ** 2 for v in stat_vec]) / len(stat_vec)

            stat_enty = {"mean": m, "std": std}

            tracking_json[exp_key][sub_rate][stat_key] = stat_enty

        save_tracking_json(tracking_json)

    _confirm_baseline(baseline_result)
    _confirm_baseline(smallest_result)


def test_kmeans():

    exp_key = "kmeans_exp_multi"
    exp_folder = os.path.join(output_folder, exp_key)
    make_folder(exp_folder)

    mean_exp_folder = os.path.join(output_folder, "mean_std_exp_multi")
    baseline_fname = os.path.join(mean_exp_folder, f"1.0_0.pt")
    baseline = torch.load(baseline_fname)

    emb_cache_folder = os.path.join(output_folder, "emb_cache")
    emb_files = os.listdir(emb_cache_folder)
    emb_files = sorted(emb_files, key=lambda x: int(x.split(".")[0]))
    emb_files = [os.path.join(emb_cache_folder, f) for f in emb_files]

    data_iter_args = {
        "files": emb_files,
        "worker_i": 0,
        "stan_mean": baseline["mean"],
        "stan_std": baseline["std"],
    }

    subsample_rates = [1.0]
    number_of_trials = 5
    total_trials = len(subsample_rates) * number_of_trials
    random_seeds = [10037 * k % 1999 for k in range(total_trials * 2)]
    rng_seed_iter = iter(random_seeds)

    max_iter = 150

    kmeans_params = {
        "n_clusters": 20000,
        "max_iter": max_iter,
        "verbose": True,
        "batch_size": 1000000,
        "compute_labels": False,
        "init_size": 100000,
        "reassignment_ratio": 0.001,
    }

    print()
    print(f"max iterations: {max_iter}")
    print()

    log_strs = []
    for sub_rate in subsample_rates:

        data_iter_args["sub_rate"] = sub_rate

        for t_i in range(number_of_trials):

            rng_seed = next(rng_seed_iter)
            kmeans_params["random_state"] = ratio

            subsample_str = f"kmeans with subsample rate={sub_rate}, trial number {t_i}"

            print()
            print(subsample_str)
            print()

            log_str = subsample_str + "\n\n"

            kmeans_param_str = "kmeans params\n"
            kmeans_param_str += json.dumps(kmeans_params)

            print()
            print(kmeans_param_str)
            print()

            log_str += kmeans_param_str + "\n\n"

            trial_file_name = f"{sub_rate}_{t_i}.pkl"
            trial_fname = os.path.join(exp_folder, trial_file_name)

            if os.path.isfile(trial_fname):

                done_str = f"experiment is done skipping"
                log_str += done_str + "\n\n"

                print()
                print(done_str)
                print()

                continue

            kmeans = MiniBatchKMeans(**kmeans_params)

            data_iter = EmbIter(data_iter_args)

            for i in tqdm(range(max_iter), total=max_iter, ncols=60):

                try:
                    emb_batch = next(data_iter)
                except:
                    data_iter = EmbIter(data_iter_args)
                    emb_batch = next(data_iter)

                kmeans.partial_fit(emb_batch)

                if i + 1 == max_iter:
                    break

            save_pickle(kmeans, trial_fname)


def main():

    args = parse_args()

    set_random_seeds(args.rng_seed)

    global device
    device = torch.device(args.device)

    global output_folder
    output_folder = args.output_folder

    global chunk_size
    chunk_size = 10000

    global total_sents
    total_sents = 1000000

    global max_sent_length
    max_sent_length = 100

    global layer
    layer = -1

    global num_procs
    num_procs = args.num_procs

    make_folder(output_folder)

    global tracking_json_fname
    tracking_json_fname = os.path.join(output_folder, "tracking.json")

    ai_config = load_json(args.ai_config)
    dataset_config = load_json(args.dataset_config)

    # make_embeddings(ai_config, dataset_config, args.batch_size)

    # mean_std_experiments()

    test_kmeans()


if __name__ == "__main__":
    with torch.no_grad():
        main()
