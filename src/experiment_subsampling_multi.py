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
    load_torch,
    save_json,
    save_torch,
    set_random_seeds,
)
from torch_custom_fns_multi import one_pass_mean_std_multi

from argparse import ArgumentParser
import math
import os
from random import sample
from time import time, sleep


from datasets import load_dataset
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


class FileQueue:
    """
    My file are large so this keeps the runtime smooth
    """

    queue_folder = None

    def __init__(self, queue_folder):
        self.queue_folder = queue_folder

    def _get_queue(self):

        queue_files = os.listdir(self.queue_folder)

        # sort queue by time
        queue_files = sorted(queue_files, key=lambda x: int(x.split("_")[1]))

        queue = []
        for qf in queue_files:
            v = qf.split("_")
            q = {"number": int(v[0]), "position": int(v[1]), "file": qf}
            queue.append(q)

        return queue

    def add(self, number):
        """
        add something to the end of the queue
        """

        queue = self._get_queue()

        # validate addition
        for q in queue:

            if q["number"] == number:
                err_msg = f"trying to add somethign to the queue twice {number} {queue}"
                raise ValueError(err_msg)

        # add to queue
        now = int(time() * 100)
        new_file = f"{number}_{now}"
        new_fname = os.path.join(self.queue_folder, new_file)
        # touch file
        with open(new_fname, "w") as f_out:
            pass

    def remove(self, number):
        """
        remove number from the queue if it is first, otherwise error
        """

        queue = self._get_queue()

        if queue[0]["number"] != number:
            err_msg = f"trying to remove somethign not at the front of the queue {number} {queue}"
            raise ValueError(err_msg)

        del_file = queue[0]["file"]
        del_fname = os.path.join(self.queue_folder, del_file)

        os.remove(del_fname)

    def is_front(self, number):

        queue = self._get_queue()

        is_f = queue[0]["number"] == number

        return is_f


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

    def __init__(self, args):

        if len(args) == 4:
            files, sub_rate, file_queue, i = args
            mean = None
            std = None
        elif len(args) == 6:
            files, sub_rate, file_queue, i, mean, std = args
        if std is not None:
            inv_std = std.reciprocal()
        else:
            inv_std = None

        assert 0 < sub_rate <= 1.0

        # can't divide by std if mean is not provided
        assert std is None or mean is not None

        self.files = files
        self.sub_rate = sub_rate
        self.file_queue = file_queue
        self.i = i
        self.mean = mean
        self.inv_std = inv_std

        self.batch_size = 1000
        self.file_i = -1
        self.emb_buffer = []

        if i == 0:
            print()
            print("worker 0 loaded with data iterator with args")
            print(args)
            print()

    def __iter__(self):
        return self

    def _load_embs(self):

        fname = self.files[self.file_i]

        if self.i == 0:
            print()
            print()
            print(f"loading files embs from {fname}")
            print()

        file_embs = load_torch(fname, map_location="cpu")

        if self.i == 0:
            print()
            print(f"embeddings loaded")
            print(len(file_embs))
            print()

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

        if self.i == 0:
            print()
            print(f"subsampling and standardization completed")
            print(len(subsample_embs))
            print()

        return subsample_embs

    def __next__(self):

        # if the buffer is too small expand it if we can
        if self.batch_size > len(self.emb_buffer) and self.file_i < len(self.files):
            self.file_i += 1
            self.emb_buffer += self._load_embs()

        # if the buffer is ever empty after an expansion check we are done
        if len(self.emb_buffer) == 0:
            raise StopIteration

        batch_embs = self.emb_buffer[: self.batch_size]
        self.emb_buffer = self.emb_buffer[self.batch_size :]

        batch_embs = torch.cat(batch_embs, dim=0)

        return batch_embs


def subsample_mean_std(exp_folder, sub_rate=1.0, standardization_p=None):

    assert 0 < sub_rate <= 1.0

    queue_folder = os.path.join(exp_folder, "loading_queue")
    make_folder(queue_folder)

    # clear out old values
    for file in os.listdir(queue_folder):
        fname = os.path.join(queue_folder, file)
        os.remove(fname)

    file_queue = FileQueue(queue_folder)

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
        (fc, sub_rate, file_queue, i, stan_mean, stan_std)
        for i, fc in enumerate(fname_chunks)
    ]

    mean, std, total_iters = one_pass_mean_std_multi(
        data_iter_input, EmbIter, num_procs=num_procs
    )

    # clean up temporary files (should be empty here)
    os.rmdir(queue_folder)

    result = {"mean": mean, "std": std, "total_iters": total_iters}

    return result


def _confirm_baseline(baseline_result):

    print()
    print("Confirming baseline result")
    print()

    result = subsample_mean_std("./", sub_rate=1.0, standardization_p=baseline_result)

    print()
    print()
    print("mean")
    print(result["mean"])
    print("std")
    print(result["std"])
    print()

    exit()


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

                result = subsample_mean_std(exp_folder, sub_rate=sub_rate)

                save_torch(result, exp_fname)

            if baseline_result is None:
                baseline_result = result
                _confirm_baseline(baseline_result)

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

    mean_std_experiments()


if __name__ == "__main__":
    with torch.no_grad():
        main()
