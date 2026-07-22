"""
python src/experiment_subsampling.py -o "../data/positional-SAE/experiments_subsampling" --total-sents 1000000 --sents-per-chunk 10000 --max-sent-length 100 --batch-size 32

python src\\experiment_subsampling.py -o "../data/positional-SAE/experiments_subsampling"
python src\\experiment_subsampling.py -o "../data/positional-SAE/experiments_subsampling" -

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

from argparse import ArgumentParser
import os

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


def parse_args():

    parser = ArgumentParser()

    parser.add_argument("-o", "--output-folder", required=True)

    parser.add_argument("-d", "--device", required=False, default="cpu")
    parser.add_argument(
        "-ai", "--ai-config", required=False, default="configs/ai/pythia-70m.json"
    )
    parser.add_argument(
        "-ds",
        "--dataset-config",
        required=False,
        default="configs/datasets/the_pile.json",
    )
    parser.add_argument("-t", "--total-sents", required=False, default=40000, type=int)
    parser.add_argument(
        "-c", "--sents-per-chunk", required=False, default=1000, type=int
    )
    parser.add_argument(
        "-m", "--max-sent-length", required=False, default=250, type=int
    )
    parser.add_argument("-l", "--layer", required=False, default=-1, type=int)
    parser.add_argument("-b", "--batch-size", required=False, default=32, type=int)
    parser.add_argument("-seed", "--rng-seed", required=False, default=4321, type=int)

    args = parser.parse_args()

    return args


def get_tracking_json():

    if os.path.isfile(tracking_json_fname):
        tracking_json = load_json(tracking_json_fname)
    else:
        tracking_json = {}

    return tracking_json


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

    save_json(tracking_json)


def do_mean_std_exp(subsample_rate=1.0):

    emb_cache_folder = os.path.join(output_folder, "emb_cache")

    chunk_files = os.listdir(emb_cache_folder)
    chunk_files = sorted(chunk_files, key=lambda x: int(x.split(".")[0]))
    chunk_fnames = [os.path.join(emb_cache_folder, fn) for fn in chunk_files]

    total_embs = 0
    c_i = 0
    curr_chunk = load_torch(chunk_fnames[c_i])
    # first embedding of first tensor of embeddings in a list of tensors
    d = len(curr_chunk[0][0])

    print(curr_chunk[0][0])

    print(d)

    exit()

    # TODO arg or config
    iter_depth = 2

    print()
    print(f"performing serial ")
    print()


def main():

    args = parse_args()

    set_random_seeds(args.rng_seed)

    global device
    device = torch.device(args.device)

    global chunk_size
    chunk_size = args.sents_per_chunk

    global output_folder
    output_folder = args.output_folder

    global total_sents
    total_sents = args.total_sents

    global max_sent_length
    max_sent_length = args.max_sent_length

    global layer
    layer = args.layer

    make_folder(output_folder)

    global tracking_json_fname
    tracking_json_fname = os.path.join(output_folder, "tracking.json")

    ai_config = load_json(args.ai_config)
    dataset_config = load_json(args.dataset_config)

    # make_embeddings(ai_config, dataset_config, args.batch_size)

    do_mean_std_exp()


if __name__ == "__main__":
    with torch.no_grad():
        main()
