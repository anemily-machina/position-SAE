from ai_models import load_model, load_tokenizer, get_emb_fn
from utils import load_json, make_folder

import os
from random import sample as python_sample

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DEVICE = "cpu"  # TODO remove


def _make_subsample_fn(kmeans_config):

    subsample_type = kmeans_config.get("subsample_type", None)

    if subsample_type is None:

        def subsample_fn(embeddings):
            return embeddings

    elif subsample_type == "random":

        subsample_k = kmeans_config["subsample_k"]

        assert subsample_k > 0

        if subsample_k < 1:

            def subsample_fn(embeddings):

                k = int(len(embeddings) * subsample_k)
                k = max(1, k)

                sample = python_sample(range(len(embeddings)), k=k)

                return embeddings[sample]

        else:

            def subsample_fn(embeddings):

                k = min(len(embeddings), subsample_k)

                sample = python_sample(range(len(embeddings)), k=k)

                return embeddings[sample]

    return subsample_fn


def _make_embeddings(dataloader, emb_fn, subsample_fn, kmeans_config):

    layers = kmeans_config["layers"]
    emb_folder = kmeans_config["emb_folder"]
    total = kmeans_config["N"]
    chunk_size = kmeans_config["chunk_size"]

    temp_folders = {}
    for l in layers:

        temp_folder = os.path.join(emb_folder, str(l))
        make_folder(temp_folder)

        temp_folders[l] = temp_folder

    chunk_sizes = []
    cs_i = 0
    while cs_i < total:
        next_cs_i = cs_i + chunk_size
        next_cs_i = min(next_cs_i, total)

        chunk_sizes.append(next_cs_i - cs_i)

        cs_i = next_cs_i

    chunk_i = 0
    max_buffer_size = chunk_sizes[chunk_i]
    buffer_size = 0
    buffer = {l: [] for l in layers}

    data_iter = iter(dataloader)

    while chunk_i < len(chunk_sizes):

        batch = None
        while batch is None:
            try:
                batch = next(data_iter)
            except StopIteration:
                print()
                print("End of input reached, restaring dataloader")
                print()
                data_iter = iter(dataloader)

        batch = batch["text"]

        emb_dict = emb_fn(batch)

        for l in layers:
            batch_layer_embs = emb_dict[l]
            batch_layer_embs = [subsample_fn(ble) for ble in batch_layer_embs]

            buffer[l] += batch_layer_embs

        # batch_layer_embs carried over from last iteration of above
        batch_size = sum([len(ble) for ble in batch_layer_embs])
        buffer_size += batch_size

        print()
        print(buffer_size, max_buffer_size, chunk_i, len(chunk_sizes))
        print()

        while buffer_size >= max_buffer_size:

            print()
            print(f"saving layer chunks {chunk_i}")
            print()

            for l in layers:

                layer_embs = buffer[l]
                layer_embs = torch.cat(layer_embs, dim=0)

                chunk = layer_embs[:max_buffer_size]

                if len(layer_embs) != max_buffer_size:
                    left_over = layer_embs[max_buffer_size:]
                    buffer[l] = [left_over]
                else:
                    buffer[l] = []

                tf = temp_folders[l]
                file_name = f"{chunk_i}_chunk.pt"
                fname = os.path.join(tf, file_name)

                print(fname)

                torch.save(chunk, fname)

            chunk_i += 1
            buffer_size -= max_buffer_size

            print()
            print(buffer_size, max_buffer_size, chunk_i, len(chunk_sizes))
            print()


def _compute_mean_std(kmeans_config):

    pass


def main(ai_config, dataset_config, kmeans_config):

    tokenizer = load_tokenizer(ai_config)
    ai_model = load_model(ai_config)
    ai_model.to(DEVICE)

    # if chunk size is not specified try one chunk
    kmeans_config["chunk_size"] = kmeans_config.get("chunk_size", kmeans_config["N"])

    max_length = kmeans_config["max_length"]
    layers = kmeans_config["layers"]

    tokenizer_kwargs = {"max_length": max_length}

    emb_fn = get_emb_fn(
        tokenizer=tokenizer,
        ai_model=ai_model,
        config=ai_config,
        layers=layers,
        input_device=DEVICE,
        output_device="cpu",
        tokenizer_kwargs=tokenizer_kwargs,
    )

    # NOTE: if you are streaming data make sure it is randomized or you handle that somehow
    data = load_dataset(**dataset_config)

    dataloader = DataLoader(data, batch_size=32)

    subsample_fn = _make_subsample_fn(kmeans_config)

    # _make_embeddings(dataloader, emb_fn, subsample_fn, kmeans_config)

    _compute_mean_std(kmeans_config)


if __name__ == "__main__":

    ai_config = load_json("configs/ai/pythia-70m.json")
    dataset_config = load_json("configs/datasets/the_pile.json")
    kmeans_config = load_json("configs/kmeans/test_lastlayer_random.json")

    main(ai_config, dataset_config, kmeans_config)
