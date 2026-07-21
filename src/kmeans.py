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


def _pairwise(emb_list):

    if len(emb_list) < 3:
        s = sum(emb_list)
        return s
    else:
        m = len(emb_list) // 2
        s1 = _pairwise(emb_list[:m])
        s2 = _pairwise(emb_list[m:])

        return s1 + s2


def _compute_mean(layer_folder, base_case_n):

    chunk_file_names = os.listdir(layer_folder)
    chunk_file_names = sorted(chunk_file_names, key=lambda x: int(x.split("_")[0]))
    chunk_fnames = [os.path.join(layer_folder, fn) for fn in chunk_file_names]

    buffer = []

    # compute mean
    acc_vecs = []
    total = 0
    for fname in chunk_fnames:

        chunk = torch.load(fname)

        if len(buffer) == 0:
            buffer = chunk
        else:
            buffer = torch.cat([buffer, chunk], dim=0)

        while len(buffer) >= base_case_n:

            acc_vectors = buffer[:base_case_n]
            buffer = buffer[base_case_n:]
            total += len(acc_vectors)

            acc_vectors = acc_vectors.to(torch.float64)
            acc = torch.sum(acc_vectors, dim=0)
            acc_vecs.append(acc)

    if len(buffer) > 0:
        buffer = buffer.to(torch.float64)
        total += len(buffer)
        acc = torch.sum(buffer, dim=0)
        acc_vecs.append(acc)

    total_sum = _pairwise(acc_vecs)

    mean = total_sum / total

    print()
    print(total_sum)
    print(mean)
    print(total)
    print()

    return mean


def _compute_std(mean, layer_folder, base_case_n):

    chunk_file_names = os.listdir(layer_folder)
    chunk_file_names = sorted(chunk_file_names, key=lambda x: int(x.split("_")[0]))
    chunk_fnames = [os.path.join(layer_folder, fn) for fn in chunk_file_names]

    buffer = []

    # compute mean
    acc_vecs = []
    total = 0
    for fname in chunk_fnames:

        chunk = torch.load(fname)

        if len(buffer) == 0:
            buffer = chunk
        else:
            buffer = torch.cat([buffer, chunk], dim=0)

        while len(buffer) >= base_case_n:

            acc_vectors = buffer[:base_case_n]
            buffer = buffer[base_case_n:]
            total += len(acc_vectors)

            acc_vectors = acc_vectors.to(torch.float64)
            acc_vectors = acc_vectors - mean
            acc = torch.sum(acc_vectors, dim=0)
            acc_vecs.append(acc)

    if len(buffer) > 0:
        buffer = buffer.to(torch.float64)
        total += len(buffer)
        acc = torch.sum(buffer, dim=0)
        acc_vecs.append(acc)

    total_sum = _pairwise(acc_vecs)

    mean = total_sum / total

    print()
    print(total_sum)
    print(mean)
    print(total)
    print()

    return mean


def _compute_mean_std(kmeans_config):

    # size of accumilator vectors (base case for pairwise summation)
    ACC_SIZE = 256

    layers = kmeans_config["layers"]
    emb_folder = kmeans_config["emb_folder"]

    for l in layers:

        layer_folder = os.path.join(emb_folder, str(l))

        mean = _compute_mean(layer_folder, ACC_SIZE)

        std = _compute_std(mean, layer_folder, ACC_SIZE)


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
