"""
Large parts of this are inspired of borrowed from the dictionary_learning
"""

from dotenv import load_dotenv

load_dotenv()

from config import DEBUG


import math
import os
import shutil
import time

from datasets import load_dataset
from nnsight.modeling.transformers import TransformersModel
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizer


from ai_models import load_tokenizer
from utils import load_json, load_torch, make_folder, save_torch, set_random_seeds

if DEBUG:
    tracer_kwargs = {"scan": True, "validate": True}
else:
    tracer_kwargs = {"scan": False, "validate": False}


class ActivationBuffer:
    """
    Implements a buffer of activations. The buffer stores activations from a model,
    yields them in batches, and refreshes them when the buffer is less than half full.
    """

    def __init__(
        self,
        data,
        model,
        tokenizer: PreTrainedTokenizer,
        get_activations_fns: list[callable],
        post_processing_fns: list[callable] = None,
        buffer_size: int = 65536,
        refresh_size: int = None,
        refresh_batch_size: int = 32,
        out_batch_size=8192,
        device="cpu",
        tokenizer_kwargs: dict = {},
        verbose: bool = True,
    ):
        """Initializes the ActionationBuffers

        Args:
            data (iterable): of "sentences". will be iterated over to generate text batches. Note that this is not an iterator as in dictionary_learning. Each entry should be a dictionary with a "text" key.
            model (TransformersModel): The NNSight Transfomer model that processes the text batches, or a hugging face model, pytroch model etc.
            tokenizer (PreTrainedTokenizer): The Tokenizer for the model, should be callable tokenizer(batch, **tokenizer_kwargs)
            get_activations_fns (list[callable]): Iteratively processes a tokenized batch of text. Must have at least one function. The first function will be called fn1(self.model, self.tokenizer(text_batch, self.tokenizer_kwargs)). Each other call will be fn_i(out_fn_i-1). The final result  is expected that this is a list containing tensors of activations, one for each sentence in the batch. They will be passed as torch.cat(result, dim=0)) so anything compatible with that works.
            post_processing_fns (list[callable], optional): if not None iteratively applied to the output of the the get_activations_fns step. deafult=None
            buffer_size (int, optional): The soft cap on activations. If we have at least this many we stop making more. Defaults to 65536.
            refresh_size (int, optional): If we have less than this many activations left, make more. Default to buffer_size//2
            refresh_batch_size (int, optional): The sentence batch size when making new activations. Defaults to 32.
            out_batch_size (int, optional): The number of activations to output in each activation batch. Defaults to 8192.
            device (str, optional): The device the activations are stored on. Defaults to "cpu".
            tokenizer_kwargs (dict, optional): tokenizer_kwargs to be passed to the model tokenizer as **tokenizer_kwargs. Defaults to {}
            verbose (bool, optional): show progress bar when filling activation buffer and similar things. Default=True
        """

        self.data = data
        self.data_iter = iter(data)
        self.model = model
        self.tokenizer = tokenizer
        self.get_activations_fns = get_activations_fns
        self.post_processing_fns = post_processing_fns
        self.buffer_size = buffer_size
        if refresh_size is None:
            self.refresh_size = self.buffer_size // 2
        else:
            self.refresh_size = refresh_size
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = torch.device(device)
        self.tokenizer_kwargs = tokenizer_kwargs
        self.verbose = verbose

        assert self.buffer_size >= self.out_batch_size
        assert self.buffer_size >= self.refresh_size

        self.activations = None
        self.read = None

    def __iter__(self):
        return self

    @torch.no_grad()
    def __next__(self) -> torch.Tensor:
        """
        Return a batch of activations
        """

        # check if buffer needs refreshing
        do_refresh = self.read is None or (~self.read).sum() < self.refresh_size
        if do_refresh:
            self.refresh()

        # indexes of unread activations
        unread_i = (~self.read).nonzero().squeeze()

        # get random sample of indexes of unread activations
        rand_idx = torch.randperm(unread_i.size(0), device=self.device)
        rand_idx = rand_idx[: self.out_batch_size]
        batch_i = unread_i[rand_idx]

        # create an actvivation batch
        activation_batch = self.activations[batch_i]

        # mark activations as read
        self.read[batch_i] = True

        return activation_batch

    @torch.no_grad()
    def _get_text_batch(self):
        """
        Return a batch of text sentences
        """

        batch_size = self.refresh_batch_size

        text_batch = []
        while len(text_batch) < batch_size:

            try:
                text = next(self.data_iter)["text"]
            except StopIteration:
                self.data_iter = iter(self.data)
                text = next(self.data_iter)["text"]

            text_batch.append(text)

        return text_batch

    @torch.no_grad()
    def _get_tokenized_batch(self):
        """
        Return a batch of tokenized inputs.
        """

        texts = self._get_text_batch()

        tokenized_batch = self.tokenizer(texts, **self.tokenizer_kwargs)

        return tokenized_batch

    @torch.no_grad()
    def refresh(self):

        # commenting these out, I guess  I'll figure out why they are here later
        # gc.collect()
        # t.cuda.empty_cache()

        # compact remaining activations
        if self.activations is not None:
            self.activations = self.activations[~self.read]
            cur_buffer_size = self.activations.size(0)
        else:
            cur_buffer_size = 0

        if self.verbose:

            pbar = tqdm(
                total=self.buffer_size,
                initial=cur_buffer_size,
                desc="Refreshing activations",
            )

        new_buffer = []
        while cur_buffer_size < self.buffer_size:

            # process a text batch
            tokenized_batch = self._get_tokenized_batch()

            fn1 = self.get_activations_fns[0]
            pipe = self.get_activations_fns[1:]

            batch_activations = fn1(self.model, tokenized_batch)
            for fn in pipe:
                batch_activations = fn(batch_activations)

            # update loop
            update_size = sum([len(b) for b in batch_activations])
            cur_buffer_size += update_size

            new_buffer += batch_activations

            if self.verbose:
                pbar.update(update_size)

        if self.verbose:
            pbar.close()

        new_buffer = torch.cat(new_buffer, dim=0)

        for fn in self.post_processing_fns:
            new_buffer = fn(new_buffer)

        new_buffer = new_buffer.to(self.device)

        # make one buffer
        if self.activations is not None:
            self.activations = torch.cat((new_buffer, self.self.activations), dim=0)
        else:
            self.activations = new_buffer

        # reset tracking
        self.read = torch.zeros(
            len(self.activations), dtype=torch.bool, device=self.device
        )


class SaveActivationBuffer:

    def __init__(
        self,
        data,
        model,
        tokenizer: PreTrainedTokenizer,
        get_activations_fns: list[callable],
        output_folder,
        number_of_chunks=20,
        post_processing_fns: list[callable] = None,
        buffer_size: int = 65536,
        refresh_size: int = None,
        refresh_batch_size: int = 32,
        out_batch_size=8192,
        device="cpu",
        tokenizer_kwargs: dict = {},
        verbose: bool = True,
    ):

        self.buffer = ActivationBuffer(
            data,
            model,
            tokenizer,
            get_activations_fns,
            post_processing_fns,
            buffer_size,
            refresh_size,
            refresh_batch_size,
            out_batch_size,
            device,
            tokenizer_kwargs,
            verbose,
        )

        self.output_folder = output_folder
        self.number_of_chunks = number_of_chunks

        # clear file buffer
        if os.path.isdir(output_folder):
            shutil.rmtree(output_folder)

        make_folder(output_folder)

    def save_activations(self):

        while True:

            existing_chunks = os.listdir(self.output_folder)

            if len(existing_chunks) < self.number_of_chunks:

                self.buffer.refresh()

                activations = self.buffer.activations

                start_time = time.time()

                file_name = f"{start_time*10000000}.pt"
                fname = os.path.join(self.output_folder, file_name)

                save_torch(activations, fname)

                total_time = time.time() - start_time

                print(total_time)

                self.buffer.activations = None

            time.sleep(1)


class LoadActivationBuffer:

    def __init__(
        self,
        output_folder,
        buffer_size: int = 65536,
        refresh_size: int = None,
        out_batch_size=8192,
        device="cpu",
    ):
        self.output_folder = output_folder
        self.buffer_size = buffer_size
        if refresh_size is None:
            self.refresh_size = self.buffer_size // 2
        else:
            self.refresh_size = refresh_size
        self.out_batch_size = out_batch_size
        self.device = torch.device(device)

        self.activations = None
        self.read = None

    def __iter__(self):
        return self

    @torch.no_grad()
    def __next__(self) -> torch.Tensor:
        """
        Return a batch of activations
        """

        # check if buffer needs refreshing
        do_refresh = self.read is None or (~self.read).sum() < self.refresh_size
        if do_refresh:
            self.refresh()

        # indexes of unread activations
        unread_i = (~self.read).nonzero().squeeze()

        # get random sample of indexes of unread activations
        rand_idx = torch.randperm(unread_i.size(0), device=self.device)
        rand_idx = rand_idx[: self.out_batch_size]
        batch_i = unread_i[rand_idx]

        # create an actvivation batch
        activation_batch = self.activations[batch_i]

        # mark activations as read
        self.read[batch_i] = True

        return activation_batch

    @torch.no_grad()
    def refresh(self):

        # commenting these out, I guess  I'll figure out why they are here later
        # gc.collect()
        # t.cuda.empty_cache()

        # compact remaining activations
        if self.activations is not None:
            self.activations = self.activations[~self.read]
            cur_buffer_size = self.activations.size(0)
        else:
            cur_buffer_size = 0

        new_buffer = []
        while cur_buffer_size < self.buffer_size:

            # load file when it exists
            files = os.listdir(self.output_folder)
            while len(files) == 0:
                files = os.listdir(self.output_folder)

            files = sorted(files)  # make it deterministic

            fname = os.path.join(self.output_folder, files[0])

            new_activations = load_torch(fname)
            new_activations = new_activations.to(self.device)

            new_buffer.append(new_activations)
            cur_buffer_size += len(new_activations)

            os.remove(fname)

        if self.activations is not None:
            new_buffer += [self.activations]

        self.activations = torch.cat(new_buffer, dim=0)

        # reset tracking
        self.read = torch.zeros(
            len(self.activations), dtype=torch.bool, device=self.device
        )


def make_subsample_activations_fn(sub_rate, max_length):

    assert 0 < sub_rate and sub_rate <= 1.0
    assert 0 < max_length

    num_embs = math.ceil(max_length * sub_rate)

    def subsample_activations(all_activations):

        subsampled_actiavtions = []
        for activations in all_activations:

            size_a = len(activations)
            if size_a == 0:
                continue

            _num_embs = min(num_embs, size_a)
            rand_idx = torch.randperm(_num_embs, device=activations[0].device)
            rand_idx = rand_idx[:_num_embs]
            activations = activations[rand_idx]

            subsampled_actiavtions.append(activations)

        return subsampled_actiavtions

    return subsample_activations


def make_standardize_activations_fn(mean, std):

    inv_std = std.reciprocal()

    def standardize_activations(activations):

        centered_activations = activations - mean
        standardized_activations = centered_activations * inv_std

        return standardized_activations

    return standardize_activations


def make_subsampling_load_buffer():

    output_folder = "../data/positional-SAE/temp_activations"

    buffer = LoadActivationBuffer(output_folder)

    return buffer


# TODO make this better
def make_subsampling_save_buffer(sub_rate):

    assert isinstance(sub_rate, float)
    assert 0 < sub_rate and sub_rate <= 1.0

    data = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

    output_folder = "../data/positional-SAE/temp_activations"

    model_device = torch.device("cuda:1")

    config_fname = "configs/ai/pythia-70m.json"
    config = load_json(config_fname)

    tokenizer = load_tokenizer(config)

    model_name = "EleutherAI/pythia-70m"
    model_name = model_name.lower()
    model_name_f = model_name.replace("/", "_")

    revision = "main"
    cache_dir = f"./.cache/{model_name_f}_{revision}"

    acitvations_device = torch.device("cpu")

    max_length = 128
    tokenizer_kwargs = {
        "return_tensors": "pt",
        "max_length": max_length,
        "padding": True,
        "truncation": True,
    }

    # model_kwargs = {
    #     "repo_id": model_name,
    #     "automodel": AutoModelForCausalLM,
    #     "revision": revision,
    #     "cache_dir": cache_dir,
    #     "device_map": model_device,
    # }

    # nn_model = TransformersModel(**model_kwargs)

    from ai_models import load_model

    nn_model = load_model(config)
    nn_model = nn_model.gpt_neox
    nn_model.to(model_device)

    def save_activation_fn(model: TransformersModel, batch):

        batch = {k: v.to(model_device) for k, v in batch.items()}

        attention_mask = batch["attention_mask"]

        with model.trace(**batch, **tracer_kwargs, use_cache=False) as tracer:

            output = model.gpt_neox.final_layer_norm.output.save()

            tracer.stop()

        result = {"attention_mask": attention_mask, "output": output}

        return result

    def save_activation_fn2(model: TransformersModel, batch):

        batch = {k: v.to(model_device) for k, v in batch.items()}

        output = model(**batch, use_cache=False)

        output = output.last_hidden_state

        attention_mask = batch["attention_mask"]

        result = {"attention_mask": attention_mask, "output": output}

        return result

    def process_activations_fn(result):

        attention_mask = result["attention_mask"]
        output = result["output"]

        lengths = torch.sum(attention_mask, dim=1)

        all_activations = []

        for example_activations, length in zip(output, lengths):

            example_activations = example_activations[:length]

            all_activations.append(example_activations)

        return all_activations

    subsample_activations_fn = make_subsample_activations_fn(sub_rate, max_length)

    fname = "../data/positional-SAE/experiments_subsampling/mean_std_exp_multi/0.2_0.pt"

    mean_std = load_torch(fname)
    mean = mean_std["mean"]
    mean = mean.to(model_device)
    std = mean_std["std"]
    std = std.to(model_device)

    standardize_activations_fn = make_standardize_activations_fn(mean, std)

    pipe = [
        save_activation_fn2,
        process_activations_fn,
        subsample_activations_fn,
    ]

    post_pipe = [
        standardize_activations_fn,
    ]

    buffer = SaveActivationBuffer(
        data,
        nn_model,
        tokenizer,
        pipe,
        output_folder=output_folder,
        post_processing_fns=post_pipe,
        buffer_size=131072,
        tokenizer_kwargs=tokenizer_kwargs,
        device=acitvations_device,
    )

    return buffer


@torch.no_grad()
def _main():
    """
    Test Activation Buffer
    """

    set_random_seeds(54321)

    output_folder = "../data/positional-SAE/temp_activations"

    buffer = make_subsampling_save_buffer(0.2)

    buffer.save_activations()


if __name__ == "__main__":
    _main()
