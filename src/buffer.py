"""
Large parts of this are inspired of borrowed from the dictionary_learning
"""

from config import DEBUG


from nnsight.modeling.transformers import TransformersModel
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer

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
            get_activations_fns (list[callable]): Iteratively processes a tokenized batch of text. Must have at least one function. The first function will be called fn1(self.model, self.tokenizer(text_batch, self.tokenizer_kwargs)). Each other call will be fn_i(out_fn_i-1). The final result  is expected that this is a list containing tensors of activations, one for each sentence in the batch. They will be passed as torch.cat((existing_buffer,*result), dim=0)) so anything compatible with that works.
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
    def __next__(self):
        """
        Return a batch of activations
        """

        # check if buffer needs refreshing
        do_refresh = self.read is None or (~self.read).sum() < self.refresh_size
        if do_refresh:
            self._refresh()

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
    def _refresh(self):

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

            result = fn1(self.model, tokenized_batch)
            for fn in pipe:
                result = fn(result)

            # puch activations to the correct device
            batch_activations = []
            for activations in result:
                activations = activations.to(self.device)
                batch_activations.append(activations)

            # update loop
            update_size = sum([len(b) for b in batch_activations])
            cur_buffer_size += update_size

            new_buffer += batch_activations

            if self.verbose:
                pbar.update(update_size)

        if self.verbose:
            pbar.close()

        # make one buffer
        if self.activations is not None:
            new_buffer += [self.activations]

        self.activations = torch.cat(new_buffer, dim=0)

        # reset tracking
        self.read = torch.zeros(
            len(self.activations), dtype=torch.bool, device=self.device
        )


@torch.no_grad()
def _main():
    """
    Test Activation Buffer
    """

    from dotenv import load_dotenv

    from ai_models import load_tokenizer
    from utils import load_json, set_random_seeds, load_torch

    import math

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM

    load_dotenv()
    set_random_seeds(54321)

    data = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    data_iter = iter(data)

    config_fname = "configs/ai/pythia-70m.json"
    config = load_json(config_fname)

    tokenizer = load_tokenizer(config)

    model_name = "EleutherAI/pythia-70m"
    model_name = model_name.lower()
    model_name_f = model_name.replace("/", "_")

    revision = "main"
    cache_dir = f"./.cache/{model_name_f}_{revision}"

    device = torch.device("cpu")

    tokenizer_kwargs = {
        "return_tensors": "pt",
        "max_length": 128,
        "padding": True,
        "truncation": True,
    }

    model_device = torch.device("cuda:1")
    model_kwargs = {
        "repo_id": model_name,
        "automodel": AutoModelForCausalLM,
        "revision": revision,
        "cache_dir": cache_dir,
        "device_map": model_device,
    }

    nn_model = TransformersModel(**model_kwargs)

    def tokenizer_fn(text_batch, **kwargs):

        tokenized_batch = tokenizer(text_batch, **kwargs)

        tokenized_batch = {k: v.to(model_device) for k, v in tokenized_batch.items()}

        return tokenized_batch

    def save_activation_fn(model: TransformersModel, batch):

        with model.trace(**batch, **tracer_kwargs) as tracer:

            input = model.gpt_neox.inputs.save()

            output = model.gpt_neox.final_layer_norm.output.save()

            tracer.stop()

        result = {"input": input, "output": output}

        return result

    def process_activations_fn(result):

        input = result["input"]
        output = result["output"]

        attention_mask = input[1]["attention_mask"]

        lengths = torch.sum(attention_mask, dim=1)

        all_activations = []

        for example_activations, length in zip(output, lengths):

            example_activations = example_activations[:length]

            all_activations.append(example_activations)

        return all_activations

    def make_subsample_activations_fn(sub_rate):

        assert 0 < sub_rate and sub_rate <= 1.0

        def subsample_activations(all_activations):

            subsampled_actiavtions = []
            for activations in all_activations:

                num_embs = math.ceil(activations.size(0) * sub_rate)
                rand_idx = torch.randperm(num_embs, device=activations[0].device)
                rand_idx = rand_idx[:num_embs]
                activations = activations[rand_idx]

                subsampled_actiavtions.append(activations)

            return subsampled_actiavtions

        return subsample_activations

    subsample_activations_fn = make_subsample_activations_fn(0.2)

    fname = "../data/positional-SAE/experiments_subsampling/mean_std_exp_multi/0.2_0.pt"

    mean_std = load_torch(fname)
    mean = mean_std["mean"]
    mean = mean.to(model_device)
    std = mean_std["std"]
    std = std.to(model_device)

    def make_standardize_activations_fn(mean, std):

        inv_std = std.reciprocal()

        def standardize_activations(all_activations):

            standardized_activations = []
            for activations in all_activations:

                sub_embs = activations - mean
                sub_embs = sub_embs * inv_std

                standardized_activations.append(sub_embs)

            return standardized_activations

        return standardize_activations

    standardize_activations_fn = make_standardize_activations_fn(mean, std)

    pipe = [
        save_activation_fn,
        process_activations_fn,
        subsample_activations_fn,
        standardize_activations_fn,
    ]

    buffer = ActivationBuffer(
        data,
        nn_model,
        tokenizer_fn,
        pipe,
        tokenizer_kwargs=tokenizer_kwargs,
        device=device,
    )

    for batch in buffer:

        print(batch)
        print(batch.size())

        exit()


if __name__ == "__main__":
    _main()
