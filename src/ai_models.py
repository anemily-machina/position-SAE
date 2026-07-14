from utils import load_json

from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM, PreTrainedTokenizer

load_dotenv()


def _process_config_for_loading(config):

    model_name = config["model_name"]
    model_name = model_name.lower()
    model_name_f = model_name.replace("/", "_")

    revision = config.get("revision", "main")

    cache_folder = f"./.cache/{model_name_f}_{revision}"

    loading_params = {
        "pretrained_model_name_or_path": model_name,
        "revision": revision,
        "cache_dir": cache_folder,
    }

    return loading_params


def _load_basic_tokenizer(config, tokenizer_class, **kwargs) -> PreTrainedTokenizer:
    """
    basic tokenizer_class.from_pretrained

    kwargs can override config, passed to from_pretrained function
    """

    loading_params = _process_config_for_loading(config)

    for k in kwargs:
        if k in loading_params:
            loading_params.pop(k)

    tokenizer = tokenizer_class.from_pretrained(**loading_params, **kwargs)

    return tokenizer


def _load_basic_model(config, ai_class, **kwargs):
    """
    basic ai_class.from_pretrained

    kwargs can override config, passed to from_pretrained function
    """

    loading_params = _process_config_for_loading(config)

    for k in kwargs:
        if k in loading_params:
            loading_params.pop(k)

    ai_model = ai_class.from_pretrained(**loading_params, **kwargs)

    return ai_model


def _load_autotokenizer_model(config, **kwargs) -> PreTrainedTokenizer:

    tokenizer = _load_basic_tokenizer(config, AutoTokenizer, **kwargs)

    return tokenizer


def _load_pythia_model(config, **kwargs):
    """
    kwargs can override config, passed to from_pretrained function
    """

    ai_model = _load_basic_model(config, GPTNeoXForCausalLM, **kwargs)

    return ai_model


def _make_basic_emb_fn(
    tokenizer,
    ai_model,
    config,
    layers,
    device="cpu",
    tokenizer_kwargs={},
    model_kwargs={},
):

    tk_default = {"padding": True, "return_tensors": "pt", "truncation": True}
    mk_default = {"output_hidden_states": True}

    for k in tokenizer_kwargs:
        if k in tk_default:
            tk_default.pop(k)

    for k in model_kwargs:
        if k in mk_default:
            mk_default.pop(k)

    device = torch.device(device)

    def emb_fn(
        batch,
    ):

        with torch.no_grad():

            # tokenize batch
            batch_t = tokenizer(batch, **tk_default, **tokenizer_kwargs)

            # calculate non-masked positions
            attention_mask = batch_t["attention_mask"]
            batch_dims = [am.nonzero(as_tuple=True) for am in attention_mask]

            # process batch
            batch_t = {k: v.to(device) for k, v in batch_t.items()}
            output = ai_model(**batch_t, **mk_default, **model_kwargs)
            hidden_states = output.hidden_states

            # extra layer hidden states
            emb_dict = {}
            for l in layers:

                raw_embs = hidden_states[l]

                embs = []
                for bd, re in zip(batch_dims, raw_embs):

                    e = re[bd]

                    embs.append(e)

                emb_dict[l] = embs

            return emb_dict

    return emb_fn


family2info = {
    "pythia": {
        "load_ai_fn": _load_pythia_model,
        "load_tokenizer_fn": _load_autotokenizer_model,
        "make_emb_fn": _make_basic_emb_fn,
    }
}


def load_tokenizer(config) -> PreTrainedTokenizer:

    model_family = config["model_family"]

    load_tokenizer_fn = family2info[model_family]["load_tokenizer_fn"]

    tokenizer = load_tokenizer_fn(config)

    return tokenizer


def load_model(config):

    model_family = config["model_family"]

    load_fn = family2info[model_family]["load_ai_fn"]

    ai_model = load_fn(config)

    return ai_model


def get_emb_fn(
    tokenizer,
    ai_model,
    config,
    layers,
    device="cpu",
    tokenizer_kwargs={},
    model_kwargs={},
):

    model_family = config["model_family"]

    make_emb_fn = family2info[model_family]["make_emb_fn"]

    emb_fn = make_emb_fn(
        tokenizer, ai_model, config, layers, device, tokenizer_kwargs, model_kwargs
    )

    return emb_fn


def main():

    fname = "configs/ai/pythia-70m.json"

    config = load_json(fname)

    tokenizer = load_tokenizer(config)

    ai_model = load_model(config)

    emb_fn = get_emb_fn(tokenizer, ai_model, layers=[-1], config=config)

    batch = [
        "this is a sentence",
        "the large dog went to the park",
        "I banked my plane but still crashed into the bank",
    ]

    emb_dict = emb_fn(batch)

    print()
    print()
    print(emb_dict)
    print([e.size() for e in emb_dict[-1]])
    print()
    print()


if __name__ == "__main__":
    main()
