from utils import load_json

from dotenv import load_dotenv

from transformers import GPTNeoXForCausalLM, AutoTokenizer

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


def _load_pythia_model(config, **kwargs):
    """
    kwargs can override config, passed to from_pretrained function
    """

    _load_basic_model(config, GPTNeoXForCausalLM, **kwargs)


family2info = {"pythia": {"load_fn": _load_pythia_model}}


def load_model(config):

    model_family = config["model_family"]

    load_fn = family2info[model_family]["load_fn"]

    ai_model = load_fn(load_fn)

    return ai_model


def main():

    fname = "configs/pythia-70m.json"

    config = load_json(fname)

    load_pythia_model(config)


if __name__ == "__main__":
    main()
