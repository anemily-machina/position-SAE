from ai_models import load_model, load_tokenizer, get_emb_fn
from utils import load_json

from datasets import load_dataset
import torch

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DEVICE = "cpu"  # TODO remove


def main(ai_config, dataset_config, kmeans_config):

    tokenizer = load_tokenizer(ai_config)
    ai_model = load_model(ai_config)
    ai_model.to(DEVICE)

    max_length = kmeans_config["max_length"]
    layers = kmeans_config["layers"]

    tokenizer_kwargs = {"max_length": max_length}

    emb_fn = get_emb_fn(
        tokenizer=tokenizer,
        ai_model=ai_model,
        config=ai_config,
        layers=layers,
        device=DEVICE,
        tokenizer_kwargs=tokenizer_kwargs,
    )

    batch = [
        "this is a sentence",
        "the large dog went to the park",
        "I banked my plane but still crashed into the bank",
    ]

    print(emb_fn(batch))


if __name__ == "__main__":

    ai_config = load_json("configs/ai/pythia-70m.json")
    dataset_config = load_json("configs/datasets/the_pile.json")
    kmeans_config = load_json("configs/kmeans/test_lastlayer_random.json")

    main(ai_config, dataset_config, kmeans_config)
