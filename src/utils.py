import json
import os
import pathlib
from random import randint, seed as python_seed
import shutil

from numpy.random import seed as numpy_seed
import torch
from torch.random import manual_seed as pytorch_seed


def _make_temp_file(base_file, max_range=1000000):

    make_folder("./.temp")

    # create a unique filename
    ri = randint(0, max_range)
    temp_fname = f"./.temp/{ri}_{base_file}"

    while os.path.isfile(temp_fname):
        ri = randint(0, max_range)
        temp_fname = f"./.temp/{ri}_{base_file}"

    return temp_fname


def _save_file(save_fn, data, fname, keep_tmp_on_fail, **kwargs):
    """
    saves a file in a way that failing won't overwrite existing data

    """

    temp_fname = _make_temp_file("temp_save_file")

    # try to save the temp file
    try:
        save_fn(data, temp_fname, **kwargs)

    except Exception as e:

        # delete temp file on exception if required
        if not keep_tmp_on_fail and os.path.isfile(temp_fname):
            os.remove(temp_fname)

        err_msg = f"failed to save temp file {temp_fname}"
        raise Exception(err_msg) from e

    # tempory file was saved successfully, move it to the target destination
    shutil.copyfile(temp_fname, fname)

    # remove temp file
    os.remove(temp_fname)


def make_folder(folder):

    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)


def load_json(fname):

    with open(fname, "r") as f_in:
        json_data = json.load(f_in)

    return json_data


def save_json(json_data, fname, keep_tmp_on_fail=False, **kwargs):

    def save_fn(d, f, **k):
        with open(f, "w") as f_out:
            json.dump(d, f_out, **k)

    _save_file(save_fn, json_data, fname, keep_tmp_on_fail, **kwargs)


def load_torch(fname):

    torch_data = torch.load(fname)

    return torch_data


def save_torch(torch_data, fname, keep_tmp_on_fail=False, **kwargs):

    def save_fn(d, f, **k):
        torch.save(d, f, **k)

    _save_file(save_fn, torch_data, fname, keep_tmp_on_fail, **kwargs)


def set_random_seeds(seed):
    numpy_seed(seed)
    python_seed(seed)
    pytorch_seed(seed)
