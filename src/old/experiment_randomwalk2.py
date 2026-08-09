"""
100%|███████████| 1000000/1000000 [04:31<00:00, 3686.58it/s]
100%|███████████▉| 999962/1000000 [04:31<00:00, 3708.33it/s]


0.9972585986328125
129304
  0%|                       | 0/1 [04:31<?, ?it/s]
"""

from utils import load_json, set_random_seeds, save_json


import json
from math import prod
import os
from random import randint, sample

from tqdm import tqdm


def run_experiment(k, steps):

    d = 1024

    accs = [0 for _ in range(d)]

    results = []
    best = 0
    for s in tqdm(range(steps), total=steps, ncols=60):

        rands = [randint(-k, k) for _ in range(d)]

        accs = [a + r for a, r in zip(accs, rands)]

        good = sum([a > k or -k > a for a in accs])

        if good == d:
            best += 1

        good = good / d

        results.append(good)

    return results, best


def make_results(num_experiments, k, steps):

    # r_fname = "./results.json"
    # if os.path.isfile(r_fname):
    #     results = load_json(r_fname)
    #     return results

    results = []

    for _ in tqdm(range(num_experiments), total=num_experiments, ncols=50):

        k, best = run_experiment(k, steps)

        print()
        print()
        print()
        print(sum(k) / len(k))
        print(best)

        exit()

        p = k / steps

        results.append(p)

    # save_json(results, r_fname, indent=2)

    return results


def main():

    # set_random_seeds(4321)

    k = 100000
    steps = 1000000
    num_experiments = 1  # 00000

    scale = 100

    results = make_results(num_experiments, k, steps)

    hist = {k: 0 for k in range(0, scale + 1)}
    for p in results:

        hist_k = round(scale * p)
        hist[hist_k] += 1

    for k in list(hist.keys()):

        hist[k] = hist[k] / num_experiments

    save_json(hist, f"./hists_1.json", indent=2)

    upsamples = [256, 512, 1024, 2048]

    hists = {d: {k: 0 for k in range(0, scale + 1)} for d in upsamples}

    for d in upsamples:

        ne = num_experiments * d

        print()
        print(f"simulating dimension {d} using upsampling")
        print(f"{ne} samples of original probs created")
        print()

        d_hist = hists[d]

        for _ in tqdm(range(ne), total=ne, ncols=50):

            probs = sample(results, d)

            p = prod(probs)

            hist_k = round(scale * p)
            d_hist[hist_k] += 1

        for k in list(d_hist.keys()):

            d_hist[k] = d_hist[k] / ne

        save_json(d_hist, f"./hists_{d}.json", indent=2)


if __name__ == "__main__":
    main()
