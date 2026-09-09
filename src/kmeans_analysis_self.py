"""
This code runs very slowly as the assignment problem is solved in
a single thread

functioning this out so i can run each rate + random in parallel

python src/kmeans_analysis_self.py -sub_rate 1.0
python src/kmeans_analysis_self.py -sub_rate 0.8
python src/kmeans_analysis_self.py -sub_rate 0.6
python src/kmeans_analysis_self.py -sub_rate 0.4
python src/kmeans_analysis_self.py -sub_rate 0.2
python src/kmeans_analysis_self.py -sub_rate 0.05
python src/kmeans_analysis_self.py -sub_rate random

[1.0, 0.8, 0.6, 0.4, 0.2, 0.05]

"""

from utils import (
    make_folder,
    load_pickle,
    save_pickle,
    set_random_seeds,
)


from argparse import ArgumentParser
import os
from time import time


from scipy.optimize import linear_sum_assignment
import torch
from tqdm import tqdm

OUTPUT_FOLDER = "../data/positional-SAE/experiments_subsampling"


def parse_args():

    parser = ArgumentParser()

    parser.add_argument("-sub_rate", "--sub_rate", required=False, type=str)

    parser.add_argument("-seed", "--rng-seed", required=False, default=4321, type=int)

    args = parser.parse_args()

    return args


def compare_clusters(cluster1, cluster2):

    # tested and torch was faster than numpy (30 vs 28)
    cluster1 = torch.tensor(cluster1, dtype=torch.float16)
    cluster2 = torch.tensor(cluster2, dtype=torch.float16)

    l1 = []
    l2 = []
    cosine = []

    norm_cluster_2 = torch.nn.functional.normalize(cluster2, 2, dim=1)

    for vec_i in tqdm(cluster1, total=len(cluster1), ncols=50):

        diff = cluster2 - vec_i

        abs = diff.abs()
        l1_i = abs.mean(dim=1)
        l1.append(l1_i)

        sqr = abs.pow(2)
        l2_i = sqr.mean(dim=1)
        l2.append(l2_i)

        vec_i_norm = torch.nn.functional.normalize(vec_i, 2, dim=0)
        vec_i_norm_t = vec_i_norm.t()
        cosine_i = norm_cluster_2 @ vec_i_norm_t
        cosine.append(cosine_i)

    score_info = {
        "L1": {
            "cost_matrix": l1,
            "maximize": False,
        },
        "L2": {
            "cost_matrix": l2,
            "maximize": False,
        },
        "cosine_sim": {
            "cost_matrix": cosine,
            "maximize": True,
        },
    }

    scores = {}
    for score_key, la_params in score_info.items():

        print()
        print(score_key)

        cost_matrix = la_params.pop("cost_matrix")
        cost_matrix = torch.stack(cost_matrix)

        start_time = time()

        row_ind, col_ind = linear_sum_assignment(cost_matrix, **la_params)

        total_time = time() - start_time
        total_time = total_time / 60

        print(f"total time: {total_time}m")

        best_match_costs = cost_matrix[row_ind, col_ind]
        score = best_match_costs.sum() / len(row_ind)
        score = float(score)
        print()
        print(score)
        print()

        scores[score_key] = score


def main():

    args = parse_args()

    set_random_seeds(args.rng_seed)

    sub_rate = args.sub_rate
    assert sub_rate in ["1.0", "0.8", "0.6", "0.4", "0.2", "0.05", "random"]
    number_of_trials = 5

    exp_key = "kmeans_exp_multi"
    exp_folder = os.path.join(OUTPUT_FOLDER, exp_key)

    stats_folder = "kmeans_exp_multi_stats_self"
    make_folder(stats_folder)

    if sub_rate == "random":
        fake_vectors = torch.randn((5, 32000, 512))

    # compute stability for each sub rate
    for t_i in range(number_of_trials - 1):

        if sub_rate != "random":
            trial_file_name_i = f"{sub_rate}_{t_i}.pkl"
            trial_fname_i = os.path.join(exp_folder, trial_file_name_i)
            kmeans_i = load_pickle(trial_fname_i)
            clusters_i = kmeans_i.cluster_centers_

        else:
            clusters_i = fake_vectors[t_i]

        for t_j in range(t_i + 1, number_of_trials):

            if sub_rate != "random":
                trial_file_name_j = f"{sub_rate}_{t_j}.pkl"
                trial_fname_j = os.path.join(exp_folder, trial_file_name_j)
                kmeans_j = load_pickle(trial_fname_j)
                clusters_j = kmeans_j.cluster_centers_
            else:
                clusters_j = fake_vectors[t_j]

            stats_file_name = f"{sub_rate}_{t_i}_{t_j}.pkl"
            stats_fname = os.path.join(stats_folder, stats_file_name)

            print()
            print(stats_file_name)
            print()

            if os.path.isfile(stats_fname):
                print()
                print("file exists skipping")
                print()
                continue

            scores = compare_clusters(cluster1=clusters_i, cluster2=clusters_j)
            save_pickle(scores, stats_fname)


if __name__ == "__main__":
    with torch.no_grad():
        main()
