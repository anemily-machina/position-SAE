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
import gc
import math
import os
from time import time


from scipy.optimize import linear_sum_assignment
import torch
from tqdm import tqdm

OUTPUT_FOLDER = "../data/positional-SAE/experiments_subsampling"
NUMBER_OF_TRIALS = 5


def parse_args():

    parser = ArgumentParser()

    parser.add_argument("-sub_rate", "--sub_rate", required=False, type=str)

    parser.add_argument("-seed", "--rng-seed", required=False, default=4321, type=int)

    args = parser.parse_args()

    return args


def _linear_sum_score(cost_matrix, maximize):

    row_ind, col_ind = linear_sum_assignment(cost_matrix=cost_matrix, maximize=maximize)

    best_match_costs = cost_matrix[row_ind, col_ind]

    score = best_match_costs.sum() / len(row_ind)

    score = float(score)

    return score


def _compute_cosine(cluster1, cluster2):

    norm_cluster_1 = torch.nn.functional.normalize(cluster1, 2, dim=1)
    norm_cluster_2 = torch.nn.functional.normalize(cluster2, 2, dim=1)

    cost_matrix = norm_cluster_1 @ norm_cluster_2.t()

    score = _linear_sum_score(cost_matrix=cost_matrix, maximize=True)

    return score


def _compute_L1(cluster1: torch.Tensor, cluster2: torch.Tensor):

    vec_size = len(cluster1[0])

    l1 = []

    diff_matrix = torch.zeros(
        cluster2.size(), dtype=cluster1.dtype, device=cluster1.device
    )
    sum_vec = torch.zeros((len(cluster2)), dtype=cluster1.dtype, device=cluster1.device)
    cost_matrix = torch.zeros((len(cluster1), len(cluster2)))

    for v_i, vec_i in tqdm(enumerate(cluster1), total=len(cluster1), ncols=50):

        torch.subtract(cluster2, vec_i, out=diff_matrix)

        torch.abs_(diff_matrix)

        # L1 amoratized across dimensions
        # why is this so much faster than using torch.mean 150it/s vs 450it/s?
        torch.sum(diff_matrix, dim=1, out=sum_vec)
        cost_matrix[v_i] = sum_vec / vec_size

    score = _linear_sum_score(cost_matrix=cost_matrix, maximize=False)

    return score


def _compute_L2(cluster1, cluster2):

    l2 = []
    for vec_i in tqdm(cluster1, total=len(cluster1), ncols=50):

        diff = cluster2 - vec_i

        sqr = diff * diff
        sqr_sum = torch.sum(sqr, dim=1)
        l2_i = torch.sqrt(sqr_sum)

        # L2 amoratized across dimensions
        l2_i = l2_i / len(vec_i)
        l2.append(l2_i)

    # cost_matrix = torch.stack(l2)

    # score = _linear_sum_score(cost_matrix=cost_matrix, maximize=False)

    # return score

    exit()


def compare_clusters(cluster1, cluster2, file_prefix):

    print()
    print(file_prefix)
    print()

    if isinstance(cluster1, torch.Tensor):
        cluster1 = cluster1.clone()
    else:
        cluster1 = torch.from_numpy(cluster1).half().clone()

    if isinstance(cluster2, torch.Tensor):
        cluster2 = cluster2.clone()
    else:
        cluster2 = torch.from_numpy(cluster2).half().clone()

    score2fn = {
        "cosine": _compute_cosine,
        "L1": _compute_L1,
        "L2": _compute_L2,
    }
    for score_key, score_fn in score2fn.items():

        print()
        print(score_key)

        start_time = time()

        score = None
        if file_prefix is not None:

            temp_fname = f"{file_prefix}_{score_key}.pkl"

            if os.path.isfile(temp_fname):

                print("score exists")

                score = load_pickle(temp_fname)

        if score is None:
            score = score_fn(cluster1, cluster2)

        #     save_pickle(score, temp_fname)

        # total_time = time() - start_time
        # total_time = total_time / 60

        # print(f"score: {score}")
        # print(f"total time: {total_time}m")


def calc_scores_self(sub_rate):

    exp_key = "kmeans_exp_multi"
    exp_folder = os.path.join(OUTPUT_FOLDER, exp_key)

    stats_key = "kmeans_exp_multi_stats_self"
    stats_folder = os.path.join(OUTPUT_FOLDER, stats_key)
    make_folder(stats_folder)

    if sub_rate == "random":
        fake_vectors = torch.randn((5, 32000, 512))

    # compute stability for each sub rate
    for t_i in range(NUMBER_OF_TRIALS - 1):

        if sub_rate != "random":
            trial_file_name_i = f"{sub_rate}_{t_i}.pkl"
            trial_fname_i = os.path.join(exp_folder, trial_file_name_i)
            kmeans_i = load_pickle(trial_fname_i)
            clusters_i = kmeans_i.cluster_centers_

        else:
            clusters_i = fake_vectors[t_i]

        for t_j in range(t_i + 1, NUMBER_OF_TRIALS):

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


def display_scores_self():

    stats_key = "kmeans_exp_multi_stats_self"
    stats_folder = os.path.join(OUTPUT_FOLDER, stats_key)

    scores = {}
    for sub_rate in ["1.0", "0.8", "0.6", "0.4", "0.2", "0.05", "random"]:

        raw_scores = {}

        for t_i in range(NUMBER_OF_TRIALS - 1):
            for t_j in range(t_i + 1, NUMBER_OF_TRIALS):

                stats_file_name = f"{sub_rate}_{t_i}_{t_j}.pkl"
                stats_fname = os.path.join(stats_folder, stats_file_name)

                stats = load_pickle(stats_fname)

                for score_key, score in stats.items():
                    if score_key not in raw_scores:
                        raw_scores[score_key] = []

                    raw_scores[score_key].append(score)

        scores[sub_rate] = {}
        for score_key, raw_scores in raw_scores.items():

            sum_s = sum(raw_scores)
            mean_s = sum_s / len(raw_scores)

            std_v = [(s - mean_s) ** 2 for s in raw_scores]
            std_sum = sum(std_v)
            std_avg = std_sum / len(raw_scores)
            std_s = math.sqrt(std_avg)

            score_entry = {"mean": mean_s, "std": std_s}

            scores[sub_rate][score_key] = score_entry

    print()
    print()
    print(scores)
    print()
    print()


def calc_scores_baseline(sub_rate):

    exp_key = "kmeans_exp_multi"
    exp_folder = os.path.join(OUTPUT_FOLDER, exp_key)

    stats_key = "kmeans_exp_multi_stats_baseline"
    stats_folder = os.path.join(OUTPUT_FOLDER, stats_key)
    make_folder(stats_folder)

    if sub_rate == "random":
        fake_vectors = torch.randn((5, 32000, 512))

    # compute stability for each sub rate
    for t_i in range(NUMBER_OF_TRIALS):

        trial_file_name_i = f"1.0_{t_i}.pkl"
        trial_fname_i = os.path.join(exp_folder, trial_file_name_i)
        kmeans_i = load_pickle(trial_fname_i)
        clusters_i = kmeans_i.cluster_centers_

        for t_j in range(NUMBER_OF_TRIALS):

            if sub_rate != "random":
                trial_file_name_j = f"{sub_rate}_{t_j}.pkl"
                trial_fname_j = os.path.join(exp_folder, trial_file_name_j)
                kmeans_j = load_pickle(trial_fname_j)
                clusters_j = kmeans_j.cluster_centers_
            else:
                clusters_j = fake_vectors[t_j]

            stats_file_prefix = f"1.0_{sub_rate}_{t_i}_{t_j}"

            compare_clusters(
                cluster1=clusters_i,
                cluster2=clusters_j,
                file_prefix=stats_file_prefix,
            )


def display_scores_baseline():

    stats_key = "kmeans_exp_multi_stats_baseline"
    stats_folder = os.path.join(OUTPUT_FOLDER, stats_key)

    scores = {}
    for sub_rate in ["0.8", "0.6", "0.4", "0.2", "0.05", "random"]:

        raw_scores = {}

        for t_i in range(NUMBER_OF_TRIALS):
            for t_j in range(NUMBER_OF_TRIALS):

                stats_file_name = f"1.0_{sub_rate}_{t_i}_{t_j}.pkl"
                stats_fname = os.path.join(stats_folder, stats_file_name)

                stats = load_pickle(stats_fname)

                for score_key, score in stats.items():
                    if score_key not in raw_scores:
                        raw_scores[score_key] = []

                    raw_scores[score_key].append(score)

        scores[sub_rate] = {}
        for score_key, raw_scores in raw_scores.items():

            sum_s = sum(raw_scores)
            mean_s = sum_s / len(raw_scores)

            std_v = [(s - mean_s) ** 2 for s in raw_scores]
            std_sum = sum(std_v)
            std_avg = std_sum / len(raw_scores)
            std_s = math.sqrt(std_avg)

            score_entry = {"mean": mean_s, "std": std_s}

            scores[sub_rate][score_key] = score_entry

    print()
    print()
    print(scores)
    print()
    print()


def main():
    args = parse_args()

    set_random_seeds(args.rng_seed)

    sub_rate = args.sub_rate
    assert sub_rate in ["1.0", "0.8", "0.6", "0.4", "0.2", "0.05", "random"]

    # calc_scores_self(sub_rate)

    calc_scores_baseline(sub_rate)

    # display_scores_self()
    # display_scores_baseline()


if __name__ == "__main__":
    with torch.no_grad():
        main()
