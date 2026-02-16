"""
Simple script to parse & convert PGMORL run results into something that can be
processed by rliable
"""

import warnings
import argparse
import json
import pathlib
from copy import deepcopy
from typing import Callable, List, Union

import numpy as np
import numpy.typing as npt
import pymoo.indicators.hv
import pymoo.util.ref_dirs
from pymoo.indicators.distance_indicator import (
    DistanceIndicator as MooDistanceIndicator,
)
from pymoo.indicators.igd_plus import IGDPlus as MooIGDPlus


def _igd_plus_max_dist(z, a, norm=None):
    """Define the IGD+ metric for the maximization task. Essentially, this
    is the equation 17 from Ishibuchi, Hisao, Hiroyuki Masuda, Yuki Tanigaki,
    and Yusuke Nojima. 2015. “Modified Distance Calculation in Generational
    Distance and Inverted Generational Distance.” In Evolutionary Multi-Criterion
    Optimization, https://doi.org/10.1007/978-3-319-15892-1_8. where a-z is
    changed to z-a.

    Parameters
    ----------
    z : npt.NDArray
        The reference set
    a : npt.NDArray
        The current pareto-front approximat
    norm : float
        The value used for normalizing the values

    Returns
    -------
    float
        The IGD+ metric for a maximization task
    """
    d = z - a
    d[d < 0] = 0
    d = d / norm
    return np.sqrt((d**2).sum(axis=1))


class IGDPlusMax(MooDistanceIndicator):
    def __init__(self, pf, **kwargs):
        super().__init__(pf, _igd_plus_max_dist, 1, **kwargs)


def get_igd_plus_max(ref_set: npt.NDArray, points: List[npt.ArrayLike]) -> float:
    """Calculate the IGD+ metric for a maximization task.

    Parameters
    ----------
    ref_set : npt.NDArray
        The set of reference points.
    points : List[npt.ArrayLike]
        The current approximation of the pareto-front.

    Returns
    -------
    float
        The IGB+ metric for a maximization task.
    """
    return IGDPlusMax(ref_set)(np.array(points))


def get_igd_plus(ref_set: npt.NDArray, points: List[npt.ArrayLike]) -> float:
    """Calculate the IGB+ metric for a minization task.

    Parameters
    ----------
    ref_set : npt.NDArray
        The set of reference points.
    points : List[npt.ArrayLike]
        The current approximation of the pareto-front.

    Returns
    -------
    float
        The IGB+ metric for a minimization task.
    """
    return MooIGDPlus(ref_set)(np.array(points))


def get_equally_spaced_prefs(dim: int, n: int, seed: int = 42):
    return list(
        pymoo.util.ref_dirs.get_reference_directions("energy", dim, n, seed=seed)
    )


def get_expected_utility(
    front: List[np.ndarray], weights_set: List[np.ndarray], utility: Callable = np.dot
) -> float:
    """Expected Utility Metric.

    Expected utility of the policies on the PF for various weights.
    Similar to R-Metrics in MOO. But only needs one PF approximation.
    Paper: L. M. Zintgraf, T. V. Kanters, D. M. Roijers, F. A. Oliehoek, and P. Beau, “Quality Assessment of MORL Algorithms: A Utility-Based Approach,” 2015.

    Args:
        front: current pareto front to compute the eum on
        weights_set: weights to use for the utility computation
        utility: utility function to use (default: dot product)

    Returns:
        float: eum metric
    """
    maxs = []
    for weights in weights_set:
        scalarized_front = np.array([utility(weights, point) for point in front])
        maxs.append(np.max(scalarized_front))

    return np.mean(np.array(maxs), axis=0)


def get_hypervol(ref_point, points):
    return pymoo.indicators.hv.HV(ref_point=-1 * ref_point)(np.array(points) * -1)


def get_sparsity(front: List[np.ndarray]) -> float:
    """Sparsity metric from PGMORL.

    Basically, the sparsity is the average distance between each point in the front.

    Args:
        front: current pareto front to compute the sparsity on

    Returns:
        float: sparsity metric
    """
    if len(front) < 2:
        return 0.0

    sparsity_value = 0.0
    m = len(front[0])
    front = np.asarray(front)
    for dim in range(m):
        objs_i = np.sort(deepcopy(front.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity_value += np.square(objs_i[i] - objs_i[i - 1])
    sparsity_value /= len(front) - 1

    return sparsity_value


def get_non_pareto_dominated_inds(
    candidates: Union[np.ndarray, List], remove_duplicates: bool = True
) -> np.ndarray:
    """A batched and fast version of the Pareto coverage set algorithm.

    Args:
        candidates (ndarray): A numpy array of vectors.
        remove_duplicates (bool, optional): Whether to remove duplicate vectors. Defaults to True.

    Returns:
        ndarray: The indices of the elements that should be kept to form the Pareto front or coverage set.
    """
    candidates = np.array(candidates)
    uniques, indcs, invs, counts = np.unique(
        candidates, return_index=True, return_inverse=True, return_counts=True, axis=0
    )

    res_eq = np.all(candidates[:, None, None] <= candidates, axis=-1).squeeze()
    res_g = np.all(candidates[:, None, None] < candidates, axis=-1).squeeze()
    c1 = np.sum(res_eq, axis=-1) == counts[invs]
    c2 = np.any(~res_g, axis=-1)
    if remove_duplicates:
        to_keep = np.zeros(len(candidates), dtype=bool)
        to_keep[indcs] = 1
    else:
        to_keep = np.ones(len(candidates), dtype=bool)

    return np.logical_and(c1, c2) & to_keep


def filter_pareto_dominated(
    candidates: Union[np.ndarray, List], remove_duplicates: bool = True
) -> np.ndarray:
    """A batched and fast version of the Pareto coverage set algorithm.

    Args:
        candidates (ndarray): A numpy array of vectors.
        remove_duplicates (bool, optional): Whether to remove duplicate vectors. Defaults to True.

    Returns:
        ndarray: A Pareto coverage set.
    """
    candidates = np.array(candidates)
    if len(candidates) < 2:
        return candidates
    return candidates[
        get_non_pareto_dominated_inds(candidates, remove_duplicates=remove_duplicates)
    ]


def _convert_val(value):
    try:
        value = int(value)
        return value
    except ValueError:
        pass

    try:
        value = float(value)
        return value
    except ValueError:
        pass

    return value


def load_args(fp):
    import ast

    data = fp.read_text()
    data = ast.literal_eval(data)
    key, value = None, None
    out = {}
    for item in data:
        if key is not None and "--" in item:
            key = key.strip("--")
            out[key] = True
            key = None
        elif "=" in item:
            key, value = item.split("=")
            key = key.strip("--")
            out[key] = _convert_val(value)
            key, value = None, None
        elif "--" in item and key is None:
            key = item.strip("--")
        elif key is not None:
            out[key] = _convert_val(item)
            key, value = None, None
    return out


def get_parser():
    parser = argparse.ArgumentParser(
        description="Upload PGMORL results into WANDB",
        formatter_class=argparse.HelpFormatter,
    )
    parser.add_argument("result_dir", type=str)
    parser.add_argument("experiment_name", type=str)
    parser.add_argument("experiment_group", type=str)
    # parser.add_argument("--use-disc-rets", action="store_true")
    # parser.add_argument("--entity", type=str, default="santeri-heiskanen")
    # parser.add_argument("--project-name", type=str, default="msc_bench")
    parser.add_argument("--target-step", type=int, default=int(1.2e6))
    parser.add_argument(
        "--reward-dim",
        type=int,
        default=2,
        help="Reward dimension. Default %(default)s",
    )
    parser.add_argument(
        "--n-sample-weights",
        type=int,
        default=100,
        help="Amount of weights used to calculate EUM. Default %(default)s",
    )
    # parser.add_argument(
    #     "--plot-spec", type=str, default="santeriheiskanen/pareto-front"
    # )
    parser.add_argument(
        "--ref-point",
        nargs="*",
        default=np.array([0, 0]),
        help="Reference point for calculating hypervolume Default %(default)s",
    )
    # parser.add_argument(
    #     "--ref-set-path",
    #     type=pathlib.Path,
    #     default=None,
    #     help="Path to the reference set file for calculating IGB+",
    # )
    parser.add_argument(
        "--save-path",
        type=pathlib.Path,
        default=None,
        help="Path to the save directory",
    )
    return parser


def load_run(run_dir, ref_point: npt.NDArray, reward_dim: int, num_weights: int):
    last_ep = 0
    global_step = 0

    eval_weights = get_equally_spaced_prefs(reward_dim, n=num_weights)

    # load the options used to train the agents
    if (args_path := (run_dir / "args.json")).is_file():
        with args_path.open("r") as ifstream:
            opts = json.load(ifstream)
    else:
        opts = load_args(run_dir / "args.txt")

    if args.ref_set_path is not None:
        payload = json.loads(args.ref_set_path.read_text())
        ref_set = np.asarray([[obj["x"], obj["y"]] for obj in payload])
    else:
        ref_set = None

    num_steps = opts["num_steps"]
    num_tasks = opts["num_tasks"]
    num_procs = opts["num_processes"]
    max_step = opts["num_env_steps"] * num_tasks

    # The scale used to modify the global-step metric
    scale = args.target_step / max_step

    directories = [
        fp for fp in run_dir.iterdir() if fp.is_dir() and "final" not in fp.stem
    ]
    directories.sort(key=lambda fp: int(fp.stem))

    eval_data = {
        "eval_undiscounted/hypervolume": [],
        "eval_undiscounted/eum": [],
        "eval_undiscounted/sparsity": [],
        "eval_discounted/hypervolume": [],
        "eval_discounted/eum": [],
        "eval_discounted/sparsity": [],
        "log_step": [],
        "episode": [],
    }
    pareto_front_table = []

    for fp in directories:
        ep = int(fp.stem)
        global_step += (ep - last_ep) * num_tasks * num_steps * num_procs
        last_ep = ep
        print(
            (
                f"EP: {ep} | (scaled) global_step {int(global_step * scale)} |  "
                f"Real step {global_step} | Total num env steps {max_step}"
            )
        )

        res_path = fp / "population" / "objs.json"
        if res_path.exists() and res_path.is_file():
            with res_path.open("r") as ifstream:
                objs = json.load(ifstream)

            avg_objs = []
            std_objs = []
            avg_disc_objs = []
            std_disc_objs = []

            for o in objs:
                avg_objs.append([o[f"avg_obj_{i}"] for i in range(reward_dim)])
                std_objs.append([o[f"std_obj_{i}"] for i in range(reward_dim)])

                avg_disc_objs.append(
                    [o[f"avg_disc_obj_{i}"] for i in range(reward_dim)]
                )
                std_disc_objs.append(
                    [o[f"std_disc_obj_{i}"] for i in range(reward_dim)]
                )

            avg_objs = np.asarray(avg_objs)
            std_objs = np.asarray(std_objs)
            avg_disc_objs = np.asarray(avg_disc_objs)
            std_disc_objs = np.asarray(std_disc_objs)

            # avg_objs = np.array(
            #     [[o[avg_key.format(0)], o[avg_key.format(1)]] for o in objs]
            # )
            # std_objs = np.array(
            #     [[o[sd_key.format(0)], o[sd_key.format(1)]] for o in objs]
            # )
            # Lets find the pareto-front
            pareto_ind = get_non_pareto_dominated_inds(avg_objs, remove_duplicates=True)

            avg_objs = avg_objs[pareto_ind]
            std_objs = std_objs[pareto_ind]

            avg_disc_objs = avg_disc_objs[pareto_ind]
            std_disc_objs = std_disc_objs[pareto_ind]

            hv = get_hypervol(args.ref_point, avg_objs)
            sparsity = get_sparsity(avg_objs)
            eum = get_expected_utility(avg_objs, eval_weights)

            disc_hv = get_hypervol(args.ref_point, avg_disc_objs)
            disc_sparsity = get_sparsity(avg_disc_objs)
            disc_eum = get_expected_utility(avg_disc_objs, eval_weights)

            eval_data["eval_undiscounted/hypervolume"].append(hv)
            eval_data["eval_undiscounted/eum"].append(eum)
            eval_data["eval_undiscounted/sparsity"].append(sparsity)
            eval_data["eval_discounted/hypervolume"].append(disc_hv)
            eval_data["eval_discounted/eum"].append(disc_eum)
            eval_data["eval_discounted/sparsity"].append(disc_sparsity)
            eval_data["log_step"].append(global_step)
            eval_data["episode"].append(ep)

            if ref_set is not None:
                igd_plus_max = get_igd_plus_max(ref_set, avg_objs)
                igd_plus = get_igd_plus(ref_set, avg_objs)
                print(f"IGD+ {igd_plus} | IGD+ max {igd_plus_max}")

            for i in range(avg_objs.shape[0]):
                pareto_front_table.append(
                    {
                        "step": global_step,
                        "episode": ep,
                        "avg_obj1": avg_objs[i, 0],
                        "avg_obj2": avg_objs[i, 1],
                        "std_obj1": std_objs[i, 0],
                        "std_obj2": std_objs[i, 1],
                    }
                )
        else:
            warnings.warn(f"Missing discounted returns for {fp!s}")
            objs = np.loadtxt(fp / "population" / "objs.txt", delimiter=",")

            # Find the pareto front
            pareto_ind = get_non_pareto_dominated_inds(objs, remove_duplicates=True)
            objs = objs[pareto_ind]
            hv = get_hypervol(args.ref_point, objs)
            sparsity = get_sparsity(objs)
            eum = get_expected_utility(
                objs,
                eval_weights,
            )

            eval_data["eval_undiscounted/hypervolume"].append(hv)
            eval_data["eval_undiscounted/sparsity"].append(sparsity)
            eval_data["eval_undiscounted/eum"].append(eum)
            eval_data["episode"].append(ep)
            eval_data["log_step"].append(global_step)

            for i in range(objs.shape[0]):
                pareto_front_table.append(
                    {
                        "step": global_step,
                        "episode": ep,
                        "avg_obj1": objs[i, 0],
                        "avg_obj2": objs[i, 1],
                        "std_obj1": np.nan,
                        "std_obj2": np.nan,
                    }
                )

    pareto_data = list(map(lambda row: list(row.values()), pareto_front_table))
    return eval_data


def upload_data(args):
    # Create npz files that contain results for each seed?

    dirpath = pathlib.Path(args.result_dir)

    # Lets extract some info from the path
    parts = dirpath.stem.split("-")
    algo = parts[0]
    assert algo == "pgmorl", f"Unknown algo {algo!r}"
    env = parts[1]
    assert env in (
        "ant",
        "swimmer",
        "hopper",
        "hopper-v3",
        "humanoid",
        "walker2d",
        "halfcheetah",
    ), f"Unknown env {env!r}"
    date = "-".join(parts[2:5])

    save_dir = pathlib.Path(args.save_dir)
    save_dir = save_dir / f"{algo}-{env}-{date}"
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving processed results to {save_dir!s}")

    for run_dir in dirpath.glob("run-*"):
        seed = int(run_dir.stem.split("-")[-1])
        seed_data = load_run(
            run_dir,
            ref_point=args.ref_point,
            reward_dim=args.reward_dim,
            num_weights=args.n_sample_weights,
        )
        print(f"Seed {seed} done!")
        np.savez_compressed(save_dir / f"run-{seed}", *seed_data)

    print("Processing done!")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    upload_data(args)
