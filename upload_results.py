""" Simple script to upload PGMORL run results into WANDB"""

import argparse
import json
import pathlib
from copy import deepcopy
from typing import Callable, List, Union

import numpy as np
import numpy.typing as npt
import pymoo.indicators.hv
import pymoo.util.ref_dirs
import wandb
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
    front = np.array(front)
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
    parser = argparse.ArgumentParser(description="Upload PGMORL results into WANDB")
    parser.add_argument("result_dir", type=str)
    parser.add_argument("experiment_name", type=str)
    parser.add_argument("experiment_group", type=str)
    parser.add_argument("--use-disc-rets", action="store_true")
    parser.add_argument("--entity", type=str, default="santeri-heiskanen")
    parser.add_argument("--project-name", type=str, default="msc_bench")
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
        default=50,
        help="Amount of weights used to calculate EUM. Default %(default)s",
    )
    parser.add_argument(
        "--plot-spec", type=str, default="santeriheiskanen/pareto-front"
    )
    parser.add_argument(
        "--ref-point",
        nargs="*",
        default=np.array([-100, -100]),
        help="Reference point for calculating hypervolume Default %(default)s",
    )
    parser.add_argument(
            "--ref-set-path", type=pathlib.Path, default=None,
            help="Path to the reference set file for calculating IGB+"
    )
    return parser


def upload_data(args):
    dirpath = pathlib.Path(args.result_dir)
    last_ep = 0
    global_step = 0

    pareto_front_table = []

    # load the options used to train the agents
    if (args_path := (dirpath / "args.json")).is_file():
        with args_path.open("r") as ifstream:
            opts = json.load(ifstream)
    else:
        opts = load_args(dirpath / "args.txt")


    if args.ref_set_path is not None:
        payload = json.loads(args.ref_set_path.read_text())
        ref_set = np.asarray([[obj["x"], obj["y"]] for obj in payload])
    else:
        ref_set = None

    wandb_run = wandb.init(
        entity=args.entity,
        project=args.project_name,
        sync_tensorboard=True,
        config=opts,
        name=args.experiment_name,
        group=args.experiment_group,
    )
    wandb_run.define_metric("*", step_metric="true_global_step")
    wandb_run.define_metric("scaled_steps", step_metric="global_step")

    avg_key = "avg_obj_{}" if not args.use_disc_rets else "avg_disc_obj_{}"
    sd_key = "std_obj_{}" if not args.use_disc_rets else "std_disc_obj_{}"

    wandb_run.log(
        {
            "true_global_step": global_step,
            "global_step": global_step,
            "episode": 0,
            "eval/hypervolume": 0.0,
            "eval/sparsity": 100.0,
            "eval/eum": 0.0
        }
    )

    num_steps = opts["num_steps"]
    num_tasks = opts["num_tasks"]
    num_procs = opts["num_processes"]
    max_step = opts["num_env_steps"] * num_tasks

    # The scale used to modify the global-step metric
    scale = args.target_step / max_step

    directories = [
        fp for fp in dirpath.iterdir() if fp.is_dir() and "final" not in fp.stem
    ]
    directories.sort(key=lambda fp: int(fp.stem))

    for fp in directories:
        ep = int(fp.stem)
        global_step += (ep - last_ep) * num_tasks * num_steps * num_procs
        last_ep = ep
        print(
            (f"EP: {ep} | (scaled) global_step {int(global_step*scale)} |  "
             f"Real step {global_step} | Total num env steps {max_step}")
        )

        res_path = fp / "population" / "objs.json"
        if res_path.exists() and res_path.is_file():
            with res_path.open("r") as ifstream:
                objs = json.load(ifstream)

            avg_objs = np.array([[o[avg_key.format(0)], o[avg_key.format(1)]] for o in objs])
            std_objs = np.array([[o[sd_key.format(0)], o[sd_key.format(1)]] for o in objs])
            # Lets find the pareto-front
            pareto_ind = get_non_pareto_dominated_inds(
                    avg_objs, remove_duplicates=True
            )

            avg_objs = avg_objs[pareto_ind]
            std_objs = std_objs[pareto_ind]
            hv = get_hypervol(args.ref_point, avg_objs)
            sparsity = get_sparsity(avg_objs)
            eum = get_expected_utility(
                avg_objs,
                get_equally_spaced_prefs(args.reward_dim, args.n_sample_weights),
            )

            wandb_run.log(
                {
                    "true_global_step": global_step,
                    "global_step": int(scale*global_step),
                    "episode": ep,
                    "eval/hypervolume": hv,
                    "eval/sparsity": sparsity,
                    "eval/eum": eum
                }, commit = False
            )

            if ref_set is not None:
                igd_plus_max = get_igd_plus_max(ref_set, avg_objs)
                igd_plus = get_igd_plus(ref_set, avg_objs)
                print(f"IGD+ {igd_plus} | IGD+ max {igd_plus_max}")

                wandb_run.log({
                    "eval/igd_plus_max": igd_plus_max,
                    "eval/igd_plus": igd_plus
                })


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
            objs = np.loadtxt(fp / "population" / "objs.txt", delimiter=",")

            # Find the pareto front
            pareto_ind = get_non_pareto_dominated_inds(objs, remove_duplicates=True)
            objs = objs[pareto_ind]
            hv = get_hypervol(args.ref_point, objs)
            sparsity = get_sparsity(objs)
            eum = get_expected_utility(
                objs, get_equally_spaced_prefs(args.reward_dim, args.n_sample_weights)
            )
            wandb_run.log(
                {
                    "true_global_step": global_step,
                    "global_step": int(scale*global_step),
                    "episode": ep,
                    "eval/hypervolume": hv,
                    "eval/sparsity": sparsity,
                    "eval/eum": eum
                }
            )

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
    eval_table = wandb.Table(
        columns=["step", "episode", "avg_obj1", "avg_obj2", "std_obj1", "std_obj2"],
        data=pareto_data,
    )
    wandb_run.log({"eval/pareto-front": eval_table})
    wandb_run.plot_table(
        vega_spec_name=args.plot_spec,
        data_table=eval_table,
        fields={
            "x": "avg_obj1",
            "y": "avg_obj2",
        },
        string_fields={
            "title": "Pareto-front",
            "size": "step",
            "size_legend": "Step",
            "x-label": "Average speed reward",
            "y-label": "Average energy efficiency"
        }
    )
    wandb_run.finish()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    upload_data(args)
