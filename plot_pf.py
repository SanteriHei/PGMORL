import argparse
import pathlib
import plotly.express as px
import numpy as np
import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=str)
    return parser


def read_data(args):
    dirpath = pathlib.Path(args.result_dir)
    objs = []
    for fp in dirpath.iterdir():
        if fp.is_file() or "final" in fp.stem:
            continue
        ep = int(fp.stem)
        print(ep)
        perf = np.loadtxt(fp / "ep" / "objs.txt", delimiter=",")
        print(perf.shape, perf[:, 0].shape)
        for i in range(perf.shape[0]):
            objs.append({"ep": ep, "obj_1": perf[i, 0], "obj_2": perf[i, 1]})

    return pd.DataFrame(objs)

def plot_pf(data):
    fig = px.scatter(data, x="obj_1", y="obj_2", color="ep", hover_data=["ep"])
    fig.show()



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    df = read_data(args)
    print(df)
    plot_pf(df)
