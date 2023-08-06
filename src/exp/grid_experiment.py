import sys
import time
from typing import Iterator, List, Optional, Dict
from collections import defaultdict

import json
import itertools
import subprocess
import multiprocessing
import argparse


def expand_hparams_grid(hparams_grid: Dict[str, List]) -> List[Dict]:
    """
    Returns a list of dicts that are all the possible value combinations for hyper-parameters.
    :param hparams_grid:
    :return:
    """
    # itertools.product([[a, b], [c, d], [g, h]) = [a, c, g], [a, c, h], [a, d, g], [a, d, h],
    # [b, c, g], [b, c, h], [b, d, g], [b, d, h]
    print([dict(zip(hparams_grid.keys(), values)) for values in itertools.product(*hparams_grid.values())])
    return [dict(zip(hparams_grid.keys(), values)) for values in itertools.product(*hparams_grid.values())]


def build_command_string(script_path: str, dataset: str, args: dict) -> str:
    cmd_str = f'python {script_path}'
    cmd_str += f' --dataset {dataset}'
    for field, value in args.items():

        if isinstance(value, bool):
            if value:
                cmd_str += f' --{field.replace("_", "-")}'
        else:
            cmd_str += f' --{field.replace("_", "-")} {value}'
    return cmd_str


device_ids_cycle_g: Optional[Iterator[int]] = None


def device_next_id() -> int:
    return next(device_ids_cycle_g)


parser = argparse.ArgumentParser(
    description="Grid Search Script"
)
parser.add_argument(
    '--config', help="Experiments grid search configuration file"
)

parser.add_argument(
    '--gpu', default=None, help="Device where to run the experiments"
)

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'r') as fp:
        config = json.load(fp)

    datasets = config["datasets"]
    script_path = config["script_path"]
    hparams_grid = config['hparams_grid']
    hparams = expand_hparams_grid(hparams_grid)  # Get the list of hyperparameters
    model_dir = config["model_dir"]
    tensorboard_dir = config["tensorboard_dir"]

    # Produce the list of commands
    commands: Dict[str, List] = {}
    num_runs_on_dataset = len(hparams)

    for dataset in config['datasets']:
        commands[dataset] = list()
        # Get the hyperparameters grid, based on the dataset
        hparams_grid_datasets = config['hparams_grid'].keys()

        for hps in hparams:
            cmd = build_command_string(script_path, dataset, hps)
            cmd += f' --model-dir={model_dir}'
            cmd += f' --tensorboard-dir={tensorboard_dir}'
            if args.gpu is not None:
                cmd += f" --gpu {args.gpu}"
            commands[dataset].append(cmd)

            # subprocess.run(cmd)
    with multiprocessing.Pool(processes=1) as pool:
        for i in range(num_runs_on_dataset):
            for dataset in datasets:
                print(commands[dataset][i])
                pool.apply_async(subprocess.run, args=[commands[dataset][i].split()],
                                 callback=lambda x: print(x), error_callback=lambda x: print(x))  # stdout=subprocess.DEVNULL
        pool.close()
        pool.join()
