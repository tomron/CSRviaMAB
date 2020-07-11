"""Run multiple  simulations"""

import sys
import argparse

from itertools import product

import BlackBoxAlgorithm

def parse_arguments(args):
    """
    Parse program arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir', type=str, default='.', help='output directory')
    return parser.parse_args(args)


def main(args):
    opt = parse_arguments(args)
    lambdas = [0, 0.4, 0.8]
    arms = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
    horizon = [1000, 5000, 10000, 50000]#, 100000, 200000]
    repeat = 100
    functions = ['lin', 'const', 'softmax']
    for lam, horizon, func_name in product(lambdas, horizon, functions):
        args = [
            "--output_dir", opt.output_dir,
            "--fairness_function", func_name,
            "--lambda", str(lam),
            "--repeat", str(repeat),
            "--T", str(horizon),
            "--output_dir", opt.output_dir,
            "--arms"]
        args.extend([str(a) for a in arms])
        BlackBoxAlgorithm.main(args)


if __name__ == "__main__":
    main(sys.argv[1:])
