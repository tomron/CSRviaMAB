"""Create graphs"""

import argparse
import os
import sys
import pandas as pd


def parse_arguments(args):
    """
    Parse program arguments
    """
    parser = argparse.ArgumentParser(description='Generate graphs')
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Simulation input directory',
        default='.')
    parser.add_argument(
        '--output_dir',
        type=str,
        help='pdf files output directory',
        default='.')
    parser.add_argument(
        '--templates_dir',
        type=str,
        help='Tex files template directory',
        default='templates')

    return parser.parse_args(args)


def main(args):
    opt = parse_arguments(args)
    data = pd.read_csv(os.path.join(opt.input_dir, "thin_results.csv"))

    lambdas = data['lambda'].unique()
    functions = data['f'].unique()
    max_lambda = max(lambdas)

    for lam in lambdas:
        command = (
            f"pdflatex -output-directory {opt.output_dir} "
            f"-interaction batchmode "
            f"-jobname=\"lam_{lam}\" "
            f"\"\\def\\IncludeLegend{{{1 if lam==max_lambda else 0}}} "
            f"\\def\\resultsDir{{{opt.input_dir}}} "
            f"\\def\\lam{{{lam}}} "
            f"\\input{{{ os.path.join(opt.templates_dir,'lam_0.0')}}}\""
        )
        os.system(command)
    for func_name in functions:
        for lam in lambdas:
            command = (
                f"pdflatex -output-directory {opt.output_dir} "
                f"-interaction batchmode "
                f"-jobname=\"pull_dist_{lam}_{func_name}\" "
                f"\"\\def\\funcName{{{func_name}}} "
                f"\\def\\IncludeLegend{{{1 if lam==max_lambda else 0}}} "
                f"\\def\\resultsDir{{{opt.input_dir}}} "
                f"\\def\\lam{{{lam}}} "
                f"\\input{{{ os.path.join(opt.templates_dir,'pulls_dist')}}}\""
            )
            os.system(command)


if __name__ == "__main__":
    main(sys.argv[1:])
