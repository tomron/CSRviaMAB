"""
Analyze model results
"""

import os
import sys
import json
import argparse

import pandas as pd
import numpy as np

FORMAT_F = {
    "MaxReward_alpha=1": "MaxReward1",
    "CartesianEqualPortion_p=1": "CartesianEqualPortion1",
    "CartesianMuPortion_alpha=1": "CartesianMuPortion1",
    "TempSoftmax_temp=1_p=1": "Softmax"
}


def parse_arguments(args):
    """
    Parse program arguments
    """
    parser = argparse.ArgumentParser(description='Analyze resluts')
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Input directory (simulation results output directory)',
        default='.')
    parser.add_argument(
        '--output_dir',
        type=str, help='Output directory', default='.')

    return parser.parse_args(args)


def read_results(raw_data_dir):
    """Reading simluation result files and returning dataframe
    """
    records = []
    for file_name in os.listdir(raw_data_dir):
        file_path = os.path.join(raw_data_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        with open(file_path) as open_file:
            for line in open_file:
                record = json.loads(line)
                records.append(record)

    data = pd.DataFrame(records)
    data['rel_utility'] = data['utility'] / data['T']
    return data


def get_log_expression(num):
    """Format a number as exponent

    Arguments:
        num {[int]} -- [number]

    Returns:
        [string] -- [number in format z \\cdot 10^y]
    """
    power = int(np.log10(num))
    multiplier = int(num/np.power(10, power))
    if multiplier != 1:
        expression = f"{multiplier}\\cdot 10^{power}"
    else:
        expression = f"10^{power}"
    return f"${expression}$"


def main(args):
    """ Read simulation results and write summary files
    """
    opt = parse_arguments(args)
    data = read_results(os.path.join(opt.input_dir, 'raw'))
    data.f = data.f.apply(lambda x: FORMAT_F[x])
    data.to_csv(
        os.path.join(
            opt.output_dir, "thin_results.csv"), index_label='index')

    # Create round dist files
    for lam in data['lambda'].unique():
        for func_name in data['f'].unique():
            tmp_df = data[(data['f'] == func_name) & (data['lambda'] == lam)]
            tmp_df = tmp_df[
                ['T', 'delta_approx', 'f_approx',
                 'opportunities_rounds', 'alg_rounds']]
            tmp_df = tmp_df.groupby('T').mean().reset_index()
            tmp_df['norm_delta_approx'] = tmp_df['delta_approx']/tmp_df['T']
            tmp_df['norm_f_approx'] = tmp_df['f_approx']/tmp_df['T']
            tmp_df['norm_approx'] = \
                tmp_df['norm_delta_approx'] + tmp_df['norm_f_approx']
            tmp_df['norm_opportunities_rounds'] = \
                tmp_df['opportunities_rounds'] / tmp_df['T']
            tmp_df['norm_alg_rounds'] = tmp_df['alg_rounds']/tmp_df['T']
            tmp_df = tmp_df[tmp_df['T'] >= 10000]
            tmp_df['T'] = tmp_df['T'].apply(get_log_expression)
            tmp_df.to_csv(
                os.path.join(
                    opt.output_dir, f"{func_name}_{lam}_pulls_dist.csv"), index=False)

    # Create lambda files
    for lam in data['lambda'].unique():
        tmp_df = data[data['lambda'] == lam]
        tmp_df = tmp_df[['f', 'rel_utility', 'T']]
        agg_df = tmp_df.pivot_table(
            index='T', values='rel_utility',
            columns='f', aggfunc=['mean', 'std'])
        agg_df.columns = [
            f"{y}_{x}" for x, y in
            zip(agg_df.columns.get_level_values(0),
                agg_df.columns.get_level_values(1))]
        agg_df.to_csv(os.path.join(opt.output_dir, f"lambda_{lam}.csv"))


if __name__ == "__main__":
    main(sys.argv[1:])
