import argparse
from datetime import datetime
import json
import sys
import os

import numpy as np

from BernoulliArm import BernoulliArm
from RoMab import RoMab
from SimulationRun import SimulationRun
import FairnessFunctions
from utils import get_confidence_interval


def parse_arguments(args):
    """
    Parse program arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lambda', type=float, dest='lam',
        default=0.1, help='penalty term')
    parser.add_argument(
        '--repeat', type=int, default=10,
        help='how many times to repeat the experiment')
    parser.add_argument(
        '--T', type=int, default=1000, help='horizon')
    parser.add_argument(
        '--arms', type=float, nargs='+',
        help='Arms probabilities', required=True)
    parser.add_argument(
        '--output_dir', type=str, default='.', help='output directory')
    parser.add_argument(
        '--fairness_function', type=str, default='const',
        help='fairness function', choices=['lin', 'const', 'softmax'])
    parser.add_argument(
        '--fairness_portion', type=float, default=1,
        help='multiplier of the fairness function')
    return parser.parse_args(args)


def ucb1(simulation_run):
    ucb_vals_idx = zip(range(simulation_run.K), simulation_run.ucb_vals)
    sorted_ucb_vals = sorted(ucb_vals_idx, key=lambda x: -x[1])
    return sorted_ucb_vals[0][0]


def get_fairness_function(opt, K):
    if opt.fairness_function == 'lin':
        return FairnessFunctions.CartesianMuPortion(opt.fairness_portion, K)
    if opt.fairness_function == 'const':
        return FairnessFunctions.CartesianEqualPortion(opt.fairness_portion, K)
    if opt.fairness_function == 'softmax':
        return FairnessFunctions.TempSoftmax(
            portion=1, temp=opt.fairness_portion)
    return None


def get_deltas_lcb_ucb(simulation_run, T):
    expected_rewards = simulation_run.expected_rewards
    max_expected_reward = max(expected_rewards)
    max_idx = expected_rewards.index(max_expected_reward)
    confidence_max = get_confidence_interval(
        simulation_run.counters[max_idx], T)
    deltas_lcb, deltas_ucb = [], []
    for idx in range(simulation_run.K):
        if (simulation_run.counters[idx] == 0 or
                simulation_run.counters[max_idx] == 0):
            deltas_lcb.append(0)
            deltas_ucb.append(1)
        else:
            delta = max_expected_reward - expected_rewards[idx]
            confidence_arm = get_confidence_interval(
                simulation_run.counters[idx], T)
            deltas_lcb.append(max(0, delta - confidence_arm - confidence_max))
            deltas_ucb.append(min(1, delta + confidence_arm + confidence_max))
    return deltas_lcb, deltas_ucb


def pull_all_arms_once(simulation_run, romab):
    for idx in range(0, simulation_run.K):
        r_t = romab.arms[idx].draw()
        simulation_run.update_arm(i=idx, reward=r_t)
        if simulation_run.t >= romab.T:
            return


def approx_f(simulation_run, romab, alpha):
    finished = True
    deltas_lcb, _ = get_deltas_lcb_ucb(simulation_run, romab.T)

    for idx in range(0, simulation_run.K):
        if deltas_lcb[idx] >= romab.lam:
            continue
        max_diff, _ = romab.f.get_max_diff_arm(
            simulation_run.expected_rewards,
            simulation_run.counters, T=romab.T, i=idx)
        if max_diff < alpha:
            continue
        finished = False
        pull_all_arms_once(simulation_run, romab)
        break
    return finished


def approx_delta(simulation_run, romab, beta):
    finished = True
    if romab.lam == 0:
        return True
    deltas_lcb, deltas_ucb = get_deltas_lcb_ucb(simulation_run, romab.T)
    upper_bound = min(1, romab.lam+beta)
    lower_bound = max(0, romab.lam-beta)
    for _, (lcb, ucb) in enumerate(zip(deltas_lcb, deltas_ucb)):
        if lcb <= lower_bound and ucb >= upper_bound:
            finished = False
            pull_all_arms_once(simulation_run, romab)
            break
    return finished


def runSimulation(romab, simulation_run, alg):
    """Run a single simulation
    """
    T = romab.T
    alpha = np.power(T, -1.0/3) * \
        np.power(np.log(T), 1.0/3) * \
        np.power(romab.get_l(), 2.0/3) *\
        simulation_run.K
    beta = np.power(T, -1.0/3) * np.power(np.log(T), 1.0/3)

    finished = False
    while not finished:
        finished = approx_delta(simulation_run, romab, beta)
        if simulation_run.t >= T:
            break

    delta_approx = simulation_run.t
    simulation_run.set_delta_approx(simulation_run.t)

    finished = False
    while not finished:
        finished = approx_f(simulation_run, romab, alpha)
        if simulation_run.t >= T:
            break

    f_approx = simulation_run.t
    simulation_run.set_f_approx(simulation_run.t - delta_approx)

    deltas_lcb, _ = get_deltas_lcb_ucb(simulation_run, T)
    f1_results = romab.f.f1(simulation_run.expected_rewards, t=T)
    for idx in range(0, simulation_run.K):
        if deltas_lcb[idx] < romab.lam:
            # floor
            diff = int(f1_results[idx] - simulation_run.counters[idx])
            for _ in range(diff):
                r_t = romab.arms[idx].draw()
                simulation_run.update_arm(i=idx, reward=r_t)
                if simulation_run.t >= T:
                    break
            if simulation_run.t >= T:
                break

    opportunities = simulation_run.t
    simulation_run.set_opportunities_rounds(simulation_run.t - f_approx)

    while simulation_run.t < T:
        i_t = alg(simulation_run)
        r_t = romab.arms[i_t].draw()
        simulation_run.update_arm(i=i_t, reward=r_t)

    simulation_run.set_alg_rounds(simulation_run.t-opportunities)

    f1_results = romab.f.f1(simulation_run.expected_rewards, t=T)
    penalty = 0
    for true_portion, actual in zip(f1_results, simulation_run.counters):
        penalty += romab.lam * max(0, true_portion-actual)
    utility = sum(simulation_run.total_rewards) - penalty
    simulation_run.set_utility(utility)


def main(args):
    opt = parse_arguments(args)
    output_file = open(
        os.path.join(opt.output_dir, "raw",
                     f"results_{datetime.now().isoformat()}.txt"), "w")
    arms_probs = opt.arms
    arms = [BernoulliArm(prob) for prob in arms_probs]
    K = len(arms)
    f1 = get_fairness_function(opt, K)

    for repeat_idx in range(opt.repeat):
        if repeat_idx % 50 == 0:
            print(f"{datetime.now()} \
                    Repateation [{repeat_idx}/{opt.repeat}] for \
                        {opt.T}, {opt.lam}, {f1}")
            output_file.flush()

        romab = RoMab(arms, T=opt.T, f=f1, lam=opt.lam)
        simulation_run = SimulationRun(K, opt.T, sequence=False)
        runSimulation(romab, simulation_run, ucb1)

        output_file.write((json.dumps({
            "arms": arms_probs, "lambda": opt.lam, "T": opt.T,
            "counter": simulation_run.counters,
            "total_reward": simulation_run.get_total_reward(),
            "observed_arms": simulation_run.expected_rewards,
            "K": len(arms_probs), "f": str(romab.f),
            "last_pulls": simulation_run.last_pulls,
            "f_approx": simulation_run.f_approx,
            "delta_approx": simulation_run.delta_approx,
            "alg_rounds": simulation_run.alg_rounds,
            "opportunities_rounds": simulation_run.opportunities_rounds,
            "utility": simulation_run.utility})) + "\n")
    output_file.close()


if __name__ == "__main__":
    main(sys.argv[1:])
