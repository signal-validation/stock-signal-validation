"""
Test #02: Monte Carlo Timing
==============================
Tests whether the engine's signal timing contributes genuine edge,
or if the same win rate could be achieved by random entry dates.

Null hypothesis:
  "The engine's win rate could be achieved by entering at random dates
   rather than the specific dates the engine chooses, given the same
   stocks and number of signals per stock."

Method:
  1. REAL: Compute overall WR on de-duplicated signals.
  2. For each stock, we have N_k de-duplicated signals. We also know
     the stock's full set of trading days from the backtest.
  3. SHUFFLE (10,000 iterations): For each stock, randomly pick N_k
     trading days from that stock's available date range. Look up
     the 20-day forward return for each random entry. Compute WR.
  4. p-value: fraction of shuffled WRs >= real WR.

This tests the TIMING component of the engine — does it matter WHEN
within a stock's history the engine fires?

Usage:
  python3 test02_mc_timing.py
"""

import json
import math
import os
import random
import time
from collections import defaultdict
from datetime import date as Date

# ============================================================
# CONFIG
# ============================================================
SEED = 42
N_ITERATIONS = 10000
DEDUP_DAYS = 28
RETURN_COL = 'return_20d'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'test02_mc_timing_results.json')


# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    with open(os.path.join(DATA_DIR, 'signals_public.json')) as f:
        d = json.load(f)
    sigs = d['daily']['signals']
    tickers = set(s['ticker'] for s in sigs)
    return sigs, tickers, set()


def parse_date(s):
    return Date(int(s[:4]), int(s[5:7]), int(s[8:10]))


# ============================================================
# BUILD PER-STOCK DATE->RETURN MAPS + DE-DUPLICATED SIGNAL COUNTS
# ============================================================
def prepare_stock_data(signals, ticker_set):
    """
    For each stock, build:
    1. Full map of date -> return_20d (all signals, including non-deduped)
    2. Count of de-duplicated signals (the N_k to sample)
    3. De-duplicated WR
    """
    # All signals by stock
    by_ticker = defaultdict(list)
    for s in signals:
        if s['ticker'] not in ticker_set:
            continue
        if s[RETURN_COL] is None:
            continue
        by_ticker[s['ticker']].append({
            'date_obj': parse_date(s['date']),
            'date': s['date'],
            'won': 1 if s[RETURN_COL] > 0 else 0,
            'return': s[RETURN_COL],
        })

    stock_data = {}
    total_deduped = 0
    total_deduped_wins = 0

    for tk, sigs in by_ticker.items():
        sigs.sort(key=lambda x: x['date_obj'])

        # Build date -> outcome map (ALL signals for this stock)
        date_map = {}
        for s in sigs:
            date_map[s['date']] = s['won']

        # De-duplicate to get target count
        deduped_dates = []
        last_taken = None
        for s in sigs:
            if last_taken is None or (s['date_obj'] - last_taken).days > DEDUP_DAYS:
                deduped_dates.append(s['date'])
                total_deduped += 1
                total_deduped_wins += s['won']
                last_taken = s['date_obj']

        stock_data[tk] = {
            'all_dates': list(date_map.keys()),
            'date_outcomes': date_map,
            'n_deduped': len(deduped_dates),
            'deduped_wins': sum(date_map[d] for d in deduped_dates),
        }

    real_wr = total_deduped_wins / total_deduped * 100 if total_deduped > 0 else 0
    return stock_data, total_deduped, total_deduped_wins, real_wr


# ============================================================
# MONTE CARLO
# ============================================================
def run_mc(stock_data, total_deduped, total_deduped_wins, real_wr, label, n_iter=N_ITERATIONS):
    """Run Monte Carlo timing test."""
    print(f'\n{"="*70}')
    print(f'  {label}')
    print(f'{"="*70}')

    n_stocks = len(stock_data)
    total_all_dates = sum(len(sd['all_dates']) for sd in stock_data.values())

    print(f'  Stocks: {n_stocks}')
    print(f'  De-duplicated signals: {total_deduped}')
    print(f'  Total available signal dates: {total_all_dates}')
    print(f'  Real WR: {real_wr:.1f}%')

    # Pre-compute for speed: for each stock, the list of outcomes (ordered by date)
    stock_outcomes_list = {}
    for tk, sd in stock_data.items():
        outcomes = [sd['date_outcomes'][d] for d in sd['all_dates']]
        stock_outcomes_list[tk] = outcomes

    rng = random.Random(SEED)
    mc_wrs = []
    mc_wins = []

    t0 = time.time()
    for i in range(n_iter):
        total_wins = 0
        total_n = 0

        for tk, sd in stock_data.items():
            n_to_sample = sd['n_deduped']
            outcomes = stock_outcomes_list[tk]
            n_available = len(outcomes)

            if n_to_sample >= n_available:
                # Take all
                total_wins += sum(outcomes)
                total_n += n_available
            else:
                # Random sample without replacement
                sampled = rng.sample(outcomes, n_to_sample)
                total_wins += sum(sampled)
                total_n += n_to_sample

        mc_wr = total_wins / total_n * 100 if total_n > 0 else 0
        mc_wrs.append(mc_wr)
        mc_wins.append(total_wins)

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f'    [{i+1}/{n_iter}] {elapsed:.0f}s elapsed')

    elapsed_total = time.time() - t0

    # Analysis
    mc_wrs.sort()
    mean_mc = sum(mc_wrs) / n_iter
    std_mc = math.sqrt(sum((w - mean_mc)**2 for w in mc_wrs) / n_iter)

    # p-value: fraction of random timing WRs >= real WR
    p_value = sum(1 for w in mc_wrs if w >= real_wr) / n_iter

    # Percentile of real WR in MC distribution
    percentile = sum(1 for w in mc_wrs if w < real_wr) / n_iter * 100

    # CI
    ci_lo = mc_wrs[int(n_iter * 0.025)]
    ci_hi = mc_wrs[int(n_iter * 0.975)]

    # Z-score
    z = (real_wr - mean_mc) / std_mc if std_mc > 0 else 0

    # Timing alpha = real WR - mean random WR
    timing_alpha = real_wr - mean_mc

    print(f'\n  MONTE CARLO RESULTS ({n_iter} iterations, {elapsed_total:.0f}s):')
    print(f'    Real WR:           {real_wr:.2f}%')
    print(f'    Random timing WR:  {mean_mc:.2f}% +/- {std_mc:.2f}%')
    print(f'    95% CI:            [{ci_lo:.2f}%, {ci_hi:.2f}%]')
    print(f'    Timing alpha:      {timing_alpha:+.2f}pp')
    print(f'    z-score:           {z:.2f}')
    print(f'    Percentile:        {percentile:.1f}th')
    print(f'    p-value:           {p_value:.4f}')

    if p_value < 0.01:
        verdict = 'STRONG PASS — engine timing adds genuine edge over random entry'
    elif p_value < 0.05:
        verdict = 'PASS — engine timing likely contributes genuine edge (p < 0.05)'
    elif p_value < 0.10:
        verdict = 'MARGINAL — some timing edge, but not conclusive'
    else:
        verdict = 'FAIL — engine timing does not beat random entry dates for the same stocks'

    print(f'\n  VERDICT: {verdict}')
    print(f'\n  INTERPRETATION:')
    if p_value < 0.05:
        print(f'    The engine achieves {timing_alpha:+.1f}pp above what random timing')
        print(f'    within the same stocks would produce. This means the WHEN matters,')
        print(f'    not just the WHICH stocks.')
    else:
        print(f'    The engine\'s WR ({real_wr:.1f}%) is within the range achievable by')
        print(f'    random entry dates ({ci_lo:.1f}%-{ci_hi:.1f}%). The edge may come')
        print(f'    primarily from stock selection, not timing.')

    # Per-stock timing analysis: which stocks benefit most from timing?
    stock_timing_alpha = {}
    for tk, sd in stock_data.items():
        if sd['n_deduped'] < 5:
            continue
        stock_base_wr = sum(stock_outcomes_list[tk]) / len(stock_outcomes_list[tk]) * 100
        stock_dedup_wr = sd['deduped_wins'] / sd['n_deduped'] * 100
        stock_timing_alpha[tk] = {
            'timing_alpha': round(stock_dedup_wr - stock_base_wr, 1),
            'dedup_wr': round(stock_dedup_wr, 1),
            'base_wr': round(stock_base_wr, 1),
            'n_deduped': sd['n_deduped'],
            'n_all': len(sd['all_dates']),
        }

    # Sort by timing alpha
    top_timing = sorted(stock_timing_alpha.items(), key=lambda x: -x[1]['timing_alpha'])[:10]
    worst_timing = sorted(stock_timing_alpha.items(), key=lambda x: x[1]['timing_alpha'])[:10]

    print(f'\n  TOP 10 STOCKS BY TIMING ALPHA:')
    print(f'    {"Ticker":<8} {"Dedup WR":>9} {"Base WR":>8} {"Alpha":>7} {"N_dd":>5}')
    for tk, d in top_timing:
        print(f'    {tk:<8} {d["dedup_wr"]:>8.1f}% {d["base_wr"]:>7.1f}% {d["timing_alpha"]:>+6.1f}pp {d["n_deduped"]:>5}')

    print(f'\n  WORST 10 STOCKS BY TIMING ALPHA:')
    print(f'    {"Ticker":<8} {"Dedup WR":>9} {"Base WR":>8} {"Alpha":>7} {"N_dd":>5}')
    for tk, d in worst_timing:
        print(f'    {tk:<8} {d["dedup_wr"]:>8.1f}% {d["base_wr"]:>7.1f}% {d["timing_alpha"]:>+6.1f}pp {d["n_deduped"]:>5}')

    return {
        'label': label,
        'n_stocks': n_stocks,
        'n_deduped': total_deduped,
        'real_wr': round(real_wr, 2),
        'monte_carlo': {
            'n_iterations': n_iter,
            'seed': SEED,
            'mean_wr': round(mean_mc, 2),
            'std_wr': round(std_mc, 2),
            'ci_95': [round(ci_lo, 2), round(ci_hi, 2)],
            'timing_alpha': round(timing_alpha, 2),
            'z_score': round(z, 2),
            'percentile': round(percentile, 1),
            'p_value': round(p_value, 4),
            'elapsed_seconds': round(elapsed_total, 1),
        },
        'verdict': verdict,
        'top_10_timing_alpha': [{'ticker': tk, **d} for tk, d in top_timing],
        'worst_10_timing_alpha': [{'ticker': tk, **d} for tk, d in worst_timing],
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print('Test #02: Monte Carlo Timing')
    print('=' * 70)
    print('Loading data...')

    signals, ins_set, exp_set = load_data()
    print(f'Total signals: {len(signals)}')

    results = {
        'test': 'Test #02: Monte Carlo Timing',
        'description': (
            'Tests whether signal timing matters by comparing the engine\'s WR '
            'against randomly sampling the same number of entry dates per stock '
            'from the stock\'s full signal history.'
        ),
        'methodology': {
            'null_hypothesis': 'Random entry dates within each stock achieve the same WR',
            'shuffle': 'For each stock, sample N_k dates (without replacement) from all available signal dates',
            'n_iterations': N_ITERATIONS,
            'seed': SEED,
            'dedup_days': DEDUP_DAYS,
            'return_column': RETURN_COL,
        },
    }

    # Primary: In-sample
    ins_data, ins_n, ins_wins, ins_wr = prepare_stock_data(signals, ins_set)
    results['in_sample'] = run_mc(ins_data, ins_n, ins_wins, ins_wr, 'IN-SAMPLE (primary)')

    # Secondary: Expansion
    exp_data, exp_n, exp_wins, exp_wr = prepare_stock_data(signals, exp_set)
    # Expansion omitted (public data is in-sample only)
    # results.get('expansion', {}) = run_mc(exp_data, exp_n, exp_wins, exp_wr, 'EXPANSION')

    # Write results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults written to {RESULTS_FILE}')


if __name__ == '__main__':
    main()
