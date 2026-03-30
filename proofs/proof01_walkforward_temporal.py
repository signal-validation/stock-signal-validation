"""
Proof #01: Walk-Forward Temporal Out-of-Sample
===============================================
The killer proof: train on 2006-2015, test on 2016-2025, SAME stocks.
If the engine works in the future half, it's not overfit.

Usage:
  python3 proof01_walkforward_temporal.py
"""

import json
import math
import os
import random
from collections import defaultdict
from datetime import date as Date

SEED = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'proof01_walkforward_temporal_results.json')
DEDUP_DAYS = 28
RETURN_COL = 'return_20d'
N_BOOTSTRAP = 10000
BLOCK_SIZE = 63


def load_data():
    with open(os.path.join(DATA_DIR, 'signals_public.json')) as f:
        d = json.load(f)
    sigs = d.get('daily', {}).get('signals', d.get('signals', []))
    return sigs, set(s['ticker'] for s in sigs)


def parse_date(s):
    return Date(int(s[:4]), int(s[5:7]), int(s[8:10]))


def dedup_signals(signals, ticker_set):
    by_ticker = defaultdict(list)
    for s in signals:
        if s['ticker'] not in ticker_set:
            continue
        if s[RETURN_COL] is None:
            continue
        by_ticker[s['ticker']].append(s)
    deduped = []
    for tk, sigs in by_ticker.items():
        sigs.sort(key=lambda x: x['date'])
        last_taken = None
        for s in sigs:
            d = parse_date(s['date'])
            if last_taken is None or (d - last_taken).days > DEDUP_DAYS:
                deduped.append(s)
                last_taken = d
    deduped.sort(key=lambda x: x['date'])
    return deduped


def compute_stats(signals_list, label=''):
    n = len(signals_list)
    if n == 0:
        return {'n': 0, 'wr': 0, 'alpha': 0, 'z': 0, 'p': 1.0, 'label': label}
    wins = sum(1 for s in signals_list if s[RETURN_COL] > 0)
    wr = wins / n * 100
    alphas = [s.get('alpha', 0) or 0 for s in signals_list]
    mean_alpha = sum(alphas) / len(alphas)
    avg_ret = sum(s[RETURN_COL] for s in signals_list) / n
    avg_spy = sum((s.get('spy_return_20d', 0) or 0) for s in signals_list) / n
    se = math.sqrt(0.5 * 0.5 / n) if n > 0 else 1
    z = (wr / 100 - 0.5) / se
    p = 0.5 * math.erfc(z / math.sqrt(2)) if z > 0 else 1.0
    return {
        'label': label, 'n': n, 'wins': wins, 'wr': round(wr, 1),
        'mean_alpha': round(mean_alpha, 2), 'mean_return': round(avg_ret, 2),
        'mean_spy_return': round(avg_spy, 2), 'z': round(z, 2), 'p': p,
    }


def block_bootstrap_ci(signals_list, n_boot=N_BOOTSTRAP, block_size=BLOCK_SIZE):
    if len(signals_list) < 30:
        return None
    all_dates = sorted(set(s['date'] for s in signals_list))
    date_to_block = {d: i // block_size for i, d in enumerate(all_dates)}
    blocks = defaultdict(list)
    for s in signals_list:
        blocks[date_to_block[s['date']]].append(s)
    block_list = [blocks[k] for k in sorted(blocks.keys())]
    if len(block_list) < 3:
        return None
    rng = random.Random(SEED)
    boot_wrs = []
    for _ in range(n_boot):
        sampled = rng.choices(block_list, k=len(block_list))
        w = sum(1 for b in sampled for s in b if s[RETURN_COL] > 0)
        t = sum(len(b) for b in sampled)
        if t > 0:
            boot_wrs.append(w / t * 100)
    boot_wrs.sort()
    n = len(boot_wrs)
    return {
        'ci_95': [round(boot_wrs[int(n * 0.025)], 1), round(boot_wrs[int(n * 0.975)], 1)],
        'se': round((sum((w - sum(boot_wrs)/n)**2 for w in boot_wrs) / n)**0.5, 2),
    }


def main():
    print('Proof #01: Walk-Forward Temporal Out-of-Sample')
    print('=' * 70)
    signals, ins_set = load_data()
    deduped = dedup_signals(signals, ins_set)
    print(f'De-duplicated in-sample signals: {len(deduped)}')

    results = {
        'analysis': 'Walk-Forward Temporal OOS',
        'description': 'Split by TIME (not stock): train on first half, test on second half, same 237 stocks.',
    }

    # === PRIMARY: 50/50 temporal split ===
    train = [s for s in deduped if int(s['date'][:4]) <= 2015]
    test = [s for s in deduped if int(s['date'][:4]) >= 2016]

    train_stats = compute_stats(train, 'Train 2006-2015')
    test_stats = compute_stats(test, 'Test 2016-2025')
    test_ci = block_bootstrap_ci(test)

    print(f'\n  PRIMARY SPLIT (50/50 temporal):')
    print(f'    {"Period":<18} {"N":>6} {"WR":>7} {"Alpha":>8} {"z":>7} {"p":>11}')
    print(f'    {"-"*60}')
    for s in [train_stats, test_stats]:
        print(f'    {s["label"]:<18} {s["n"]:>6} {s["wr"]:>6.1f}% {s["mean_alpha"]:>+7.2f}% {s["z"]:>7.2f} {s["p"]:>11.2e}')

    if test_ci:
        print(f'\n    Test period 95% CI: [{test_ci["ci_95"][0]}%, {test_ci["ci_95"][1]}%]')
        print(f'    CI excludes 50%: {"YES" if test_ci["ci_95"][0] > 50 else "NO"}')

    delta = test_stats['wr'] - train_stats['wr']
    print(f'\n    Train->Test WR change: {delta:+.1f}pp')
    if abs(delta) < 3:
        print(f'    -> STABLE across time periods')
    elif delta < -3:
        print(f'    -> DEGRADATION in test period')
    else:
        print(f'    -> IMPROVEMENT in test period')

    results['primary_split'] = {
        'train': train_stats, 'test': test_stats,
        'test_bootstrap_ci': test_ci, 'wr_delta': round(delta, 1),
    }

    # === 5-YEAR ROLLING WINDOWS ===
    windows_5yr = [
        ('2006-2010', 2006, 2010), ('2011-2015', 2011, 2015),
        ('2016-2020', 2016, 2020), ('2021-2025', 2021, 2025),
    ]
    print(f'\n  5-YEAR ROLLING WINDOWS:')
    print(f'    {"Window":<12} {"N":>6} {"WR":>7} {"Alpha":>8} {"z":>7} {"p":>11}')
    print(f'    {"-"*55}')
    window_results = []
    for label, y_start, y_end in windows_5yr:
        w_sigs = [s for s in deduped if y_start <= int(s['date'][:4]) <= y_end]
        w_stats = compute_stats(w_sigs, label)
        print(f'    {label:<12} {w_stats["n"]:>6} {w_stats["wr"]:>6.1f}% {w_stats["mean_alpha"]:>+7.2f}% {w_stats["z"]:>7.2f} {w_stats["p"]:>11.2e}')
        window_results.append(w_stats)

    results['rolling_5yr'] = window_results
    above_50 = sum(1 for w in window_results if w['wr'] > 50)
    print(f'\n    Windows above 50%: {above_50}/{len(window_results)}')

    # === 3 OVERLAPPING SPLITS ===
    splits_3 = [
        ('2006-2012', 2006, 2012), ('2013-2019', 2013, 2019), ('2020-2025', 2020, 2025),
    ]
    print(f'\n  3-WAY OVERLAPPING SPLITS:')
    print(f'    {"Window":<12} {"N":>6} {"WR":>7} {"Alpha":>8}')
    print(f'    {"-"*36}')
    split3_results = []
    for label, y_start, y_end in splits_3:
        w_sigs = [s for s in deduped if y_start <= int(s['date'][:4]) <= y_end]
        w_stats = compute_stats(w_sigs, label)
        print(f'    {label:<12} {w_stats["n"]:>6} {w_stats["wr"]:>6.1f}% {w_stats["mean_alpha"]:>+7.2f}%')
        split3_results.append(w_stats)
    results['splits_3way'] = split3_results

    # === VERDICT ===
    print(f'\n{"="*70}')
    print(f'  VERDICT')
    print(f'{"="*70}')
    if test_stats['wr'] > 55 and test_stats['p'] < 0.05:
        verdict = 'STRONG PASS -- test period (2016-2025) shows significant edge'
    elif test_stats['wr'] > 50 and test_stats['p'] < 0.10:
        verdict = 'PASS -- test period shows edge above 50%'
    else:
        verdict = 'FAIL -- edge does not hold in test period'
    print(f'  {verdict}')
    print(f'  Train (2006-2015): {train_stats["wr"]:.1f}% WR on {train_stats["n"]} signals')
    print(f'  Test  (2016-2025): {test_stats["wr"]:.1f}% WR on {test_stats["n"]} signals')
    if test_ci:
        print(f'  Test 95% CI: [{test_ci["ci_95"][0]}%, {test_ci["ci_95"][1]}%]')
    results['verdict'] = verdict

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  Results written to {RESULTS_FILE}')


if __name__ == '__main__':
    main()
