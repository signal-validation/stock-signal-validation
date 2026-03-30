"""
Test #03: Block Bootstrap Confidence Intervals
================================================
Constructs confidence intervals for the de-duplicated in-sample WR claim
(60.3%) using block bootstrap, which respects temporal structure.

Why block bootstrap instead of standard bootstrap?
  Even after 28-day de-duplication, signals from different stocks on the
  same dates share market conditions. A standard bootstrap would
  underestimate variance by breaking this cross-sectional dependence.
  Block bootstrap resamples contiguous TIME BLOCKS, preserving both
  within-stock and cross-stock temporal structure.

Method:
  1. Partition all de-duplicated signals into non-overlapping time blocks
     of B trading days (default: 63 trading days ≈ 3 months).
  2. Resample blocks WITH replacement to form bootstrap samples of the
     same total size.
  3. Compute WR for each bootstrap sample.
  4. Report bootstrap 95% CI, standard error, and bias.
  5. Also compute CI at 90% and 99% levels.
  6. Test multiple block sizes for robustness (21, 42, 63, 126, 252 days).

Output:
  - Console summary with CIs
  - test03_block_bootstrap_results.json

Usage:
  python3 test03_block_bootstrap.py
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
N_BOOTSTRAP = 10000
DEDUP_DAYS = 28
RETURN_COL = 'return_20d'
PRIMARY_BLOCK_SIZE = 63  # ~3 months of trading days
BLOCK_SIZES = [21, 42, 63, 126, 252]  # 1m, 2m, 3m, 6m, 1y

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'test03_block_bootstrap_results.json')


# ============================================================
# DATA LOADING & DE-DUPLICATION
# ============================================================
def load_data():
    with open(os.path.join(DATA_DIR, 'signals_public.json')) as f:
        d = json.load(f)
    sigs = d['daily']['signals']
    tickers = set(s['ticker'] for s in sigs)
    return sigs, tickers, set()


def parse_date(s):
    return Date(int(s[:4]), int(s[5:7]), int(s[8:10]))


def dedup_signals(signals, ticker_set):
    """De-duplicate: keep first per stock per DEDUP_DAYS."""
    by_ticker = defaultdict(list)
    for s in signals:
        if s['ticker'] not in ticker_set:
            continue
        if s[RETURN_COL] is None:
            continue
        by_ticker[s['ticker']].append({
            'date': s['date'],
            'date_obj': parse_date(s['date']),
            'ticker': s['ticker'],
            'won': 1 if s[RETURN_COL] > 0 else 0,
            'return': s[RETURN_COL],
        })

    deduped = []
    for tk, sigs in by_ticker.items():
        sigs.sort(key=lambda x: x['date_obj'])
        last_taken = None
        for s in sigs:
            if last_taken is None or (s['date_obj'] - last_taken).days > DEDUP_DAYS:
                deduped.append(s)
                last_taken = s['date_obj']

    deduped.sort(key=lambda x: x['date'])
    return deduped


# ============================================================
# BLOCK PARTITIONING
# ============================================================
def partition_into_blocks(deduped_signals, block_size):
    """
    Partition signals into non-overlapping time blocks.

    We assign each signal to a block based on its chronological rank
    among all unique signal dates. Block i contains signals from
    dates ranked [i*block_size, (i+1)*block_size).
    """
    # Get all unique dates, sorted
    all_dates = sorted(set(s['date'] for s in deduped_signals))
    date_to_block = {}
    for i, d in enumerate(all_dates):
        date_to_block[d] = i // block_size

    # Group signals by block
    blocks = defaultdict(list)
    for s in deduped_signals:
        block_id = date_to_block[s['date']]
        blocks[block_id].append(s)

    # Convert to list of lists, sorted by block_id
    block_list = [blocks[k] for k in sorted(blocks.keys())]
    return block_list


# ============================================================
# BLOCK BOOTSTRAP
# ============================================================
def run_bootstrap(deduped_signals, label, block_size=PRIMARY_BLOCK_SIZE, n_boot=N_BOOTSTRAP):
    """Run block bootstrap for a given block size."""
    blocks = partition_into_blocks(deduped_signals, block_size)
    n_blocks = len(blocks)
    total_signals = len(deduped_signals)
    total_wins = sum(s['won'] for s in deduped_signals)
    real_wr = total_wins / total_signals * 100

    # Block sizes
    block_sizes_actual = [len(b) for b in blocks]
    mean_block_size = sum(block_sizes_actual) / len(block_sizes_actual)

    print(f'\n    Block size: {block_size} dates -> {n_blocks} blocks')
    print(f'    Mean signals per block: {mean_block_size:.1f}')
    print(f'    Min/Max block size: {min(block_sizes_actual)}/{max(block_sizes_actual)}')

    rng = random.Random(SEED)
    boot_wrs = []

    t0 = time.time()
    for i in range(n_boot):
        # Resample blocks with replacement
        sampled_blocks = rng.choices(blocks, k=n_blocks)

        # Compute WR on resampled signals
        total_w = 0
        total_n = 0
        for block in sampled_blocks:
            total_w += sum(s['won'] for s in block)
            total_n += len(block)

        boot_wr = total_w / total_n * 100 if total_n > 0 else 0
        boot_wrs.append(boot_wr)

    elapsed = time.time() - t0
    boot_wrs.sort()

    # Statistics
    mean_boot = sum(boot_wrs) / n_boot
    std_boot = math.sqrt(sum((w - mean_boot)**2 for w in boot_wrs) / n_boot)
    bias = mean_boot - real_wr

    # Confidence intervals (percentile method)
    ci_90 = (boot_wrs[int(n_boot * 0.05)], boot_wrs[int(n_boot * 0.95)])
    ci_95 = (boot_wrs[int(n_boot * 0.025)], boot_wrs[int(n_boot * 0.975)])
    ci_99 = (boot_wrs[int(n_boot * 0.005)], boot_wrs[int(n_boot * 0.995)])

    # BCa (bias-corrected and accelerated) — simplified: bias-corrected percentile
    # z0 = proportion of boot WRs < real WR → normal quantile
    frac_below = sum(1 for w in boot_wrs if w < real_wr) / n_boot
    if 0 < frac_below < 1:
        z0 = -math.sqrt(2) * math.erfc(2 * frac_below) / 2  # Approx
        # For simplicity, just use percentile method (BCa needs jackknife)
    else:
        z0 = 0

    # Does 50% fall within CI?
    excludes_50 = ci_95[0] > 50.0

    print(f'    Bootstrap SE: {std_boot:.2f}pp')
    print(f'    Bias: {bias:+.2f}pp')
    print(f'    90% CI: [{ci_90[0]:.2f}%, {ci_90[1]:.2f}%]')
    print(f'    95% CI: [{ci_95[0]:.2f}%, {ci_95[1]:.2f}%]')
    print(f'    99% CI: [{ci_99[0]:.2f}%, {ci_99[1]:.2f}%]')
    print(f'    95% CI excludes 50%: {"YES" if excludes_50 else "NO"}')
    print(f'    ({elapsed:.1f}s)')

    return {
        'block_size': block_size,
        'n_blocks': n_blocks,
        'mean_signals_per_block': round(mean_block_size, 1),
        'n_bootstrap': n_boot,
        'real_wr': round(real_wr, 2),
        'bootstrap_mean': round(mean_boot, 2),
        'bootstrap_se': round(std_boot, 2),
        'bias': round(bias, 2),
        'ci_90': [round(ci_90[0], 2), round(ci_90[1], 2)],
        'ci_95': [round(ci_95[0], 2), round(ci_95[1], 2)],
        'ci_99': [round(ci_99[0], 2), round(ci_99[1], 2)],
        'excludes_50_at_95': excludes_50,
        'elapsed_seconds': round(elapsed, 1),
    }


def run_full_analysis(deduped_signals, label):
    """Run block bootstrap with multiple block sizes."""
    print(f'\n{"="*70}')
    print(f'  {label}')
    print(f'{"="*70}')

    total_signals = len(deduped_signals)
    total_wins = sum(s['won'] for s in deduped_signals)
    real_wr = total_wins / total_signals * 100
    n_stocks = len(set(s['ticker'] for s in deduped_signals))

    # Date range
    dates = sorted(set(s['date'] for s in deduped_signals))
    print(f'  Signals: {total_signals}, Stocks: {n_stocks}')
    print(f'  WR: {real_wr:.1f}%')
    print(f'  Date range: {dates[0]} to {dates[-1]}')
    print(f'  Unique dates: {len(dates)}')

    results_by_block = {}
    for bs in BLOCK_SIZES:
        results_by_block[str(bs)] = run_bootstrap(deduped_signals, label, block_size=bs)

    # Primary result
    primary = results_by_block[str(PRIMARY_BLOCK_SIZE)]

    # Robustness: are CIs stable across block sizes?
    print(f'\n  BLOCK SIZE ROBUSTNESS:')
    print(f'    {"Block":>6} {"Blocks":>7} {"SE":>6} {"95% CI":>22} {"Excl 50%":>9}')
    print(f'    {"-"*55}')
    for bs in BLOCK_SIZES:
        r = results_by_block[str(bs)]
        ci_str = f'[{r["ci_95"][0]:.1f}%, {r["ci_95"][1]:.1f}%]'
        print(f'    {bs:>6} {r["n_blocks"]:>7} {r["bootstrap_se"]:>5.1f}pp {ci_str:>22} {"YES" if r["excludes_50_at_95"] else "NO":>9}')

    # Verdict
    all_exclude_50 = all(results_by_block[str(bs)]['excludes_50_at_95'] for bs in BLOCK_SIZES)
    if all_exclude_50:
        verdict = 'STRONG PASS — 95% CI excludes 50% at ALL block sizes'
    elif primary['excludes_50_at_95']:
        verdict = 'PASS — 95% CI excludes 50% at primary block size (63 days)'
    else:
        verdict = 'FAIL — 95% CI includes 50%'

    print(f'\n  VERDICT: {verdict}')

    # Normal approximation comparison
    se_normal = math.sqrt(real_wr/100 * (1 - real_wr/100) / total_signals) * 100
    print(f'\n  COMPARISON TO NAIVE SE:')
    print(f'    Naive SE (iid assumption): {se_normal:.2f}pp')
    print(f'    Block bootstrap SE (63d):  {primary["bootstrap_se"]:.2f}pp')
    print(f'    Ratio: {primary["bootstrap_se"]/se_normal:.1f}x')
    print(f'    (>1x means naive underestimates uncertainty)')

    return {
        'label': label,
        'n_signals': total_signals,
        'n_stocks': n_stocks,
        'real_wr': round(real_wr, 2),
        'date_range': [dates[0], dates[-1]],
        'unique_dates': len(dates),
        'primary_block_size': PRIMARY_BLOCK_SIZE,
        'primary_result': primary,
        'robustness': results_by_block,
        'all_exclude_50': all_exclude_50,
        'naive_se': round(se_normal, 2),
        'se_inflation_ratio': round(primary['bootstrap_se'] / se_normal, 2) if se_normal > 0 else None,
        'verdict': verdict,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print('Test #03: Block Bootstrap Confidence Intervals')
    print('=' * 70)
    print('Loading data...')

    signals, ins_set, exp_set = load_data()
    print(f'Total signals: {len(signals)}')

    ins_deduped = dedup_signals(signals, ins_set)
    exp_deduped = dedup_signals(signals, exp_set)

    print(f'De-duplicated: in-sample={len(ins_deduped)}, expansion={len(exp_deduped)}')

    results = {
        'test': 'Test #03: Block Bootstrap Confidence Intervals',
        'description': (
            'Constructs confidence intervals for de-duplicated win rates using '
            'block bootstrap, which preserves temporal structure (cross-sectional '
            'dependence from shared market conditions).'
        ),
        'methodology': {
            'block_sizes_tested': BLOCK_SIZES,
            'primary_block_size': PRIMARY_BLOCK_SIZE,
            'n_bootstrap': N_BOOTSTRAP,
            'seed': SEED,
            'dedup_days': DEDUP_DAYS,
            'return_column': RETURN_COL,
            'ci_method': 'percentile',
        },
    }

    # Primary: In-sample
    results['in_sample'] = run_full_analysis(ins_deduped, 'IN-SAMPLE (primary — 60.3% claim)')

    # Secondary: Expansion
    # Expansion omitted (public data is in-sample only)
    # results.get('expansion', {}) = run_full_analysis(exp_deduped, 'EXPANSION (55.4% claim)')

    # ============================================================
    # PUBLISHABLE SUMMARY
    # ============================================================
    ins = results['in_sample']
    ins_p = ins['primary_result']

    print(f'\n{"="*70}')
    print(f'  PUBLISHABLE SUMMARY')
    print(f'{"="*70}')
    print(f'')
    print(f'  In-sample (de-duplicated): {ins["real_wr"]:.1f}% WR')
    print(f'    Block bootstrap 95% CI: [{ins_p["ci_95"][0]:.1f}%, {ins_p["ci_95"][1]:.1f}%]')
    print(f'    Bootstrap SE: {ins_p["bootstrap_se"]:.1f}pp (vs naive {ins["naive_se"]:.1f}pp, {ins["se_inflation_ratio"]:.1f}x inflation)')
    print(f'    CI excludes 50% at all block sizes: {"YES" if ins["all_exclude_50"] else "NO"}')

    # Write results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results written to {RESULTS_FILE}')


if __name__ == '__main__':
    main()
