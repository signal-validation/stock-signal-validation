"""
Test #01: Monte Carlo Stock Selection
======================================
Tests whether the engine's stock-level win rates are genuine or could arise
from random assignment of outcomes to stocks.

Null hypothesis:
  "The engine's per-stock win rates are indistinguishable from randomly
   reassigning all win/loss outcomes across stocks (preserving overall WR
   and per-stock signal counts)."

Method:
  1. REAL: Compute WR on de-duplicated in-sample signals. Record overall WR
     and the cross-sectional standard deviation of per-stock WRs.
  2. SHUFFLE (10,000 iterations): Pool all de-duplicated outcomes, randomly
     reassign them to stocks (preserving each stock's signal count).
     Recompute overall WR (unchanged by construction) and cross-sectional
     std dev of per-stock WRs.
  3. Also test: top-quartile stock WR (do the best stocks beat random?),
     WR spread (max - min), and fraction of stocks with WR > 55%.

If p < 0.05: stock-level differentiation is real — some stocks are genuinely
better for this engine than others.

Usage:
  python3 test01_mc_stock_selection.py
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
MIN_SIGNALS_PER_STOCK = 3  # Only include stocks with enough signals

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'test01_mc_stock_selection_results.json')


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


def dedup_and_prepare(signals, ticker_set):
    """De-duplicate signals and return list of {ticker, won} dicts."""
    by_ticker = defaultdict(list)
    for s in signals:
        if s['ticker'] not in ticker_set:
            continue
        if s[RETURN_COL] is None:
            continue
        by_ticker[s['ticker']].append({
            'date_obj': parse_date(s['date']),
            'ticker': s['ticker'],
            'won': 1 if s[RETURN_COL] > 0 else 0,
        })

    # De-duplicate: keep first signal per stock per DEDUP_DAYS
    deduped = []
    for tk, sigs in by_ticker.items():
        sigs.sort(key=lambda x: x['date_obj'])
        last_taken = None
        for s in sigs:
            if last_taken is None or (s['date_obj'] - last_taken).days > DEDUP_DAYS:
                deduped.append({'ticker': s['ticker'], 'won': s['won']})
                last_taken = s['date_obj']

    return deduped


# ============================================================
# METRICS
# ============================================================
def compute_metrics(signals_list):
    """Compute stock-selection metrics from a list of {ticker, won}."""
    by_stock = defaultdict(list)
    for s in signals_list:
        by_stock[s['ticker']].append(s['won'])

    # Filter to stocks with enough signals
    stock_wrs = {}
    for tk, outcomes in by_stock.items():
        if len(outcomes) >= MIN_SIGNALS_PER_STOCK:
            stock_wrs[tk] = sum(outcomes) / len(outcomes) * 100

    if not stock_wrs:
        return {}

    wrs = list(stock_wrs.values())
    overall_wr = sum(s['won'] for s in signals_list) / len(signals_list) * 100

    # Cross-sectional std dev of per-stock WRs
    mean_wr = sum(wrs) / len(wrs)
    std_wr = math.sqrt(sum((w - mean_wr)**2 for w in wrs) / len(wrs))

    # Top quartile
    wrs_sorted = sorted(wrs, reverse=True)
    q1_idx = max(1, len(wrs_sorted) // 4)
    top_q_wr = sum(wrs_sorted[:q1_idx]) / q1_idx

    # Bottom quartile
    bot_q_wr = sum(wrs_sorted[-q1_idx:]) / q1_idx

    # Spread
    spread = max(wrs) - min(wrs)

    # Fraction > 55%
    frac_above_55 = sum(1 for w in wrs if w > 55) / len(wrs)

    # Fraction > 60%
    frac_above_60 = sum(1 for w in wrs if w > 60) / len(wrs)

    return {
        'overall_wr': overall_wr,
        'n_stocks': len(stock_wrs),
        'n_signals': len(signals_list),
        'mean_stock_wr': mean_wr,
        'std_stock_wr': std_wr,
        'top_quartile_wr': top_q_wr,
        'bottom_quartile_wr': bot_q_wr,
        'spread': spread,
        'frac_above_55': frac_above_55,
        'frac_above_60': frac_above_60,
    }


# ============================================================
# MONTE CARLO
# ============================================================
def run_mc(deduped_signals, label, n_iter=N_ITERATIONS):
    """Run Monte Carlo stock-selection test."""
    print(f'\n{"="*70}')
    print(f'  {label}')
    print(f'{"="*70}')

    # Real metrics
    real = compute_metrics(deduped_signals)
    print(f'  Signals: {real["n_signals"]}, Stocks (>={MIN_SIGNALS_PER_STOCK} signals): {real["n_stocks"]}')
    print(f'  Overall WR: {real["overall_wr"]:.1f}%')
    print(f'  Mean stock WR: {real["mean_stock_wr"]:.1f}%')
    print(f'  Std stock WR: {real["std_stock_wr"]:.1f}pp')
    print(f'  Top quartile WR: {real["top_quartile_wr"]:.1f}%')
    print(f'  Bottom quartile WR: {real["bottom_quartile_wr"]:.1f}%')
    print(f'  Spread (max-min): {real["spread"]:.1f}pp')
    print(f'  Stocks > 55%: {real["frac_above_55"]*100:.0f}%')
    print(f'  Stocks > 60%: {real["frac_above_60"]*100:.0f}%')

    # Build stock-size map (preserve signal counts per stock)
    stock_sizes = defaultdict(int)
    for s in deduped_signals:
        stock_sizes[s['ticker']] += 1
    stock_order = sorted(stock_sizes.keys())
    sizes = [stock_sizes[tk] for tk in stock_order]
    all_outcomes = [s['won'] for s in deduped_signals]

    # Shuffle: randomly reassign outcomes to stocks preserving counts
    rng = random.Random(SEED)
    mc_std = []
    mc_top_q = []
    mc_bot_q = []
    mc_spread = []
    mc_frac_55 = []
    mc_frac_60 = []

    t0 = time.time()
    for i in range(n_iter):
        # Shuffle all outcomes
        shuffled_outcomes = all_outcomes.copy()
        rng.shuffle(shuffled_outcomes)

        # Reassign to stocks based on original sizes
        idx = 0
        stock_wrs = []
        for sz in sizes:
            chunk = shuffled_outcomes[idx:idx + sz]
            idx += sz
            if sz >= MIN_SIGNALS_PER_STOCK:
                stock_wrs.append(sum(chunk) / sz * 100)

        if not stock_wrs:
            continue

        mean_sw = sum(stock_wrs) / len(stock_wrs)
        std_sw = math.sqrt(sum((w - mean_sw)**2 for w in stock_wrs) / len(stock_wrs))
        mc_std.append(std_sw)

        wrs_sorted = sorted(stock_wrs, reverse=True)
        q1_idx = max(1, len(wrs_sorted) // 4)
        mc_top_q.append(sum(wrs_sorted[:q1_idx]) / q1_idx)
        mc_bot_q.append(sum(wrs_sorted[-q1_idx:]) / q1_idx)
        mc_spread.append(max(stock_wrs) - min(stock_wrs))
        mc_frac_55.append(sum(1 for w in stock_wrs if w > 55) / len(stock_wrs))
        mc_frac_60.append(sum(1 for w in stock_wrs if w > 60) / len(stock_wrs))

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f'    [{i+1}/{n_iter}] {elapsed:.0f}s elapsed')

    elapsed_total = time.time() - t0

    # Compute p-values (fraction of shuffled >= real)
    def pval(real_val, mc_dist):
        return sum(1 for d in mc_dist if d >= real_val) / len(mc_dist)

    def mean_sd(dist):
        m = sum(dist) / len(dist)
        s = math.sqrt(sum((x - m)**2 for x in dist) / len(dist))
        return m, s

    p_std = pval(real['std_stock_wr'], mc_std)
    p_top_q = pval(real['top_quartile_wr'], mc_top_q)
    p_spread = pval(real['spread'], mc_spread)
    p_frac_55 = pval(real['frac_above_55'], mc_frac_55)
    p_frac_60 = pval(real['frac_above_60'], mc_frac_60)
    # Bottom quartile: test if real is LOWER than shuffled (worse worst stocks)
    p_bot_q = sum(1 for d in mc_bot_q if d <= real['bottom_quartile_wr']) / len(mc_bot_q)

    mean_std, sd_std = mean_sd(mc_std)
    mean_top, sd_top = mean_sd(mc_top_q)
    mean_bot, sd_bot = mean_sd(mc_bot_q)
    mean_spr, sd_spr = mean_sd(mc_spread)
    mean_f55, sd_f55 = mean_sd(mc_frac_55)
    mean_f60, sd_f60 = mean_sd(mc_frac_60)

    print(f'\n  MONTE CARLO RESULTS ({n_iter} iterations, {elapsed_total:.0f}s):')
    print(f'    {"Metric":<25} {"Real":>8} {"MC Mean":>8} {"MC SD":>7} {"p-value":>8} {"Sig?":>5}')
    print(f'    {"-"*60}')
    print(f'    {"Std(stock WR)":<25} {real["std_stock_wr"]:>7.1f}pp {mean_std:>7.1f}pp {sd_std:>6.1f}pp {p_std:>8.4f} {"***" if p_std<0.01 else "**" if p_std<0.05 else "":>5}')
    print(f'    {"Top quartile WR":<25} {real["top_quartile_wr"]:>7.1f}% {mean_top:>7.1f}% {sd_top:>6.1f}pp {p_top_q:>8.4f} {"***" if p_top_q<0.01 else "**" if p_top_q<0.05 else "":>5}')
    print(f'    {"Bottom quartile WR":<25} {real["bottom_quartile_wr"]:>7.1f}% {mean_bot:>7.1f}% {sd_bot:>6.1f}pp {p_bot_q:>8.4f} {"***" if p_bot_q<0.01 else "**" if p_bot_q<0.05 else "":>5}')
    print(f'    {"Spread (max-min)":<25} {real["spread"]:>7.1f}pp {mean_spr:>7.1f}pp {sd_spr:>6.1f}pp {p_spread:>8.4f} {"***" if p_spread<0.01 else "**" if p_spread<0.05 else "":>5}')
    print(f'    {"Frac > 55%":<25} {real["frac_above_55"]*100:>7.0f}% {mean_f55*100:>7.0f}% {sd_f55*100:>6.0f}pp {p_frac_55:>8.4f} {"***" if p_frac_55<0.01 else "**" if p_frac_55<0.05 else "":>5}')
    print(f'    {"Frac > 60%":<25} {real["frac_above_60"]*100:>7.0f}% {mean_f60*100:>7.0f}% {sd_f60*100:>6.0f}pp {p_frac_60:>8.4f} {"***" if p_frac_60<0.01 else "**" if p_frac_60<0.05 else "":>5}')

    # Verdict
    sig_count = sum(1 for p in [p_std, p_top_q, p_spread, p_frac_60] if p < 0.05)
    if sig_count >= 3:
        verdict = 'STRONG PASS — stock selection contributes genuine edge beyond random assignment'
    elif sig_count >= 1:
        verdict = 'PARTIAL PASS — some stock-level differentiation is real'
    else:
        verdict = 'FAIL — stock-level WRs are consistent with random outcome assignment'

    print(f'\n  VERDICT: {verdict}')

    return {
        'label': label,
        'real': {k: round(v, 4) if isinstance(v, float) else v for k, v in real.items()},
        'monte_carlo': {
            'n_iterations': n_iter,
            'seed': SEED,
            'elapsed_seconds': round(elapsed_total, 1),
            'std_stock_wr': {'real': round(real['std_stock_wr'], 2), 'mc_mean': round(mean_std, 2), 'mc_sd': round(sd_std, 2), 'p_value': round(p_std, 4)},
            'top_quartile_wr': {'real': round(real['top_quartile_wr'], 2), 'mc_mean': round(mean_top, 2), 'mc_sd': round(sd_top, 2), 'p_value': round(p_top_q, 4)},
            'bottom_quartile_wr': {'real': round(real['bottom_quartile_wr'], 2), 'mc_mean': round(mean_bot, 2), 'mc_sd': round(sd_bot, 2), 'p_value': round(p_bot_q, 4)},
            'spread': {'real': round(real['spread'], 2), 'mc_mean': round(mean_spr, 2), 'mc_sd': round(sd_spr, 2), 'p_value': round(p_spread, 4)},
            'frac_above_55': {'real': round(real['frac_above_55'], 4), 'mc_mean': round(mean_f55, 4), 'mc_sd': round(sd_f55, 4), 'p_value': round(p_frac_55, 4)},
            'frac_above_60': {'real': round(real['frac_above_60'], 4), 'mc_mean': round(mean_f60, 4), 'mc_sd': round(sd_f60, 4), 'p_value': round(p_frac_60, 4)},
        },
        'verdict': verdict,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print('Test #01: Monte Carlo Stock Selection')
    print('=' * 70)
    print('Loading data...')

    signals, ins_set, exp_set = load_data()
    print(f'Total signals: {len(signals)}')

    # De-duplicate
    ins_deduped = dedup_and_prepare(signals, ins_set)
    exp_deduped = dedup_and_prepare(signals, exp_set)
    all_deduped = dedup_and_prepare(signals, ins_set | exp_set)

    print(f'De-duplicated: in-sample={len(ins_deduped)}, expansion={len(exp_deduped)}, combined={len(all_deduped)}')

    results = {
        'test': 'Test #01: Monte Carlo Stock Selection',
        'description': (
            'Tests whether per-stock win rate variation is genuine or could arise from '
            'randomly assigning outcomes to stocks while preserving overall WR and '
            'per-stock signal counts.'
        ),
        'methodology': {
            'null_hypothesis': 'Per-stock WR variation is random noise from finite sampling',
            'shuffle': 'Pool all outcomes, reassign to stocks preserving per-stock signal counts',
            'n_iterations': N_ITERATIONS,
            'seed': SEED,
            'dedup_days': DEDUP_DAYS,
            'min_signals_per_stock': MIN_SIGNALS_PER_STOCK,
            'return_column': RETURN_COL,
        },
    }

    # Primary: In-sample
    results['in_sample'] = run_mc(ins_deduped, 'IN-SAMPLE (primary)')

    # Secondary: Expansion
    # Expansion omitted (public data is in-sample only)
    # results.get('expansion', {}) = run_mc(exp_deduped, 'EXPANSION')

    # Write results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults written to {RESULTS_FILE}')


if __name__ == '__main__':
    main()
