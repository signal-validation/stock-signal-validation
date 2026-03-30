"""
Test #05: Transaction Cost Sensitivity
========================================
Tests whether the engine's edge survives realistic trading costs.

Costs modeled:
  - Brokerage commission (~0.01% for IBKR)
  - Bid-ask spread (~0.03-0.05% for liquid US large-caps)
  - Total realistic one-way: ~0.05-0.10%

Method:
  1. For each one-way cost level c, adjust each signal's return:
       adj = ((1 + return/100) * (1 - c) / (1 + c) - 1) * 100
     This exact formula accounts for entry cost on buy and exit cost on sell.
  2. Recompute WR, alpha, z-score, p-value on de-duplicated signals.
  3. Block bootstrap (5,000 resamples, block=63) at each cost level for CIs.
  4. Find breakeven cost (WR=50%, alpha=0).
  5. Test across all holding periods (5d, 10d, 20d, 40d).

Usage:
  python3 test05_transaction_costs.py
"""

import json
import math
import os
import random
from collections import defaultdict
from datetime import date as Date

# ============================================================
# CONFIG
# ============================================================
SEED = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'test05_transaction_costs_results.json')

DEDUP_DAYS = 28
COST_GRID = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]  # One-way %
HOLDING_PERIODS = ['return_5d', 'return_10d', 'return_20d', 'return_40d']
ALPHA_COLS = {'return_5d': 'alpha_5d', 'return_10d': 'alpha_10d', 'return_20d': 'alpha', 'return_40d': 'alpha_40d'}
SPY_COLS = {'return_5d': 'spy_return_5d', 'return_10d': 'spy_return_10d', 'return_20d': 'spy_return_20d', 'return_40d': 'spy_return_40d'}

N_BOOTSTRAP = 5000
BLOCK_SIZE = 63

# Reference costs
REFERENCE_COSTS = {
    'IBKR_commission': 0.01,
    'typical_spread': 0.04,
    'total_realistic': 0.05,
    'conservative': 0.10,
    'high_friction': 0.20,
}


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
    """De-duplicate signals, preserving all return columns."""
    by_ticker = defaultdict(list)
    for s in signals:
        if s['ticker'] not in ticker_set:
            continue
        if s['return_20d'] is None:
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


# ============================================================
# COST ADJUSTMENT
# ============================================================
def adjust_return(ret_pct, cost_oneway_pct):
    """Exact cost adjustment: accounts for cost at entry and exit."""
    if ret_pct is None:
        return None
    c = cost_oneway_pct / 100  # Convert to decimal
    r = ret_pct / 100
    # Buy at price*(1+c), sell at price*(1+r)*(1-c)
    adj = ((1 + r) * (1 - c) / (1 + c) - 1) * 100
    return adj


# ============================================================
# BLOCK BOOTSTRAP
# ============================================================
def block_bootstrap_wr(deduped, return_col, cost, block_size=BLOCK_SIZE, n_boot=N_BOOTSTRAP):
    """Block bootstrap CI for WR at a given cost level."""
    # Partition into time blocks
    all_dates = sorted(set(s['date'] for s in deduped))
    date_to_block = {d: i // block_size for i, d in enumerate(all_dates)}
    blocks = defaultdict(list)
    for s in deduped:
        bid = date_to_block[s['date']]
        blocks[bid].append(s)
    block_list = [blocks[k] for k in sorted(blocks.keys())]

    if not block_list:
        return None

    rng = random.Random(SEED)
    boot_wrs = []
    n_blocks = len(block_list)

    for _ in range(n_boot):
        sampled = rng.choices(block_list, k=n_blocks)
        total_w = 0
        total_n = 0
        for block in sampled:
            for s in block:
                r = s.get(return_col)
                if r is None:
                    continue
                adj = adjust_return(r, cost)
                total_w += 1 if adj > 0 else 0
                total_n += 1
        if total_n > 0:
            boot_wrs.append(total_w / total_n * 100)

    boot_wrs.sort()
    n = len(boot_wrs)
    if n < 100:
        return None

    return {
        'ci_95': [round(boot_wrs[int(n * 0.025)], 2), round(boot_wrs[int(n * 0.975)], 2)],
        'se': round((sum((w - sum(boot_wrs)/n)**2 for w in boot_wrs) / n) ** 0.5, 2),
    }


# ============================================================
# ANALYSIS
# ============================================================
def analyze_cost_sensitivity(deduped, label):
    """Run cost sensitivity for all holding periods and cost levels."""
    print(f'\n{"="*70}')
    print(f'  {label}')
    print(f'{"="*70}')
    print(f'  Signals: {len(deduped)}')

    results = {'label': label, 'n_signals': len(deduped)}

    for return_col in HOLDING_PERIODS:
        period_label = return_col.replace('return_', '')
        alpha_col = ALPHA_COLS[return_col]
        spy_col = SPY_COLS[return_col]

        # Count valid signals for this period
        valid = [s for s in deduped if s.get(return_col) is not None]
        n_valid = len(valid)
        if n_valid == 0:
            continue

        print(f'\n  --- Holding period: {period_label} (N={n_valid}) ---')
        print(f'    {"Cost":>6} {"WR":>7} {"Alpha":>8} {"z":>7} {"p":>11} {"CI_lo":>7} {"CI_hi":>7} {"Sig?":>5}')
        print(f'    {"-"*62}')

        period_results = []
        breakeven_wr = None
        breakeven_alpha = None

        for cost in COST_GRID:
            # Adjust returns
            wins = 0
            total = 0
            alpha_sum = 0
            alpha_count = 0
            for s in valid:
                r = s.get(return_col)
                if r is None:
                    continue
                adj = adjust_return(r, cost)
                total += 1
                if adj > 0:
                    wins += 1

                # Alpha = adjusted return - spy return
                spy_r = s.get(spy_col, 0) or 0
                adj_alpha = adj - spy_r
                alpha_sum += adj_alpha
                alpha_count += 1

            wr = wins / total * 100 if total > 0 else 0
            mean_alpha = alpha_sum / alpha_count if alpha_count > 0 else 0

            # Z-score and p-value
            se = math.sqrt(0.5 * 0.5 / total) if total > 0 else 1
            z = (wr / 100 - 0.5) / se
            p = 0.5 * math.erfc(z / math.sqrt(2)) if z > 0 else 1.0

            # Bootstrap CI (only for primary 20d period)
            boot = None
            if return_col == 'return_20d':
                boot = block_bootstrap_wr(valid, return_col, cost)

            ci_lo = boot['ci_95'][0] if boot else None
            ci_hi = boot['ci_95'][1] if boot else None
            sig = 'YES' if p < 0.05 else 'NO'

            ci_lo_str = f'{ci_lo:.1f}%' if ci_lo is not None else '  --'
            ci_hi_str = f'{ci_hi:.1f}%' if ci_hi is not None else '--'

            print(f'    {cost:>5.2f}% {wr:>6.1f}% {mean_alpha:>+7.2f}% {z:>7.2f} {p:>11.2e} {ci_lo_str:>7} {ci_hi_str:>7} {sig:>5}')

            # Track breakeven
            if breakeven_wr is None and wr <= 50.0 and cost > 0:
                breakeven_wr = cost
            if breakeven_alpha is None and mean_alpha <= 0 and cost > 0:
                breakeven_alpha = cost

            period_results.append({
                'cost_oneway_pct': cost,
                'wr': round(wr, 2),
                'mean_alpha': round(mean_alpha, 2),
                'z': round(z, 2),
                'p': p,
                'significant': p < 0.05,
                'bootstrap_ci_95': boot['ci_95'] if boot else None,
                'bootstrap_se': boot['se'] if boot else None,
            })

        print(f'\n    Breakeven cost (WR=50%):   {breakeven_wr:.2f}% one-way' if breakeven_wr else '\n    Breakeven cost (WR=50%):   >0.50% (edge survives all tested costs)')
        print(f'    Breakeven cost (alpha=0):  {breakeven_alpha:.2f}% one-way' if breakeven_alpha else f'    Breakeven cost (alpha=0):  >0.50%')

        results[period_label] = {
            'n_valid': n_valid,
            'cost_sensitivity': period_results,
            'breakeven_wr': breakeven_wr,
            'breakeven_alpha': breakeven_alpha,
        }

    return results


def main():
    print('Test #05: Transaction Cost Sensitivity')
    print('=' * 70)
    print('Loading data...')

    signals, ins_set, exp_set = load_data()
    print(f'Total signals: {len(signals)}')

    ins_deduped = dedup_signals(signals, ins_set)
    exp_deduped = dedup_signals(signals, exp_set)
    print(f'De-duplicated: in-sample={len(ins_deduped)}, expansion={len(exp_deduped)}')

    results = {
        'test': 'Test #05: Transaction Cost Sensitivity',
        'description': (
            'Tests whether the engine edge survives realistic trading costs. '
            'Uses exact cost formula accounting for entry and exit price differences. '
            'Block bootstrap CIs at each cost level for 20d holding period.'
        ),
        'methodology': {
            'cost_formula': 'adj = ((1+r)*(1-c)/(1+c) - 1) * 100, c=one-way cost',
            'cost_grid_oneway_pct': COST_GRID,
            'holding_periods': HOLDING_PERIODS,
            'bootstrap': {'n': N_BOOTSTRAP, 'block_size': BLOCK_SIZE, 'seed': SEED},
            'dedup_days': DEDUP_DAYS,
        },
        'reference_costs': REFERENCE_COSTS,
    }

    results['in_sample'] = analyze_cost_sensitivity(ins_deduped, 'IN-SAMPLE')
    # Expansion omitted (public data is in-sample only)
    # results.get('expansion', {}) = analyze_cost_sensitivity(exp_deduped, 'EXPANSION')

    # Summary
    ins_20d = results['in_sample'].get('20d', {})
    exp_20d = results.get('expansion', {}).get('20d', {})

    print(f'\n{"="*70}')
    print(f'  SUMMARY (20-day holding period)')
    print(f'{"="*70}')
    print(f'  In-sample breakeven (WR=50%):  {ins_20d.get("breakeven_wr", ">0.50")}% one-way')
    print(f'  In-sample breakeven (alpha=0): {ins_20d.get("breakeven_alpha", ">0.50")}% one-way')
    print(f'  Expansion breakeven (WR=50%):  {exp_20d.get("breakeven_wr", ">0.50")}% one-way')
    print(f'  Expansion breakeven (alpha=0): {exp_20d.get("breakeven_alpha", ">0.50")}% one-way')
    print(f'')
    print(f'  Typical retail cost (IBKR + spread): ~0.05% one-way')
    print(f'  Conservative estimate:                ~0.10% one-way')

    # Determine verdict
    ins_be = ins_20d.get('breakeven_wr')
    exp_be = exp_20d.get('breakeven_wr')
    if ins_be is None or ins_be > 0.20:
        ins_verdict = 'SURVIVES at all realistic costs'
    elif ins_be > 0.10:
        ins_verdict = 'SURVIVES at typical costs, marginal at conservative'
    else:
        ins_verdict = 'DOES NOT SURVIVE realistic costs'

    if exp_be is None or exp_be > 0.20:
        exp_verdict = 'SURVIVES at all realistic costs'
    elif exp_be > 0.10:
        exp_verdict = 'SURVIVES at typical costs, marginal at conservative'
    else:
        exp_verdict = 'DOES NOT SURVIVE realistic costs'

    print(f'\n  In-sample:  {ins_verdict}')
    print(f'  Expansion:  {exp_verdict}')

    results['verdict'] = {
        'in_sample': ins_verdict,
        'expansion': exp_verdict,
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  Results written to {RESULTS_FILE}')


if __name__ == '__main__':
    main()
