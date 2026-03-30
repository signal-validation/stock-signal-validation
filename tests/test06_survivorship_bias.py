"""
Test #06: Survivorship Bias Simulation
========================================
Bounds the potential impact of survivorship bias on our backtest results.

Problem:
  Our universe is 100% survivors. Stocks that went bankrupt, were delisted,
  or merged during 2006-2025 are excluded. These stocks likely had poor
  returns, biasing our WR upward.

Method:
  1. Parametric Monte Carlo: For each year, randomly "delist" some fraction
     of stocks and replace their outcomes with worse synthetic ones.
  2. Test a grid of (delisting_rate, delisted_stock_WR) assumptions.
  3. Find the breakeven: what assumptions would kill the edge?
  4. Separate analysis: compare WR of "true survivors" (stocks present
     across all 20 years) vs partial-history stocks.

Limitation:
  This is a SIMULATION of bias, not a measurement. Without actual delisted
  stock data (requires Norgate Data, $300/yr), we can only bound the bias
  under reasonable assumptions. Stated explicitly in all output.

Usage:
  python3 test06_survivorship_bias.py
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
N_ITERATIONS = 10000
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'test06_survivorship_bias_results.json')

DEDUP_DAYS = 28
RETURN_COL = 'return_20d'

# Parameter grid
DELISTING_RATES = [0.01, 0.02, 0.03, 0.05]  # Annual fraction of stocks "delisted"
DELISTED_WRS = [0.20, 0.30, 0.40, 0.45]     # WR assigned to delisted stocks

# Reference: S&P 500 turnover
REFERENCE_NOTES = {
    'sp500_turnover': '~4-5% annual turnover (but includes mergers/acquisitions with positive returns)',
    'bankruptcy_rate': '~1-2% of large/mid-cap stocks per year',
    'delisted_wr_note': 'Bankruptcies ~0-20% WR; acquisitions ~40-60% WR; blended realistic ~30-40%',
}


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
# BUILD PER-STOCK-YEAR SIGNAL DATA (DE-DUPLICATED)
# ============================================================
def build_stock_year_data(signals, ticker_set):
    """
    Build de-duplicated signal data organized by (stock, year).
    Returns:
      - stock_year_outcomes: {(ticker, year): [won, won, ...]}
      - all_deduped: list of {ticker, year, won}
      - stock_active_years: {ticker: set of years}
    """
    by_ticker = defaultdict(list)
    for s in signals:
        if s['ticker'] not in ticker_set:
            continue
        if s[RETURN_COL] is None:
            continue
        by_ticker[s['ticker']].append({
            'date_obj': parse_date(s['date']),
            'year': int(s['date'][:4]),
            'ticker': s['ticker'],
            'won': 1 if s[RETURN_COL] > 0 else 0,
        })

    # De-duplicate
    stock_year_outcomes = defaultdict(list)
    all_deduped = []
    stock_active_years = defaultdict(set)

    for tk, sigs in by_ticker.items():
        sigs.sort(key=lambda x: x['date_obj'])
        last_taken = None
        for s in sigs:
            if last_taken is None or (s['date_obj'] - last_taken).days > DEDUP_DAYS:
                stock_year_outcomes[(tk, s['year'])].append(s['won'])
                all_deduped.append({'ticker': tk, 'year': s['year'], 'won': s['won']})
                stock_active_years[tk].add(s['year'])
                last_taken = s['date_obj']

    return dict(stock_year_outcomes), all_deduped, dict(stock_active_years)


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================
def run_survivorship_mc(stock_year_outcomes, stock_active_years, all_deduped,
                        delisting_rate, delisted_wr, label, n_iter=N_ITERATIONS):
    """
    For each MC iteration:
    1. For each year, randomly select delisting_rate fraction of active stocks
    2. For "delisted" stocks, replace outcomes from delisting year onward
       with draws from Bernoulli(delisted_wr)
    3. Recompute overall WR
    """
    real_wins = sum(s['won'] for s in all_deduped)
    real_n = len(all_deduped)
    real_wr = real_wins / real_n * 100

    years = sorted(set(s['year'] for s in all_deduped))
    stocks_by_year = defaultdict(set)
    for tk, yrs in stock_active_years.items():
        for y in yrs:
            stocks_by_year[y].add(tk)

    rng = random.Random(SEED)
    mc_wrs = []
    mc_delisted_counts = []

    for _ in range(n_iter):
        # Track which stocks are "delisted" and from which year
        delisted = {}  # ticker -> delisting year

        for year in years:
            active = stocks_by_year[year] - set(delisted.keys())
            if not active:
                continue
            n_to_delist = max(1, int(len(active) * delisting_rate))
            if n_to_delist >= len(active):
                continue
            newly_delisted = rng.sample(sorted(active), n_to_delist)
            for tk in newly_delisted:
                delisted[tk] = year

        mc_delisted_counts.append(len(delisted))

        # Recompute WR with delisted stocks' outcomes replaced
        total_wins = 0
        total_n = 0
        for s in all_deduped:
            if s['ticker'] in delisted and s['year'] >= delisted[s['ticker']]:
                # Replace with synthetic outcome
                total_wins += 1 if rng.random() < delisted_wr else 0
            else:
                total_wins += s['won']
            total_n += 1

        mc_wrs.append(total_wins / total_n * 100)

    mc_wrs.sort()
    mean_wr = sum(mc_wrs) / n_iter
    std_wr = math.sqrt(sum((w - mean_wr)**2 for w in mc_wrs) / n_iter)
    ci_lo = mc_wrs[int(n_iter * 0.025)]
    ci_hi = mc_wrs[int(n_iter * 0.975)]
    mean_delisted = sum(mc_delisted_counts) / n_iter
    wr_drop = real_wr - mean_wr

    return {
        'delisting_rate': delisting_rate,
        'delisted_wr': delisted_wr,
        'real_wr': round(real_wr, 2),
        'adjusted_wr_mean': round(mean_wr, 2),
        'adjusted_wr_std': round(std_wr, 2),
        'adjusted_wr_ci_95': [round(ci_lo, 2), round(ci_hi, 2)],
        'wr_drop': round(wr_drop, 2),
        'mean_stocks_delisted': round(mean_delisted, 1),
        'still_above_50': mean_wr > 50.0,
        'still_significant': ci_lo > 50.0,
    }


# ============================================================
# TRUE SURVIVOR ANALYSIS
# ============================================================
def survivor_analysis(stock_year_outcomes, stock_active_years, all_deduped, years_range):
    """Compare WR of stocks present across ALL years vs partial."""
    all_years = set(range(years_range[0], years_range[1] + 1))
    # True survivors: stocks with signals in early, middle, AND late periods
    early = set(range(years_range[0], years_range[0] + 5))
    late = set(range(years_range[1] - 4, years_range[1] + 1))

    true_survivors = set()
    partial = set()
    for tk, yrs in stock_active_years.items():
        has_early = bool(yrs & early)
        has_late = bool(yrs & late)
        if has_early and has_late:
            true_survivors.add(tk)
        else:
            partial.add(tk)

    surv_signals = [s for s in all_deduped if s['ticker'] in true_survivors]
    part_signals = [s for s in all_deduped if s['ticker'] in partial]

    surv_wr = sum(s['won'] for s in surv_signals) / len(surv_signals) * 100 if surv_signals else 0
    part_wr = sum(s['won'] for s in part_signals) / len(part_signals) * 100 if part_signals else 0

    return {
        'true_survivors': len(true_survivors),
        'partial_history': len(partial),
        'survivor_signals': len(surv_signals),
        'partial_signals': len(part_signals),
        'survivor_wr': round(surv_wr, 2),
        'partial_wr': round(part_wr, 2),
        'wr_gap': round(surv_wr - part_wr, 2),
    }


# ============================================================
# MAIN
# ============================================================
def run_full_analysis(signals, ticker_set, label):
    """Run full survivorship analysis for a universe cut."""
    print(f'\n{"="*70}')
    print(f'  {label}')
    print(f'{"="*70}')

    stock_year_outcomes, all_deduped, stock_active_years = build_stock_year_data(signals, ticker_set)
    real_wr = sum(s['won'] for s in all_deduped) / len(all_deduped) * 100

    print(f'  Signals (de-duped): {len(all_deduped)}')
    print(f'  Stocks: {len(stock_active_years)}')
    print(f'  Real WR: {real_wr:.1f}%')

    years = sorted(set(s['year'] for s in all_deduped))
    print(f'  Year range: {years[0]}-{years[-1]}')

    # --- Grid simulation ---
    print(f'\n  SURVIVORSHIP BIAS GRID (mean adjusted WR, {N_ITERATIONS} MC iterations each):')
    print(f'    {"":>12}', end='')
    for dwr in DELISTED_WRS:
        print(f'  WR={dwr*100:.0f}%', end='')
    print()
    print(f'    {"-"*52}')

    grid_results = {}
    breakeven = None

    for dr in DELISTING_RATES:
        print(f'    delist={dr*100:.0f}%/yr', end='')
        for dwr in DELISTED_WRS:
            result = run_survivorship_mc(
                stock_year_outcomes, stock_active_years, all_deduped,
                dr, dwr, label, n_iter=N_ITERATIONS
            )
            key = f'{dr}_{dwr}'
            grid_results[key] = result
            adj_wr = result['adjusted_wr_mean']
            marker = '*' if not result['still_above_50'] else ''
            print(f'  {adj_wr:5.1f}%{marker}', end='')

            if breakeven is None and not result['still_above_50']:
                breakeven = (dr, dwr)
        print()

    print(f'    (* = adjusted WR below 50%)')

    # --- WR drop summary ---
    print(f'\n  WR DROP FROM REAL ({real_wr:.1f}%):')
    print(f'    {"":>12}', end='')
    for dwr in DELISTED_WRS:
        print(f'  WR={dwr*100:.0f}%', end='')
    print()
    print(f'    {"-"*52}')

    for dr in DELISTING_RATES:
        print(f'    delist={dr*100:.0f}%/yr', end='')
        for dwr in DELISTED_WRS:
            key = f'{dr}_{dwr}'
            drop = grid_results[key]['wr_drop']
            print(f'  {drop:+5.1f}pp', end='')
        print()

    # --- Survivor vs partial analysis ---
    surv = survivor_analysis(stock_year_outcomes, stock_active_years, all_deduped, (years[0], years[-1]))
    print(f'\n  TRUE SURVIVOR ANALYSIS:')
    print(f'    True survivors (early+late signals): {surv["true_survivors"]} stocks, {surv["survivor_signals"]} signals, WR={surv["survivor_wr"]:.1f}%')
    print(f'    Partial history stocks:              {surv["partial_history"]} stocks, {surv["partial_signals"]} signals, WR={surv["partial_wr"]:.1f}%')
    print(f'    WR gap (survivor - partial):         {surv["wr_gap"]:+.1f}pp')
    if surv['wr_gap'] > 2:
        print(f'    -> Survivors OUTPERFORM partial-history stocks -> some survivorship bias likely')
    elif surv['wr_gap'] < -2:
        print(f'    -> Survivors UNDERPERFORM partial-history stocks -> survivorship bias unlikely')
    else:
        print(f'    -> No material difference -> survivorship bias minimal for this metric')

    # --- Breakeven ---
    if breakeven:
        print(f'\n  BREAKEVEN: adjusted WR drops below 50% at delisting={breakeven[0]*100:.0f}%/yr, delisted WR={breakeven[1]*100:.0f}%')
    else:
        print(f'\n  BREAKEVEN: adjusted WR stays above 50% at ALL tested scenarios')

    # Most realistic scenario (2% delisting, 35% WR — blended bankruptcy+M&A)
    realistic_key = None
    for dr in [0.02, 0.03]:
        for dwr in [0.40, 0.30]:
            key = f'{dr}_{dwr}'
            if key in grid_results:
                realistic_key = key
                break
        if realistic_key:
            break

    if realistic_key:
        r = grid_results[realistic_key]
        print(f'\n  MOST REALISTIC SCENARIO (delist={r["delisting_rate"]*100:.0f}%/yr, delisted WR={r["delisted_wr"]*100:.0f}%):')
        print(f'    Adjusted WR: {r["adjusted_wr_mean"]:.1f}% (drop: {r["wr_drop"]:+.1f}pp)')
        print(f'    95% CI: [{r["adjusted_wr_ci_95"][0]:.1f}%, {r["adjusted_wr_ci_95"][1]:.1f}%]')
        print(f'    Still above 50%: {"YES" if r["still_above_50"] else "NO"}')
        print(f'    CI excludes 50%: {"YES" if r["still_significant"] else "NO"}')

    return {
        'label': label,
        'n_signals': len(all_deduped),
        'n_stocks': len(stock_active_years),
        'real_wr': round(real_wr, 2),
        'grid_results': grid_results,
        'survivor_analysis': surv,
        'breakeven': {'delisting_rate': breakeven[0], 'delisted_wr': breakeven[1]} if breakeven else None,
        'realistic_scenario': grid_results.get(realistic_key),
    }


def main():
    print('Test #06: Survivorship Bias Simulation')
    print('=' * 70)
    print(f'IMPORTANT: This is a SIMULATION of bias, not a measurement.')
    print(f'Without actual delisted stock data (Norgate Data, $300/yr),')
    print(f'we can only bound the bias under reasonable assumptions.')
    print()
    print('Loading data...')

    signals, ins_set, exp_set = load_data()
    print(f'Total signals: {len(signals)}')

    results = {
        'test': 'Test #06: Survivorship Bias Simulation',
        'description': (
            'Bounds potential survivorship bias by simulating random stock delistings '
            'at various rates and assigning delisted stocks a specified win rate. '
            'Tests a grid of assumptions from optimistic to pessimistic.'
        ),
        'limitation': (
            'This is a simulation, not a measurement. The actual bias depends on which '
            'stocks were delisted and their true returns, which requires Norgate Data. '
            'Results should be interpreted as plausible bounds, not point estimates.'
        ),
        'methodology': {
            'delisting_rates': DELISTING_RATES,
            'delisted_wrs': DELISTED_WRS,
            'n_iterations': N_ITERATIONS,
            'seed': SEED,
            'dedup_days': DEDUP_DAYS,
            'return_column': RETURN_COL,
        },
        'reference_notes': REFERENCE_NOTES,
    }

    results['in_sample'] = run_full_analysis(signals, ins_set, 'IN-SAMPLE')
    # Expansion omitted (public data is in-sample only)
    # results.get('expansion', {}) = run_full_analysis(signals, exp_set, 'EXPANSION')

    # Final verdict
    ins_realistic = results['in_sample'].get('realistic_scenario')
    exp_realistic = results.get('expansion', {}).get('realistic_scenario')

    print(f'\n{"="*70}')
    print(f'  FINAL VERDICT')
    print(f'{"="*70}')
    if ins_realistic and ins_realistic['still_above_50']:
        print(f'  In-sample: ROBUST to survivorship bias under realistic assumptions')
        print(f'    (adjusted WR={ins_realistic["adjusted_wr_mean"]:.1f}%, drop={ins_realistic["wr_drop"]:+.1f}pp)')
    else:
        print(f'  In-sample: VULNERABLE to survivorship bias')

    if exp_realistic and exp_realistic['still_above_50']:
        print(f'  Expansion: ROBUST to survivorship bias under realistic assumptions')
        print(f'    (adjusted WR={exp_realistic["adjusted_wr_mean"]:.1f}%, drop={exp_realistic["wr_drop"]:+.1f}pp)')
    else:
        print(f'  Expansion: VULNERABLE to survivorship bias')

    results['verdict'] = {
        'in_sample': 'ROBUST' if ins_realistic and ins_realistic['still_above_50'] else 'VULNERABLE',
        'expansion': 'ROBUST' if exp_realistic and exp_realistic['still_above_50'] else 'VULNERABLE',
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results written to {RESULTS_FILE}')


if __name__ == '__main__':
    main()
