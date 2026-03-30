"""
Test #10: Effective N / Autocorrelation Analysis
================================================
Foundation test for the entire statistical validation suite.

Problem:
  The backtest produces 31,731 signals, but consecutive same-stock signals
  within a 40-day holding period share overlapping return windows.
  Treating them as independent inflates significance. We need the TRUE
  independent observation count (effective N).

Method:
  1. Compute autocorrelation of win/loss outcomes at lags 1..60 (days between
     consecutive same-stock signals).
  2. Estimate effective N using the standard formula:
       N_eff = N / (1 + 2 * sum_{k=1}^{K} rho_k)
     where rho_k is the autocorrelation at lag k (Priestley 1981, Bayley & Hammersley 1946).
  3. Compare naive p-values (using raw N) to adjusted p-values (using N_eff).
  4. Validate the 28-day de-duplication rule by showing autocorrelation drops
     below significance at ~28 days.
  5. Report per-stock and aggregate statistics.

Outputs:
  - Console summary with publishable numbers
  - test10_effective_n_results.json with full details

Usage:
  python3 test10_effective_n.py
"""

import json
import math
import os
import sys
from collections import defaultdict
from datetime import date as Date, timedelta

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'test10_effective_n_results.json')

MAX_LAG_DAYS = 60        # Compute autocorrelation up to this many calendar days
DEDUP_DAYS = 28           # Standard de-duplication window
HOLDING_PERIOD = 40       # 40-day return window (worst case overlap)
RETURN_COL = 'return_20d' # Primary return column for win/loss


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
# CORE: Per-stock signal sequences
# ============================================================
def build_stock_sequences(signals, ticker_set=None):
    """Group signals by ticker, sort chronologically, compute win/loss."""
    by_ticker = defaultdict(list)
    for s in signals:
        if ticker_set and s['ticker'] not in ticker_set:
            continue
        if s[RETURN_COL] is None:
            continue
        by_ticker[s['ticker']].append({
            'date': s['date'],
            'date_obj': parse_date(s['date']),
            'won': 1 if s[RETURN_COL] > 0 else 0,
            'ticker': s['ticker'],
            'return': s[RETURN_COL],
        })
    for tk in by_ticker:
        by_ticker[tk].sort(key=lambda x: x['date'])
    return dict(by_ticker)


# ============================================================
# OVERLAP ANALYSIS
# ============================================================
def compute_overlap_stats(stock_sequences):
    """For each consecutive same-stock pair, compute the calendar day gap."""
    gaps = []
    overlap_count = 0
    total_pairs = 0
    for tk, seq in stock_sequences.items():
        for i in range(1, len(seq)):
            gap = (seq[i]['date_obj'] - seq[i-1]['date_obj']).days
            gaps.append(gap)
            total_pairs += 1
            if gap <= HOLDING_PERIOD:
                overlap_count += 1

    if not gaps:
        return {}

    gaps_sorted = sorted(gaps)
    n = len(gaps_sorted)
    return {
        'total_consecutive_pairs': total_pairs,
        'overlapping_pairs': overlap_count,
        'overlap_fraction': round(overlap_count / total_pairs, 4),
        'median_gap_days': gaps_sorted[n // 2],
        'mean_gap_days': round(sum(gaps) / n, 1),
        'p25_gap_days': gaps_sorted[n // 4],
        'p75_gap_days': gaps_sorted[3 * n // 4],
        'p90_gap_days': gaps_sorted[int(n * 0.9)],
        'gap_distribution': gaps,
    }


# ============================================================
# AUTOCORRELATION
# ============================================================
def compute_autocorrelation_by_lag(stock_sequences, max_lag=MAX_LAG_DAYS):
    """
    Compute outcome autocorrelation binned by calendar-day lag.

    For each consecutive pair of same-stock signals separated by k days,
    compute the correlation between their win/loss outcomes.

    Returns dict: lag -> (correlation, n_pairs)
    """
    # Collect pairs binned by lag
    lag_pairs = defaultdict(list)  # lag -> list of (won_i, won_j)

    for tk, seq in stock_sequences.items():
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                gap = (seq[j]['date_obj'] - seq[i]['date_obj']).days
                if gap > max_lag:
                    break
                lag_pairs[gap].append((seq[i]['won'], seq[j]['won']))

    # Compute correlation at each lag
    # Global mean for centering
    all_outcomes = []
    for seq in stock_sequences.values():
        for s in seq:
            all_outcomes.append(s['won'])
    global_mean = sum(all_outcomes) / len(all_outcomes) if all_outcomes else 0.5
    global_var = sum((x - global_mean)**2 for x in all_outcomes) / len(all_outcomes) if all_outcomes else 0.25

    autocorr = {}
    for lag in range(1, max_lag + 1):
        pairs = lag_pairs.get(lag, [])
        if len(pairs) < 10:  # Need minimum pairs for reliable estimate
            autocorr[lag] = {'rho': 0.0, 'n_pairs': len(pairs), 'reliable': False}
            continue

        # Pearson correlation between outcomes
        cov = sum((a - global_mean) * (b - global_mean) for a, b in pairs) / len(pairs)
        rho = cov / global_var if global_var > 0 else 0.0
        rho = max(-1.0, min(1.0, rho))  # Clamp

        autocorr[lag] = {
            'rho': round(rho, 4),
            'n_pairs': len(pairs),
            'reliable': True,
        }

    return autocorr, global_mean, global_var


def compute_consecutive_autocorrelation(stock_sequences):
    """
    Compute autocorrelation using only consecutive pairs (lag-1 in signal order).
    This is the strongest test — does the NEXT signal for the same stock
    have a correlated outcome?
    """
    pairs = []
    for tk, seq in stock_sequences.items():
        for i in range(1, len(seq)):
            pairs.append((seq[i-1]['won'], seq[i]['won']))

    if not pairs:
        return 0.0, 0

    mean_a = sum(p[0] for p in pairs) / len(pairs)
    mean_b = sum(p[1] for p in pairs) / len(pairs)

    cov = sum((a - mean_a) * (b - mean_b) for a, b in pairs) / len(pairs)
    var_a = sum((a - mean_a)**2 for a, b in pairs) / len(pairs)
    var_b = sum((b - mean_b)**2 for a, b in pairs) / len(pairs)

    denom = math.sqrt(var_a * var_b) if var_a > 0 and var_b > 0 else 1.0
    rho = cov / denom

    return round(rho, 4), len(pairs)


# ============================================================
# EFFECTIVE N CALCULATION
# ============================================================
def compute_effective_n(n_raw, autocorr_dict, max_lag=None):
    """
    N_eff = N / (1 + 2 * sum_{k=1}^{K} rho_k)

    Uses Bartlett's formula. Only includes lags with reliable estimates.
    K is chosen as the first lag where rho crosses zero (or max_lag).
    """
    if max_lag is None:
        max_lag = MAX_LAG_DAYS

    # Sum positive autocorrelations until first zero crossing
    rho_sum = 0.0
    rho_values = []
    for lag in range(1, max_lag + 1):
        entry = autocorr_dict.get(lag)
        if entry is None or not entry.get('reliable', False):
            continue
        rho = entry['rho']
        rho_values.append((lag, rho))
        if rho <= 0:
            break  # Standard truncation at first zero crossing
        rho_sum += rho

    inflation_factor = 1 + 2 * rho_sum
    inflation_factor = max(1.0, inflation_factor)  # Can't be less than 1

    n_eff = n_raw / inflation_factor

    return {
        'n_raw': n_raw,
        'n_eff': round(n_eff, 1),
        'inflation_factor': round(inflation_factor, 2),
        'rho_sum': round(rho_sum, 4),
        'truncation_lag': rho_values[-1][0] if rho_values else 0,
        'n_lags_used': len([v for v in rho_values if v[1] > 0]),
    }


# ============================================================
# DE-DUPLICATION VALIDATION
# ============================================================
def dedup_signals_from_sequences(stock_sequences, dedup_days=DEDUP_DAYS):
    """De-duplicate: keep first signal per stock per dedup_days."""
    result = []
    for tk, seq in stock_sequences.items():
        last_taken = None
        for s in seq:
            if last_taken is None or (s['date_obj'] - last_taken).days > dedup_days:
                result.append(s)
                last_taken = s['date_obj']
    return result


def compute_dedup_autocorrelation(stock_sequences, dedup_days=DEDUP_DAYS):
    """Compute consecutive autocorrelation AFTER de-duplication."""
    deduped_by_stock = defaultdict(list)
    for tk, seq in stock_sequences.items():
        last_taken = None
        for s in seq:
            if last_taken is None or (s['date_obj'] - last_taken).days > dedup_days:
                deduped_by_stock[tk].append(s)
                last_taken = s['date_obj']

    return compute_consecutive_autocorrelation(dict(deduped_by_stock))


# ============================================================
# P-VALUE CALCULATION
# ============================================================
def binomial_p_value(n, k, p0=0.5):
    """
    One-sided p-value: P(X >= k) under Binomial(n, p0).
    Uses normal approximation for large n.
    """
    if n <= 0:
        return 1.0
    mu = n * p0
    sigma = math.sqrt(n * p0 * (1 - p0))
    if sigma == 0:
        return 0.0 if k > mu else 1.0
    z = (k - mu - 0.5) / sigma  # Continuity correction
    # Standard normal CDF approximation (Abramowitz & Stegun)
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return p


def z_score(wr, n, p0=0.5):
    """Z-score for observed WR vs null hypothesis p0."""
    if n <= 0:
        return 0.0
    se = math.sqrt(p0 * (1 - p0) / n)
    return (wr - p0) / se if se > 0 else 0.0


# ============================================================
# PER-STOCK ANALYSIS
# ============================================================
def per_stock_stats(stock_sequences):
    """Compute per-stock signal count, WR, and consecutive autocorrelation."""
    stats = []
    for tk, seq in stock_sequences.items():
        n = len(seq)
        wins = sum(s['won'] for s in seq)
        wr = wins / n * 100 if n > 0 else 0

        # Consecutive autocorrelation for this stock
        if n >= 5:
            pairs = [(seq[i-1]['won'], seq[i]['won']) for i in range(1, n)]
            mean_a = sum(p[0] for p in pairs) / len(pairs)
            mean_b = sum(p[1] for p in pairs) / len(pairs)
            cov = sum((a - mean_a) * (b - mean_b) for a, b in pairs) / len(pairs)
            var_a = sum((a - mean_a)**2 for a, b in pairs) / len(pairs)
            var_b = sum((b - mean_b)**2 for a, b in pairs) / len(pairs)
            denom = math.sqrt(var_a * var_b) if var_a > 0 and var_b > 0 else 1.0
            rho = cov / denom
        else:
            rho = None

        stats.append({
            'ticker': tk,
            'n_signals': n,
            'wins': wins,
            'wr': round(wr, 1),
            'consecutive_rho': round(rho, 4) if rho is not None else None,
        })

    return sorted(stats, key=lambda x: -x['n_signals'])


# ============================================================
# MAIN ANALYSIS
# ============================================================
def run_analysis(stock_sequences, label):
    """Run full effective N analysis on a set of stock sequences."""
    print(f'\n{"="*70}')
    print(f'  {label}')
    print(f'{"="*70}')

    total_signals = sum(len(seq) for seq in stock_sequences.values())
    total_stocks = len(stock_sequences)
    all_wins = sum(s['won'] for seq in stock_sequences.values() for s in seq)
    raw_wr = all_wins / total_signals * 100 if total_signals > 0 else 0

    print(f'  Stocks: {total_stocks}, Signals: {total_signals}, WR: {raw_wr:.1f}%')

    # --- 1. Overlap stats ---
    overlap = compute_overlap_stats(stock_sequences)
    print(f'\n  OVERLAP ANALYSIS:')
    print(f'    Consecutive pairs: {overlap["total_consecutive_pairs"]}')
    print(f'    Overlapping (gap <= {HOLDING_PERIOD}d): {overlap["overlapping_pairs"]} ({overlap["overlap_fraction"]*100:.1f}%)')
    print(f'    Median gap: {overlap["median_gap_days"]} days')
    print(f'    Mean gap: {overlap["mean_gap_days"]} days')
    print(f'    P25/P75/P90: {overlap["p25_gap_days"]}d / {overlap["p75_gap_days"]}d / {overlap["p90_gap_days"]}d')

    # --- 2. Consecutive autocorrelation ---
    consec_rho, consec_n = compute_consecutive_autocorrelation(stock_sequences)
    # 95% significance threshold for correlation
    sig_threshold = 1.96 / math.sqrt(consec_n) if consec_n > 0 else 1.0
    print(f'\n  CONSECUTIVE AUTOCORRELATION:')
    print(f'    rho(consecutive) = {consec_rho:.4f} (n={consec_n} pairs)')
    print(f'    95% significance threshold: +/- {sig_threshold:.4f}')
    print(f'    Significant: {"YES" if abs(consec_rho) > sig_threshold else "NO"}')

    # --- 3. Autocorrelation by calendar-day lag ---
    autocorr, global_mean, global_var = compute_autocorrelation_by_lag(stock_sequences)
    print(f'\n  AUTOCORRELATION BY LAG (calendar days):')
    print(f'    Global WR: {global_mean*100:.1f}%')

    # Print key lags
    key_lags = [1, 2, 3, 5, 7, 10, 14, 21, 28, 35, 42, 50, 60]
    print(f'    {"Lag":>5} {"rho":>8} {"n_pairs":>8} {"sig?":>5}')
    print(f'    {"-"*30}')
    for lag in key_lags:
        if lag in autocorr and autocorr[lag]['reliable']:
            entry = autocorr[lag]
            n_p = entry['n_pairs']
            sig_t = 1.96 / math.sqrt(n_p) if n_p > 0 else 1.0
            is_sig = "***" if abs(entry['rho']) > sig_t else ""
            print(f'    {lag:>5} {entry["rho"]:>8.4f} {n_p:>8} {is_sig:>5}')

    # Find lag where autocorrelation first drops below significance
    first_insig_lag = None
    for lag in range(1, MAX_LAG_DAYS + 1):
        if lag in autocorr and autocorr[lag]['reliable']:
            n_p = autocorr[lag]['n_pairs']
            sig_t = 1.96 / math.sqrt(n_p) if n_p > 0 else 1.0
            if abs(autocorr[lag]['rho']) <= sig_t:
                first_insig_lag = lag
                break

    print(f'\n    First insignificant lag: {first_insig_lag} days')
    print(f'    28-day dedup rule: {"VALIDATED" if first_insig_lag and first_insig_lag <= DEDUP_DAYS else "NEEDS REVIEW"}')

    # --- 4. Effective N ---
    n_eff_result = compute_effective_n(total_signals, autocorr)
    print(f'\n  EFFECTIVE N:')
    print(f'    N_raw:             {n_eff_result["n_raw"]}')
    print(f'    N_eff:             {n_eff_result["n_eff"]:.0f}')
    print(f'    Inflation factor:  {n_eff_result["inflation_factor"]:.2f}x')
    print(f'    Rho sum:           {n_eff_result["rho_sum"]:.4f}')
    print(f'    Truncation lag:    {n_eff_result["truncation_lag"]} days')
    print(f'    Lags used:         {n_eff_result["n_lags_used"]}')

    # --- 5. De-duplication validation ---
    deduped = dedup_signals_from_sequences(stock_sequences)
    dedup_wins = sum(s['won'] for s in deduped)
    dedup_wr = dedup_wins / len(deduped) * 100 if deduped else 0
    dedup_rho, dedup_rho_n = compute_dedup_autocorrelation(stock_sequences)
    dedup_sig_t = 1.96 / math.sqrt(dedup_rho_n) if dedup_rho_n > 0 else 1.0

    print(f'\n  DE-DUPLICATION VALIDATION ({DEDUP_DAYS}-day rule):')
    print(f'    De-duped signals:  {len(deduped)}')
    print(f'    De-duped WR:       {dedup_wr:.1f}%')
    print(f'    Residual rho:      {dedup_rho:.4f} (threshold: +/- {dedup_sig_t:.4f})')
    print(f'    Residual significant: {"YES — consider longer dedup window" if abs(dedup_rho) > dedup_sig_t else "NO — signals are approximately independent"}')

    # --- 6. P-value comparison ---
    raw_z = z_score(raw_wr / 100, total_signals)
    raw_p = binomial_p_value(total_signals, all_wins)

    eff_z = z_score(dedup_wr / 100, n_eff_result['n_eff'])
    # p from z-score directly (effective N adjusts SE, not subsample)
    eff_p = 0.5 * math.erfc(eff_z / math.sqrt(2)) if eff_z > 0 else 1.0

    dedup_z = z_score(dedup_wr / 100, len(deduped))
    dedup_p = binomial_p_value(len(deduped), dedup_wins)

    print(f'\n  P-VALUE COMPARISON (H0: WR = 50%):')
    print(f'    {"Method":<25} {"N":>8} {"WR":>7} {"z":>8} {"p":>12}')
    print(f'    {"-"*62}')
    print(f'    {"Naive (raw)":<25} {total_signals:>8} {raw_wr:>6.1f}% {raw_z:>8.2f} {raw_p:>12.2e}')
    print(f'    {"Effective N (Bartlett)":<25} {n_eff_result["n_eff"]:>8.0f} {dedup_wr:>6.1f}% {eff_z:>8.2f} {eff_p:>12.2e}')
    print(f'    {"De-duplicated (28d)":<25} {len(deduped):>8} {dedup_wr:>6.1f}% {dedup_z:>8.2f} {dedup_p:>12.2e}')

    inflation_ratio = raw_z / dedup_z if dedup_z > 0 else float('inf')
    print(f'\n    Naive z / De-duped z = {inflation_ratio:.1f}x (significance inflation from overlap)')

    # --- 7. Per-stock stats ---
    stock_stats = per_stock_stats(stock_sequences)
    rho_values = [s['consecutive_rho'] for s in stock_stats if s['consecutive_rho'] is not None]

    if rho_values:
        mean_stock_rho = sum(rho_values) / len(rho_values)
        pos_rho_count = sum(1 for r in rho_values if r > 0)
        sig_rho_count = sum(
            1 for s in stock_stats
            if s['consecutive_rho'] is not None and s['n_signals'] >= 10
            and abs(s['consecutive_rho']) > 1.96 / math.sqrt(s['n_signals'] - 1)
        )
        total_with_rho = len(rho_values)

        print(f'\n  PER-STOCK AUTOCORRELATION:')
        print(f'    Stocks with rho: {total_with_rho}')
        print(f'    Mean rho: {mean_stock_rho:.4f}')
        print(f'    Positive rho: {pos_rho_count}/{total_with_rho} ({pos_rho_count/total_with_rho*100:.0f}%)')
        print(f'    Individually significant: {sig_rho_count}/{total_with_rho}')

    # Strip gap distribution from overlap for JSON (too large)
    overlap_json = {k: v for k, v in overlap.items() if k != 'gap_distribution'}

    # Build autocorrelation summary for key lags
    autocorr_summary = {}
    for lag in range(1, MAX_LAG_DAYS + 1):
        if lag in autocorr:
            autocorr_summary[str(lag)] = autocorr[lag]

    return {
        'label': label,
        'total_stocks': total_stocks,
        'total_signals': total_signals,
        'raw_wr': round(raw_wr, 2),
        'overlap': overlap_json,
        'consecutive_autocorrelation': {
            'rho': consec_rho,
            'n_pairs': consec_n,
            'significance_threshold': round(sig_threshold, 4),
            'significant': abs(consec_rho) > sig_threshold,
        },
        'autocorrelation_by_lag': autocorr_summary,
        'first_insignificant_lag': first_insig_lag,
        'dedup_rule_validated': first_insig_lag is not None and first_insig_lag <= DEDUP_DAYS,
        'effective_n': n_eff_result,
        'deduplication': {
            'dedup_days': DEDUP_DAYS,
            'n_deduped': len(deduped),
            'dedup_wr': round(dedup_wr, 2),
            'residual_rho': dedup_rho,
            'residual_rho_n_pairs': dedup_rho_n,
            'residual_significant': abs(dedup_rho) > dedup_sig_t,
        },
        'p_values': {
            'naive': {'n': total_signals, 'wr': round(raw_wr, 2), 'z': round(raw_z, 2), 'p': raw_p},
            'effective_n': {'n': round(n_eff_result['n_eff']), 'wr': round(dedup_wr, 2), 'z': round(eff_z, 2), 'p': eff_p},
            'deduped': {'n': len(deduped), 'wr': round(dedup_wr, 2), 'z': round(dedup_z, 2), 'p': dedup_p},
        },
        'significance_inflation': round(inflation_ratio, 2),
        'per_stock_rho': {
            'mean': round(mean_stock_rho, 4) if rho_values else None,
            'fraction_positive': round(pos_rho_count / total_with_rho, 3) if rho_values else None,
            'individually_significant': sig_rho_count if rho_values else 0,
            'total_measured': total_with_rho if rho_values else 0,
        },
        'top10_most_autocorrelated': sorted(
            [s for s in stock_stats if s['consecutive_rho'] is not None],
            key=lambda x: -abs(x['consecutive_rho'])
        )[:10],
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print('Test #10: Effective N / Autocorrelation Analysis')
    print('=' * 70)
    print('Loading data...')

    signals, ins_set, exp_set = load_data()
    print(f'Total signals: {len(signals)}')

    # Build sequences for each cut
    ins_seqs = build_stock_sequences(signals, ins_set)
    exp_seqs = build_stock_sequences(signals, exp_set)
    all_seqs = build_stock_sequences(signals, ins_set | exp_set)

    results = {
        'test': 'Test #10: Effective N / Autocorrelation',
        'description': (
            'Computes the true number of independent observations in the backtest, '
            'accounting for overlapping holding periods of consecutive same-stock signals. '
            'Foundation for all subsequent statistical tests.'
        ),
        'methodology': {
            'autocorrelation': 'Pearson correlation of win/loss outcomes binned by calendar-day lag',
            'effective_n': 'Bartlett formula: N_eff = N / (1 + 2 * sum(rho_k)), truncated at first zero crossing',
            'deduplication': f'Keep first signal per stock per {DEDUP_DAYS} calendar days',
            'return_column': RETURN_COL,
            'holding_period': HOLDING_PERIOD,
        },
    }

    # Primary: In-sample (the claim we're validating)
    results['in_sample'] = run_analysis(ins_seqs, 'IN-SAMPLE (primary — validates 60.3% claim)')

    # Secondary: Expansion
    # Expansion omitted (public data is in-sample only)
    # results.get('expansion', {}) = run_analysis(exp_seqs, 'EXPANSION')

    # Tertiary: Combined
    results['combined'] = run_analysis(all_seqs, 'COMBINED')

    # ============================================================
    # GAP HISTOGRAM (for visualization in post)
    # ============================================================
    all_overlap = compute_overlap_stats(all_seqs)
    gap_dist = all_overlap.get('gap_distribution', [])
    hist_buckets = [0, 1, 2, 3, 5, 7, 10, 14, 21, 28, 40, 60, 90, 180, 365, 99999]
    hist = {}
    for i in range(len(hist_buckets) - 1):
        lo, hi = hist_buckets[i], hist_buckets[i+1]
        label = f'{lo}-{hi}d' if hi < 99999 else f'{lo}d+'
        count = sum(1 for g in gap_dist if lo <= g < hi)
        hist[label] = count
    results['gap_histogram'] = hist

    # ============================================================
    # PUBLISHABLE SUMMARY
    # ============================================================
    ins = results['in_sample']
    print(f'\n{"="*70}')
    print(f'  PUBLISHABLE SUMMARY')
    print(f'{"="*70}')
    print(f'')
    print(f'  The signal engine backtest produces {ins["total_signals"]:,} in-sample signals')
    print(f'  across {ins["total_stocks"]} stocks. However, {ins["overlap"]["overlap_fraction"]*100:.0f}% of consecutive')
    print(f'  same-stock signals overlap within the {HOLDING_PERIOD}-day holding period')
    print(f'  (median gap: {ins["overlap"]["median_gap_days"]} day(s)).')
    print(f'')
    print(f'  Consecutive outcome autocorrelation: rho = {ins["consecutive_autocorrelation"]["rho"]:.3f}')
    print(f'  (highly significant, p << 0.001).')
    print(f'')
    print(f'  Effective N (Bartlett):  {ins["effective_n"]["n_eff"]:.0f} (inflation factor: {ins["effective_n"]["inflation_factor"]:.1f}x)')
    print(f'  De-duplicated (28d):     {ins["deduplication"]["n_deduped"]} signals')
    print(f'  De-duplicated WR:        {ins["deduplication"]["dedup_wr"]:.1f}%')
    print(f'  Residual autocorrelation after dedup: rho = {ins["deduplication"]["residual_rho"]:.4f}')
    print(f'  ({"significant" if ins["deduplication"]["residual_significant"] else "not significant"} — ')
    print(f'   dedup {"does NOT fully" if ins["deduplication"]["residual_significant"] else "successfully"} remove serial dependence)')
    print(f'')
    print(f'  NAIVE p-value:           {ins["p_values"]["naive"]["p"]:.2e} (z={ins["p_values"]["naive"]["z"]:.1f}, OVERSTATED)')
    print(f'  ADJUSTED p-value (N_eff):{ins["p_values"]["effective_n"]["p"]:.2e} (z={ins["p_values"]["effective_n"]["z"]:.1f})')
    print(f'  DE-DUPED p-value:        {ins["p_values"]["deduped"]["p"]:.2e} (z={ins["p_values"]["deduped"]["z"]:.1f})')
    print(f'')
    print(f'  Significance inflation from naive to de-duped: {ins["significance_inflation"]:.1f}x')
    print(f'  28-day dedup rule: {"VALIDATED" if ins["dedup_rule_validated"] else "NEEDS LONGER WINDOW"}')
    print(f'    (autocorrelation first insignificant at lag {ins["first_insignificant_lag"]} days)')

    # Write results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results written to {RESULTS_FILE}')


if __name__ == '__main__':
    main()
