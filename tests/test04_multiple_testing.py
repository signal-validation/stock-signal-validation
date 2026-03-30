"""
Test #04: Multiple Testing Correction
======================================
Corrects for the fact that we iterated through multiple engine configurations
to arrive at the current version. Without correction,
reporting the best result among K attempts inflates significance.

Methods:
  1. Bonferroni: p_adj = min(1, p_raw * K). Controls FWER. Most conservative.
  2. Sidak: p_adj = 1 - (1 - p_raw)^K. Exact for independent tests.
  3. Critical K: The K that would make p_adj = 0.05.
     "How many configurations would we have needed to try?"

Applied at three levels of conservatism:
  - K=8:  Engine versions that changed signal logic
  - K=20: Including hyperparameter sweeps
  - K=100: Extreme — assume far more exploration than recorded

Usage:
  python3 test04_multiple_testing.py
"""

import json
import math
import os
from collections import defaultdict
from datetime import date as Date

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'test04_multiple_testing_results.json')

DEDUP_DAYS = 28
RETURN_COL = 'return_20d'

# Engine versions that changed buy signal quality
VERSION_HISTORY = {f'cfg{i}': f'Configuration {i}' for i in range(1, 14)}

# K values to test
K_LEVELS = {
    'conservative_8': {
        'K': 8,
        'description': 'Engine configurations that changed signal logic (K=13)',
    },
    'aggressive_20': {
        'K': 20,
        'description': 'Including parameter sweeps and hyperparameter search',
    },
    'extreme_100': {
        'K': 100,
        'description': 'Assume extensive unreported exploration',
    },
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


def dedup_and_compute(signals, ticker_set):
    """De-duplicate and compute WR + p-value."""
    by_ticker = defaultdict(list)
    for s in signals:
        if s['ticker'] not in ticker_set:
            continue
        if s[RETURN_COL] is None:
            continue
        by_ticker[s['ticker']].append({
            'date_obj': parse_date(s['date']),
            'won': 1 if s[RETURN_COL] > 0 else 0,
        })

    deduped = []
    for tk, sigs in by_ticker.items():
        sigs.sort(key=lambda x: x['date_obj'])
        last_taken = None
        for s in sigs:
            if last_taken is None or (s['date_obj'] - last_taken).days > DEDUP_DAYS:
                deduped.append(s)
                last_taken = s['date_obj']

    n = len(deduped)
    wins = sum(s['won'] for s in deduped)
    wr = wins / n * 100 if n > 0 else 0

    # Z-score and p-value (one-sided, H0: WR=50%)
    se = math.sqrt(0.5 * 0.5 / n) if n > 0 else 1
    z = (wr / 100 - 0.5) / se
    p = 0.5 * math.erfc(z / math.sqrt(2)) if z > 0 else 1.0

    return n, wins, wr, z, p


# ============================================================
# CORRECTION METHODS
# ============================================================
def bonferroni(p, K):
    """Bonferroni: p_adj = min(1, p * K)"""
    return min(1.0, p * K)


def sidak(p, K):
    """Sidak: p_adj = 1 - (1 - p)^K"""
    if p >= 1.0:
        return 1.0
    if p <= 0.0:
        return 0.0
    # For very small p, use log approximation to avoid numerical issues
    if p < 1e-15:
        # 1 - (1-p)^K ≈ K*p for small p
        return min(1.0, K * p)
    return 1.0 - (1.0 - p) ** K


def critical_k(p, alpha=0.05):
    """Find K such that Bonferroni-adjusted p = alpha."""
    if p <= 0:
        return float('inf')
    return alpha / p


# ============================================================
# MAIN
# ============================================================
def main():
    print('Test #04: Multiple Testing Correction')
    print('=' * 70)
    print('Loading data...')

    signals, ins_set, exp_set = load_data()

    # Compute raw p-values
    ins_n, ins_wins, ins_wr, ins_z, ins_p = dedup_and_compute(signals, ins_set)
    exp_n, exp_wins, exp_wr, exp_z, exp_p = dedup_and_compute(signals, exp_set)

    print(f'\nRAW RESULTS (de-duplicated, {DEDUP_DAYS}-day rule):')
    print(f'  In-sample:  N={ins_n}, WR={ins_wr:.1f}%, z={ins_z:.2f}, p={ins_p:.2e}')
    print(f'  Expansion:  N={exp_n}, WR={exp_wr:.1f}%, z={exp_z:.2f}, p={exp_p:.2e}')

    # Version history
    print(f'\nENGINE VERSION HISTORY ({len(VERSION_HISTORY)} buy-engine versions):')
    for ver, desc in VERSION_HISTORY.items():
        print(f'  {ver}: {desc}')

    # Apply corrections
    results_table = []

    print(f'\n{"="*70}')
    print(f'  MULTIPLE TESTING CORRECTIONS')
    print(f'{"="*70}')

    for level_name, level_info in K_LEVELS.items():
        K = level_info['K']
        desc = level_info['description']

        ins_bonf = bonferroni(ins_p, K)
        ins_sidak = sidak(ins_p, K)
        exp_bonf = bonferroni(exp_p, K)
        exp_sidak = sidak(exp_p, K)

        print(f'\n  K = {K} ({desc})')
        print(f'  {"":25} {"Bonferroni":>14} {"Sidak":>14} {"Still sig?":>12}')
        print(f'  {"-"*68}')
        print(f'  {"In-sample  (p=" + f"{ins_p:.2e})":25} {ins_bonf:>14.2e} {ins_sidak:>14.2e} {"YES" if ins_bonf < 0.05 else "NO":>12}')
        print(f'  {"Expansion  (p=" + f"{exp_p:.2e})":25} {exp_bonf:>14.2e} {exp_sidak:>14.2e} {"YES" if exp_bonf < 0.05 else "NO":>12}')

        results_table.append({
            'level': level_name,
            'K': K,
            'description': desc,
            'in_sample': {
                'raw_p': ins_p,
                'bonferroni': ins_bonf,
                'sidak': ins_sidak,
                'significant_at_005': ins_bonf < 0.05,
            },
            'expansion': {
                'raw_p': exp_p,
                'bonferroni': exp_bonf,
                'sidak': exp_sidak,
                'significant_at_005': exp_bonf < 0.05,
            },
        })

    # Critical K analysis
    ins_crit_k = critical_k(ins_p)
    exp_crit_k = critical_k(exp_p)

    print(f'\n  CRITICAL K (Bonferroni threshold = 0.05):')
    print(f'    In-sample:  K_crit = {ins_crit_k:.2e}')
    print(f'      -> You would need to have tried {ins_crit_k:.0e} configurations')
    print(f'         to make the in-sample result non-significant.')
    print(f'    Expansion:  K_crit = {exp_crit_k:.0f}')
    print(f'      -> You would need to have tried {exp_crit_k:.0f} configurations')
    print(f'         to make the expansion result non-significant.')

    # Extended K sweep
    print(f'\n  EXTENDED K SWEEP (expansion, Bonferroni):')
    print(f'    {"K":>10} {"p_adj":>12} {"Significant?":>14}')
    print(f'    {"-"*38}')
    for K in [1, 5, 8, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
        p_adj = bonferroni(exp_p, K)
        print(f'    {K:>10} {p_adj:>12.4e} {"YES" if p_adj < 0.05 else "NO":>14}')

    # Verdict
    max_k_tested = max(level['K'] for level in K_LEVELS.values())
    all_survive_insample = all(
        r['in_sample']['significant_at_005'] for r in results_table
    )
    all_survive_expansion = all(
        r['expansion']['significant_at_005'] for r in results_table
    )

    if all_survive_insample and all_survive_expansion:
        verdict = f'STRONG PASS -- both cuts survive correction at all K levels (up to K={max_k_tested})'
    elif all_survive_insample:
        verdict = f'PARTIAL PASS -- in-sample survives all corrections; expansion fails at K>={exp_crit_k:.0f}'
    else:
        verdict = 'FAIL -- corrections invalidate significance'

    print(f'\n  VERDICT: {verdict}')

    # Assemble results
    results = {
        'test': 'Test #04: Multiple Testing Correction',
        'description': (
            'Corrects de-duplicated p-values for the number of engine configurations '
            'tested. Applies Bonferroni and Sidak corrections at K=8/20/100. '
            'Reports critical K: how many tests would invalidate significance.'
        ),
        'methodology': {
            'corrections': ['Bonferroni (FWER)', 'Sidak (exact for independent tests)'],
            'significance_threshold': 0.05,
            'version_history': VERSION_HISTORY,
            'dedup_days': DEDUP_DAYS,
            'return_column': RETURN_COL,
        },
        'raw_results': {
            'in_sample': {'n': ins_n, 'wins': ins_wins, 'wr': round(ins_wr, 2), 'z': round(ins_z, 2), 'p': ins_p},
            'expansion': {'n': exp_n, 'wins': exp_wins, 'wr': round(exp_wr, 2), 'z': round(exp_z, 2), 'p': exp_p},
        },
        'corrections': results_table,
        'critical_k': {
            'in_sample': ins_crit_k,
            'expansion': round(exp_crit_k, 0),
            'interpretation_in_sample': f'Need {ins_crit_k:.0e} configurations to invalidate',
            'interpretation_expansion': f'Need {exp_crit_k:.0f} configurations to invalidate',
        },
        'verdict': verdict,
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  Results written to {RESULTS_FILE}')


if __name__ == '__main__':
    main()
