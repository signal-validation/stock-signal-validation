"""
Proof #05: Return Distribution + Risk Profile
===============================================
Win/loss asymmetry, tail risk, expected value, Kelly criterion.

Usage:
  python3 proof05_return_distribution.py
"""

import json
import math
import os
from collections import defaultdict
from datetime import date as Date

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'proof05_return_distribution_results.json')
DEDUP_DAYS = 28
RETURN_COL = 'return_20d'


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


def percentile(sorted_vals, pct):
    idx = int(len(sorted_vals) * pct / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def main():
    print('Proof #05: Return Distribution + Risk Profile')
    print('=' * 70)
    signals, ins_set = load_data()
    deduped = dedup_signals(signals, ins_set)
    print(f'De-duplicated in-sample signals: {len(deduped)}')

    returns = sorted([s[RETURN_COL] for s in deduped])
    alphas = sorted([(s.get('alpha', 0) or 0) for s in deduped])
    n = len(returns)

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    # === DISTRIBUTION ===
    buckets = [(-999, -20), (-20, -10), (-10, -5), (-5, -2), (-2, 0), (0, 2), (2, 5), (5, 10), (10, 20), (20, 999)]
    bucket_labels = ['<-20%', '-20 to -10%', '-10 to -5%', '-5 to -2%', '-2 to 0%',
                     '0 to 2%', '2 to 5%', '5 to 10%', '10 to 20%', '>20%']

    hist_return = {}
    hist_alpha = {}
    for (lo, hi), label in zip(buckets, bucket_labels):
        hist_return[label] = sum(1 for r in returns if lo <= r < hi)
        hist_alpha[label] = sum(1 for a in alphas if lo <= a < hi)

    print(f'\n  RETURN DISTRIBUTION (20-day forward returns):')
    print(f'    {"Bucket":<16} {"Count":>6} {"%":>7} {"||":>3}')
    print(f'    {"-"*35}')
    for label in bucket_labels:
        count = hist_return[label]
        pct = count / n * 100
        bar = '#' * int(pct / 2)
        print(f'    {label:<16} {count:>6} {pct:>6.1f}% {bar}')

    # === WIN/LOSS ASYMMETRY ===
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    median_win = sorted(wins)[len(wins)//2] if wins else 0
    median_loss = sorted(losses)[len(losses)//2] if losses else 0
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    wr = len(wins) / n * 100

    print(f'\n  WIN/LOSS ASYMMETRY:')
    print(f'    Win rate:         {wr:.1f}% ({len(wins)} wins, {len(losses)} losses)')
    print(f'    Average win:      +{avg_win:.2f}%')
    print(f'    Average loss:     {avg_loss:.2f}%')
    print(f'    Win/loss ratio:   {wl_ratio:.2f}x')
    print(f'    Median win:       +{median_win:.2f}%')
    print(f'    Median loss:      {median_loss:.2f}%')

    # === TAIL ANALYSIS ===
    worst_5pct = returns[:int(n * 0.05)]
    best_5pct = returns[int(n * 0.95):]
    worst_1pct = returns[:int(n * 0.01)]
    best_1pct = returns[int(n * 0.99):]

    var_95 = percentile(returns, 5)  # 5th percentile = 95% VaR
    var_99 = percentile(returns, 1)
    cvar_95 = sum(worst_5pct) / len(worst_5pct) if worst_5pct else 0

    print(f'\n  TAIL RISK ANALYSIS:')
    print(f'    Worst 5% of signals:  avg return = {sum(worst_5pct)/len(worst_5pct):.2f}%')
    print(f'    Best 5% of signals:   avg return = +{sum(best_5pct)/len(best_5pct):.2f}%')
    print(f'    Worst signal:         {returns[0]:.2f}%')
    print(f'    Best signal:          +{returns[-1]:.2f}%')
    print(f'    VaR (95%):            {var_95:.2f}% (5% of trades lose more than this)')
    print(f'    VaR (99%):            {var_99:.2f}% (1% of trades lose more than this)')
    print(f'    CVaR/ES (95%):        {cvar_95:.2f}% (avg loss in worst 5%)')

    # === POSITIVE EXPECTANCY ===
    ev = sum(returns) / n
    ev_alpha = sum(alphas) / n

    # Kelly criterion: f* = (p*b - q) / b where b = avg_win/avg_loss, p = WR, q = 1-WR
    p = len(wins) / n
    q = 1 - p
    b = abs(avg_win / avg_loss) if avg_loss != 0 else 1
    kelly = (p * b - q) / b if b > 0 else 0
    kelly = max(0, kelly)  # Can't be negative

    print(f'\n  POSITIVE EXPECTANCY:')
    print(f'    Expected value per trade:  +{ev:.2f}%')
    print(f'    Expected alpha per trade:  +{ev_alpha:.2f}%')
    print(f'    Kelly optimal fraction:    {kelly*100:.1f}% of capital per trade')
    print(f'    Half-Kelly (conservative): {kelly*50:.1f}% per trade')
    print(f'')
    print(f'    At half-Kelly on $50,000 portfolio:')
    trades_per_year = 120  # ~10/month
    annual_ev = ev * trades_per_year * kelly * 0.5
    print(f'      Position size: ${50000 * kelly * 0.5:.0f} per trade')
    print(f'      ~{trades_per_year} trades/year')
    print(f'      Expected annual gain: ~${50000 * annual_ev / 100:.0f}')


    # === SUMMARY STATS ===
    mean_ret = sum(returns) / n
    std_ret = math.sqrt(sum((r - mean_ret)**2 for r in returns) / n)
    skew = sum((r - mean_ret)**3 for r in returns) / (n * std_ret**3) if std_ret > 0 else 0
    kurt = sum((r - mean_ret)**4 for r in returns) / (n * std_ret**4) - 3 if std_ret > 0 else 0

    print(f'\n  DISTRIBUTION STATISTICS:')
    print(f'    Mean:     +{mean_ret:.2f}%')
    print(f'    Std Dev:  {std_ret:.2f}%')
    print(f'    Skewness: {skew:+.3f} ({"right-skewed" if skew > 0.2 else "left-skewed" if skew < -0.2 else "symmetric"})')
    print(f'    Kurtosis: {kurt:+.3f} ({"fat tails" if kurt > 1 else "thin tails" if kurt < -1 else "normal tails"})')
    print(f'    Median:   +{percentile(returns, 50):.2f}%')
    print(f'    P10:      {percentile(returns, 10):.2f}%')
    print(f'    P90:      +{percentile(returns, 90):.2f}%')

    results = {
        'analysis': 'Return Distribution + Risk Profile',
        'n_signals': n,
        'histogram_returns': hist_return,
        'histogram_alpha': hist_alpha,
        'win_loss': {
            'wr': round(wr, 1), 'n_wins': len(wins), 'n_losses': len(losses),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'median_win': round(median_win, 2), 'median_loss': round(median_loss, 2),
            'wl_ratio': round(wl_ratio, 2),
        },
        'tail_risk': {
            'var_95': round(var_95, 2), 'var_99': round(var_99, 2),
            'cvar_95': round(cvar_95, 2),
            'worst_signal': round(returns[0], 2), 'best_signal': round(returns[-1], 2),
        },
        'expectancy': {
            'ev_per_trade': round(ev, 2), 'ev_alpha_per_trade': round(ev_alpha, 2),
            'kelly_fraction': round(kelly, 4), 'half_kelly': round(kelly * 0.5, 4),
        },
        'distribution_stats': {
            'mean': round(mean_ret, 2), 'std': round(std_ret, 2),
            'skewness': round(skew, 3), 'kurtosis': round(kurt, 3),
            'median': round(percentile(returns, 50), 2),
            'p10': round(percentile(returns, 10), 2), 'p90': round(percentile(returns, 90), 2),
        },
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results written to {RESULTS_FILE}')


if __name__ == '__main__':
    main()
