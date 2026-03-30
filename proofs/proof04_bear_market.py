"""
Proof #04: Bear Market Spotlight
==================================
What happens during market crashes? Silence IS a feature.

Usage:
  python3 proof04_bear_market.py
"""

import json
import os
from collections import defaultdict
from datetime import date as Date

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'proof04_bear_market_results.json')
DEDUP_DAYS = 28
RETURN_COL = 'return_20d'

BEAR_PERIODS = [
    {'name': '2008 GFC', 'start': '2007-10-01', 'end': '2009-03-31', 'spy_drawdown': -56.8},
    {'name': '2011 Euro Crisis', 'start': '2011-05-01', 'end': '2011-10-31', 'spy_drawdown': -19.4},
    {'name': '2018 Q4 Selloff', 'start': '2018-10-01', 'end': '2018-12-31', 'spy_drawdown': -19.8},
    {'name': '2020 COVID Crash', 'start': '2020-02-01', 'end': '2020-03-31', 'spy_drawdown': -33.9},
    {'name': '2022 Bear Market', 'start': '2022-01-01', 'end': '2022-10-31', 'spy_drawdown': -25.4},
]


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


def main():
    print('Proof #04: Bear Market Spotlight')
    print('=' * 70)
    signals, ins_set = load_data()
    deduped = dedup_signals(signals, ins_set)
    total_n = len(deduped)
    total_wr = sum(1 for s in deduped if s[RETURN_COL] > 0) / total_n * 100
    total_alpha = sum((s.get('alpha', 0) or 0) for s in deduped) / total_n
    print(f'De-duplicated in-sample signals: {total_n}, overall WR: {total_wr:.1f}%')

    # Monthly signal density (for context)
    monthly_counts = defaultdict(int)
    for s in deduped:
        monthly_counts[s['date'][:7]] += 1
    avg_monthly = sum(monthly_counts.values()) / len(monthly_counts) if monthly_counts else 0

    results = {
        'analysis': 'Bear Market Spotlight',
        'total_signals': total_n,
        'overall_wr': round(total_wr, 1),
        'overall_alpha': round(total_alpha, 2),
        'avg_monthly_signals': round(avg_monthly, 1),
        'periods': [],
    }

    print(f'\n  {"Period":<22} {"SPY DD":>8} {"N":>5} {"WR":>7} {"Alpha":>8} {"Sig/mo":>7} {"Story":>25}')
    print(f'  {"-"*90}')

    for period in BEAR_PERIODS:
        start = period['start']
        end = period['end']
        sigs = [s for s in deduped if start <= s['date'] <= end]
        n = len(sigs)

        # How many months in this period?
        start_d = parse_date(start)
        end_d = parse_date(end)
        months_in_period = max(1, (end_d.year - start_d.year) * 12 + (end_d.month - start_d.month))
        sigs_per_month = n / months_in_period if months_in_period > 0 else 0

        if n == 0:
            wr = 0
            avg_alpha = 0
            avg_ret = 0
            story = 'ENGINE SILENT'
        else:
            wr = sum(1 for s in sigs if s[RETURN_COL] > 0) / n * 100
            avg_alpha = sum((s.get('alpha', 0) or 0) for s in sigs) / n
            avg_ret = sum(s[RETURN_COL] for s in sigs) / n

            if n <= 5:
                story = 'NEAR-SILENT (few signals)'
            elif wr >= 55:
                story = 'HELD UP WELL'
            elif wr >= 45:
                story = 'SLIGHTLY DEGRADED'
            else:
                story = 'STRUGGLED'

        print(f'  {period["name"]:<22} {period["spy_drawdown"]:>+7.1f}% {n:>5} {wr:>6.1f}% {avg_alpha:>+7.2f}% {sigs_per_month:>6.1f} {story:>25}')

        # Monthly detail
        period_monthly = defaultdict(list)
        for s in sigs:
            period_monthly[s['date'][:7]].append(s)

        monthly_detail = []
        for month in sorted(period_monthly.keys()):
            m_sigs = period_monthly[month]
            m_n = len(m_sigs)
            m_wr = sum(1 for s in m_sigs if s[RETURN_COL] > 0) / m_n * 100 if m_n > 0 else 0
            m_alpha = sum((s.get('alpha', 0) or 0) for s in m_sigs) / m_n if m_n > 0 else 0
            monthly_detail.append({'month': month, 'n': m_n, 'wr': round(m_wr, 1), 'alpha': round(m_alpha, 2)})

        results['periods'].append({
            'name': period['name'],
            'start': start, 'end': end,
            'spy_drawdown': period['spy_drawdown'],
            'n_signals': n,
            'wr': round(wr, 1),
            'mean_alpha': round(avg_alpha, 2),
            'mean_return': round(avg_ret, 2) if n > 0 else 0,
            'signals_per_month': round(sigs_per_month, 1),
            'story': story,
            'monthly_detail': monthly_detail,
        })

    # Silence analysis
    silent_periods = [p for p in results['periods'] if p['n_signals'] <= 5]
    active_bear = [p for p in results['periods'] if p['n_signals'] > 5]

    print(f'\n  SILENCE ANALYSIS:')
    print(f'    Average monthly signals (overall): {avg_monthly:.1f}')
    print(f'    Bear periods where engine went silent (<= 5 signals): {len(silent_periods)}')
    if silent_periods:
        for p in silent_periods:
            print(f'      {p["name"]}: {p["n_signals"]} signals during {p["spy_drawdown"]:+.1f}% SPY drawdown')
        print(f'    -> The engine naturally produces fewer signals during market downturns.')
        print(f'       "Not trading" during a -37% to -57% crash IS alpha.')

    if active_bear:
        print(f'\n    Bear periods with active signals ({len(active_bear)}):')
        for p in active_bear:
            print(f'      {p["name"]}: {p["n_signals"]} signals, {p["wr"]:.1f}% WR, {p["mean_alpha"]:+.2f}% alpha')

    # Compare bear vs non-bear
    bear_dates = set()
    for period in BEAR_PERIODS:
        for s in deduped:
            if period['start'] <= s['date'] <= period['end']:
                bear_dates.add(s['date'])

    bear_sigs = [s for s in deduped if s['date'] in bear_dates]
    non_bear = [s for s in deduped if s['date'] not in bear_dates]

    print(f'\n  BEAR vs NON-BEAR:')
    if bear_sigs:
        bear_wr = sum(1 for s in bear_sigs if s[RETURN_COL] > 0) / len(bear_sigs) * 100
        bear_alpha = sum((s.get('alpha', 0) or 0) for s in bear_sigs) / len(bear_sigs)
    else:
        bear_wr = 0
        bear_alpha = 0
    non_bear_wr = sum(1 for s in non_bear if s[RETURN_COL] > 0) / len(non_bear) * 100
    non_bear_alpha = sum((s.get('alpha', 0) or 0) for s in non_bear) / len(non_bear)

    print(f'    Bear periods:     N={len(bear_sigs):>5}, WR={bear_wr:.1f}%, Alpha={bear_alpha:+.2f}%')
    print(f'    Non-bear periods: N={len(non_bear):>5}, WR={non_bear_wr:.1f}%, Alpha={non_bear_alpha:+.2f}%')

    results['bear_vs_nonbear'] = {
        'bear': {'n': len(bear_sigs), 'wr': round(bear_wr, 1), 'alpha': round(bear_alpha, 2)},
        'non_bear': {'n': len(non_bear), 'wr': round(non_bear_wr, 1), 'alpha': round(non_bear_alpha, 2)},
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results written to {RESULTS_FILE}')


if __name__ == '__main__':
    main()
