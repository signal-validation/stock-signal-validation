"""
Proof #03: Year-by-Year Performance
=====================================
Shows consistency: 20 bars of annual WR. Bull/bear/flat split.

Usage:
  python3 proof03_yearly_performance.py
"""

import json
import math
import os
from collections import defaultdict
from datetime import date as Date

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'proof03_yearly_performance_results.json')
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


# Approximate annual SPY returns (total return, %)
SPY_ANNUAL = {
    2006: 15.8, 2007: 5.5, 2008: -37.0, 2009: 26.5, 2010: 15.1,
    2011: 2.1, 2012: 16.0, 2013: 32.4, 2014: 13.7, 2015: 1.4,
    2016: 12.0, 2017: 21.8, 2018: -4.4, 2019: 31.5, 2020: 18.4,
    2021: 28.7, 2022: -18.1, 2023: 26.3, 2024: 25.0, 2025: 2.0,
}


def main():
    print('Proof #03: Year-by-Year Performance')
    print('=' * 70)
    signals, ins_set = load_data()
    deduped = dedup_signals(signals, ins_set)
    print(f'De-duplicated in-sample signals: {len(deduped)}')

    yearly = defaultdict(list)
    for s in deduped:
        yearly[int(s['date'][:4])].append(s)

    years = sorted(yearly.keys())
    yearly_data = []

    print(f'\n  {"Year":>6} {"N":>5} {"WR":>7} {"Alpha":>8} {"SPY":>8} {"Market":>8} {"Beat 50%":>9}')
    print(f'  {"-"*58}')

    above_50 = 0
    above_55 = 0
    above_60 = 0

    for yr in years:
        sigs = yearly[yr]
        n = len(sigs)
        wins = sum(1 for s in sigs if s[RETURN_COL] > 0)
        wr = wins / n * 100
        avg_alpha = sum((s.get('alpha', 0) or 0) for s in sigs) / n
        avg_ret = sum(s[RETURN_COL] for s in sigs) / n
        avg_spy_sig = sum((s.get('spy_return_20d', 0) or 0) for s in sigs) / n
        spy_annual = SPY_ANNUAL.get(yr, 0)

        if spy_annual > 10:
            market = 'BULL'
        elif spy_annual < -5:
            market = 'BEAR'
        else:
            market = 'FLAT'

        beat = 'YES' if wr > 50 else 'NO'
        if wr > 50: above_50 += 1
        if wr > 55: above_55 += 1
        if wr > 60: above_60 += 1

        print(f'  {yr:>6} {n:>5} {wr:>6.1f}% {avg_alpha:>+7.2f}% {spy_annual:>+7.1f}% {market:>8} {beat:>9}')

        yearly_data.append({
            'year': yr, 'n': n, 'wins': wins, 'wr': round(wr, 1),
            'mean_alpha': round(avg_alpha, 2), 'mean_return': round(avg_ret, 2),
            'spy_annual_return': spy_annual, 'market_condition': market,
            'beat_50': wr > 50,
        })

    n_years = len(years)
    print(f'\n  CONSISTENCY:')
    print(f'    Years above 50% WR: {above_50}/{n_years} ({above_50/n_years*100:.0f}%)')
    print(f'    Years above 55% WR: {above_55}/{n_years} ({above_55/n_years*100:.0f}%)')
    print(f'    Years above 60% WR: {above_60}/{n_years} ({above_60/n_years*100:.0f}%)')

    # Bull/bear/flat
    bull_sigs = [s for s in deduped if SPY_ANNUAL.get(int(s['date'][:4]), 0) > 10]
    bear_sigs = [s for s in deduped if SPY_ANNUAL.get(int(s['date'][:4]), 0) < -5]
    flat_sigs = [s for s in deduped if -5 <= SPY_ANNUAL.get(int(s['date'][:4]), 0) <= 10]

    print(f'\n  MARKET CONDITION SPLIT:')
    for label, sigs in [('Bull (SPY>10%)', bull_sigs), ('Bear (SPY<-5%)', bear_sigs), ('Flat', flat_sigs)]:
        if not sigs:
            continue
        n = len(sigs)
        wr = sum(1 for s in sigs if s[RETURN_COL] > 0) / n * 100
        alpha = sum((s.get('alpha', 0) or 0) for s in sigs) / n
        print(f'    {label:<20} N={n:>5}, WR={wr:.1f}%, Alpha={alpha:+.2f}%')

    # Best/worst years
    best = max(yearly_data, key=lambda x: x['wr'])
    worst_with_data = [y for y in yearly_data if y['n'] >= 10]
    worst = min(worst_with_data, key=lambda x: x['wr']) if worst_with_data else None

    print(f'\n  BEST YEAR:  {best["year"]} — {best["wr"]:.1f}% WR on {best["n"]} signals (SPY {best["spy_annual_return"]:+.1f}%)')
    if worst:
        print(f'  WORST YEAR: {worst["year"]} — {worst["wr"]:.1f}% WR on {worst["n"]} signals (SPY {worst["spy_annual_return"]:+.1f}%)')

    results = {
        'analysis': 'Year-by-Year Performance',
        'n_signals': len(deduped),
        'n_years': n_years,
        'consistency': {
            'above_50': above_50, 'above_55': above_55, 'above_60': above_60,
            'pct_above_50': round(above_50/n_years*100, 0),
        },
        'yearly': yearly_data,
        'condition_split': {
            'bull': {'n': len(bull_sigs), 'wr': round(sum(1 for s in bull_sigs if s[RETURN_COL]>0)/len(bull_sigs)*100, 1) if bull_sigs else 0},
            'bear': {'n': len(bear_sigs), 'wr': round(sum(1 for s in bear_sigs if s[RETURN_COL]>0)/len(bear_sigs)*100, 1) if bear_sigs else 0},
            'flat': {'n': len(flat_sigs), 'wr': round(sum(1 for s in flat_sigs if s[RETURN_COL]>0)/len(flat_sigs)*100, 1) if flat_sigs else 0},
        },
        'best_year': best,
        'worst_year': worst,
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results written to {RESULTS_FILE}')


if __name__ == '__main__':
    main()
