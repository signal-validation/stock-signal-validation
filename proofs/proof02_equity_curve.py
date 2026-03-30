"""
Proof #02: Equity Curve + Drawdown Analysis
=============================================
The visual proof: cumulative returns of following every signal vs SPY.
Includes Sharpe ratio, max drawdown, Calmar ratio.

Usage:
  python3 proof02_equity_curve.py
"""

import json
import math
import os
from collections import defaultdict
from datetime import date as Date

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'proof02_equity_curve_results.json')
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


def main():
    print('Proof #02: Equity Curve + Drawdown Analysis')
    print('=' * 70)
    signals, ins_set = load_data()
    deduped = dedup_signals(signals, ins_set)
    print(f'De-duplicated in-sample signals: {len(deduped)}')

    # Group by month
    monthly = defaultdict(list)
    for s in deduped:
        month_key = s['date'][:7]  # YYYY-MM
        monthly[month_key].append(s)

    months_sorted = sorted(monthly.keys())
    print(f'Months with signals: {len(months_sorted)} ({months_sorted[0]} to {months_sorted[-1]})')

    # Compute monthly returns (average of signal returns in that month)
    strategy_equity = 100.0
    spy_equity = 100.0
    alpha_equity = 100.0
    peak_strategy = 100.0
    peak_alpha = 100.0

    curve = []
    monthly_returns = []
    monthly_alpha_returns = []
    max_dd_strategy = 0
    max_dd_alpha = 0
    dd_start = None
    longest_dd_months = 0
    current_dd_months = 0

    for month in months_sorted:
        sigs = monthly[month]
        n = len(sigs)

        # Average returns for this month's signals
        avg_ret = sum(s[RETURN_COL] for s in sigs) / n
        avg_spy = sum((s.get('spy_return_20d', 0) or 0) for s in sigs) / n
        avg_alpha = sum((s.get('alpha', 0) or 0) for s in sigs) / n

        # Compound
        strategy_equity *= (1 + avg_ret / 100)
        spy_equity *= (1 + avg_spy / 100)
        alpha_equity *= (1 + avg_alpha / 100)

        monthly_returns.append(avg_ret)
        monthly_alpha_returns.append(avg_alpha)

        # Drawdown tracking (strategy)
        if strategy_equity > peak_strategy:
            peak_strategy = strategy_equity
            current_dd_months = 0
        dd = (peak_strategy - strategy_equity) / peak_strategy * 100
        max_dd_strategy = max(max_dd_strategy, dd)
        if dd > 0:
            current_dd_months += 1
            longest_dd_months = max(longest_dd_months, current_dd_months)

        # Drawdown tracking (alpha)
        if alpha_equity > peak_alpha:
            peak_alpha = alpha_equity
        dd_alpha = (peak_alpha - alpha_equity) / peak_alpha * 100
        max_dd_alpha = max(max_dd_alpha, dd_alpha)

        curve.append({
            'month': month,
            'n_signals': n,
            'avg_return': round(avg_ret, 2),
            'avg_spy_return': round(avg_spy, 2),
            'avg_alpha': round(avg_alpha, 2),
            'strategy_equity': round(strategy_equity, 2),
            'spy_equity': round(spy_equity, 2),
            'alpha_equity': round(alpha_equity, 2),
            'drawdown_pct': round(dd, 2),
            'drawdown_alpha_pct': round(dd_alpha, 2),
        })

    # Risk metrics
    n_months = len(monthly_returns)
    n_years = n_months / 12

    total_return = (strategy_equity / 100 - 1) * 100
    spy_total = (spy_equity / 100 - 1) * 100
    alpha_total = (alpha_equity / 100 - 1) * 100

    cagr = ((strategy_equity / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    spy_cagr = ((spy_equity / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    alpha_cagr = ((alpha_equity / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    # Monthly volatility -> annualized
    mean_monthly = sum(monthly_returns) / n_months
    var_monthly = sum((r - mean_monthly)**2 for r in monthly_returns) / n_months
    monthly_vol = math.sqrt(var_monthly)
    annual_vol = monthly_vol * math.sqrt(12)

    mean_alpha_monthly = sum(monthly_alpha_returns) / n_months
    var_alpha_monthly = sum((r - mean_alpha_monthly)**2 for r in monthly_alpha_returns) / n_months
    alpha_monthly_vol = math.sqrt(var_alpha_monthly)
    alpha_annual_vol = alpha_monthly_vol * math.sqrt(12)

    # Sharpe (using monthly alpha returns, annualized)
    sharpe = (mean_alpha_monthly * 12) / alpha_annual_vol if alpha_annual_vol > 0 else 0

    # Sortino (downside deviation only)
    downside = [r for r in monthly_alpha_returns if r < 0]
    downside_vol = math.sqrt(sum(r**2 for r in downside) / len(downside)) * math.sqrt(12) if downside else 0
    sortino = (mean_alpha_monthly * 12) / downside_vol if downside_vol > 0 else 0

    # Calmar
    calmar = cagr / max_dd_strategy if max_dd_strategy > 0 else 0

    # Winning months
    winning_months = sum(1 for r in monthly_returns if r > 0)

    print(f'\n  EQUITY CURVE SUMMARY ({months_sorted[0]} to {months_sorted[-1]}):')
    print(f'    {"":25} {"Strategy":>12} {"SPY":>12} {"Alpha":>12}')
    print(f'    {"-"*63}')
    print(f'    {"Total return":25} {total_return:>+11.1f}% {spy_total:>+11.1f}% {alpha_total:>+11.1f}%')
    print(f'    {"CAGR":25} {cagr:>+11.2f}% {spy_cagr:>+11.2f}% {alpha_cagr:>+11.2f}%')
    print(f'    {"Final equity ($100)":25} {"$"+f"{strategy_equity:.0f}":>12} {"$"+f"{spy_equity:.0f}":>12} {"$"+f"{alpha_equity:.0f}":>12}')

    print(f'\n  RISK METRICS:')
    print(f'    Annual volatility:     {annual_vol:.1f}% (alpha: {alpha_annual_vol:.1f}%)')
    print(f'    Sharpe ratio (alpha):  {sharpe:.2f}')
    print(f'    Sortino ratio (alpha): {sortino:.2f}')
    print(f'    Max drawdown:          {max_dd_strategy:.1f}% (alpha: {max_dd_alpha:.1f}%)')
    print(f'    Calmar ratio:          {calmar:.2f}')
    print(f'    Longest drawdown:      {longest_dd_months} months')
    print(f'    Winning months:        {winning_months}/{n_months} ({winning_months/n_months*100:.0f}%)')

    # Year-by-year
    yearly = defaultdict(list)
    for s in deduped:
        yearly[int(s['date'][:4])].append(s)

    print(f'\n  YEAR-BY-YEAR RETURNS:')
    print(f'    {"Year":>6} {"N":>5} {"WR":>7} {"Return":>8} {"SPY":>8} {"Alpha":>8}')
    print(f'    {"-"*45}')
    yearly_data = []
    for yr in sorted(yearly.keys()):
        sigs = yearly[yr]
        n = len(sigs)
        wr = sum(1 for s in sigs if s[RETURN_COL] > 0) / n * 100
        ret = sum(s[RETURN_COL] for s in sigs) / n
        spy = sum((s.get('spy_return_20d', 0) or 0) for s in sigs) / n
        alpha = sum((s.get('alpha', 0) or 0) for s in sigs) / n
        print(f'    {yr:>6} {n:>5} {wr:>6.1f}% {ret:>+7.2f}% {spy:>+7.2f}% {alpha:>+7.2f}%')
        yearly_data.append({'year': yr, 'n': n, 'wr': round(wr, 1), 'return': round(ret, 2),
                           'spy_return': round(spy, 2), 'alpha': round(alpha, 2)})

    results = {
        'analysis': 'Equity Curve + Drawdown',
        'period': f'{months_sorted[0]} to {months_sorted[-1]}',
        'n_signals': len(deduped),
        'n_months': n_months,
        'n_years': round(n_years, 1),
        'summary': {
            'total_return': round(total_return, 1),
            'spy_total_return': round(spy_total, 1),
            'alpha_total_return': round(alpha_total, 1),
            'cagr': round(cagr, 2),
            'spy_cagr': round(spy_cagr, 2),
            'alpha_cagr': round(alpha_cagr, 2),
            'final_equity': round(strategy_equity, 0),
            'spy_final_equity': round(spy_equity, 0),
        },
        'risk': {
            'annual_volatility': round(annual_vol, 1),
            'alpha_annual_volatility': round(alpha_annual_vol, 1),
            'sharpe': round(sharpe, 2),
            'sortino': round(sortino, 2),
            'max_drawdown': round(max_dd_strategy, 1),
            'max_drawdown_alpha': round(max_dd_alpha, 1),
            'calmar': round(calmar, 2),
            'longest_drawdown_months': longest_dd_months,
            'winning_months': winning_months,
            'total_months': n_months,
        },
        'monthly_curve': curve,
        'yearly': yearly_data,
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results written to {RESULTS_FILE}')


if __name__ == '__main__':
    main()
