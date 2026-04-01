"""
Test #11: Per-Regime Carhart 4-Factor Regression
==================================================
Tests whether signal alpha persists separately in bear and bull market
regimes after controlling for market, size, value, and momentum factors.

Motivation:
  A common critique of accumulation-based signals is that apparent bear-market
  outperformance may simply reflect momentum crash avoidance rather than
  genuine regime-adaptive alpha. If the momentum factor (Mom) explains the
  bear-market edge, the Carhart alpha in the bear subsample will be zero.

Method:
  1. Classify each signal date as "bear" (market below 50-day moving average)
     or "bull" using the cumulative market return from Fama-French MktRF + RF.
  2. Run Carhart 4-factor regression separately on each subsample.
  3. Compare alpha, t-stat, and momentum loading between regimes.
  4. Run bear subsample at all 4 holding periods (5d, 10d, 20d, 40d).

Regime definition:
  Cumulative daily market return (MktRF + RF) below its trailing 50-day
  simple moving average. This is equivalent to "SPY < SMA50" but derived
  entirely from the public Fama-French factor data.

Data sources:
  - Fama-French 3 factors daily: Kenneth French Data Library
  - Momentum factor daily: Kenneth French Data Library
  - No external price data required (regime derived from factors)

Usage:
  python3 test11_regime_factor_regression.py
"""

import json
import math
import os
import io
import zipfile
import urllib.request
from collections import defaultdict
from datetime import date as Date, timedelta

import numpy as np

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CACHE_DIR = os.path.join(SCRIPT_DIR, 'data')
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'test11_regime_factor_regression_results.json')

DEDUP_DAYS = 28
REGIME_SMA_WINDOW = 50

HOLDING_PERIODS = {
    'return_5d': 5,
    'return_10d': 10,
    'return_20d': 20,
    'return_40d': 40,
}

FF3_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'
MOM_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip'
FF3_CACHE = os.path.join(CACHE_DIR, 'ff3_daily.json')
MOM_CACHE = os.path.join(CACHE_DIR, 'mom_daily.json')


# ============================================================
# FACTOR DATA (shared with test07)
# ============================================================
def download_and_parse_ff3():
    if os.path.exists(FF3_CACHE):
        with open(FF3_CACHE) as f:
            return json.load(f)
    print('  Downloading Fama-French 3-factor daily data...')
    req = urllib.request.Request(FF3_URL, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=30)
    z = zipfile.ZipFile(io.BytesIO(resp.read()))
    content = z.read(z.namelist()[0]).decode('utf-8')
    factors = {}
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 5:
            continue
        try:
            ds = parts[0]
            if len(ds) != 8:
                continue
            dk = f'{ds[:4]}-{ds[4:6]}-{ds[6:8]}'
            factors[dk] = {
                'MktRF': float(parts[1]), 'SMB': float(parts[2]),
                'HML': float(parts[3]), 'RF': float(parts[4]),
            }
        except (ValueError, IndexError):
            continue
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(FF3_CACHE, 'w') as f:
        json.dump(factors, f)
    return factors


def download_and_parse_mom():
    if os.path.exists(MOM_CACHE):
        with open(MOM_CACHE) as f:
            return json.load(f)
    print('  Downloading Momentum factor daily data...')
    req = urllib.request.Request(MOM_URL, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=30)
    z = zipfile.ZipFile(io.BytesIO(resp.read()))
    content = z.read(z.namelist()[0]).decode('utf-8')
    factors = {}
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 2:
            continue
        try:
            ds = parts[0]
            if len(ds) != 8:
                continue
            dk = f'{ds[:4]}-{ds[4:6]}-{ds[6:8]}'
            mom = float(parts[1])
            if mom == -99.99 or mom == -999:
                continue
            factors[dk] = {'Mom': mom}
        except (ValueError, IndexError):
            continue
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(MOM_CACHE, 'w') as f:
        json.dump(factors, f)
    return factors


# ============================================================
# REGIME CLASSIFICATION FROM FACTOR DATA
# ============================================================
def compute_market_regime(ff3):
    """
    Classify each trading day as bear or bull using cumulative market return
    relative to its 50-day SMA. Equivalent to SPY < SMA50 but derived
    entirely from Fama-French MktRF + RF (public data, no price feed needed).
    """
    dates = sorted(d for d in ff3.keys() if d >= '2005-01-01')
    cum = 100.0
    cum_vals = []
    cum_dates = []
    for d in dates:
        mkt_ret = (ff3[d]['MktRF'] + ff3[d]['RF']) / 100
        cum *= (1 + mkt_ret)
        cum_vals.append(cum)
        cum_dates.append(d)

    regime = {}  # date -> True if bear
    cum_arr = np.array(cum_vals)
    for i in range(REGIME_SMA_WINDOW - 1, len(cum_dates)):
        sma = np.mean(cum_arr[i - REGIME_SMA_WINDOW + 1:i + 1])
        regime[cum_dates[i]] = cum_arr[i] < sma  # True = bear

    return regime


# ============================================================
# DATA LOADING & DEDUP
# ============================================================
def parse_date(s):
    return Date(int(s[:4]), int(s[5:7]), int(s[8:10]))


def load_data():
    with open(os.path.join(DATA_DIR, 'signals_public.json')) as f:
        d = json.load(f)
    return d['daily']['signals']


def dedup_signals(signals):
    by_ticker = defaultdict(list)
    for s in signals:
        if s.get('return_20d') is None:
            continue
        by_ticker[s['ticker']].append(s)
    deduped = []
    for tk, sigs in by_ticker.items():
        sigs.sort(key=lambda x: x['date'])
        last = None
        for s in sigs:
            d = parse_date(s['date'])
            if last is None or (d - last).days > DEDUP_DAYS:
                deduped.append(s)
                last = d
    deduped.sort(key=lambda x: x['date'])
    return deduped


# ============================================================
# COMPOUND FACTOR RETURNS
# ============================================================
def compute_signal_factors(signals, ff3, mom, trading_dates, holding_days, return_col):
    date_idx = {d: i for i, d in enumerate(trading_dates)}
    records = []
    skipped = 0
    for s in signals:
        if s.get(return_col) is None:
            skipped += 1
            continue
        entry_date = s['date']
        if entry_date not in date_idx:
            found = False
            for offset in range(5):
                check = (parse_date(entry_date) + timedelta(days=offset)).isoformat()
                if check in date_idx:
                    entry_date = check
                    found = True
                    break
            if not found:
                skipped += 1
                continue
        start_idx = date_idx[entry_date]
        end_idx = start_idx + holding_days
        if end_idx >= len(trading_dates):
            skipped += 1
            continue
        cum_mktrf = cum_smb = cum_hml = cum_mom = cum_rf = 1.0
        valid_days = 0
        for idx in range(start_idx, end_idx):
            td = trading_dates[idx]
            ff = ff3.get(td)
            m = mom.get(td)
            if ff:
                cum_mktrf *= (1 + ff['MktRF'] / 100)
                cum_smb *= (1 + ff['SMB'] / 100)
                cum_hml *= (1 + ff['HML'] / 100)
                cum_rf *= (1 + ff['RF'] / 100)
                valid_days += 1
            if m:
                cum_mom *= (1 + m['Mom'] / 100)
        if valid_days < holding_days * 0.8:
            skipped += 1
            continue
        rf_ret = (cum_rf - 1) * 100
        records.append({
            'excess_return': s[return_col] - rf_ret,
            'MktRF': (cum_mktrf - 1) * 100,
            'SMB': (cum_smb - 1) * 100,
            'HML': (cum_hml - 1) * 100,
            'Mom': (cum_mom - 1) * 100,
        })
    return records, skipped


# ============================================================
# OLS WITH HC1 ROBUST SE
# ============================================================
def ols_robust(y, X, factor_names):
    n, k = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y)
    y_hat = X @ beta
    resid = y - y_hat
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    resid_sq = resid ** 2
    meat = X.T @ np.diag(resid_sq) @ X
    robust_var = XtX_inv @ meat @ XtX_inv * (n / (n - k))
    robust_se = np.sqrt(np.diag(robust_var))
    t_stats = beta / robust_se
    p_values = np.array([2 * 0.5 * math.erfc(abs(t) / math.sqrt(2)) for t in t_stats])
    results = {}
    names = ['alpha'] + factor_names
    for i, name in enumerate(names):
        results[name] = {
            'coef': round(float(beta[i]), 4),
            'se': round(float(robust_se[i]), 4),
            't_stat': round(float(t_stats[i]), 2),
            'p_value': float(p_values[i]),
        }
    return results, round(float(r2), 4), n


def run_carhart(records, label):
    if len(records) < 30:
        print(f'  {label}: INSUFFICIENT DATA (N={len(records)})')
        return None
    y = np.array([r['excess_return'] for r in records])
    mktrf = np.array([r['MktRF'] for r in records])
    smb = np.array([r['SMB'] for r in records])
    hml = np.array([r['HML'] for r in records])
    mom = np.array([r['Mom'] for r in records])
    ones = np.ones(len(records))
    X = np.column_stack([ones, mktrf, smb, hml, mom])
    coefs, r2, n = ols_robust(y, X, ['MktRF', 'SMB', 'HML', 'Mom'])
    return coefs, r2, n


def print_regression(coefs, r2, n, label):
    print(f'\n  {label} (N={n}, R2={r2})')
    for name in ['alpha', 'MktRF', 'SMB', 'HML', 'Mom']:
        c = coefs[name]
        sig = '***' if c['p_value'] < 0.01 else '**' if c['p_value'] < 0.05 else '*' if c['p_value'] < 0.10 else ''
        print(f'    {name:>6}: {c["coef"]:+.4f}  (se={c["se"]:.4f})  t={c["t_stat"]:>6.2f}  p={c["p_value"]:.4f} {sig}')


# ============================================================
# MAIN
# ============================================================
def main():
    print('Test #11: Per-Regime Carhart 4-Factor Regression')
    print('=' * 70)

    # Load factors
    print('\nStep 1: Loading factor data...')
    ff3 = download_and_parse_ff3()
    mom = download_and_parse_mom()
    trading_dates = sorted(d for d in (set(ff3.keys()) & set(mom.keys())) if d >= '2006-01-01')
    print(f'  Trading days (2006+): {len(trading_dates)}')

    # Compute regime
    print('\nStep 2: Computing market regime from factor data...')
    regime = compute_market_regime(ff3)
    bear_days = sum(1 for v in regime.values() if v and '2006' <= list(regime.keys())[0])
    total_days = len([d for d in regime if d >= '2006-01-01'])
    bear_pct = sum(1 for d in regime if d >= '2006-01-01' and regime[d]) / total_days * 100
    print(f'  Bear days (market < 50d SMA): {bear_pct:.1f}% of trading days')

    # Load and dedup signals
    print('\nStep 3: Loading signals...')
    signals = load_data()
    deduped = dedup_signals(signals)
    print(f'  De-duplicated: {len(deduped)}')

    # Split by regime
    bear_sigs = []
    bull_sigs = []
    unclassified = 0
    for s in deduped:
        # Find nearest regime classification
        found = False
        for offset in range(5):
            check = (parse_date(s['date']) - timedelta(days=offset)).isoformat()
            if check in regime:
                if regime[check]:
                    bear_sigs.append(s)
                else:
                    bull_sigs.append(s)
                found = True
                break
        if not found:
            unclassified += 1

    print(f'  Bear signals: {len(bear_sigs)}')
    print(f'  Bull signals: {len(bull_sigs)}')
    if unclassified > 0:
        print(f'  Unclassified: {unclassified}')

    # ============================================================
    # PART 1: 20-day Carhart on each regime
    # ============================================================
    print('\n' + '=' * 70)
    print('  PART 1: 20-DAY CARHART BY REGIME')
    print('=' * 70)

    all_results = {
        'test': 'Test #11: Per-Regime Carhart 4-Factor Regression',
        'description': (
            'Tests whether signal alpha persists separately in bear and bull regimes '
            'after controlling for market, size, value, and momentum factors. '
            'Bear regime defined as cumulative market return below 50-day SMA '
            '(derived from Fama-French MktRF + RF).'
        ),
        'regime_definition': f'Market cumulative return below {REGIME_SMA_WINDOW}-day SMA',
        'bear_signals': len(bear_sigs),
        'bull_signals': len(bull_sigs),
        'regimes': {},
    }

    for label, subset in [('ALL', deduped), ('BEAR', bear_sigs), ('BULL', bull_sigs)]:
        recs, skip = compute_signal_factors(subset, ff3, mom, trading_dates, 20, 'return_20d')
        result = run_carhart(recs, label)
        if result:
            coefs, r2, n = result
            print_regression(coefs, r2, n, f'{label} (20-day)')
            all_results['regimes'][label] = {
                'holding_days': 20,
                'coefficients': coefs,
                'r2': r2,
                'n': n,
            }

    # Comparison
    if 'BEAR' in all_results['regimes'] and 'BULL' in all_results['regimes']:
        ba = all_results['regimes']['BEAR']['coefficients']['alpha']
        bua = all_results['regimes']['BULL']['coefficients']['alpha']
        bm = all_results['regimes']['BEAR']['coefficients']['Mom']
        bum = all_results['regimes']['BULL']['coefficients']['Mom']

        print(f'\n  REGIME COMPARISON:')
        print(f'    Bear alpha: {ba["coef"]:+.4f}% (t={ba["t_stat"]:.2f}, p={ba["p_value"]:.4f})')
        print(f'    Bull alpha: {bua["coef"]:+.4f}% (t={bua["t_stat"]:.2f}, p={bua["p_value"]:.4f})')
        print(f'    Alpha spread: {ba["coef"] - bua["coef"]:+.4f}pp')
        print(f'    Bear Mom beta: {bm["coef"]:+.4f} (t={bm["t_stat"]:.2f}, p={bm["p_value"]:.4f})')
        print(f'    Bull Mom beta: {bum["coef"]:+.4f} (t={bum["t_stat"]:.2f}, p={bum["p_value"]:.4f})')

        all_results['comparison'] = {
            'alpha_spread_pp': round(ba['coef'] - bua['coef'], 4),
            'bear_alpha_significant': ba['p_value'] < 0.05,
            'bull_alpha_significant': bua['p_value'] < 0.05,
            'bear_mom_significant': bm['p_value'] < 0.05,
            'bull_mom_significant': bum['p_value'] < 0.05,
        }

    # ============================================================
    # PART 2: Bear alpha across holding periods
    # ============================================================
    print('\n' + '=' * 70)
    print('  PART 2: BEAR ALPHA ACROSS HOLDING PERIODS')
    print('=' * 70)

    all_results['bear_by_holding_period'] = {}
    for ret_col, hold_days in sorted(HOLDING_PERIODS.items(), key=lambda x: x[1]):
        valid = [s for s in bear_sigs if s.get(ret_col) is not None]
        recs, skip = compute_signal_factors(valid, ff3, mom, trading_dates, hold_days, ret_col)
        result = run_carhart(recs, f'BEAR {hold_days}d')
        if result:
            coefs, r2, n = result
            a = coefs['alpha']
            m = coefs['Mom']
            sig = '***' if a['p_value'] < 0.01 else '**' if a['p_value'] < 0.05 else '*' if a['p_value'] < 0.10 else 'ns'
            print(f'  {hold_days:>2}d: alpha={a["coef"]:+.4f}% t={a["t_stat"]:>5.2f} p={a["p_value"]:.4f} [{sig}]  |  Mom={m["coef"]:+.4f} t={m["t_stat"]:.2f}')
            all_results['bear_by_holding_period'][f'{hold_days}d'] = {
                'coefficients': coefs, 'r2': r2, 'n': n,
            }

    # ============================================================
    # VERDICT
    # ============================================================
    print('\n' + '=' * 70)
    print('  VERDICT')
    print('=' * 70)

    if 'BEAR' in all_results['regimes']:
        ba = all_results['regimes']['BEAR']['coefficients']['alpha']
        bm = all_results['regimes']['BEAR']['coefficients']['Mom']
        bear_survives = ba['p_value'] < 0.05
        mom_significant = bm['p_value'] < 0.05

        if bear_survives and not mom_significant:
            verdict = ('PASS -- Bear alpha survives factor control (p={:.4f}). '
                       'Momentum loading is NOT significant (p={:.4f}). '
                       'The bear-market outperformance is genuine alpha, '
                       'not momentum crash avoidance.').format(ba['p_value'], bm['p_value'])
        elif bear_survives and mom_significant:
            verdict = ('PARTIAL PASS -- Bear alpha survives (p={:.4f}) but momentum '
                       'loading is also significant (p={:.4f}). Alpha is genuine '
                       'but partially co-travels with momentum.').format(ba['p_value'], bm['p_value'])
        else:
            verdict = ('FAIL -- Bear alpha does NOT survive factor control (p={:.4f}). '
                       'The bear-market outperformance is explained by factor exposures, '
                       'likely momentum crash avoidance.').format(ba['p_value'])

        print(f'\n  {verdict}')
        all_results['verdict'] = verdict

    # Save
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n  Results saved to {RESULTS_FILE}')


if __name__ == '__main__':
    main()
