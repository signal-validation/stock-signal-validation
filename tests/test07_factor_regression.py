"""
Test #07: Fama-MacBeth Factor Regression
==========================================
Tests whether the engine's alpha is explained by known risk factors
(market, size, value, momentum) or is genuinely novel.

Method:
  1. Download Fama-French 3-factor + Momentum daily data from
     Kenneth French's Data Library (official CRSP-based source).
  2. For each de-duplicated signal, compound daily factor returns
     over the exact 20 trading-day holding period.
  3. Pooled OLS regression:
       excess_return = a + b1*MktRF + b2*SMB + b3*HML + b4*Mom + e
     with White's HC1 heteroskedasticity-robust standard errors.
  4. The intercept (alpha) is the return not explained by any factor.
  5. Run CAPM (1-factor), FF3 (3-factor), and Carhart (4-factor) models.
  6. Run on in-sample and expansion separately.

Data sources:
  - Fama-French 3 factors daily: Kenneth French Data Library (Dartmouth)
  - Momentum factor daily: Kenneth French Data Library
  - Trading calendar: SPY.json candle dates

Usage:
  python3 test07_factor_regression.py
"""

import json
import math
import os
import io
import zipfile
import urllib.request
from collections import defaultdict
from datetime import date as Date

import numpy as np

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CANDLE_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CACHE_DIR = os.path.join(SCRIPT_DIR, 'data')
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'test07_factor_regression_results.json')

DEDUP_DAYS = 28
HOLDING_DAYS = 20  # Trading days
RETURN_COL = 'return_20d'

FF3_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'
MOM_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip'

FF3_CACHE = os.path.join(CACHE_DIR, 'ff3_daily.json')
MOM_CACHE = os.path.join(CACHE_DIR, 'mom_daily.json')


# ============================================================
# FACTOR DATA DOWNLOAD & PARSING
# ============================================================
def download_and_parse_ff3():
    """Download and parse Fama-French 3-factor daily data."""
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
            date_str = parts[0]
            if len(date_str) != 8:
                continue
            date_key = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}'
            mkt_rf = float(parts[1])
            smb = float(parts[2])
            hml = float(parts[3])
            rf = float(parts[4])
            factors[date_key] = {'MktRF': mkt_rf, 'SMB': smb, 'HML': hml, 'RF': rf}
        except (ValueError, IndexError):
            continue

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(FF3_CACHE, 'w') as f:
        json.dump(factors, f)
    print(f'    Cached {len(factors)} daily observations to {FF3_CACHE}')
    return factors


def download_and_parse_mom():
    """Download and parse Momentum factor daily data."""
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
            date_str = parts[0]
            if len(date_str) != 8:
                continue
            date_key = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}'
            mom = float(parts[1])
            if mom == -99.99 or mom == -999:
                continue
            factors[date_key] = {'Mom': mom}
        except (ValueError, IndexError):
            continue

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(MOM_CACHE, 'w') as f:
        json.dump(factors, f)
    print(f'    Cached {len(factors)} daily observations to {MOM_CACHE}')
    return factors


# ============================================================
# TRADING CALENDAR FROM SPY
# ============================================================
def load_trading_calendar():
    """Build trading calendar from Fama-French factor dates (no SPY candles needed)."""
    # Use FF3 daily dates as trading calendar (these are all NYSE trading days)
    if os.path.exists(FF3_CACHE):
        with open(FF3_CACHE) as f:
            ff3 = json.load(f)
        dates = sorted(ff3.keys())
        # Filter to 2006+ (our signal range)
        dates = [d for d in dates if d >= '2006-01-01']
        return dates
    return []


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
    """De-duplicate and return list with all fields."""
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
    return deduped


# ============================================================
# COMPOUND FACTOR RETURNS OVER HOLDING WINDOW
# ============================================================
def compute_signal_factors(deduped, ff3, mom, trading_dates):
    """For each signal, compound factor returns over 20 trading days."""
    # Build date->index map for trading calendar
    date_idx = {d: i for i, d in enumerate(trading_dates)}

    records = []
    skipped = 0

    for s in deduped:
        entry_date = s['date']

        # Find entry in trading calendar (exact or next trading day)
        if entry_date not in date_idx:
            # Find nearest trading day at or after entry
            found = False
            for offset in range(5):
                d_obj = parse_date(entry_date)
                from datetime import timedelta
                check = (d_obj + timedelta(days=offset)).isoformat()
                if check in date_idx:
                    entry_date = check
                    found = True
                    break
            if not found:
                skipped += 1
                continue

        start_idx = date_idx[entry_date]
        end_idx = start_idx + HOLDING_DAYS

        if end_idx >= len(trading_dates):
            skipped += 1
            continue

        # Compound factor returns over holding window
        cum_mktrf = 1.0
        cum_smb = 1.0
        cum_hml = 1.0
        cum_mom = 1.0
        cum_rf = 1.0
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

        if valid_days < HOLDING_DAYS * 0.8:  # Require at least 80% of days
            skipped += 1
            continue

        # Convert to percentage returns
        mktrf_ret = (cum_mktrf - 1) * 100
        smb_ret = (cum_smb - 1) * 100
        hml_ret = (cum_hml - 1) * 100
        mom_ret = (cum_mom - 1) * 100
        rf_ret = (cum_rf - 1) * 100

        # Signal's excess return = return_20d - risk-free
        excess_ret = s[RETURN_COL] - rf_ret

        records.append({
            'excess_return': excess_ret,
            'return_20d': s[RETURN_COL],
            'MktRF': mktrf_ret,
            'SMB': smb_ret,
            'HML': hml_ret,
            'Mom': mom_ret,
            'RF': rf_ret,
            'ticker': s['ticker'],
            'date': s['date'],
        })

    return records, skipped


# ============================================================
# OLS WITH WHITE'S HC1 ROBUST STANDARD ERRORS
# ============================================================
def ols_robust(y, X, factor_names):
    """
    OLS regression with White's HC1 robust standard errors.
    y: (n,) array
    X: (n, k) array (includes intercept column)
    Returns: coefficients, robust standard errors, t-stats, p-values, R2
    """
    n, k = X.shape

    # OLS: beta = (X'X)^-1 X'y
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y)

    # Residuals
    y_hat = X @ beta
    resid = y - y_hat

    # R-squared
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # White's HC1 robust variance: (X'X)^-1 X' diag(e^2) X (X'X)^-1 * n/(n-k)
    resid_sq = resid ** 2
    meat = X.T @ np.diag(resid_sq) @ X
    robust_var = XtX_inv @ meat @ XtX_inv * (n / (n - k))
    robust_se = np.sqrt(np.diag(robust_var))

    # t-stats and p-values
    t_stats = beta / robust_se
    # Two-sided p-values from normal approximation (large n)
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


# ============================================================
# RUN REGRESSIONS
# ============================================================
def run_regressions(records, label):
    """Run CAPM, FF3, and Carhart 4-factor regressions."""
    print(f'\n{"="*70}')
    print(f'  {label}')
    print(f'{"="*70}')
    print(f'  Signals with matched factors: {len(records)}')

    if len(records) < 30:
        print(f'  INSUFFICIENT DATA for regression')
        return None

    # Build arrays
    y = np.array([r['excess_return'] for r in records])
    mktrf = np.array([r['MktRF'] for r in records])
    smb = np.array([r['SMB'] for r in records])
    hml = np.array([r['HML'] for r in records])
    mom = np.array([r['Mom'] for r in records])
    ones = np.ones(len(records))

    mean_excess = float(np.mean(y))
    print(f'  Mean excess return: {mean_excess:.2f}%')

    model_results = {}

    # --- CAPM (1-factor) ---
    X_capm = np.column_stack([ones, mktrf])
    capm_coefs, capm_r2, capm_n = ols_robust(y, X_capm, ['MktRF'])
    model_results['CAPM'] = {'coefficients': capm_coefs, 'r2': capm_r2, 'n': capm_n}

    print(f'\n  CAPM (1-factor):')
    print(f'    alpha:  {capm_coefs["alpha"]["coef"]:+.4f}% (t={capm_coefs["alpha"]["t_stat"]:.2f}, p={capm_coefs["alpha"]["p_value"]:.4f})')
    print(f'    MktRF:  {capm_coefs["MktRF"]["coef"]:+.4f}  (t={capm_coefs["MktRF"]["t_stat"]:.2f})')
    print(f'    R2:     {capm_r2:.4f}')

    # --- FF3 (3-factor) ---
    X_ff3 = np.column_stack([ones, mktrf, smb, hml])
    ff3_coefs, ff3_r2, ff3_n = ols_robust(y, X_ff3, ['MktRF', 'SMB', 'HML'])
    model_results['FF3'] = {'coefficients': ff3_coefs, 'r2': ff3_r2, 'n': ff3_n}

    print(f'\n  Fama-French 3-factor:')
    print(f'    alpha:  {ff3_coefs["alpha"]["coef"]:+.4f}% (t={ff3_coefs["alpha"]["t_stat"]:.2f}, p={ff3_coefs["alpha"]["p_value"]:.4f})')
    print(f'    MktRF:  {ff3_coefs["MktRF"]["coef"]:+.4f}  (t={ff3_coefs["MktRF"]["t_stat"]:.2f})')
    print(f'    SMB:    {ff3_coefs["SMB"]["coef"]:+.4f}  (t={ff3_coefs["SMB"]["t_stat"]:.2f})')
    print(f'    HML:    {ff3_coefs["HML"]["coef"]:+.4f}  (t={ff3_coefs["HML"]["t_stat"]:.2f})')
    print(f'    R2:     {ff3_r2:.4f}')

    # --- Carhart 4-factor ---
    X_4f = np.column_stack([ones, mktrf, smb, hml, mom])
    c4_coefs, c4_r2, c4_n = ols_robust(y, X_4f, ['MktRF', 'SMB', 'HML', 'Mom'])
    model_results['Carhart4'] = {'coefficients': c4_coefs, 'r2': c4_r2, 'n': c4_n}

    print(f'\n  Carhart 4-factor (with Momentum):')
    print(f'    alpha:  {c4_coefs["alpha"]["coef"]:+.4f}% (t={c4_coefs["alpha"]["t_stat"]:.2f}, p={c4_coefs["alpha"]["p_value"]:.4f})')
    print(f'    MktRF:  {c4_coefs["MktRF"]["coef"]:+.4f}  (t={c4_coefs["MktRF"]["t_stat"]:.2f})')
    print(f'    SMB:    {c4_coefs["SMB"]["coef"]:+.4f}  (t={c4_coefs["SMB"]["t_stat"]:.2f})')
    print(f'    HML:    {c4_coefs["HML"]["coef"]:+.4f}  (t={c4_coefs["HML"]["t_stat"]:.2f})')
    print(f'    Mom:    {c4_coefs["Mom"]["coef"]:+.4f}  (t={c4_coefs["Mom"]["t_stat"]:.2f})')
    print(f'    R2:     {c4_r2:.4f}')

    # --- Summary ---
    print(f'\n  ALPHA COMPARISON ACROSS MODELS:')
    print(f'    {"Model":<20} {"Alpha":>8} {"t-stat":>8} {"p-value":>10} {"Significant?":>14}')
    print(f'    {"-"*62}')
    for model_name in ['CAPM', 'FF3', 'Carhart4']:
        a = model_results[model_name]['coefficients']['alpha']
        sig = 'YES ***' if a['p_value'] < 0.01 else 'YES **' if a['p_value'] < 0.05 else 'YES *' if a['p_value'] < 0.10 else 'NO'
        print(f'    {model_name:<20} {a["coef"]:>+7.3f}% {a["t_stat"]:>8.2f} {a["p_value"]:>10.4f} {sig:>14}')

    # Factor exposure interpretation
    mom_coef = c4_coefs['Mom']['coef']
    mom_t = c4_coefs['Mom']['t_stat']
    c4_alpha = c4_coefs['alpha']['coef']
    capm_alpha = capm_coefs['alpha']['coef']
    alpha_drop = capm_alpha - c4_alpha

    print(f'\n  FACTOR EXPOSURE INTERPRETATION:')
    print(f'    Momentum loading: {mom_coef:+.4f} (t={mom_t:.2f})')
    if abs(mom_t) > 1.96:
        print(f'    -> Engine has SIGNIFICANT momentum exposure')
    else:
        print(f'    -> Engine does NOT have significant momentum exposure')

    print(f'    Alpha drop (CAPM -> 4-factor): {alpha_drop:+.3f}pp')
    if abs(alpha_drop) > 0.3 * abs(capm_alpha) and abs(capm_alpha) > 0.1:
        print(f'    -> Factors explain {alpha_drop/capm_alpha*100:.0f}% of CAPM alpha')
    else:
        print(f'    -> Factors explain minimal portion of the alpha')

    c4_alpha_p = c4_coefs['alpha']['p_value']
    if c4_alpha_p < 0.05:
        verdict = 'PASS -- alpha survives 4-factor model (genuinely novel signal)'
    elif c4_alpha_p < 0.10:
        verdict = 'MARGINAL -- alpha weakly significant after factor control'
    else:
        verdict = 'FAIL -- alpha explained by standard factors (momentum/size/value)'

    print(f'\n  VERDICT: {verdict}')

    return {
        'label': label,
        'n_signals': len(records),
        'mean_excess_return': round(mean_excess, 2),
        'models': {k: {'coefficients': v['coefficients'], 'r2': v['r2'], 'n': v['n']}
                   for k, v in model_results.items()},
        'alpha_drop_capm_to_4f': round(alpha_drop, 4),
        'verdict': verdict,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print('Test #07: Fama-MacBeth Factor Regression')
    print('=' * 70)

    # Step 1: Download factor data
    print('\nStep 1: Loading factor data...')
    ff3 = download_and_parse_ff3()
    mom = download_and_parse_mom()

    # Merge
    factor_dates = set(ff3.keys()) & set(mom.keys())
    print(f'  FF3 dates: {len(ff3)}, Mom dates: {len(mom)}, Overlap: {len(factor_dates)}')

    # Check date range
    sorted_dates = sorted(factor_dates)
    print(f'  Factor data range: {sorted_dates[0]} to {sorted_dates[-1]}')

    # Step 2: Trading calendar
    print('\nStep 2: Loading trading calendar...')
    trading_dates = load_trading_calendar()
    print(f'  SPY trading days: {len(trading_dates)} ({trading_dates[0]} to {trading_dates[-1]})')

    # Step 3: Load signals
    print('\nStep 3: Loading signals...')
    signals, ins_set, exp_set = load_data()
    ins_deduped = dedup_signals(signals, ins_set)
    exp_deduped = dedup_signals(signals, exp_set)
    print(f'  De-duplicated: in-sample={len(ins_deduped)}, expansion={len(exp_deduped)}')

    # Step 4: Match factors to signals
    print('\nStep 4: Matching factors to signal holding windows...')
    ins_records, ins_skipped = compute_signal_factors(ins_deduped, ff3, mom, trading_dates)
    exp_records, exp_skipped = compute_signal_factors(exp_deduped, ff3, mom, trading_dates)
    print(f'  In-sample: {len(ins_records)} matched, {ins_skipped} skipped')
    print(f'  Expansion: {len(exp_records)} matched, {exp_skipped} skipped')

    # Step 5: Run regressions
    results = {
        'test': 'Test #07: Fama-MacBeth Factor Regression',
        'description': (
            'Tests whether the engine alpha is explained by standard risk factors '
            '(market, size, value, momentum). Uses official Fama-French factor data '
            'from Kenneth French Data Library and White HC1 robust standard errors.'
        ),
        'methodology': {
            'data_source': 'Kenneth French Data Library (CRSP-based)',
            'models': ['CAPM (1-factor)', 'Fama-French 3-factor', 'Carhart 4-factor'],
            'standard_errors': 'White HC1 heteroskedasticity-robust',
            'holding_period': f'{HOLDING_DAYS} trading days',
            'dedup_days': DEDUP_DAYS,
        },
        'factor_data': {
            'ff3_dates': len(ff3),
            'mom_dates': len(mom),
            'overlap_dates': len(factor_dates),
            'range': [sorted_dates[0], sorted_dates[-1]],
        },
    }

    results['in_sample'] = run_regressions(ins_records, 'IN-SAMPLE')
    # Expansion omitted (public data is in-sample only)
    # results.get('expansion', {}) = run_regressions(exp_records, 'EXPANSION')

    # Final summary
    print(f'\n{"="*70}')
    print(f'  PUBLISHABLE SUMMARY')
    print(f'{"="*70}')

    for cut in ['in_sample']:  # expansion omitted in public data
        r = results[cut]
        if r is None:
            continue
        c4 = r['models']['Carhart4']['coefficients']
        print(f'\n  {r["label"]}:')
        print(f'    4-factor alpha: {c4["alpha"]["coef"]:+.3f}% per 20 days (t={c4["alpha"]["t_stat"]:.2f}, p={c4["alpha"]["p_value"]:.4f})')
        print(f'    Momentum loading: {c4["Mom"]["coef"]:+.3f} (t={c4["Mom"]["t_stat"]:.2f})')
        print(f'    R2: {r["models"]["Carhart4"]["r2"]:.1%}')
        print(f'    {r["verdict"]}')

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  Results written to {RESULTS_FILE}')


if __name__ == '__main__':
    main()
