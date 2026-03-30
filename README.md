# Statistical Validation Suite for a Stock Signal Engine

A reproducible, open-source statistical validation suite for a proprietary stock signal engine. This repo contains **only** the validation scripts and anonymized signal outcome data — no engine scoring logic, no proprietary methodology, no trade secrets.

Anyone can clone this repo, run the tests, and verify every statistical claim independently.

## What This Is

A stock signal engine produces buy signals on US equities. Each signal predicts that the stock will rise over the next 20 trading days. This suite tests whether that prediction has statistically significant edge using 8 standard tests and 5 visual proof analyses.

**What the data contains:** For each signal — the date it fired, the ticker, and the actual forward returns at 5, 10, 20, and 40 trading days (plus S&P 500 returns over the same windows). This is everything needed to evaluate whether the signals work. No engine internals are included.

## Quick Start

```bash
git clone https://github.com/signal-validation/stock-signal-validation.git
cd statistical-validation
pip install numpy pandas matplotlib
python tests/run_all_tests.py      # 8 statistical tests (~5 min)
python proofs/run_all_proofs.py    # 5 visual analyses (~15 sec)
```

## Requirements

- Python 3.10+
- numpy
- pandas (optional, not used by core tests)
- matplotlib (for figures only)

All tests use `seed=42` for deterministic, reproducible results.

## The 8 Statistical Tests

| Test | File | What It Proves |
|---|---|---|
| **#10 Effective N** | `test10_effective_n.py` | Quantifies signal overlap — only 2,375 of 19,558 raw signals are independent (28-day de-duplication validated by autocorrelation decay) |
| **#01 MC Stock Selection** | `test01_mc_stock_selection.py` | Tests whether per-stock win rate variation is real or sampling noise (result: p=0.50, edge is broad-based) |
| **#02 MC Timing** | `test02_mc_timing.py` | Tests whether signal timing matters vs random entry dates (result: timing adds +2.5pp on out-of-sample stocks, p=0.015) |
| **#03 Block Bootstrap** | `test03_block_bootstrap.py` | Confidence interval for win rate preserving temporal structure (result: 95% CI [57.6%, 63.2%], excludes 50% at all 5 block sizes) |
| **#04 Multiple Testing** | `test04_multiple_testing.py` | Bonferroni correction for testing multiple engine versions (result: survives correction to K=10^22) |
| **#05 Transaction Costs** | `test05_transaction_costs.py` | Tests whether edge survives realistic trading costs (result: breakeven > 0.50% one-way) |
| **#06 Survivorship Bias** | `test06_survivorship_bias.py` | Simulates impact of delisted stocks (result: -1.2pp under realistic assumptions, still above 50%) |
| **#07 Factor Regression** | `test07_factor_regression.py` | Carhart 4-factor regression using official Fama-French data (result: alpha +0.97% per trade, t=5.60, not explained by market/size/value/momentum) |

## The 5 Visual Proofs

| Proof | File | What It Shows |
|---|---|---|
| **Walk-Forward** | `proof01_walkforward_temporal.py` | Train on 2006-2015 (60.3%), test on 2016-2025 (60.3%) — identical across temporal halves |
| **Equity Curve** | `proof02_equity_curve.py` | Cumulative alpha vs S&P 500 over 19 years with drawdown analysis |
| **Year-by-Year** | `proof03_yearly_performance.py` | Win rate above 50% in 17 of 20 years, bull/bear split |
| **Bear Markets** | `proof04_bear_market.py` | Engine performance during 2008, 2020, 2022 crashes (engine goes silent = protective) |
| **Return Distribution** | `proof05_return_distribution.py` | Average win +6.35%, average loss -5.15%, W/L ratio 1.23x, expected value +1.79% per trade |

## Data Schema

Each signal in `data/signals_public.json` contains exactly 14 fields:

```
date            Signal date (YYYY-MM-DD)
ticker          Stock ticker symbol
return_5d       5-day forward return (%)
return_10d      10-day forward return (%)
return_20d      20-day forward return (%) — primary metric
return_40d      40-day forward return (%)
spy_return_5d   S&P 500 return over same 5-day window
spy_return_10d  S&P 500 return over same 10-day window
spy_return_20d  S&P 500 return over same 20-day window
spy_return_40d  S&P 500 return over same 40-day window
alpha           20-day excess return (return_20d - spy_return_20d)
alpha_5d        5-day excess return
alpha_10d       10-day excess return
alpha_40d       40-day excess return
```

The data file contains 19,664 daily signals and 4,837 weekly signals across 234 stocks from 2006 to 2026.

## What Is NOT Included

- Engine scoring logic or source code
- Signal generation methodology
- Feature weights, thresholds, or parameters
- Stock selection criteria (in-sample vs out-of-sample classification)
- Internal signal quality metrics (scores, confidence, etc.)

The engine is proprietary. This repo validates its **outputs**, not its internals.

## Headline Results

**Daily signals (de-duplicated, 2,375 independent observations):**
- Win rate: 60.3% (95% CI: [57.6%, 63.2%])
- Walk-forward: train 2006-2015 = 60.3%, test 2016-2025 = 60.3%
- 4-factor alpha: +0.97% per trade (t=5.60, p<0.0001)
- Survives transaction costs, survivorship bias, and multiple testing correction

**Weekly signals (de-duplicated, 1,317 independent observations):**
- Win rate: 58.9% (95% CI: [55.1%, 63.2%])
- 4-factor alpha: +0.50% per trade (t=2.20, p=0.028)

## Honest Disclaimers

- **Survivorship bias:** The stock universe contains only current survivors. Simulated impact: -1.2 percentage points under realistic assumptions (2% annual delisting, 40% delisted win rate).
- **In-sample universe:** The 234 stocks were selected because the engine was developed on them. Out-of-sample (expansion) stocks achieve 55.4% — weaker but still significant.
- **No live track record:** Signal logging started March 2026. Backtest results only. Past performance is not indicative of future results.
- **Transaction costs excluded:** Breakeven is above 0.50% one-way (realistic retail costs are ~0.05-0.10%).
- **De-duplication:** Raw signal count is 19,664 but 90% overlap within holding periods. Only 2,375 are statistically independent after 28-day de-duplication.

## Methodology

All tests use standard, published statistical methods:

- **Block bootstrap:** Kunsch (1989), preserves temporal structure
- **Fama-MacBeth factor regression:** Fama & MacBeth (1973), with White's HC1 robust standard errors
- **Bonferroni correction:** Most conservative family-wise error rate control
- **Factor data:** Kenneth French Data Library (official CRSP-based, auto-downloaded on first run)

## License

MIT License. See [LICENSE](LICENSE).

## Citation

If you use this validation methodology in academic work:

```
Statistical Validation Suite for a Stock Signal Engine (2026).
8-test validation framework for evaluating buy signal predictions.
https://github.com/signal-validation/stock-signal-validation
```
