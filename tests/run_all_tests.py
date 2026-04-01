"""
Statistical Validation Suite — Runner
=======================================
Executes all 9 tests sequentially, captures output, combines results.

Tests:
  #10: Effective N / Autocorrelation (foundation)
  #01: Monte Carlo Stock Selection
  #02: Monte Carlo Timing
  #03: Block Bootstrap Confidence Intervals
  #04: Multiple Testing Correction
  #05: Transaction Cost Sensitivity
  #06: Survivorship Bias Simulation
  #07: Fama-MacBeth Factor Regression
  #11: Per-Regime Factor Regression

Usage:
  python3 run_all_tests.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, 'validation_suite_log.txt')
COMBINED_RESULTS = os.path.join(SCRIPT_DIR, 'validation_suite_results.json')

TESTS = [
    ('test10', 'test10_effective_n.py', 'test10_effective_n_results.json', 'Effective N / Autocorrelation'),
    ('test01', 'test01_mc_stock_selection.py', 'test01_mc_stock_selection_results.json', 'MC Stock Selection'),
    ('test02', 'test02_mc_timing.py', 'test02_mc_timing_results.json', 'MC Timing'),
    ('test03', 'test03_block_bootstrap.py', 'test03_block_bootstrap_results.json', 'Block Bootstrap'),
    ('test04', 'test04_multiple_testing.py', 'test04_multiple_testing_results.json', 'Multiple Testing Correction'),
    ('test05', 'test05_transaction_costs.py', 'test05_transaction_costs_results.json', 'Transaction Cost Sensitivity'),
    ('test06', 'test06_survivorship_bias.py', 'test06_survivorship_bias_results.json', 'Survivorship Bias Simulation'),
    ('test07', 'test07_factor_regression.py', 'test07_factor_regression_results.json', 'Factor Regression'),
    ('test11', 'test11_regime_factor_regression.py', 'test11_regime_factor_regression_results.json', 'Per-Regime Factor Regression'),
]


def run_test(test_id, script, results_file, description):
    """Run a single test, return (success, elapsed, results_dict)."""
    script_path = os.path.join(SCRIPT_DIR, script)
    results_path = os.path.join(SCRIPT_DIR, results_file)

    print(f'\n{"="*70}')
    print(f'  RUNNING: {test_id} — {description}')
    print(f'  Script: {script}')
    print(f'{"="*70}')

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per test
            cwd=SCRIPT_DIR,
        )
        elapsed = time.time() - t0

        # Write output to log
        output = result.stdout + (f'\nSTDERR:\n{result.stderr}' if result.stderr else '')

        if result.returncode != 0:
            print(f'  FAILED (exit code {result.returncode}, {elapsed:.1f}s)')
            print(f'  STDERR: {result.stderr[:500]}')
            return False, elapsed, None, output

        # Load results JSON
        results_dict = None
        if os.path.exists(results_path):
            with open(results_path) as f:
                results_dict = json.load(f)

        print(f'  PASSED ({elapsed:.1f}s)')
        return True, elapsed, results_dict, output

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f'  TIMEOUT after {elapsed:.0f}s')
        return False, elapsed, None, 'TIMEOUT'
    except Exception as e:
        elapsed = time.time() - t0
        print(f'  ERROR: {e}')
        return False, elapsed, None, str(e)


def main():
    start_time = datetime.now()
    print(f'Statistical Validation Suite')
    print(f'Started: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Tests: {len(TESTS)}')

    all_output = []
    all_results = {
        'suite': 'Statistical Validation Suite',
        'started': start_time.isoformat(),
        'tests': {},
        'summary': {},
    }

    passed = 0
    failed = 0
    total_elapsed = 0

    for test_id, script, results_file, description in TESTS:
        success, elapsed, results_dict, output = run_test(test_id, script, results_file, description)
        total_elapsed += elapsed

        all_output.append(f'\n{"#"*70}')
        all_output.append(f'# {test_id}: {description}')
        all_output.append(f'# Status: {"PASSED" if success else "FAILED"} ({elapsed:.1f}s)')
        all_output.append(f'{"#"*70}')
        all_output.append(output or '')

        all_results['tests'][test_id] = {
            'description': description,
            'script': script,
            'success': success,
            'elapsed_seconds': round(elapsed, 1),
        }

        if results_dict:
            all_results['tests'][test_id]['results'] = results_dict

        if success:
            passed += 1
        else:
            failed += 1

    end_time = datetime.now()
    all_results['completed'] = end_time.isoformat()
    all_results['total_elapsed_seconds'] = round(total_elapsed, 1)
    all_results['summary'] = {
        'total': len(TESTS),
        'passed': passed,
        'failed': failed,
    }

    # Write log
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_output))
    print(f'\nLog written to {LOG_FILE}')

    # Write combined results
    with open(COMBINED_RESULTS, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'Combined results written to {COMBINED_RESULTS}')

    # Final summary
    print(f'\n{"="*70}')
    print(f'  VALIDATION SUITE COMPLETE')
    print(f'{"="*70}')
    print(f'  Started:  {start_time.strftime("%H:%M:%S")}')
    print(f'  Finished: {end_time.strftime("%H:%M:%S")}')
    print(f'  Elapsed:  {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)')
    print(f'  Tests:    {passed} passed, {failed} failed, {len(TESTS)} total')
    print()

    print(f'  {"Test":<10} {"Description":<35} {"Status":<10} {"Time":>8}')
    print(f'  {"-"*65}')
    for test_id, script, results_file, description in TESTS:
        t = all_results['tests'][test_id]
        status = 'PASS' if t['success'] else 'FAIL'
        print(f'  {test_id:<10} {description:<35} {status:<10} {t["elapsed_seconds"]:>7.1f}s')

    if failed > 0:
        print(f'\n  WARNING: {failed} test(s) failed. Check {LOG_FILE} for details.')
        sys.exit(1)
    else:
        print(f'\n  All tests passed.')


if __name__ == '__main__':
    main()
