"""
Proof Suite Runner — 5 Visual Analyses
========================================
Usage:
  python3 run_all_proofs.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, 'proof_suite_log.txt')
COMBINED_RESULTS = os.path.join(SCRIPT_DIR, 'proof_suite_results.json')

PROOFS = [
    ('proof01', 'proof01_walkforward_temporal.py', 'proof01_walkforward_temporal_results.json', 'Walk-Forward Temporal OOS'),
    ('proof02', 'proof02_equity_curve.py', 'proof02_equity_curve_results.json', 'Equity Curve + Drawdown'),
    ('proof03', 'proof03_yearly_performance.py', 'proof03_yearly_performance_results.json', 'Year-by-Year Performance'),
    ('proof04', 'proof04_bear_market.py', 'proof04_bear_market_results.json', 'Bear Market Spotlight'),
    ('proof05', 'proof05_return_distribution.py', 'proof05_return_distribution_results.json', 'Return Distribution'),
]


def run_proof(proof_id, script, results_file, description):
    script_path = os.path.join(SCRIPT_DIR, script)
    results_path = os.path.join(SCRIPT_DIR, results_file)

    print(f'\n{"="*70}')
    print(f'  RUNNING: {proof_id} -- {description}')
    print(f'{"="*70}')

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=120, cwd=SCRIPT_DIR,
        )
        elapsed = time.time() - t0
        output = result.stdout + (f'\nSTDERR:\n{result.stderr}' if result.stderr else '')

        if result.returncode != 0:
            print(f'  FAILED (exit code {result.returncode}, {elapsed:.1f}s)')
            print(f'  STDERR: {result.stderr[:500]}')
            return False, elapsed, None, output

        results_dict = None
        if os.path.exists(results_path):
            with open(results_path) as f:
                results_dict = json.load(f)

        print(f'  PASSED ({elapsed:.1f}s)')
        return True, elapsed, results_dict, output

    except subprocess.TimeoutExpired:
        return False, time.time() - t0, None, 'TIMEOUT'
    except Exception as e:
        return False, time.time() - t0, None, str(e)


def main():
    start_time = datetime.now()
    print(f'Proof Suite: 5 Visual Analyses')
    print(f'Started: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')

    all_output = []
    all_results = {'suite': 'Proof Suite', 'started': start_time.isoformat(), 'proofs': {}}
    passed = failed = 0
    total_elapsed = 0

    for proof_id, script, results_file, description in PROOFS:
        success, elapsed, results_dict, output = run_proof(proof_id, script, results_file, description)
        total_elapsed += elapsed
        all_output.append(f'\n{"#"*70}\n# {proof_id}: {description}\n# {"PASSED" if success else "FAILED"} ({elapsed:.1f}s)\n{"#"*70}\n{output or ""}')
        all_results['proofs'][proof_id] = {'description': description, 'success': success, 'elapsed_seconds': round(elapsed, 1)}
        if results_dict:
            all_results['proofs'][proof_id]['results'] = results_dict
        if success: passed += 1
        else: failed += 1

    all_results['completed'] = datetime.now().isoformat()
    all_results['total_elapsed_seconds'] = round(total_elapsed, 1)
    all_results['summary'] = {'total': len(PROOFS), 'passed': passed, 'failed': failed}

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_output))
    with open(COMBINED_RESULTS, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f'\n{"="*70}')
    print(f'  PROOF SUITE COMPLETE ({total_elapsed:.0f}s)')
    print(f'{"="*70}')
    print(f'  {"Proof":<10} {"Description":<35} {"Status":<8} {"Time":>7}')
    print(f'  {"-"*62}')
    for proof_id, _, _, description in PROOFS:
        p = all_results['proofs'][proof_id]
        print(f'  {proof_id:<10} {description:<35} {"PASS" if p["success"] else "FAIL":<8} {p["elapsed_seconds"]:>6.1f}s')
    print(f'\n  {passed} passed, {failed} failed')

    if failed:
        sys.exit(1)

if __name__ == '__main__':
    main()
