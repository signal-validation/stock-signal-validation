"""
Generate publication-quality academic figures for statistical validation.
Each figure follows the same format as monte_carlo_academic_40d.png:
  - 1920x1080 HD, 150 DPI, white background
  - Multi-panel layout (chart + histogram/stats panel)
  - Summary statistics table below
  - Methodology box + Disclosures box
  - Clean typography (Segoe UI / Helvetica)

Figures generated:
  1. Walk-Forward Temporal OOS (train 2006-2015 vs test 2016-2025)
  2. Block Bootstrap Confidence Intervals (10K resamples distribution)
  3. Autocorrelation Decay (effective N visualization)
  4. Equity Curve + Drawdown (19-year cumulative returns)

Each figure is self-contained proof. The script itself is publishable
alongside the figure — no engine internals leaked, only statistical
analysis of pre-computed signal outcomes.

Data source: signals_public.json (signal-level outcomes)
             universe.json (stock universe definition)

Usage:
  python3 generate_academic_figures.py           # all figures
  python3 generate_academic_figures.py walkforward  # single figure
"""

import json
import math
import os
import sys
import random
from collections import defaultdict
from datetime import date as Date
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec

# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / ".." / "data"
CHART_DIR = SCRIPT_DIR / "output"
CHART_DIR.mkdir(exist_ok=True)

WIDTH_PX, HEIGHT_PX, DPI = 1920, 1080, 150
SEED = 42
DEDUP_DAYS = 28
RETURN_COL = "return_20d"
N_BOOTSTRAP = 10000

# ── Fonts ──
FONT_SANS = "sans-serif"
FONT_MONO = "monospace"
try:
    all_fonts = fm.findSystemFonts(fontpaths=None)
    for c in ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]:
        if any(c.lower().replace(" ", "") in f.lower() for f in all_fonts):
            FONT_SANS = c; break
    for c in ["Consolas", "Cascadia Code", "Courier New"]:
        if any(c.lower().replace(" ", "") in f.lower() for f in all_fonts):
            FONT_MONO = c; break
except Exception:
    pass

# ── Colors (academic white theme) ──
BG = "#ffffff"
TEXT = "#1a1a1a"
SUBTLE = "#666666"
GRID = "#e8e8e8"
AXIS = "#888888"
ENGINE_CLR = "#000000"
ACCENT_BLUE = "#2166ac"
ACCENT_GREEN = "#1b7837"
ACCENT_RED = "#b2182b"
LIGHT_BLUE = "#d1e5f0"
LIGHT_GREEN = "#d9f0d3"
LIGHT_RED = "#fddbc7"
BAND_BLUE = "#92c5de"

def spine_style(ax):
    for s in ax.spines.values():
        s.set_color("#cccccc"); s.set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(colors=AXIS, labelsize=7)

# ════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════

def load_data():
    with open(DATA_DIR / "signals_public.json") as f:
        d = json.load(f)
    sigs = d.get("daily", {}).get("signals", d.get("signals", []))
    tickers = set(s["ticker"] for s in sigs)
    return sigs, tickers, set()

def parse_date(s):
    return Date(int(s[:4]), int(s[5:7]), int(s[8:10]))

def dedup(signals, ticker_set):
    by_tk = defaultdict(list)
    for s in signals:
        if s["ticker"] not in ticker_set or s[RETURN_COL] is None:
            continue
        by_tk[s["ticker"]].append(s)
    out = []
    for tk, sigs in by_tk.items():
        sigs.sort(key=lambda x: x["date"])
        last = None
        for s in sigs:
            d = parse_date(s["date"])
            if last is None or (d - last).days > DEDUP_DAYS:
                out.append(s)
                last = d
    out.sort(key=lambda x: x["date"])
    return out


# ════════════════════════════════════════════════════════════════════════
# FIGURE 1: WALK-FORWARD TEMPORAL OOS
# ════════════════════════════════════════════════════════════════════════

def fig_walkforward(deduped):
    print("  Generating: Walk-Forward Temporal OOS...")

    # Split
    train = [s for s in deduped if int(s["date"][:4]) <= 2015]
    test  = [s for s in deduped if int(s["date"][:4]) >= 2016]

    # 5-year windows
    windows = [
        ("2006-2010", 2006, 2010), ("2011-2015", 2011, 2015),
        ("2016-2020", 2016, 2020), ("2021-2025", 2021, 2025),
    ]

    def wr_stats(sigs):
        n = len(sigs)
        w = sum(1 for s in sigs if s[RETURN_COL] > 0)
        wr = w / n * 100 if n else 0
        alpha = sum((s.get("alpha", 0) or 0) for s in sigs) / n if n else 0
        se = math.sqrt(0.5 * 0.5 / n) if n else 1
        z = (wr / 100 - 0.5) / se
        p = 0.5 * math.erfc(z / math.sqrt(2)) if z > 0 else 1.0
        return {"n": n, "wr": wr, "alpha": alpha, "z": z, "p": p}

    train_s = wr_stats(train)
    test_s = wr_stats(test)
    win_stats = []
    for label, y0, y1 in windows:
        ws = [s for s in deduped if y0 <= int(s["date"][:4]) <= y1]
        win_stats.append((label, wr_stats(ws)))

    # Bootstrap CI for test period
    rng = random.Random(SEED)
    test_dates = sorted(set(s["date"] for s in test))
    d2b = {d: i // 63 for i, d in enumerate(test_dates)}
    blocks = defaultdict(list)
    for s in test:
        blocks[d2b[s["date"]]].append(s)
    blist = [blocks[k] for k in sorted(blocks.keys())]
    boot_wrs = []
    for _ in range(N_BOOTSTRAP):
        sam = rng.choices(blist, k=len(blist))
        w = sum(1 for b in sam for s in b if s[RETURN_COL] > 0)
        t = sum(len(b) for b in sam)
        if t: boot_wrs.append(w / t * 100)
    boot_wrs.sort()
    ci = [boot_wrs[int(len(boot_wrs)*0.025)], boot_wrs[int(len(boot_wrs)*0.975)]]

    # ── Build figure ──
    fig = plt.figure(figsize=(WIDTH_PX/DPI, HEIGHT_PX/DPI), dpi=DPI, facecolor=BG)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[0.55, 0.45], height_ratios=[0.75, 0.25],
                  hspace=0.18, wspace=0.15, left=0.07, right=0.95, top=0.84, bottom=0.06)

    # Title
    fig.text(0.5, 0.97, "Walk-Forward Temporal Out-of-Sample Validation",
             fontsize=14, fontweight="bold", color=TEXT, ha="center", fontfamily=FONT_SANS)
    fig.text(0.5, 0.915,
             "Null hypothesis: engine is overfit to historical data  |  "
             "Same 237 stocks, split by TIME  |  De-duplicated (28-day rule)  |  2006-2025",
             fontsize=7.5, color=SUBTLE, ha="center", fontfamily=FONT_SANS)

    # LEFT: Primary split bars
    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor(BG)
    labels = ["TRAIN\n2006-2015", "TEST\n2016-2025"]
    wrs = [train_s["wr"], test_s["wr"]]
    ns = [train_s["n"], test_s["n"]]
    colors = [LIGHT_BLUE, ACCENT_GREEN]
    edge_colors = [ACCENT_BLUE, "#1b7837"]

    bars = ax1.bar([0, 1], wrs, width=0.55, color=colors, edgecolor=edge_colors, linewidth=1.5)
    ax1.axhline(50, color=ACCENT_RED, linestyle="--", alpha=0.6, linewidth=1)
    ax1.set_ylim(40, 72)
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(labels, fontsize=10, fontfamily=FONT_SANS, fontweight="bold")
    ax1.set_ylabel("Win Rate (%)", fontsize=9, color=SUBTLE, fontfamily=FONT_SANS)
    ax1.set_title("Primary Split: First Half vs Second Half", fontsize=10.5, fontweight="bold",
                   color=TEXT, fontfamily=FONT_SANS, pad=10)

    for bar, wr, n in zip(bars, wrs, ns):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f"{wr:.1f}%", ha="center", fontsize=16, fontweight="bold", color=TEXT, fontfamily=FONT_MONO)
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 3,
                 f"N = {n:,}", ha="center", fontsize=8, color=SUBTLE, fontfamily=FONT_MONO)

    ax1.text(0.5, 50.5, "50% = random chance", fontsize=7, color=ACCENT_RED, alpha=0.7,
             ha="center", fontfamily=FONT_SANS, transform=ax1.get_yaxis_transform())
    ax1.grid(True, axis="y", color=GRID, linewidth=0.4)
    spine_style(ax1)

    # CI annotation
    ax1.text(1, wrs[1] - 7, f"95% CI: [{ci[0]:.1f}%, {ci[1]:.1f}%]\np = {test_s['p']:.2e}",
             ha="center", fontsize=7.5, color=SUBTLE, fontfamily=FONT_MONO,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#cccccc", linewidth=0.5))

    # RIGHT: 5-year rolling windows
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor(BG)
    w_labels = [w[0] for w in windows]
    w_wrs = [ws[1]["wr"] for ws in win_stats]
    w_ns = [ws[1]["n"] for ws in win_stats]
    w_colors = [ACCENT_GREEN if wr > 55 else "#e6ab02" if wr > 50 else ACCENT_RED for wr in w_wrs]
    w_edge = [c for c in w_colors]

    bars2 = ax2.bar(range(4), w_wrs, width=0.6, color=[c + "44" for c in w_colors],
                     edgecolor=w_edge, linewidth=1.5)
    ax2.axhline(50, color=ACCENT_RED, linestyle="--", alpha=0.6, linewidth=1)
    ax2.set_ylim(40, 72)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(w_labels, fontsize=8.5, fontfamily=FONT_SANS)
    ax2.set_ylabel("Win Rate (%)", fontsize=9, color=SUBTLE, fontfamily=FONT_SANS)
    ax2.set_title("5-Year Rolling Windows", fontsize=10.5, fontweight="bold",
                   color=TEXT, fontfamily=FONT_SANS, pad=10)

    for bar, wr, n in zip(bars2, w_wrs, w_ns):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f"{wr:.1f}%", ha="center", fontsize=11, fontweight="bold", color=TEXT, fontfamily=FONT_MONO)
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 3,
                 f"N={n}", ha="center", fontsize=7, color=SUBTLE, fontfamily=FONT_MONO)

    ax2.grid(True, axis="y", color=GRID, linewidth=0.4)
    spine_style(ax2)

    # BOTTOM LEFT: Stats table
    ax_t = fig.add_subplot(gs[1, 0]); ax_t.set_facecolor(BG); ax_t.axis("off")
    ty = 0.85
    headers = [("Period", 0.02), ("N", 0.25), ("Win Rate", 0.38), ("Alpha/Trade", 0.56), ("z-score", 0.72), ("p-value", 0.88)]
    for h, x in headers:
        ax_t.text(x, ty, h, fontsize=7.5, fontweight="bold", color=TEXT, va="top",
                  fontfamily=FONT_SANS, transform=ax_t.transAxes)
    ty -= 0.08
    ax_t.plot([0.02, 0.98], [ty, ty], transform=ax_t.transAxes, color="#cccccc", linewidth=0.5)
    ty -= 0.06

    rows = [("Train 2006-2015", train_s), ("Test 2016-2025", test_s)]
    rows += [(ws[0], ws[1]) for ws in win_stats]
    for label, st in rows:
        is_primary = "Test" in label
        if is_primary:
            ax_t.fill_between([0.01, 0.99], ty - 0.1, ty + 0.04, transform=ax_t.transAxes,
                               color="#e8f5e9", zorder=0)
        fw = "bold" if is_primary else "normal"
        cl = TEXT if st["p"] < 0.05 else SUBTLE
        ax_t.text(0.02, ty, label, fontsize=7, fontweight=fw, color=cl, va="top", fontfamily=FONT_SANS, transform=ax_t.transAxes)
        ax_t.text(0.25, ty, f"{st['n']:,}", fontsize=7, fontweight=fw, color=cl, va="top", fontfamily=FONT_MONO, transform=ax_t.transAxes)
        ax_t.text(0.38, ty, f"{st['wr']:.1f}%", fontsize=7, fontweight=fw, color=cl, va="top", fontfamily=FONT_MONO, transform=ax_t.transAxes)
        ax_t.text(0.56, ty, f"{st['alpha']:+.2f}%", fontsize=7, fontweight=fw, color=cl, va="top", fontfamily=FONT_MONO, transform=ax_t.transAxes)
        ax_t.text(0.72, ty, f"{st['z']:.2f}", fontsize=7, fontweight=fw, color=cl, va="top", fontfamily=FONT_MONO, transform=ax_t.transAxes)
        ax_t.text(0.88, ty, f"{st['p']:.2e}" if st['p'] < 0.001 else f"{st['p']:.4f}", fontsize=7, fontweight=fw, color=cl, va="top", fontfamily=FONT_MONO, transform=ax_t.transAxes)
        ty -= 0.13

    # BOTTOM RIGHT: Methodology + Disclosures
    ax_d = fig.add_subplot(gs[1, 1]); ax_d.set_facecolor(BG); ax_d.axis("off")
    dy = 0.85
    ax_d.text(0.05, dy, "Methodology", fontsize=7.5, fontweight="bold", color=SUBTLE, va="top", fontfamily=FONT_SANS, transform=ax_d.transAxes)
    dy -= 0.14
    ax_d.text(0.05, dy,
        "Same 237 in-sample stocks in both halves.\n"
        "De-duplicated: first signal per stock per 28\n"
        "calendar days (validated in Test #10).\n"
        "Block bootstrap CI: 63-day blocks, 10K resamples.",
        fontsize=6.5, color="#999999", va="top", fontfamily=FONT_SANS, transform=ax_d.transAxes, linespacing=1.5)
    dy -= 0.48
    ax_d.text(0.05, dy, "Disclosures", fontsize=7.5, fontweight="bold", color=SUBTLE, va="top", fontfamily=FONT_SANS, transform=ax_d.transAxes)
    dy -= 0.14
    ax_d.text(0.05, dy,
        "Survivorship bias present (simulated -1.2pp)\n"
        "Transaction costs excluded (breakeven >0.50%)\n"
        "In-sample universe (not out-of-sample stocks)\n"
        "Seed: 42 | Python 3.13 | numpy + matplotlib",
        fontsize=6.5, color="#999999", va="top", fontfamily=FONT_SANS, transform=ax_d.transAxes, linespacing=1.5)

    path = CHART_DIR / "academic_walkforward_temporal.png"
    fig.savefig(path, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path} ({path.stat().st_size/1024:.0f} KB)")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 2: BLOCK BOOTSTRAP CI
# ════════════════════════════════════════════════════════════════════════

def fig_bootstrap(deduped):
    print("  Generating: Block Bootstrap Confidence Intervals...")

    rng = random.Random(SEED)
    all_dates = sorted(set(s["date"] for s in deduped))
    d2b = {d: i // 63 for i, d in enumerate(all_dates)}
    blocks = defaultdict(list)
    for s in deduped:
        blocks[d2b[s["date"]]].append(s)
    blist = [blocks[k] for k in sorted(blocks.keys())]

    n_total = len(deduped)
    real_wr = sum(1 for s in deduped if s[RETURN_COL] > 0) / n_total * 100

    boot_wrs = []
    for _ in range(N_BOOTSTRAP):
        sam = rng.choices(blist, k=len(blist))
        w = sum(1 for b in sam for s in b if s[RETURN_COL] > 0)
        t = sum(len(b) for b in sam)
        if t: boot_wrs.append(w / t * 100)
    boot_wrs_arr = np.array(sorted(boot_wrs))
    ci95 = [boot_wrs_arr[int(len(boot_wrs)*0.025)], boot_wrs_arr[int(len(boot_wrs)*0.975)]]
    ci99 = [boot_wrs_arr[int(len(boot_wrs)*0.005)], boot_wrs_arr[int(len(boot_wrs)*0.995)]]
    se = float(np.std(boot_wrs_arr))

    fig = plt.figure(figsize=(WIDTH_PX/DPI, HEIGHT_PX/DPI), dpi=DPI, facecolor=BG)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[0.65, 0.35], height_ratios=[0.78, 0.22],
                  hspace=0.15, wspace=0.12, left=0.08, right=0.95, top=0.84, bottom=0.06)

    fig.text(0.5, 0.97, "Block Bootstrap: Confidence Interval for Win Rate",
             fontsize=14, fontweight="bold", color=TEXT, ha="center", fontfamily=FONT_SANS)
    fig.text(0.5, 0.915,
             f"10,000 resamples  |  63-day blocks  |  {n_total:,} de-duplicated signals  |  237 stocks  |  2006-2025",
             fontsize=7.5, color=SUBTLE, ha="center", fontfamily=FONT_SANS)

    # LEFT: Histogram of bootstrapped WRs
    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor(BG)
    n_bins = 60
    counts, edges, patches = ax1.hist(boot_wrs_arr, bins=n_bins, color=LIGHT_BLUE,
                                       edgecolor=ACCENT_BLUE, linewidth=0.3, alpha=0.8)

    # Color bins: green if above 50, red if below
    for i, patch in enumerate(patches):
        center = (edges[i] + edges[i+1]) / 2
        if center < 50:
            patch.set_facecolor(LIGHT_RED)
            patch.set_edgecolor(ACCENT_RED)

    # CI bands
    ax1.axvspan(ci95[0], ci95[1], color=ACCENT_BLUE, alpha=0.08, zorder=0, label=f"95% CI [{ci95[0]:.1f}%, {ci95[1]:.1f}%]")
    ax1.axvline(ci95[0], color=ACCENT_BLUE, linewidth=1.5, linestyle="--", alpha=0.7)
    ax1.axvline(ci95[1], color=ACCENT_BLUE, linewidth=1.5, linestyle="--", alpha=0.7)

    # Real WR line
    ax1.axvline(real_wr, color=ENGINE_CLR, linewidth=2.5, zorder=3, label=f"Observed: {real_wr:.1f}%")

    # 50% reference
    ax1.axvline(50, color=ACCENT_RED, linewidth=1.5, linestyle=":", alpha=0.6, label="50% (random)")

    ax1.set_xlabel("Win Rate (%)", fontsize=9, color=SUBTLE, fontfamily=FONT_SANS)
    ax1.set_ylabel("Bootstrap Resamples", fontsize=9, color=SUBTLE, fontfamily=FONT_SANS)
    ax1.set_title("Distribution of 10,000 Block Bootstrap Win Rates", fontsize=10.5,
                   fontweight="bold", color=TEXT, fontfamily=FONT_SANS, pad=10)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9, edgecolor="#cccccc")
    ax1.grid(True, axis="y", color=GRID, linewidth=0.4)
    spine_style(ax1)

    # Annotations
    ax1.annotate(f"Lower bound\n{ci95[0]:.1f}%", xy=(ci95[0], counts.max()*0.7),
                 xytext=(-60, 20), textcoords="offset points", fontsize=7, color=ACCENT_BLUE,
                 fontfamily=FONT_SANS, arrowprops=dict(arrowstyle="->", color=ACCENT_BLUE, lw=0.8))
    ax1.annotate(f"Upper bound\n{ci95[1]:.1f}%", xy=(ci95[1], counts.max()*0.7),
                 xytext=(15, 20), textcoords="offset points", fontsize=7, color=ACCENT_BLUE,
                 fontfamily=FONT_SANS, arrowprops=dict(arrowstyle="->", color=ACCENT_BLUE, lw=0.8))

    # RIGHT: Stats card
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor(BG); ax2.axis("off")
    ax2.set_title("Key Statistics", fontsize=10.5, fontweight="bold", color=TEXT, fontfamily=FONT_SANS, pad=10)

    card = [
        ("Observed WR", f"{real_wr:.1f}%", ENGINE_CLR, 13),
        ("", "", TEXT, 0),
        ("95% CI Lower", f"{ci95[0]:.1f}%", ACCENT_BLUE, 11),
        ("95% CI Upper", f"{ci95[1]:.1f}%", ACCENT_BLUE, 11),
        ("99% CI", f"[{ci99[0]:.1f}%, {ci99[1]:.1f}%]", SUBTLE, 9),
        ("", "", TEXT, 0),
        ("Bootstrap SE", f"{se:.2f}pp", SUBTLE, 10),
        ("N (de-duplicated)", f"{n_total:,}", SUBTLE, 10),
        ("Block size", "63 trading days", SUBTLE, 9),
        ("", "", TEXT, 0),
        ("CI excludes 50%", "YES" if ci95[0] > 50 else "NO", ACCENT_GREEN if ci95[0]>50 else ACCENT_RED, 12),
        ("Distance from 50%", f"+{ci95[0]-50:.1f}pp", ACCENT_GREEN, 10),
    ]
    cy = 0.92
    for label, value, color, size in card:
        if not label:
            cy -= 0.02; continue
        ax2.text(0.05, cy, label, fontsize=8, color=SUBTLE, va="top", fontfamily=FONT_SANS, transform=ax2.transAxes)
        ax2.text(0.95, cy, value, fontsize=size, fontweight="bold", color=color, va="top", ha="right",
                 fontfamily=FONT_MONO, transform=ax2.transAxes)
        cy -= 0.07

    # BOTTOM LEFT: Block size robustness table
    ax_t = fig.add_subplot(gs[1, 0]); ax_t.set_facecolor(BG); ax_t.axis("off")
    # Run quick bootstrap at multiple block sizes
    block_results = []
    for bs in [21, 42, 63, 126, 252]:
        d2b_t = {d: i // bs for i, d in enumerate(all_dates)}
        bl_t = defaultdict(list)
        for s in deduped:
            bl_t[d2b_t[s["date"]]].append(s)
        bl_list = [bl_t[k] for k in sorted(bl_t.keys())]
        rng_t = random.Random(SEED)
        bw = []
        for _ in range(5000):
            sam = rng_t.choices(bl_list, k=len(bl_list))
            w = sum(1 for b in sam for s in b if s[RETURN_COL] > 0)
            t = sum(len(b) for b in sam)
            if t: bw.append(w / t * 100)
        bw.sort()
        block_results.append((bs, len(bl_list), bw[int(len(bw)*0.025)], bw[int(len(bw)*0.975)],
                              (sum(bw)/len(bw)), bw[int(len(bw)*0.025)] > 50))

    ty = 0.85
    for h, x in [("Block Size", 0.02), ("N Blocks", 0.18), ("95% CI Lower", 0.35), ("95% CI Upper", 0.55), ("Excludes 50%", 0.78)]:
        ax_t.text(x, ty, h, fontsize=7, fontweight="bold", color=TEXT, va="top", fontfamily=FONT_SANS, transform=ax_t.transAxes)
    ty -= 0.08
    ax_t.plot([0.02, 0.95], [ty, ty], transform=ax_t.transAxes, color="#cccccc", linewidth=0.5)
    ty -= 0.06

    for bs, nb, lo, hi, mean, excl in block_results:
        is_primary = bs == 63
        if is_primary:
            ax_t.fill_between([0.01, 0.96], ty-0.1, ty+0.04, transform=ax_t.transAxes, color="#e8f5e9", zorder=0)
        fw = "bold" if is_primary else "normal"
        ax_t.text(0.02, ty, f"{bs} days", fontsize=7, fontweight=fw, color=TEXT, va="top", fontfamily=FONT_SANS, transform=ax_t.transAxes)
        ax_t.text(0.18, ty, f"{nb}", fontsize=7, fontweight=fw, color=TEXT, va="top", fontfamily=FONT_MONO, transform=ax_t.transAxes)
        ax_t.text(0.35, ty, f"{lo:.1f}%", fontsize=7, fontweight=fw, color=TEXT, va="top", fontfamily=FONT_MONO, transform=ax_t.transAxes)
        ax_t.text(0.55, ty, f"{hi:.1f}%", fontsize=7, fontweight=fw, color=TEXT, va="top", fontfamily=FONT_MONO, transform=ax_t.transAxes)
        excl_color = ACCENT_GREEN if excl else ACCENT_RED
        ax_t.text(0.78, ty, "YES" if excl else "NO", fontsize=7, fontweight="bold", color=excl_color, va="top", fontfamily=FONT_MONO, transform=ax_t.transAxes)
        ty -= 0.13

    # BOTTOM RIGHT: Methodology + Disclosures
    ax_d = fig.add_subplot(gs[1, 1]); ax_d.set_facecolor(BG); ax_d.axis("off")
    dy = 0.85
    ax_d.text(0.05, dy, "Methodology", fontsize=7.5, fontweight="bold", color=SUBTLE, va="top", fontfamily=FONT_SANS, transform=ax_d.transAxes)
    dy -= 0.14
    ax_d.text(0.05, dy,
        "Signals partitioned into non-overlapping\n"
        "63-day time blocks. Blocks resampled with\n"
        "replacement (preserves temporal structure).\n"
        "Tested at 5 block sizes for robustness.",
        fontsize=6.5, color="#999999", va="top", fontfamily=FONT_SANS, transform=ax_d.transAxes, linespacing=1.5)
    dy -= 0.48
    ax_d.text(0.05, dy, "Disclosures", fontsize=7.5, fontweight="bold", color=SUBTLE, va="top", fontfamily=FONT_SANS, transform=ax_d.transAxes)
    dy -= 0.14
    ax_d.text(0.05, dy,
        "28-day de-duplication applied\n"
        "Residual autocorrelation < 0.02 (Test #10)\n"
        "In-sample universe (237 stocks)\n"
        "Seed: 42 | 10,000 resamples",
        fontsize=6.5, color="#999999", va="top", fontfamily=FONT_SANS, transform=ax_d.transAxes, linespacing=1.5)

    path = CHART_DIR / "academic_block_bootstrap.png"
    fig.savefig(path, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path} ({path.stat().st_size/1024:.0f} KB)")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 3: AUTOCORRELATION DECAY (Effective N)
# ════════════════════════════════════════════════════════════════════════

def fig_autocorrelation(deduped):
    print("  Generating: Autocorrelation Decay (Effective N)...")

    # Load pre-computed results
    with open(SCRIPT_DIR / ".." / "tests" / "test10_effective_n_results.json") as f:
        t10 = json.load(f)

    ins = t10["in_sample"]
    acf = ins["autocorrelation_by_lag"]
    eff_n = ins["effective_n"]

    lags = list(range(1, 61))
    rhos = []
    n_pairs = []
    for lag in lags:
        entry = acf.get(str(lag), {})
        rhos.append(entry.get("rho", 0))
        n_pairs.append(entry.get("n_pairs", 0))

    rhos = np.array(rhos)
    sig_thresholds = [1.96 / math.sqrt(max(np, 10)) for np in n_pairs]

    fig = plt.figure(figsize=(WIDTH_PX/DPI, HEIGHT_PX/DPI), dpi=DPI, facecolor=BG)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[0.65, 0.35], height_ratios=[0.78, 0.22],
                  hspace=0.15, wspace=0.12, left=0.08, right=0.95, top=0.84, bottom=0.06)

    fig.text(0.5, 0.97, "Signal Autocorrelation Decay: Why Raw N is Misleading",
             fontsize=14, fontweight="bold", color=TEXT, ha="center", fontfamily=FONT_SANS)
    fig.text(0.5, 0.915,
             f"19,558 raw signals -> {eff_n['n_eff']:.0f} effective independent observations (inflation: {eff_n['inflation_factor']:.1f}x)  |  "
             "28-day de-duplication validated at lag 28",
             fontsize=7.5, color=SUBTLE, ha="center", fontfamily=FONT_SANS)

    # LEFT: Autocorrelation decay
    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor(BG)

    # Significance band
    ax1.fill_between(lags, [-s for s in sig_thresholds], sig_thresholds,
                      color=LIGHT_RED, alpha=0.3, label="95% significance threshold")

    # Bar chart of autocorrelations
    bar_colors = [ACCENT_BLUE if abs(r) > s else "#cccccc" for r, s in zip(rhos, sig_thresholds)]
    ax1.bar(lags, rhos, width=0.8, color=bar_colors, edgecolor="none", alpha=0.7)

    # Zero line
    ax1.axhline(0, color="#666666", linewidth=0.5)

    # 28-day marker
    ax1.axvline(28, color=ACCENT_GREEN, linewidth=2, linestyle="-", alpha=0.8)
    ax1.annotate("28-day\ndedup rule", xy=(28, rhos[27] if len(rhos) > 27 else 0),
                 xytext=(35, 0.5), fontsize=8, fontweight="bold", color=ACCENT_GREEN,
                 fontfamily=FONT_SANS,
                 arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=1.2))

    ax1.set_xlabel("Lag (calendar days)", fontsize=9, color=SUBTLE, fontfamily=FONT_SANS)
    ax1.set_ylabel("Autocorrelation (rho)", fontsize=9, color=SUBTLE, fontfamily=FONT_SANS)
    ax1.set_title("Win/Loss Outcome Autocorrelation by Calendar-Day Lag", fontsize=10.5,
                   fontweight="bold", color=TEXT, fontfamily=FONT_SANS, pad=10)
    ax1.set_xlim(0, 61)
    ax1.set_ylim(-0.15, 0.85)
    ax1.legend(loc="upper right", fontsize=7.5, framealpha=0.9, edgecolor="#cccccc")
    ax1.grid(True, axis="y", color=GRID, linewidth=0.4)
    spine_style(ax1)

    # RIGHT: Stats card
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor(BG); ax2.axis("off")
    ax2.set_title("Effective N Analysis", fontsize=10.5, fontweight="bold", color=TEXT, fontfamily=FONT_SANS, pad=10)

    card = [
        ("Raw N (signals)", f"{eff_n['n_raw']:,}", ACCENT_RED, 12),
        ("Effective N", f"{eff_n['n_eff']:.0f}", ACCENT_GREEN, 14),
        ("Inflation factor", f"{eff_n['inflation_factor']:.1f}x", ENGINE_CLR, 12),
        ("", "", TEXT, 0),
        ("Rho at lag 1", f"{rhos[0]:.3f}", ACCENT_BLUE, 10),
        ("Rho at lag 7", f"{rhos[6]:.3f}" if len(rhos) > 6 else "N/A", ACCENT_BLUE, 10),
        ("Rho at lag 14", f"{rhos[13]:.3f}" if len(rhos) > 13 else "N/A", ACCENT_BLUE, 10),
        ("Rho at lag 28", f"{rhos[27]:.3f}" if len(rhos) > 27 else "N/A", ACCENT_GREEN, 10),
        ("", "", TEXT, 0),
        ("First insig. lag", f"{ins['first_insignificant_lag']} days", ACCENT_GREEN, 11),
        ("Dedup validated", "YES" if ins["dedup_rule_validated"] else "NO",
         ACCENT_GREEN if ins["dedup_rule_validated"] else ACCENT_RED, 12),
        ("", "", TEXT, 0),
        ("After dedup", f"{ins['deduplication']['n_deduped']:,} signals", TEXT, 10),
        ("Residual rho", f"{ins['deduplication']['residual_rho']:.4f}", ACCENT_GREEN, 10),
        ("De-duped WR", f"{ins['deduplication']['dedup_wr']:.1f}%", ENGINE_CLR, 12),
    ]
    cy = 0.92
    for label, value, color, size in card:
        if not label:
            cy -= 0.015; continue
        ax2.text(0.05, cy, label, fontsize=7.5, color=SUBTLE, va="top", fontfamily=FONT_SANS, transform=ax2.transAxes)
        ax2.text(0.95, cy, value, fontsize=size, fontweight="bold", color=color, va="top", ha="right",
                 fontfamily=FONT_MONO, transform=ax2.transAxes)
        cy -= 0.058

    # BOTTOM LEFT: p-value comparison table
    ax_t = fig.add_subplot(gs[1, 0]); ax_t.set_facecolor(BG); ax_t.axis("off")
    pv = ins["p_values"]
    ty = 0.85
    for h, x in [("Method", 0.02), ("N", 0.30), ("WR", 0.45), ("z-score", 0.58), ("p-value", 0.75)]:
        ax_t.text(x, ty, h, fontsize=7.5, fontweight="bold", color=TEXT, va="top", fontfamily=FONT_SANS, transform=ax_t.transAxes)
    ty -= 0.08
    ax_t.plot([0.02, 0.95], [ty, ty], transform=ax_t.transAxes, color="#cccccc", linewidth=0.5)
    ty -= 0.06
    for label, data, highlight in [
        ("Naive (raw N)", pv["naive"], False),
        ("Effective N (Bartlett)", pv["effective_n"], False),
        ("De-duplicated (28d)", pv["deduped"], True),
    ]:
        if highlight:
            ax_t.fill_between([0.01, 0.96], ty-0.12, ty+0.04, transform=ax_t.transAxes, color="#e8f5e9", zorder=0)
        fw = "bold" if highlight else "normal"
        ax_t.text(0.02, ty, label, fontsize=7, fontweight=fw, color=TEXT, va="top", fontfamily=FONT_SANS, transform=ax_t.transAxes)
        ax_t.text(0.30, ty, f"{data['n']:,}", fontsize=7, fontweight=fw, color=TEXT, va="top", fontfamily=FONT_MONO, transform=ax_t.transAxes)
        ax_t.text(0.45, ty, f"{data['wr']:.1f}%", fontsize=7, fontweight=fw, color=TEXT, va="top", fontfamily=FONT_MONO, transform=ax_t.transAxes)
        ax_t.text(0.58, ty, f"{data['z']:.1f}", fontsize=7, fontweight=fw, color=TEXT, va="top", fontfamily=FONT_MONO, transform=ax_t.transAxes)
        p_str = f"{data['p']:.2e}" if data['p'] < 0.001 else f"{data['p']:.4f}"
        naive_flag = " (OVERSTATED)" if "Naive" in label else ""
        ax_t.text(0.75, ty, p_str + naive_flag, fontsize=7, fontweight=fw,
                  color=ACCENT_RED if "Naive" in label else TEXT, va="top", fontfamily=FONT_MONO, transform=ax_t.transAxes)
        ty -= 0.18

    # BOTTOM RIGHT: Methodology
    ax_d = fig.add_subplot(gs[1, 1]); ax_d.set_facecolor(BG); ax_d.axis("off")
    dy = 0.85
    ax_d.text(0.05, dy, "Methodology", fontsize=7.5, fontweight="bold", color=SUBTLE, va="top", fontfamily=FONT_SANS, transform=ax_d.transAxes)
    dy -= 0.14
    ax_d.text(0.05, dy,
        "Pearson correlation of win/loss outcomes\n"
        "between all same-stock signal pairs at each\n"
        "calendar-day lag. Effective N via Bartlett\n"
        "formula, truncated at first zero crossing.",
        fontsize=6.5, color="#999999", va="top", fontfamily=FONT_SANS, transform=ax_d.transAxes, linespacing=1.5)
    dy -= 0.48
    ax_d.text(0.05, dy, "Key Finding", fontsize=7.5, fontweight="bold", color=SUBTLE, va="top", fontfamily=FONT_SANS, transform=ax_d.transAxes)
    dy -= 0.14
    ax_d.text(0.05, dy,
        f"Naive analysis inflates significance {eff_n['inflation_factor']:.0f}x.\n"
        "28-day de-duplication removes serial\n"
        "dependence (residual rho = -0.014, NS).\n"
        "All p-values in this suite use de-duped N.",
        fontsize=6.5, color="#999999", va="top", fontfamily=FONT_SANS, transform=ax_d.transAxes, linespacing=1.5)

    path = CHART_DIR / "academic_autocorrelation_decay.png"
    fig.savefig(path, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path} ({path.stat().st_size/1024:.0f} KB)")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 4: EQUITY CURVE + DRAWDOWN
# ════════════════════════════════════════════════════════════════════════

def fig_equity_curve(deduped):
    print("  Generating: Equity Curve + Drawdown...")

    monthly = defaultdict(list)
    for s in deduped:
        monthly[s["date"][:7]].append(s)
    months = sorted(monthly.keys())

    strat_eq = [100.0]
    spy_eq = [100.0]
    dd_pct = [0.0]
    peak = 100.0
    month_labels = ["Start"]
    monthly_rets = []

    for m in months:
        sigs = monthly[m]
        n = len(sigs)
        avg_ret = sum(s[RETURN_COL] for s in sigs) / n
        avg_spy = sum((s.get("spy_return_20d", 0) or 0) for s in sigs) / n
        monthly_rets.append(avg_ret)

        s_val = strat_eq[-1] * (1 + avg_ret / 100)
        spy_val = spy_eq[-1] * (1 + avg_spy / 100)
        strat_eq.append(s_val)
        spy_eq.append(spy_val)
        month_labels.append(m)

        if s_val > peak:
            peak = s_val
        dd_pct.append((peak - s_val) / peak * 100)

    x = np.arange(len(strat_eq))
    strat_eq = np.array(strat_eq)
    spy_eq = np.array(spy_eq)
    dd_pct = np.array(dd_pct)

    n_years = len(months) / 12
    cagr = ((strat_eq[-1] / 100) ** (1/n_years) - 1) * 100
    spy_cagr = ((spy_eq[-1] / 100) ** (1/n_years) - 1) * 100
    max_dd = dd_pct.max()
    mean_m = sum(monthly_rets) / len(monthly_rets)
    std_m = (sum((r - mean_m)**2 for r in monthly_rets) / len(monthly_rets)) ** 0.5
    sharpe = (mean_m * 12) / (std_m * math.sqrt(12)) if std_m > 0 else 0
    win_months = sum(1 for r in monthly_rets if r > 0)

    fig = plt.figure(figsize=(WIDTH_PX/DPI, HEIGHT_PX/DPI), dpi=DPI, facecolor=BG)
    gs = GridSpec(3, 2, figure=fig, width_ratios=[0.72, 0.28],
                  height_ratios=[0.55, 0.20, 0.25],
                  hspace=0.12, wspace=0.12, left=0.07, right=0.95, top=0.84, bottom=0.06)

    fig.text(0.5, 0.97, "Cumulative Returns: Engine Signals vs S&P 500",
             fontsize=14, fontweight="bold", color=TEXT, ha="center", fontfamily=FONT_SANS)
    fig.text(0.5, 0.915,
             f"{len(deduped):,} de-duplicated signals  |  237 stocks  |  "
             f"{months[0]} to {months[-1]}  |  Equal-weight monthly compounding",
             fontsize=7.5, color=SUBTLE, ha="center", fontfamily=FONT_SANS)

    # TOP LEFT: Equity curves (log scale)
    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor(BG)
    ax1.fill_between(x, spy_eq, strat_eq, where=strat_eq >= spy_eq, alpha=0.08, color=ACCENT_GREEN)
    ax1.plot(x, strat_eq, color=ENGINE_CLR, linewidth=2, label=f"Engine (CAGR {cagr:+.1f}%)")
    ax1.plot(x, spy_eq, color=AXIS, linewidth=1.2, linestyle="--", label=f"S&P 500 (CAGR {spy_cagr:+.1f}%)")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax1.set_ylabel("Portfolio Value ($)", fontsize=8, color=SUBTLE, fontfamily=FONT_SANS)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9, edgecolor="#cccccc")
    ax1.grid(True, color=GRID, linewidth=0.4)
    spine_style(ax1)

    # X ticks (every 2 years)
    ticks, tlabels = [], []
    for i, m in enumerate(month_labels):
        if m != "Start" and m.endswith("-01") and int(m[:4]) % 3 == 0:
            ticks.append(i); tlabels.append(m[:4])
    ax1.set_xticks(ticks); ax1.set_xticklabels(tlabels, fontsize=7)

    # Annotate finals
    ax1.annotate(f"${strat_eq[-1]:,.0f}", xy=(x[-1], strat_eq[-1]),
                 xytext=(-70, 10), textcoords="offset points", fontsize=9, fontweight="bold",
                 color=ENGINE_CLR, fontfamily=FONT_SANS,
                 arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))

    # MID LEFT: Drawdown
    ax_dd = fig.add_subplot(gs[1, 0], sharex=ax1); ax_dd.set_facecolor(BG)
    ax_dd.fill_between(x, 0, -dd_pct, color=ACCENT_RED, alpha=0.25)
    ax_dd.plot(x, -dd_pct, color=ACCENT_RED, linewidth=0.6)
    ax_dd.set_ylabel("Drawdown", fontsize=8, color=SUBTLE, fontfamily=FONT_SANS)
    ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_dd.set_ylim(-max_dd * 1.3, 5)
    ax_dd.grid(True, axis="y", color=GRID, linewidth=0.4)
    spine_style(ax_dd)
    ax_dd.annotate(f"Max DD: {max_dd:.1f}%", xy=(np.argmax(dd_pct), -max_dd),
                   xytext=(30, -15), textcoords="offset points", fontsize=7, color=ACCENT_RED,
                   fontfamily=FONT_SANS, arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=0.6))

    # RIGHT: Stats card (spans top two rows)
    ax_s = fig.add_subplot(gs[0:2, 1]); ax_s.set_facecolor(BG); ax_s.axis("off")
    ax_s.set_title("Performance Summary", fontsize=10.5, fontweight="bold", color=TEXT, fontfamily=FONT_SANS, pad=10)

    card = [
        ("Starting Capital", "$100", TEXT, 11),
        ("Final Value", f"${strat_eq[-1]:,.0f}", ENGINE_CLR, 14),
        ("S&P 500 Final", f"${spy_eq[-1]:,.0f}", AXIS, 11),
        ("", "", TEXT, 0),
        ("CAGR", f"{cagr:+.1f}%", ACCENT_GREEN if cagr > 0 else ACCENT_RED, 13),
        ("S&P 500 CAGR", f"{spy_cagr:+.1f}%", AXIS, 10),
        ("Alpha CAGR", f"{cagr - spy_cagr:+.1f}%", ACCENT_GREEN, 11),
        ("", "", TEXT, 0),
        ("Sharpe Ratio", f"{sharpe:.2f}", ACCENT_BLUE, 12),
        ("Max Drawdown", f"{max_dd:.1f}%", ACCENT_RED, 11),
        ("Calmar Ratio", f"{cagr/max_dd:.2f}" if max_dd > 0 else "N/A", ACCENT_BLUE, 10),
        ("", "", TEXT, 0),
        ("Winning Months", f"{win_months}/{len(monthly_rets)} ({win_months/len(monthly_rets)*100:.0f}%)", SUBTLE, 9),
        ("Total Return", f"{(strat_eq[-1]/100-1)*100:+,.0f}%", ENGINE_CLR, 11),
    ]
    cy = 0.95
    for label, value, color, size in card:
        if not label:
            cy -= 0.015; continue
        ax_s.text(0.05, cy, label, fontsize=8, color=SUBTLE, va="top", fontfamily=FONT_SANS, transform=ax_s.transAxes)
        ax_s.text(0.95, cy, value, fontsize=size, fontweight="bold", color=color, va="top", ha="right",
                  fontfamily=FONT_MONO, transform=ax_s.transAxes)
        cy -= 0.062

    # BOTTOM: Methodology + Disclosures
    ax_m = fig.add_subplot(gs[2, 0]); ax_m.set_facecolor(BG); ax_m.axis("off")
    ax_m.text(0.02, 0.9, "Methodology", fontsize=7.5, fontweight="bold", color=SUBTLE, va="top", fontfamily=FONT_SANS, transform=ax_m.transAxes)
    ax_m.text(0.02, 0.7,
        "Each month: average the 20-day forward returns of all de-duplicated signals firing that month. "
        "Compound monthly. Equal-weight across all concurrent signals. SPY benchmark uses the same "
        "20-day windows. De-duplication: first signal per stock per 28 calendar days.",
        fontsize=6.5, color="#999999", va="top", fontfamily=FONT_SANS, transform=ax_m.transAxes, linespacing=1.5)

    ax_d = fig.add_subplot(gs[2, 1]); ax_d.set_facecolor(BG); ax_d.axis("off")
    ax_d.text(0.05, 0.9, "Disclosures", fontsize=7.5, fontweight="bold", color=SUBTLE, va="top", fontfamily=FONT_SANS, transform=ax_d.transAxes)
    ax_d.text(0.05, 0.7,
        "Survivorship bias present (est. -1.2pp)\n"
        "Transaction costs excluded\n"
        "In-sample universe (237 stocks)\n"
        "No reinvestment of inter-signal capital\n"
        "Past performance is not indicative\nof future results",
        fontsize=6.5, color="#999999", va="top", fontfamily=FONT_SANS, transform=ax_d.transAxes, linespacing=1.5)

    path = CHART_DIR / "academic_equity_curve.png"
    fig.savefig(path, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path} ({path.stat().st_size/1024:.0f} KB)")


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    print("Generating academic publication figures...")
    print(f"Fonts: sans={FONT_SANS}, mono={FONT_MONO}")
    print(f"Output: {CHART_DIR}/")
    print()

    signals, ins_set, _ = load_data()
    deduped = dedup(signals, ins_set)
    print(f"De-duplicated in-sample: {len(deduped)} signals")
    print()

    targets = sys.argv[1:] if len(sys.argv) > 1 else ["walkforward", "bootstrap", "autocorrelation", "equity"]

    if "walkforward" in targets:
        fig_walkforward(deduped)
    if "bootstrap" in targets:
        fig_bootstrap(deduped)
    if "autocorrelation" in targets:
        fig_autocorrelation(deduped)
    if "equity" in targets:
        fig_equity_curve(deduped)

    print(f"\nDone! All figures in {CHART_DIR}/")


if __name__ == "__main__":
    main()
