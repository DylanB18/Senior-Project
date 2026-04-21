"""
plotting.py — Visualisation and summary table for simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    N_SELLERS, N_PRICE_LEVELS, N_PERIODS,
    MAX_PRICE, MARGINAL_COST, COMMISSION_RATE,
    MONOPOLY_PRICE, RECORD_EVERY, ROLLING_WINDOW,
)


def _smooth(arr: np.ndarray, window_periods: int) -> np.ndarray:
    """
    Apply a centered moving-average over `window_periods` raw periods.
    Uses mode="valid" to avoid zero-padding artifacts at the edges, then
    pads the result back to the original length by repeating the first/last value.
    """
    w = max(1, window_periods // RECORD_EVERY)
    smoothed = np.convolve(arr, np.ones(w) / w, mode="valid")
    pad = len(arr) - len(smoothed)
    left = pad // 2
    right = pad - left
    return np.concatenate([
        np.full(left, smoothed[0]),
        smoothed,
        np.full(right, smoothed[-1]),
    ])


def plot_results(
    ctrl: dict,
    treat: dict,
    save_path: str = "simulation_results.png",
    demand_label: str = "",
) -> None:
    """
    Generate a 5-panel figure comparing the control (no threshold) and
    treatment (with threshold) experiments.
    """
    n_records   = len(ctrl["avg_price"])
    x           = np.arange(n_records) * RECORD_EVERY / 1_000  # x-axis in thousands of periods
    sm          = lambda arr: _smooth(arr, ROLLING_WINDOW)
    RED, BLUE   = "#d62728", "#1f77b4"
    P_M, P_C    = MONOPOLY_PRICE, MARGINAL_COST

    demand_tag = f" — {demand_label}" if demand_label else ""
    fig = plt.figure(figsize=(15, 13))
    fig.suptitle(
        f"Collusion vs. Dynamic Threshold Mechanism{demand_tag}\n"
        f"{N_SELLERS} sellers · {N_PRICE_LEVELS} price levels · {N_PERIODS:,} periods",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

    # Average price (full-width top panel)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(x, sm(ctrl["avg_price"]),  color=RED,  lw=1.8, label="No Threshold (control)")
    ax.plot(x, sm(treat["avg_price"]), color=BLUE, lw=1.8, label="With Threshold (treatment)")
    ax.plot(x, sm(treat["threshold"]), color=BLUE, lw=1.2, ls="--", alpha=0.65, label="Threshold τ_t")
    ax.axhline(P_M, ls=":", color="gray",  lw=1.0, label=f"Monopoly  p_m = {P_M:.2f}")
    ax.axhline(P_C, ls=":", color="black", lw=1.0, label=f"Competitive p_c = {P_C:.2f}")
    ax.set(title="Average Seller Price Over Time",
           xlabel="Periods (thousands)", ylabel="Price",
           ylim=(0, MAX_PRICE + 0.06))
    ax.legend(fontsize=8, loc="upper right")

    # Collusion index
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x, sm(ctrl["collusion_index"]),  color=RED,  lw=1.8, label="No Threshold")
    ax.plot(x, sm(treat["collusion_index"]), color=BLUE, lw=1.8, label="With Threshold")
    ax.axhline(1, ls=":", color="gray",  lw=1.0, label="Monopoly (1.0)")
    ax.axhline(0, ls=":", color="black", lw=1.0, label="Competitive (0.0)")
    ax.set(title="Collusion Index\n(0 = competitive,  1 = monopoly)",
           xlabel="Periods (thousands)", ylabel="Index",
           ylim=(-0.1, 1.2))
    ax.legend(fontsize=8)

    # Consumer surplus
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x, sm(ctrl["consumer_surplus"]),  color=RED,  lw=1.8, label="No Threshold")
    ax.plot(x, sm(treat["consumer_surplus"]), color=BLUE, lw=1.8, label="With Threshold")
    ax.set(title="Consumer Surplus",
           xlabel="Periods (thousands)", ylabel="Surplus")
    ax.legend(fontsize=8)

    # Platform revenue
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(x, sm(ctrl["platform_revenue"]),  color=RED,  lw=1.8, label="No Threshold")
    ax.plot(x, sm(treat["platform_revenue"]), color=BLUE, lw=1.8, label="With Threshold")
    ax.set(title=f"Platform Revenue (r={COMMISSION_RATE})",
           xlabel="Periods (thousands)", ylabel="Revenue")
    ax.legend(fontsize=8)

    # Total seller profit
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(x, sm(ctrl["seller_profit"]),  color=RED,  lw=1.8, label="No Threshold")
    ax.plot(x, sm(treat["seller_profit"]), color=BLUE, lw=1.8, label="With Threshold")
    ax.set(title="Total Seller Profit per Period",
           xlabel="Periods (thousands)", ylabel="Profit")
    ax.legend(fontsize=8)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {save_path}")


def print_summary(
    ctrl: dict,
    treat: dict,
    demand_label: str = "",
    save_path: str | None = None,
    config_info: dict | None = None,
) -> None:
    """Print a table comparing final-period averages and optionally save to a text file."""
    tail = slice(-max(1, len(ctrl["avg_price"]) // 10), None)
    avg  = lambda d, k: float(np.mean(d[k][tail]))
    std  = lambda d, k: float(np.std(d[k][tail]))

    def pct_change(ctrl_val, treat_val):
        """Return a formatted % change string, or absolute change for index metrics."""
        if abs(ctrl_val) < 1e-9:
            return "     N/A"
        return f"{(treat_val - ctrl_val) / abs(ctrl_val) * 100:>+7.1f}%"

    rows = [
        ("Avg Seller Price",    "avg_price",       "pct"),
        ("Collusion Index",     "collusion_index",  "abs"),
        ("Consumer Surplus",    "consumer_surplus", "pct"),
        ("Platform Revenue",    "platform_revenue", "pct"),
        ("Total Seller Profit", "seller_profit",    "pct"),
    ]

    W = 74
    demand_tag = f" — {demand_label}" if demand_label else ""

    # Config block (printed only if config_info provided)
    config_lines = []
    if config_info:
        costs_arr = config_info.get("costs", np.array([]))
        costs_str = "  ".join(f"{c:.4f}" for c in costs_arr)
        config_lines = [
            "=" * W,
            "  CONFIG",
            f"  N_SELLERS : {config_info.get('n_sellers', '?'):<4}  "
            f"S_TARGET : {config_info.get('s_target', '?'):<4}  "
            f"F_FEE : {config_info.get('f_fee', 0):.4f}  "
            f"TAU_LR : {config_info.get('tau_lr', 0):.4f}",
            f"  Costs     : [{costs_str}]  mean = {costs_arr.mean():.4f}",
            f"  N_PERIODS : {config_info.get('n_periods', '?'):,}",
        ]

    result_lines = [
        "=" * W,
        f"  RESULTS{demand_tag}  (last 10% of periods)",
        "=" * W,
        f"  {'METRIC':<26} {'NO THRESHOLD':>14} {'WITH THRESHOLD':>14} {'CHANGE':>10}",
        "-" * W,
        *[
            f"  {lbl:<26} {avg(ctrl, key):>14.4f} {avg(treat, key):>14.4f} "
            f"{'  {:+.3f}'.format(avg(treat, key) - avg(ctrl, key)):>10}"
            if kind == "abs" else
            f"  {lbl:<26} {avg(ctrl, key):>14.4f} {avg(treat, key):>14.4f} "
            f"{pct_change(avg(ctrl, key), avg(treat, key)):>10}"
            for lbl, key, kind in rows
        ],
        f"  {'Avg Price Std Dev':<26} {std(ctrl, 'avg_price'):>14.4f} {std(treat, 'avg_price'):>14.4f}",
        "-" * W,
        f"  {'Monopoly price  p_m':<26} {MONOPOLY_PRICE:>14.4f} {MONOPOLY_PRICE:>14.4f}",
        f"  {'Mean cost ref  c̄':<26} {MARGINAL_COST:>14.4f} {MARGINAL_COST:>14.4f}",
        f"  {'Threshold τ  (mean)':<26} {'N/A':>14} {avg(treat, 'threshold'):>14.4f}",
        f"  {'Threshold τ  (std dev)':<26} {'N/A':>14} {std(treat, 'threshold'):>14.4f}",
        "=" * W,
    ]

    lines = config_lines + result_lines
    output = "\n".join(lines)
    print(output)
    if save_path:
        with open(save_path, "w") as f:
            f.write(output + "\n")
        print(f"Summary saved → {save_path}")
