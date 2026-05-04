"""
diagnose_oscillation.py — Diagnostic for 3-seller WTA oscillation and end-spike.

Runs a short simulation (1M periods) with N_SELLERS=3, DEMAND_MODEL="wta"

Produces two figures:
  /tmp/diag_raw_vs_smooth.png  — raw vs smoothed collusion index (last 30%)
  /tmp/diag_price_hist.png     — price action histogram at end of training
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Patch config before importing simulation
import config
config.N_SELLERS        = 3
config.N_PERIODS        = 1_000_000
config.DEMAND_MODEL     = "wta"
config.RUN_BOTH_MODELS  = False
config.S_TARGET         = config.S_TARGET_BY_SELLERS[3]

# Recompute derived constants that depend on N_SELLERS
config.N_STATES = (config.N_PRICE_LEVELS ** config.N_SELLERS) * config.N_TAU_BINS
config._qtable_mb = config.N_STATES * config.N_PRICE_LEVELS * 8 / 1e6

import simulation as sim

# Re-bind module globals that were already imported by simulation
sim.N_SELLERS   = config.N_SELLERS
sim.N_PERIODS   = config.N_PERIODS
sim.N_STATES    = config.N_STATES
sim.DEMAND_MODEL = "wta"
sim.S_TARGET    = config.S_TARGET

print("Running 3-seller WTA (1M periods) ...")
ctrl  = sim.run_simulation(use_threshold=False, seed=42)
treat = sim.run_simulation(use_threshold=True,  seed=42)

# ── Figure 1: raw vs smoothed collusion index (last 30%) ──────────────────────
from config import RECORD_EVERY, ROLLING_WINDOW

def smooth(arr, window_periods):
    w = max(1, window_periods // RECORD_EVERY)
    s = np.convolve(arr, np.ones(w) / w, mode="valid")
    pad = len(arr) - len(s)
    l = pad // 2
    return np.concatenate([np.full(l, s[0]), s, np.full(pad - l, s[-1])])

n = len(ctrl["collusion_index"])
start = int(n * 0.70)    # show last 30%
x = np.arange(n)[start:] * RECORD_EVERY / 1_000

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("3-Seller WTA — Raw vs Smoothed Collusion Index (last 30%)\n"
             "Spike = smoothing padding artifact if raw data is still oscillating",
             fontsize=11)

for ax, data, label in zip(axes,
                           [ctrl, treat],
                           ["No Threshold (control)", "With Threshold"]):
    raw = data["collusion_index"][start:]
    sm  = smooth(data["collusion_index"], ROLLING_WINDOW)[start:]
    ax.plot(x, raw, color="lightgray", lw=0.6, label="Raw")
    ax.plot(x, sm,  color="#1f77b4",   lw=1.8, label="Smoothed")
    ax.axvline(x[-int((ROLLING_WINDOW // RECORD_EVERY) // 2)],
               color="red", ls="--", lw=1.0, alpha=0.7,
               label="Smoothing pad boundary")
    ax.set(title=label, xlabel="Periods (thousands)", ylabel="Collusion Index")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("/tmp/diag_raw_vs_smooth.png", dpi=130, bbox_inches="tight")
print("Saved → /tmp/diag_raw_vs_smooth.png")

# ── Figure 2: oscillation amplitude over time ─────────────────────────────────
window = 100   # records per rolling-std window
def rolling_std(arr, w):
    return np.array([arr[max(0,i-w):i].std() for i in range(1, len(arr)+1)])

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("3-Seller WTA — Rolling Std Dev of Collusion Index\n"
             "High std = ongoing oscillation (Edgeworth cycles)", fontsize=11)

for ax, data, label in zip(axes,
                            [ctrl, treat],
                            ["No Threshold (control)", "With Threshold"]):
    x_all = np.arange(n) * RECORD_EVERY / 1_000
    ax.plot(x_all, rolling_std(data["collusion_index"], window),
            color="#d62728", lw=1.2)
    ax.set(title=label, xlabel="Periods (thousands)", ylabel="Std Dev (collusion index)")

plt.tight_layout()
plt.savefig("/tmp/diag_oscillation_std.png", dpi=130, bbox_inches="tight")
print("Saved → /tmp/diag_oscillation_std.png")
plt.show()
