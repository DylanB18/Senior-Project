"""
experiment_qinit.py — Q-table initialisation strategy experiment
=================================================================
Tests whether initialising the Q-table at monopoly prices (optimistic)
vs. zero vs. random affects collusive outcomes over a shorter training run.

Three strategies:
  monopoly  Q_INIT = π_m / (1−γ)   — optimistic, current default (Calvano)
  zero      Q_INIT = 0              — pessimistic baseline
  random    Q values ~ Uniform[0, Q_INIT]

Results are written to results_qinit/ and do not touch the main results pipeline.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import simulation
import plotting as _plotting_mod
from config import (
    N_SELLERS, N_PRICE_LEVELS, COST_MIN, COST_MAX, COST_SEED,
    MARGINAL_COST, MAX_PRICE, LOGIT_A, LOGIT_MU, LOGIT_A0,
    TAU_LR, TAU_DRIFT, F_FEE, COMMISSION_RATE,
    GAMMA, RECORD_EVERY, ROLLING_WINDOW,
    S_TARGET_BY_SELLERS, S_TARGET, DEMAND_MODEL,
)
from plotting import print_summary

# ── Experiment settings ──────────────────────────────────────────────────────
SHORT_PERIODS = 20_000_000
RESULTS_DIR   = "results_qinit"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Recompute derived constants for current N_SELLERS ────────────────────────
np.random.seed(COST_SEED)
costs = np.random.uniform(COST_MIN, COST_MAX, N_SELLERS)

_p          = np.linspace(MARGINAL_COST, MAX_PRICE, 10_000)
_u          = np.exp((LOGIT_A - _p) / LOGIT_MU)
_sym_d      = _u / (np.exp(LOGIT_A0 / LOGIT_MU) + N_SELLERS * _u)
_sym_profit = (_p - MARGINAL_COST) * _sym_d
mono_price  = float(_p[np.argmax(_sym_profit)])
mono_profit = float(_sym_profit.max())
q_init_mono = mono_profit / (1 - GAMMA)
n_states    = (N_PRICE_LEVELS ** N_SELLERS) * 10  # N_TAU_BINS = 10
s_target    = S_TARGET_BY_SELLERS.get(N_SELLERS, S_TARGET)

# Push updated values into simulation module so all functions pick them up
simulation.N_PERIODS      = SHORT_PERIODS
simulation.N_SELLERS      = N_SELLERS
simulation.N_STATES       = n_states
simulation.MONOPOLY_PRICE = mono_price
simulation.Q_INIT         = q_init_mono
simulation.S_TARGET       = s_target

_plotting_mod.N_PERIODS      = SHORT_PERIODS
_plotting_mod.MONOPOLY_PRICE = mono_price
_plotting_mod.N_SELLERS      = N_SELLERS

# ── Agent subclasses for each init strategy ──────────────────────────────────

class _MonopolyInitAgent(simulation.QLearningAgent):
    """Q = π_m / (1−γ) everywhere (monopoly prices). This is Calvano's original strategy."""
    pass  # unchanged — inherits __init__ which already uses simulation.Q_INIT


class _ZeroInitAgent(simulation.QLearningAgent):
    """All Q-values start at 0."""
    def __init__(self):
        self.Q = np.zeros(
            (simulation.N_STATES, simulation.N_PRICE_LEVELS), dtype=np.float32
        )


class _RandomInitAgent(simulation.QLearningAgent):
    """Random: Q-values drawn from Uniform[0, Q_INIT]."""
    def __init__(self):
        self.Q = np.random.uniform(
            0, simulation.Q_INIT,
            (simulation.N_STATES, simulation.N_PRICE_LEVELS),
        ).astype(np.float32)


STRATEGIES = [
    ("monopoly", _MonopolyInitAgent),
    ("zero",     _ZeroInitAgent),
    ("random",   _RandomInitAgent),
]

# ── Run experiments ───────────────────────────────────────────────────────────
print("=" * 64)
print("  Q-Init Experiment")
print("=" * 64)
print(f"  N_SELLERS    : {N_SELLERS}")
print(f"  S_TARGET     : {s_target}")
print(f"  Costs        : {np.round(costs, 4)}  (mean={costs.mean():.4f})")
print(f"  Short periods: {SHORT_PERIODS:,}  (main run: {20_000_000:,})")
print(f"  Demand model : {DEMAND_MODEL}")
print(f"  Monopoly px  : {mono_price:.4f}")
print(f"  Q_INIT       : {q_init_mono:.4f}")
print()

results = {}
for name, AgentClass in STRATEGIES:
    simulation.QLearningAgent = AgentClass
    print(f"─── Init strategy: {name} ───")

    ctrl  = simulation.run_simulation(use_threshold=False, seed=42, costs=costs)
    treat = simulation.run_simulation(use_threshold=True,  seed=42, costs=costs)
    results[name] = (ctrl, treat)

    txt_path = os.path.join(RESULTS_DIR, f"qinit_{name}_{DEMAND_MODEL}.txt")
    print_summary(
        ctrl, treat,
        demand_label=f"{DEMAND_MODEL} / {name} init",
        save_path=txt_path,
        config_info={
            "n_sellers": N_SELLERS,
            "s_target":  s_target,
            "costs":     costs,
            "f_fee":     F_FEE,
            "tau_lr":    TAU_LR,
            "n_periods": SHORT_PERIODS,
        },
    )
    print()

# ── Comparison plot ───────────────────────────────────────────────────────────
n_records = len(next(iter(results.values()))[0]["collusion_index"])
x = np.arange(n_records) * RECORD_EVERY / 1_000

def _smooth(arr):
    w = max(1, ROLLING_WINDOW // RECORD_EVERY)
    s = np.convolve(arr, np.ones(w) / w, mode="valid")
    pad = len(arr) - len(s)
    left = pad // 2
    return np.concatenate([np.full(left, s[0]), s, np.full(pad - left, s[-1])])

COLORS = {"monopoly": "#d62728", "zero": "#1f77b4", "random": "#2ca02c"}
LABELS = {"monopoly": "Monopoly init", "zero": "Zero init", "random": "Random init"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.suptitle(
    f"Q-Table Init Strategy — Collusion Index over Time\n"
    f"{N_SELLERS} sellers · {SHORT_PERIODS:,} periods · {DEMAND_MODEL} demand",
    fontsize=12, fontweight="bold",
)

panel_titles = ["No Threshold (control)", "With Threshold (treatment)"]
result_keys  = ["ctrl", "treat"]

for ax, panel_title, result_key in zip(axes, panel_titles, result_keys):
    for name, _ in STRATEGIES:
        ctrl_d, treat_d = results[name]
        data = ctrl_d if result_key == "ctrl" else treat_d
        ax.plot(x, _smooth(data["collusion_index"]),
                color=COLORS[name], lw=1.8, label=LABELS[name])
    ax.axhline(1.0, ls=":", color="gray",  lw=1.0, label="Monopoly (1.0)")
    ax.axhline(0.0, ls=":", color="black", lw=1.0, label="Competitive (0.0)")
    ax.set(title=panel_title,
           xlabel="Periods (thousands)", ylabel="Collusion Index",
           ylim=(-0.15, 1.25))
    ax.legend(fontsize=9)

plt.tight_layout()
img_path = os.path.join(RESULTS_DIR, f"qinit_comparison_{DEMAND_MODEL}.png")
plt.savefig(img_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Figure saved → {img_path}")
