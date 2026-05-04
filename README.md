# Mitigating and Simulating Algorithmic Price Collusion

**Dylan Bober — Yale University**

Algorithmic pricing agents trained via Q-learning reliably converge to tacit collusion, yet regulators currently lack effective tools to counteract this behavior. This project asks whether a platform-imposed price threshold can meaningfully reduce collusion among competing Q-learning sellers. Drawing on the Calvano et al. (2020) framework and extending it with a dynamic platform learning mechanism, we simulate markets of two to five sellers under logit and winner-take-all demand.

We find that a threshold set to exclude high-pricing sellers reduces collusion by up to 62% in five-seller logit markets, with consumer surplus rising by over 23%. The mechanism is most effective when the number of sellers meaningfully exceeds the platform's target eligible count. These results suggest that  price thresholds represent an effective, market-compatible policy 
instrument---one that regulators could plausibly mandate without requiring direct access to a firm's pricing algorithms.


The full paper is available at [`Report/Report.pdf`](Report/Report.pdf).

---

## Model overview

The platform sets a price threshold τ each period. Sellers above the threshold are excluded from the choice set (and earn zero demand). The platform adjusts τ dynamically using a proportional controller:

```
τ_{t+1} = clip(τ_t − α·S_O − δ,  c_min, p_max)
```

where S_O = S_t − S_target is the oversupply signal (observed eligible sellers minus target), α is a learning rate, and δ = 0.0005 is a constant downward drift so τ never stalls when S_O ≈ 0.

Sellers are Q-learning agents with optimistic initialization (Q_INIT = π_m / (1−γ)), which replicates the Calvano et al. result that agents discover and sustain tacit collusion.

Two demand models are compared:

| Model | Mechanism |
|---|---|
| **Logit** (default) | Each seller's demand share is proportional to exp((a − p_i)/μ); ineligible sellers removed from the choice set |
| **Winner-take-all** | Lowest-price eligible seller captures all demand |

---

## Key results

| Config | Control CI | Treatment CI | Reduction | CS Δ |
|---|---|---|---|---|
| 3-seller Logit | 0.710 | 0.422 | −0.288 | +25.2% |
| 5-seller Logit | 0.681 | 0.261 | −0.419 | +23.2% |
| 3-seller WTA   | 0.620 | 0.414 | −0.206 | −7.2%  |
| 5-seller WTA   | 0.632 | 0.242 | −0.390 | +6.0%  |

Where CI represents the Collusion Index measure.

Threshold effectiveness scales strongly with N − S_target (the number of sellers the platform can credibly exclude).

---

## Running the simulation

### Requirements

- Python 3.9+
- `numpy`, `matplotlib`, `scipy`

Install dependencies:
```bash
pip install numpy matplotlib scipy
```

### Run all seller configurations

```bash
python simulation.py
```

This runs 2-, 3-, and 5-seller markets for the configured demand model(s), and storing results in `results/`.

### Run Q-init sensitivity experiment

```bash
python experiment_qinit.py
```

Compares monopoly, random, and zero Q-table initialization strategies. Writes results to `results_qinit/`.

### Key config flags (`config.py`)

| Flag | Default | Effect |
|---|---|---|
| `DEMAND_MODEL` | `"logit"` | `"logit"` or `"wta"` |
| `RUN_BOTH_MODELS` | `True` | Runs logit and WTA back-to-back |
| `COST_MIN` / `COST_MAX` | `0.05` / `0.15` | Seller cost distribution bounds |
| `S_TARGET_BY_SELLERS` | `{2:2, 3:2, 5:3}` | Target eligible sellers per config |
| `TAU_DRIFT` | `0.0005` | Constant downward drift on τ |
| `EPS_END` | `0.005` | Final exploration probability |