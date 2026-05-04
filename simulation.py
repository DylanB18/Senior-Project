"""
Q-Learning Simulation: Algorithmic Collusion and the Threshold Mechanism
=========================================================================

Two experiments run back-to-back with identical Q-learning sellers:

  Control (no threshold):
    All sellers compete via logit demand — each earns a demand share
    proportional to their price attractiveness.  No single winner; all sellers
    earn simultaneously.
    Hypothesis: sellers learn to tacitly collude at supra-competitive prices.

  Treatment (threshold mechanism):
    Platform enforces a dynamic price threshold τ_t.  Only sellers pricing at
    or below τ_t are eligible; demand is allocated among eligible sellers via
    restricted logit (ineligible sellers are removed from consumers' choice
    set).  Lower prices within the eligible set still attract higher share.
    Threshold updates each period proportionally to seller oversupply.
    Hypothesis: threshold drives prices toward competitive equilibrium.

"""

import time
import numpy as np

from config import (
    N_SELLERS, MARGINAL_COST, COST_MIN, COST_MAX, COST_SEED, MAX_PRICE, N_PRICE_LEVELS,
    Q_LR, GAMMA, EPS_START, EPS_END, N_PERIODS,
    TAU_LR, TAU_DRIFT, S_TARGET, S_TARGET_BY_SELLERS, TAU_INIT, N_TAU_BINS,
    COMMISSION_RATE, F_FEE, RECORD_EVERY,
    PRICE_GRID, MONOPOLY_PRICE, N_STATES, _qtable_mb,
    LOGIT_A, LOGIT_MU, LOGIT_A0, Q_INIT, DEMAND_MODEL, RUN_BOTH_MODELS,
)
from plotting import plot_results, print_summary

# Helper functions for demand, welfare, and state encoding

def logit_demands(prices: np.ndarray, subset: np.ndarray | None = None) -> np.ndarray:
    """
    Logit demand share for each seller (based on Calvano paper).
    D_i = exp((a − p_i) / μ) / (outside + Σ_j exp((a − p_j) / μ))

    If `subset` is provided (an array of seller indices), only those sellers
    compete for demand; all others receive a share of zero. This implements
    restricted logit for the threshold case.
    """
    u = np.exp((LOGIT_A - prices) / LOGIT_MU)
    outside = np.exp(LOGIT_A0 / LOGIT_MU)
    if subset is None:
        return u / (outside + u.sum())
    shares = np.zeros_like(prices)
    shares[subset] = u[subset] / (outside + u[subset].sum())
    return shares


def consumer_surplus(prices: np.ndarray) -> float:
    """
    Expected consumer surplus via the logit inclusive value.
    CS = μ · log(1 + Σ_j exp((a − p_j) / μ))
    Equals 0 when no sellers participate; increases as prices fall.
    """
    u = np.exp((LOGIT_A - prices) / LOGIT_MU)
    return LOGIT_MU * np.log(1.0 + u.sum())


def collusion_index(avg_price: float) -> float:
    """
    Normalised price index:
      0  →  competitive  (avg_price = MARGINAL_COST)
      1  →  monopoly     (avg_price = MONOPOLY_PRICE)
    """
    return (avg_price - MARGINAL_COST) / (MONOPOLY_PRICE - MARGINAL_COST)

def tau_to_bin(tau: float) -> int:
    """Map a continuous threshold value to a discrete bin index in [0, N_TAU_BINS)."""
    frac = (tau - MARGINAL_COST) / (MAX_PRICE - MARGINAL_COST)
    return int(np.clip(int(frac * N_TAU_BINS), 0, N_TAU_BINS - 1))

def encode_state(price_indices: tuple, tau_bin: int) -> int:
    """
    Encode the joint state (p1_idx, p2_idx, …, pN_idx, τ_bin) as a single
    scalar index using mixed-radix encoding.
    """
    idx = 0
    for p in price_indices:
        idx = idx * N_PRICE_LEVELS + p
    return idx * N_TAU_BINS + tau_bin


# Q-Learning agent

class QLearningAgent:
    """
    Tabular Q-learning agent with greedy action selection.

    State  : encoded index representing all sellers' last prices + τ bin.
    Action : price index ∈ {0, …, N_PRICE_LEVELS − 1}.
    Reward : profit = (p_i − c) × D_i, where D_i is the seller's logit demand
             share (restricted to threshold eligible sellers).
    """

    def __init__(self) -> None:
        # Optimistic initialisation: every Q-value starts at the expected
        # long-run monopoly payoff so agents initially favour high prices.
        self.Q = np.full((N_STATES, N_PRICE_LEVELS), Q_INIT, dtype=np.float32)

    def act(self, state_idx: int, epsilon: float) -> int:
        """ε-greedy action selection."""
        if np.random.rand() < epsilon:
            return np.random.randint(N_PRICE_LEVELS)
        return int(np.argmax(self.Q[state_idx]))

    def update(self, s: int, a: int, reward: float, s_next: int) -> None:
        """Standard Q-learning update."""
        td_target = reward + GAMMA * np.max(self.Q[s_next])
        self.Q[s, a] += Q_LR * (td_target - self.Q[s, a])


# Market period step

def market_step(
    price_indices: tuple,
    tau: float,
    use_threshold: bool,
    costs: np.ndarray,
) -> tuple:
    """
    Resolve one market period.

    Without threshold (control):
        All sellers compete via full logit demand.  Each seller earns a demand
        share and profit simultaneously; there is no single winner.
        winner_price is the demand-weighted average price.

    With threshold mechanism (treatment):
        Only sellers with p_i ≤ τ_t are eligible; demand is allocated among
        them via restricted logit (ineligible sellers have share = 0).  Lower
        prices within the eligible set still attract higher share.  If no
        seller is eligible, no sale occurs and the threshold is raised.

        Threshold update:
            τ_{t+1} = clip( τ_t − α · (S_t − S_target),  c_min,  p_max )

    Returns
    -------
    rewards      : per-seller profit array  (profit = (p_i − c_i − f) × D_i)
    winner_price : demand-weighted average price among active sellers (0 if no sale)
    new_tau      : updated threshold
    sold         : whether a transaction occurred
    """
    prices = PRICE_GRID[np.array(price_indices)]
    rewards = np.zeros(N_SELLERS)
    demands = np.zeros(N_SELLERS)
    sold = False
    winner_price = 0.0

    if use_threshold:
        eligible = np.where(prices <= tau)[0]
        if len(eligible) > 0:
            if DEMAND_MODEL == "logit":
                # Restricted logit: only eligible sellers compete for demand.
                # Ineligible sellers are removed from the choice set (share = 0).
                demands = logit_demands(prices, eligible)
                for i in eligible:
                    rewards[i] = (prices[i] - costs[i] - F_FEE) * demands[i]
                winner_price = float(
                    np.dot(prices[eligible], demands[eligible] / demands[eligible].sum())
                )
            else:
                # WTA: lowest-priced eligible seller wins all demand.
                winner = eligible[np.argmin(prices[eligible])]
                total_demand = logit_demands(prices, eligible).sum()
                demands[winner] = total_demand
                rewards[winner] = (prices[winner] - costs[winner] - F_FEE) * total_demand
                winner_price = float(prices[winner])
            sold = True

        # Proportional threshold update with constant downward drift.
        # Even when S_O = 0, the platform nudges τ downward by TAU_DRIFT
        S_O = len(eligible) - S_TARGET  # Seller oversupply
        new_tau = float(np.clip(tau - TAU_LR * S_O - TAU_DRIFT, COST_MIN, MAX_PRICE))

    else:
        if DEMAND_MODEL == "logit":
            # All sellers earn simultaneously via logit demand shares.
            demands = logit_demands(prices)
            for i in range(N_SELLERS):
                rewards[i] = (prices[i] - costs[i] - F_FEE) * demands[i]
            winner_price = float(np.dot(prices, demands))  # demand-weighted average price
        else:
            # WTA: lowest-price seller wins all demand. Ties broken uniformly at random.
            min_price = prices.min()
            tied = np.where(prices == min_price)[0]
            winner = np.random.choice(tied)
            demands = logit_demands(prices)
            total_demand = demands.sum()
            demands[:] = 0.0
            demands[winner] = total_demand
            rewards[winner] = (prices[winner] - costs[winner] - F_FEE) * total_demand
            winner_price = float(prices[winner])
        sold = True
        new_tau = tau  # No threshold, so τ stays the same

    return rewards, winner_price, new_tau, sold, demands


# Simulation that loops through periods, collects metrics, and trains the agents

def run_simulation(use_threshold: bool, seed: int = 0, costs: np.ndarray | None = None) -> dict:
    """Run a full Q-learning simulation for one experiment."""
    np.random.seed(seed)
    if costs is None:
        costs = np.full(N_SELLERS, MARGINAL_COST)
    agents = [QLearningAgent() for _ in range(N_SELLERS)]
    tau = TAU_INIT

    # Initialise with random prices
    price_indices = tuple(np.random.randint(N_PRICE_LEVELS) for _ in range(N_SELLERS))

    n_periods = N_PERIODS
    n_records = n_periods // RECORD_EVERY
    keys = ["avg_price", "winning_price", "consumer_surplus",
            "platform_revenue", "seller_profit", "collusion_index", "threshold"]
    metrics = {k: np.zeros(n_records) for k in keys}
    acc = {k: 0.0 for k in keys}
    acc_n = 0
    label = "Threshold" if use_threshold else "No Threshold"
    t0 = time.time()

    for ep in range(n_periods):
        # Exponential epsilon decay 
        epsilon = EPS_START * (EPS_END / EPS_START) ** (ep / max(n_periods - 1, 1))

        # Encode current state
        tau_bin = tau_to_bin(tau)
        s = encode_state(price_indices, tau_bin)

        # Each seller selects a price (ε-greedy)
        actions = tuple(agent.act(s, epsilon) for agent in agents)

        # Market resolves
        rewards, winner_price, new_tau, sold, demands = market_step(actions, tau, use_threshold, costs)

        # Encode next state
        new_tau_bin = tau_to_bin(new_tau)
        s_next = encode_state(actions, new_tau_bin)

        # Q-update for every seller
        for i, agent in enumerate(agents):
            agent.update(s, actions[i], rewards[i], s_next)

        # Metric logging
        prices_arr = PRICE_GRID[np.array(actions)]
        if use_threshold:
            # Average over eligible sellers only; ineligible sellers are excluded
            # from the market to avoid skewing the average upward.
            eligible_prices = prices_arr[prices_arr <= tau]
            avg_p = float(eligible_prices.mean()) if len(eligible_prices) > 0 else tau
        else:
            avg_p = float(prices_arr.mean())

        acc["avg_price"] += avg_p
        acc["winning_price"] += winner_price if sold else 0.0
        if DEMAND_MODEL == "logit":
            # Logit inclusive value formula
            acc["consumer_surplus"] += consumer_surplus(prices_arr)
        else:
            # WTA: CS = v − p_winner (buyer valuation v = LOGIT_A)
            acc["consumer_surplus"] += max(0.0, LOGIT_A - winner_price) if sold else 0.0
        acc["platform_revenue"] += (
            (COMMISSION_RATE * winner_price + F_FEE) * demands.sum() if sold else 0.0
        )
        acc["seller_profit"] += float(rewards.sum())
        acc["collusion_index"] += collusion_index(avg_p)
        acc["threshold"] += tau
        acc_n += 1

        # Advance market state
        price_indices = actions
        tau = new_tau

        # Record snapshot
        if acc_n == RECORD_EVERY:
            rec = ep // RECORD_EVERY
            if rec < n_records:
                for k in keys:
                    metrics[k][rec] = acc[k] / RECORD_EVERY
            acc = {k: 0.0 for k in keys}
            acc_n = 0

        # Progress report every 20 % of periods
        if (ep + 1) % max(1, n_periods // 5) == 0:
            elapsed = time.time() - t0
            pct     = 100 * (ep + 1) / n_periods
            print(f"  [{label}] {pct:3.0f}% done  "
                  f"ε={epsilon:.3f}  τ={tau:.3f}  "
                  f"avg_p={avg_p:.3f}  "
                  f"({elapsed:.0f}s elapsed)")

    print(f"  [{label}] completed in {time.time() - t0:.1f}s\n")
    return metrics


# Multiprocessing worker

def _run_simulation_worker(args: tuple) -> dict:
    config_overrides, use_threshold, seed, costs = args
    for k, v in config_overrides.items():
        globals()[k] = v
    return run_simulation(use_threshold=use_threshold, seed=seed, costs=costs)


# Main driver code

if __name__ == "__main__":
    import os
    from concurrent.futures import ProcessPoolExecutor
    import plotting as _plotting_mod  # direct module reference to update its globals

    os.makedirs("results", exist_ok=True)

    SELLER_COUNTS  = [2, 3, 5]
    models_to_run  = ["logit", "wta"] if RUN_BOTH_MODELS else [DEMAND_MODEL]

    # Draw costs once with a fixed seed so every seller-count / demand-model
    # combination uses the same underlying cost realizations.
    # The N-seller run uses the first N draws from this array.
    np.random.seed(COST_SEED)
    all_costs = np.random.uniform(COST_MIN, COST_MAX, max(SELLER_COUNTS))

    for n_sellers in SELLER_COUNTS:
        costs = all_costs[:n_sellers]
        # Recompute derived constants for this seller count
        _p          = np.linspace(MARGINAL_COST, MAX_PRICE, 10_000)
        _u          = np.exp((LOGIT_A - _p) / LOGIT_MU)
        _sym_d      = _u / (np.exp(LOGIT_A0 / LOGIT_MU) + n_sellers * _u)
        _sym_profit = (_p - MARGINAL_COST) * _sym_d
        mono_price  = float(_p[np.argmax(_sym_profit)])
        mono_profit = float(_sym_profit.max())
        n_states    = (N_PRICE_LEVELS ** n_sellers) * N_TAU_BINS
        q_init      = mono_profit / (1 - GAMMA)
        qtable_mb   = n_states * N_PRICE_LEVELS * 4 / 1e6

        # Update simulation module globals so all functions pick up new values
        g = globals()
        g["N_SELLERS"]      = n_sellers
        g["N_STATES"]       = n_states
        g["MONOPOLY_PRICE"] = mono_price
        g["Q_INIT"]         = q_init
        g["S_TARGET"]       = S_TARGET_BY_SELLERS.get(n_sellers, S_TARGET)

        # Update plotting module globals for titles and reference lines
        _plotting_mod.N_SELLERS      = n_sellers
        _plotting_mod.MONOPOLY_PRICE = mono_price

        print("=" * 64)
        print(f"  N_SELLERS       : {n_sellers}")
        print(f"  S_TARGET        : {g['S_TARGET']}")
        print(f"  Seller costs    : {np.round(costs, 4)}  (mean={costs.mean():.4f})")
        print(f"  Fixed fee f     : {F_FEE}")
        print(f"  Price levels    : {N_PRICE_LEVELS}  ({COST_MIN:.2f} – {MAX_PRICE:.2f})")
        print(f"  Periods         : {N_PERIODS:,}")
        print(f"  Monopoly price  : {mono_price:.3f}")
        print(f"  Mean cost ref   : {MARGINAL_COST:.3f}")
        print(f"  State space     : {n_states:,} states/agent")
        print(f"  Q-table memory  : {qtable_mb:.1f} MB/agent  "
              f"({n_sellers * qtable_mb:.1f} MB total)")
        print()

        for model in models_to_run:
            g["DEMAND_MODEL"] = model

            demand_label = "Logit Demand" if model == "logit" else "Winner-Take-All"
            img_path     = f"results/sellers_{n_sellers}_{model}.png"
            txt_path     = f"results/sellers_{n_sellers}_{model}.txt"

            print(f"  Demand model    : {demand_label}")
            print(f"  Output (image)  : {img_path}")
            print(f"  Output (text)   : {txt_path}")
            print()

            # Run control and treatment in parallel — they are fully independent.
            # Progress lines from the two workers will interleave in the terminal.
            config_overrides = {
                "N_SELLERS":      n_sellers,
                "N_STATES":       n_states,
                "MONOPOLY_PRICE": mono_price,
                "Q_INIT":         q_init,
                "S_TARGET":       g["S_TARGET"],
                "DEMAND_MODEL":   model,
            }
            print("─── Running control and treatment in parallel ───")
            with ProcessPoolExecutor(max_workers=2) as executor:
                fut_ctrl  = executor.submit(_run_simulation_worker,
                                            (config_overrides, False, 42, costs))
                fut_treat = executor.submit(_run_simulation_worker,
                                            (config_overrides, True,  42, costs))
                ctrl  = fut_ctrl.result()
                treat = fut_treat.result()

            print_summary(ctrl, treat, demand_label, save_path=txt_path, config_info={
                "n_sellers": n_sellers,
                "s_target":  g["S_TARGET"],
                "costs":     costs,
                "f_fee":     F_FEE,
                "tau_lr":    TAU_LR,
                "n_periods": N_PERIODS,
            })
            plot_results(ctrl, treat, save_path=img_path, demand_label=demand_label)

            if model != models_to_run[-1]:
                print()

        if n_sellers != SELLER_COUNTS[-1]:
            print("\n" + "─" * 64 + "\n")
