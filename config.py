"""
config.py — All simulation parameters and derived constants.
Edit this file to change the experiment setup.
"""

import numpy as np

# Market structure
N_SELLERS = 5         # Number of competing seller agents
MARGINAL_COST = 0.10  # Mean seller cost — used as competitive price reference for collusion index
COST_MIN = 0.05       # Lower bound of seller cost distribution U[COST_MIN, COST_MAX]
COST_MAX = 0.15       # Upper bound of seller cost distribution
COST_SEED = 0         # RNG seed for drawing seller costs (fixed so all runs use the same draws)
MAX_PRICE = 1.00      # Upper bound of price grid
N_PRICE_LEVELS = 15   # Number of discrete price actions

# Q-learning hyperparameters
Q_LR = 0.10         # Q-table learning rate (α)
GAMMA = 0.95        # Discount factor (γ)
EPS_START = 1.0     # Initial exploration probability (ε_0)
EPS_END = 0.005     # Final exploration probability (ε_T); decay is exponential: ε_t = ε_0·(ε_T/ε_0)^(t/T)
N_PERIODS = 20_000_000  # Training steps per experiment

# Platform threshold mechanism
TAU_LR = 0.05           # Proportional adjustment rate for threshold updates
TAU_DRIFT = 0.0005       # Constant downward drift applied every period — platform always
                        # tries to push τ lower even when S_O = 0 to prevent stalling
S_TARGET = 2            # Default target eligible sellers (used for single runs; see S_TARGET_BY_SELLERS)
S_TARGET_BY_SELLERS = { # Per-seller-count target
    2: 2,               #   2 sellers: target 2 eligible
    3: 2,               #   3 sellers: target 2 eligible
    5: 3,               #   5 sellers: target 3 eligible
}
F_FEE = 0.01            # Fixed fee per transaction charged by platform (f); paid by seller
TAU_INIT = MAX_PRICE    # Starting threshold price (permissive; descends over time)
N_TAU_BINS = 10         # Discrete bins used to represent τ in the Q-table state

# Demand model — controls how market demand is allocated each period
# "logit" : demand shared via restricted logit (Calvano-style); all active sellers earn
# "wta"   : winner-take-all; lowest-price seller (or eligible seller) captures full demand
DEMAND_MODEL = "wta"      # "logit" | "wta" — ignored when RUN_BOTH_MODELS is True
RUN_BOTH_MODELS = True   # If True, run logit and wta back-to-back and save separate outputs

# Platform revenue
COMMISSION_RATE = 0.05  # r — platform earns r × p_winner × demand per transaction

# Metrics and output
RECORD_EVERY = 500  # Accumulate metrics over this many periods, then record one point
ROLLING_WINDOW = 36000  # Smoothing window for plots (in periods)

# Validation
assert N_SELLERS >= 2, "Need at least 2 sellers."
assert 1 <= S_TARGET <= N_SELLERS, "S_TARGET must be between 1 and N_SELLERS."
assert N_PRICE_LEVELS >= 4, "Need at least 4 price levels."

# Logit demand parameters
LOGIT_A = 0.5    # Product quality / mean utility
LOGIT_MU = 0.25  # Substitutability (lower = closer substitutes)
LOGIT_A0 = 0.0   # Outside option utility

# Derived constants
PRICE_GRID = np.linspace(COST_MIN, MAX_PRICE, N_PRICE_LEVELS)
N_STATES = (N_PRICE_LEVELS ** N_SELLERS) * N_TAU_BINS
_qtable_mb = N_STATES * N_PRICE_LEVELS * 8 / 1e6

#TODO: Validate math below

# Numerically compute monopoly price under logit demand (symmetric case:
# all N_SELLERS charge the same price p; find p that maximises per-seller profit)
_p_search = np.linspace(MARGINAL_COST, MAX_PRICE, 10_000)
_u = np.exp((LOGIT_A - _p_search) / LOGIT_MU)
_sym_demand = _u / (np.exp(LOGIT_A0 / LOGIT_MU) + N_SELLERS * _u)
_sym_profit = (_p_search - MARGINAL_COST) * _sym_demand
MONOPOLY_PRICE = float(_p_search[np.argmax(_sym_profit)])
_monopoly_profit = float(_sym_profit.max())

# Optimistic Q-table initialisation (Calvano):
# start every Q-value at the expected long-run monopoly payoff π_m / (1 − γ)
Q_INIT = _monopoly_profit / (1 - GAMMA)
