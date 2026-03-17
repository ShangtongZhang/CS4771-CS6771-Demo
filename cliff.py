"""
cliff.py — Cliff Gridworld demos for a graduate RL course.

Layout (4 rows × 12 cols, row 0 = top):

    .  .  .  .  .  .  .  .  .  .  .  .
    .  .  .  .  .  .  .  .  .  .  .  .
    .  .  .  .  .  .  .  .  .  .  .  .
    S  C  C  C  C  C  C  C  C  C  C  G

S = Start  (3, 0)
G = Goal   (3, 11)
C = Cliff  (3, 1) – (3, 10)   reward −100, reset to S

Every non-terminal step: reward = −1.
Episode ends upon reaching G.

Algorithms implemented
──────────────────────
Dynamic Programming
  • Bellman expectation operator (repeated application)
  • Policy Iteration
  • Value Iteration

Temporal Difference
  • TD(0) policy evaluation
  • SARSA   (on-policy control)
  • Q-learning (off-policy control)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm


# ════════════════════════════════════════════════════════════════════════════
# Environment
# ════════════════════════════════════════════════════════════════════════════

class CliffWorld:
    """Deterministic cliff gridworld (Sutton & Barto, Example 6.6)."""

    NROWS, NCOLS = 4, 12
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    ACTIONS = [0, 1, 2, 3]
    ACTION_NAMES   = ["Up", "Down", "Left", "Right"]
    ACTION_SYMBOLS = ["↑",  "↓",    "←",    "→"]
    ACTION_DELTAS  = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    START = (3, 0)
    GOAL  = (3, 11)
    CLIFF = frozenset((3, c) for c in range(1, 11))

    def __init__(self):
        self.n_states  = self.NROWS * self.NCOLS
        self.n_actions = 4

    # ── index helpers ──────────────────────────────────────────────────────

    def state_index(self, state):
        """(row, col) → flat index."""
        return state[0] * self.NCOLS + state[1]

    def index_state(self, idx):
        """Flat index → (row, col)."""
        return (idx // self.NCOLS, idx % self.NCOLS)

    def all_states(self):
        return [(r, c) for r in range(self.NROWS) for c in range(self.NCOLS)]

    # ── dynamics ───────────────────────────────────────────────────────────

    def is_terminal(self, state):
        return state == self.GOAL

    def step(self, state, action):
        """Returns (next_state, reward, done)."""
        if self.is_terminal(state):
            return state, 0.0, True

        dr, dc = self.ACTION_DELTAS[action]
        r, c = state
        nr = max(0, min(self.NROWS - 1, r + dr))
        nc = max(0, min(self.NCOLS - 1, c + dc))
        next_state = (nr, nc)

        if next_state in self.CLIFF:
            return self.START, -100.0, False
        elif self.is_terminal(next_state):
            return next_state, -1.0, True
        else:
            return next_state, -1.0, False

    def transitions(self, state, action):
        """
        Returns [(prob, next_state, reward)].
        Deterministic: always a single tuple with prob = 1.
        Used by DP algorithms.
        """
        if self.is_terminal(state):
            return [(1.0, state, 0.0)]
        next_state, reward, _ = self.step(state, action)
        return [(1.0, next_state, reward)]


# ════════════════════════════════════════════════════════════════════════════
# Dynamic Programming
# ════════════════════════════════════════════════════════════════════════════

def bellman_backup(V, state, action, env, gamma):
    """Compute Q(s, a) = Σ p(s'|s,a)[r + γ V(s')]."""
    return sum(
        p * (r + gamma * V[env.state_index(s_next)])
        for p, s_next, r in env.transitions(state, action)
    )


def apply_bellman_operator(V, policy, env, gamma):
    """
    One full sweep of the Bellman expectation operator for *policy*.
    policy : array of shape (n_states,) containing action indices.
    Returns a new value array V'.
    """
    V_new = np.zeros(env.n_states)
    for s_idx in range(env.n_states):
        state = env.index_state(s_idx)
        if env.is_terminal(state):
            continue
        V_new[s_idx] = bellman_backup(V, state, policy[s_idx], env, gamma)
    return V_new


def policy_evaluation(policy, env, gamma=1.0, theta=1e-8, max_iter=5_000):
    """
    Iterative policy evaluation via repeated Bellman backups.

    Returns
    -------
    V       : converged value function (array of shape n_states)
    history : list of V snapshots — history[k] is V after k sweeps
    """
    V = np.zeros(env.n_states)
    history = [V.copy()]
    for _ in range(max_iter):
        V_new = apply_bellman_operator(V, policy, env, gamma)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        history.append(V.copy())
        if delta < theta:
            break
    return V, history


def greedy_policy(V, env, gamma=1.0):
    """Extract the greedy policy w.r.t. V."""
    policy = np.zeros(env.n_states, dtype=int)
    for s_idx in range(env.n_states):
        state = env.index_state(s_idx)
        if env.is_terminal(state):
            continue
        qs = [bellman_backup(V, state, a, env, gamma) for a in env.ACTIONS]
        policy[s_idx] = int(np.argmax(qs))
    return policy


def policy_iteration(env, gamma=1.0, theta=1e-8):
    """
    Policy Iteration.

    Returns
    -------
    policy     : optimal policy
    V          : optimal value function
    pi_history : list of policies after each improvement step
    V_history  : list of value functions after each evaluation step
    """
    # Initialise with a "leftward sweep" policy — terminates but is far from optimal,
    # so Policy Iteration needs several improvement steps to reach the optimum.
    #
    # Rule (in order of priority):
    #   row 3            → UP   (escape cliff row)
    #   col 11           → DOWN (descend to goal)
    #   row 0            → RIGHT (traverse top row toward col 11)
    #   col 0, rows 1-2  → UP   (climb to top row)
    #   rows 1-2, cols 1-10 → LEFT (sweep left toward col 0)
    #
    # Every state eventually reaches the goal via: LEFT* → UP* → RIGHT* → DOWN*.
    policy = np.zeros(env.n_states, dtype=int)
    for s_idx in range(env.n_states):
        r, c = env.index_state(s_idx)
        if r == 3:
            policy[s_idx] = env.UP
        elif c == env.NCOLS - 1:
            policy[s_idx] = env.DOWN
        elif r == 0:
            policy[s_idx] = env.RIGHT
        elif c == 0:
            policy[s_idx] = env.UP
        else:
            policy[s_idx] = env.LEFT
    pi_history = [policy.copy()]
    V_history  = []

    while True:
        V, _ = policy_evaluation(policy, env, gamma, theta)
        V_history.append(V.copy())

        new_policy = greedy_policy(V, env, gamma)
        pi_history.append(new_policy.copy())

        if np.array_equal(new_policy, policy):
            break
        policy = new_policy

    return policy, V, pi_history, V_history


def value_iteration(env, gamma=1.0, theta=1e-8, max_iter=10_000):
    """
    Value Iteration.

    Returns
    -------
    V       : optimal value function
    policy  : extracted greedy policy
    history : list of V snapshots after each sweep
    """
    V = np.zeros(env.n_states)
    history = [V.copy()]

    for _ in range(max_iter):
        V_new = np.zeros(env.n_states)
        for s_idx in range(env.n_states):
            state = env.index_state(s_idx)
            if env.is_terminal(state):
                continue
            qs = [bellman_backup(V, state, a, env, gamma) for a in env.ACTIONS]
            V_new[s_idx] = max(qs)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        history.append(V.copy())
        if delta < theta:
            break

    policy = greedy_policy(V, env, gamma)
    return V, policy, history


# ════════════════════════════════════════════════════════════════════════════
# Temporal Difference Learning
# ════════════════════════════════════════════════════════════════════════════

def _epsilon_greedy(Q_row, epsilon, n_actions, rng):
    if rng.random() < epsilon:
        return rng.integers(n_actions)
    return int(np.argmax(Q_row))


def td_evaluation(policy, env, gamma=1.0, alpha=0.1,
                  n_episodes=500, snapshot_eps=None, seed=0):
    """
    TD(0) policy evaluation.

    Parameters
    ----------
    policy       : array (n_states,) of action indices
    alpha        : step-size
    n_episodes   : number of training episodes
    snapshot_eps : list of 1-indexed episode numbers at which to save V
                   (e.g. [1, 10, 50, 200, 500]).  Pass None to skip.

    Returns
    -------
    V               : estimated value function after all episodes
    episode_rewards : list of per-episode total rewards
    snapshots       : dict {episode: V_array}  (empty if snapshot_eps is None)
    """
    snap_set = set(snapshot_eps) if snapshot_eps is not None else set()
    V = np.zeros(env.n_states)
    episode_rewards = []
    snapshots = {}

    for ep in range(1, n_episodes + 1):
        state = env.START
        total_reward = 0.0
        while not env.is_terminal(state):
            s_idx  = env.state_index(state)
            action = policy[s_idx]
            next_state, reward, _ = env.step(state, action)
            ns_idx = env.state_index(next_state)
            V[s_idx] += alpha * (reward + gamma * V[ns_idx] - V[s_idx])
            total_reward += reward
            state = next_state
        episode_rewards.append(total_reward)
        if ep in snap_set:
            snapshots[ep] = V.copy()

    return V, episode_rewards, snapshots


def sarsa(env, gamma=1.0, alpha=0.5, epsilon=0.1,
          n_episodes=500, seed=0):
    """
    SARSA — on-policy TD control.

    Returns
    -------
    Q               : action-value function, shape (n_states, n_actions)
    episode_rewards : list of per-episode total rewards
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    episode_rewards = []

    for _ in range(n_episodes):
        state  = env.START
        s_idx  = env.state_index(state)
        action = _epsilon_greedy(Q[s_idx], epsilon, env.n_actions, rng)
        total_reward = 0.0

        while not env.is_terminal(state):
            next_state, reward, done = env.step(state, action)
            ns_idx      = env.state_index(next_state)
            next_action = _epsilon_greedy(Q[ns_idx], epsilon, env.n_actions, rng)

            # SARSA update: uses the *actual* next action (on-policy)
            Q[s_idx, action] += alpha * (
                reward + gamma * Q[ns_idx, next_action] - Q[s_idx, action]
            )

            total_reward += reward
            state  = next_state
            s_idx  = ns_idx
            action = next_action

        episode_rewards.append(total_reward)

    return Q, episode_rewards


def q_learning(env, gamma=1.0, alpha=0.5, epsilon=0.1,
               n_episodes=500, seed=0):
    """
    Q-learning — off-policy TD control.

    Returns
    -------
    Q               : action-value function, shape (n_states, n_actions)
    episode_rewards : list of per-episode total rewards
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    episode_rewards = []

    for _ in range(n_episodes):
        state = env.START
        total_reward = 0.0

        while not env.is_terminal(state):
            s_idx  = env.state_index(state)
            action = _epsilon_greedy(Q[s_idx], epsilon, env.n_actions, rng)
            next_state, reward, done = env.step(state, action)
            ns_idx = env.state_index(next_state)

            # Q-learning update: uses the *greedy* next action (off-policy)
            Q[s_idx, action] += alpha * (
                reward + gamma * np.max(Q[ns_idx]) - Q[s_idx, action]
            )

            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)

    return Q, episode_rewards


def q_to_policy(Q, env):
    """Extract greedy policy from Q-function."""
    return np.argmax(Q, axis=1)


def _greedy_trajectory(env, Q, max_steps=200):
    """
    Run one episode with the greedy (epsilon=0) policy derived from Q.
    Returns the list of states visited (including start, excluding goal).
    """
    state = env.START
    traj  = [state]
    for _ in range(max_steps):
        if env.is_terminal(state):
            break
        s_idx  = env.state_index(state)
        action = int(np.argmax(Q[s_idx]))
        state, _, _ = env.step(state, action)
        traj.append(state)
        # Safety: break if the agent is stuck in a very long loop
        if len(traj) > max_steps:
            break
    return traj


def sarsa_snapshots(env, gamma=1.0, alpha=0.5, epsilon=0.1,
                    n_episodes=500, snapshot_eps=None, seed=0):
    """
    SARSA with greedy-policy and trajectory snapshots.

    At each episode in *snapshot_eps*, records:
      • the current greedy policy  (argmax Q)
      • one greedy trajectory from the start state

    Returns
    -------
    Q               : final action-value function
    episode_rewards : per-episode total rewards
    pi_snapshots    : {ep: greedy policy array}
    traj_snapshots  : {ep: list of states visited greedily}
    """
    snap_set = set(snapshot_eps) if snapshot_eps is not None else set()
    rng = np.random.default_rng(seed)
    Q   = np.zeros((env.n_states, env.n_actions))
    episode_rewards = []
    pi_snapshots   = {}
    traj_snapshots = {}

    for ep in range(1, n_episodes + 1):
        state  = env.START
        s_idx  = env.state_index(state)
        action = _epsilon_greedy(Q[s_idx], epsilon, env.n_actions, rng)
        total_reward = 0.0

        while not env.is_terminal(state):
            next_state, reward, _ = env.step(state, action)
            ns_idx      = env.state_index(next_state)
            next_action = _epsilon_greedy(Q[ns_idx], epsilon, env.n_actions, rng)
            Q[s_idx, action] += alpha * (
                reward + gamma * Q[ns_idx, next_action] - Q[s_idx, action]
            )
            total_reward += reward
            state, s_idx, action = next_state, ns_idx, next_action

        episode_rewards.append(total_reward)
        if ep in snap_set:
            pi_snapshots[ep]   = np.argmax(Q, axis=1).copy()
            traj_snapshots[ep] = _greedy_trajectory(env, Q)

    return Q, episode_rewards, pi_snapshots, traj_snapshots


def q_learning_snapshots(env, gamma=1.0, alpha=0.5, epsilon=0.1,
                          n_episodes=500, snapshot_eps=None, seed=0):
    """
    Q-learning with greedy-policy and trajectory snapshots.

    Returns
    -------
    Q               : final action-value function
    episode_rewards : per-episode total rewards
    pi_snapshots    : {ep: greedy policy array}
    traj_snapshots  : {ep: list of states visited greedily}
    """
    snap_set = set(snapshot_eps) if snapshot_eps is not None else set()
    rng = np.random.default_rng(seed)
    Q   = np.zeros((env.n_states, env.n_actions))
    episode_rewards = []
    pi_snapshots   = {}
    traj_snapshots = {}

    for ep in range(1, n_episodes + 1):
        state = env.START
        total_reward = 0.0

        while not env.is_terminal(state):
            s_idx  = env.state_index(state)
            action = _epsilon_greedy(Q[s_idx], epsilon, env.n_actions, rng)
            next_state, reward, _ = env.step(state, action)
            ns_idx = env.state_index(next_state)
            Q[s_idx, action] += alpha * (
                reward + gamma * np.max(Q[ns_idx]) - Q[s_idx, action]
            )
            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)
        if ep in snap_set:
            pi_snapshots[ep]   = np.argmax(Q, axis=1).copy()
            traj_snapshots[ep] = _greedy_trajectory(env, Q)

    return Q, episode_rewards, pi_snapshots, traj_snapshots


def run_multiple_seeds(algo_fn, env, n_seeds=20, **kwargs):
    """
    Run an algorithm over multiple seeds and aggregate episode rewards.

    Returns
    -------
    mean_rewards : array of shape (n_episodes,)
    std_rewards  : array of shape (n_episodes,)
    """
    all_rewards = []
    for seed in range(n_seeds):
        *_, rewards = algo_fn(env, seed=seed, **kwargs)   # rewards is always the last element
        all_rewards.append(rewards)
    all_rewards = np.array(all_rewards)
    return all_rewards.mean(axis=0), all_rewards.std(axis=0)


# ════════════════════════════════════════════════════════════════════════════
# Policy Gradient Methods
# ════════════════════════════════════════════════════════════════════════════

def _softmax(x):
    """Numerically stable softmax over a 1-D array."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def reinforce(env, gamma=1.0, alpha=0.005, n_episodes=2000,
              max_steps=500, seed=0):
    """
    REINFORCE — Monte-Carlo Policy Gradient (Williams, 1992).

    Policy: tabular softmax  π(a|s; θ) = softmax(θ[s, :])

    After each complete episode, update backwards through the trajectory:
        G  ←  return from step t
        θ[s_t, :] += α · γ^t · G · ∇log π(a_t | s_t; θ)

    where the score function for the tabular softmax is:
        ∇log π(a_t | s_t; θ) = e_{a_t} − π(· | s_t; θ)

    Returns (theta, episode_rewards).
    """
    rng = np.random.default_rng(seed)
    theta = np.zeros((env.n_states, env.n_actions))
    episode_rewards = []

    for _ in range(n_episodes):
        trajectory = []          # list of (s_idx, action, reward)
        state = env.START
        total_reward = 0.0
        steps = 0

        while not env.is_terminal(state) and steps < max_steps:
            s_idx  = env.state_index(state)
            probs  = _softmax(theta[s_idx])
            action = int(rng.choice(env.n_actions, p=probs))
            next_state, reward, _ = env.step(state, action)
            trajectory.append((s_idx, action, reward))
            total_reward += reward
            state  = next_state
            steps += 1

        episode_rewards.append(total_reward)

        G = 0.0
        for t in reversed(range(len(trajectory))):
            s_idx, action, reward = trajectory[t]
            G = reward + gamma * G
            probs = _softmax(theta[s_idx])
            grad  = -probs.copy()
            grad[action] += 1.0                          # score function
            theta[s_idx] += alpha * (gamma ** t) * G * grad

    return theta, episode_rewards


def reinforce_baseline(env, gamma=1.0, alpha_theta=0.005, alpha_w=0.05,
                        n_episodes=2000, max_steps=500, seed=0):
    """
    REINFORCE with value-function baseline.

    After each complete episode, for each step t (reversed):
        G   ←  return from step t
        δ   =  G − V(s_t ; w)             (advantage, zero-mean in expectation)
        w[s_t]    += α_w · δ              (critic: tabular semi-gradient)
        θ[s_t, :] += α_θ · γ^t · δ · ∇log π(a_t | s_t; θ)   (actor)

    The baseline b(s) = V(s; w) reduces variance without biasing the gradient
    because E_π[∇log π(a|s) · b(s)] = 0.

    Returns (theta, w, episode_rewards).
    """
    rng = np.random.default_rng(seed)
    theta = np.zeros((env.n_states, env.n_actions))
    w     = np.zeros(env.n_states)
    episode_rewards = []

    for _ in range(n_episodes):
        trajectory = []
        state = env.START
        total_reward = 0.0
        steps = 0

        while not env.is_terminal(state) and steps < max_steps:
            s_idx  = env.state_index(state)
            probs  = _softmax(theta[s_idx])
            action = int(rng.choice(env.n_actions, p=probs))
            next_state, reward, _ = env.step(state, action)
            trajectory.append((s_idx, action, reward))
            total_reward += reward
            state  = next_state
            steps += 1

        episode_rewards.append(total_reward)

        G = 0.0
        for t in reversed(range(len(trajectory))):
            s_idx, action, reward = trajectory[t]
            G     = reward + gamma * G
            delta = G - w[s_idx]
            w[s_idx] += alpha_w * delta                  # critic
            probs = _softmax(theta[s_idx])
            grad  = -probs.copy()
            grad[action] += 1.0
            theta[s_idx] += alpha_theta * (gamma ** t) * delta * grad  # actor

    return theta, w, episode_rewards


def actor_critic(env, gamma=1.0, alpha_theta=0.01, alpha_w=0.1,
                  n_episodes=2000, max_steps=500, seed=0):
    """
    One-step Actor-Critic (online, semi-gradient TD).

    At every transition (no waiting for episode end):
        δ_t = R_{t+1} + γ V(s_{t+1}; w) − V(s_t; w)    (TD error ≈ advantage)
        w[s_t]    += α_w · δ_t                           (critic)
        θ[s_t, :] += α_θ · γ^t · δ_t · ∇log π(a_t|s_t) (actor)

    Compared with REINFORCE: replaces the high-variance MC return G_t with
    a bootstrapped TD estimate → lower variance at the cost of some bias.
    Online updates mean the policy improves within each episode.

    Returns (theta, w, episode_rewards).
    """
    rng = np.random.default_rng(seed)
    theta = np.zeros((env.n_states, env.n_actions))
    w     = np.zeros(env.n_states)
    episode_rewards = []

    for _ in range(n_episodes):
        state = env.START
        total_reward = 0.0
        t = 0

        while not env.is_terminal(state) and t < max_steps:
            s_idx  = env.state_index(state)
            probs  = _softmax(theta[s_idx])
            action = int(rng.choice(env.n_actions, p=probs))
            next_state, reward, _ = env.step(state, action)
            ns_idx = env.state_index(next_state)

            delta = reward + gamma * w[ns_idx] - w[s_idx]   # TD error
            w[s_idx] += alpha_w * delta                      # critic

            grad = -probs.copy()
            grad[action] += 1.0
            theta[s_idx] += alpha_theta * (gamma ** t) * delta * grad  # actor

            total_reward += reward
            state  = next_state
            t     += 1

        episode_rewards.append(total_reward)

    return theta, w, episode_rewards


def theta_to_policy(theta):
    """Extract the greedy (argmax) policy from softmax preference parameters θ."""
    return np.argmax(theta, axis=1)


def demo_policy_gradient(env, gamma=1.0,
                          alpha_reinforce=0.005,
                          alpha_theta=0.005, alpha_w=0.05,
                          n_episodes=2000, n_seeds=10):
    """
    Run REINFORCE, REINFORCE+baseline, and one-step Actor-Critic.
    No figures are created; plot from the notebook for full control.

    Returns a dict with θ tables, w tables, policies, and reward statistics.
    """
    print(f"Running REINFORCE × {n_seeds} seeds …")
    rf_mean, rf_std = run_multiple_seeds(
        reinforce, env, n_seeds=n_seeds,
        gamma=gamma, alpha=alpha_reinforce, n_episodes=n_episodes)

    print(f"Running REINFORCE + baseline × {n_seeds} seeds …")
    rfb_mean, rfb_std = run_multiple_seeds(
        reinforce_baseline, env, n_seeds=n_seeds,
        gamma=gamma, alpha_theta=alpha_theta, alpha_w=alpha_w,
        n_episodes=n_episodes)

    print(f"Running Actor-Critic × {n_seeds} seeds …")
    ac_mean, ac_std = run_multiple_seeds(
        actor_critic, env, n_seeds=n_seeds,
        gamma=gamma, alpha_theta=alpha_theta, alpha_w=alpha_w,
        n_episodes=n_episodes)

    # Single seed=0 runs for policy / value visualisation
    theta_rf,           _ = reinforce(
        env, gamma=gamma, alpha=alpha_reinforce, n_episodes=n_episodes, seed=0)
    theta_rfb, w_rfb,   _ = reinforce_baseline(
        env, gamma=gamma, alpha_theta=alpha_theta, alpha_w=alpha_w,
        n_episodes=n_episodes, seed=0)
    theta_ac,  w_ac,    _ = actor_critic(
        env, gamma=gamma, alpha_theta=alpha_theta, alpha_w=alpha_w,
        n_episodes=n_episodes, seed=0)

    return dict(
        theta_rf=theta_rf, theta_rfb=theta_rfb, theta_ac=theta_ac,
        w_rfb=w_rfb, w_ac=w_ac,
        pi_rf=theta_to_policy(theta_rf),
        pi_rfb=theta_to_policy(theta_rfb),
        pi_ac=theta_to_policy(theta_ac),
        rf_mean=rf_mean,   rf_std=rf_std,
        rfb_mean=rfb_mean, rfb_std=rfb_std,
        ac_mean=ac_mean,   ac_std=ac_std,
    )


# ════════════════════════════════════════════════════════════════════════════
# Visualisation Utilities
# ════════════════════════════════════════════════════════════════════════════

def _base_grid(env, ax):
    """Draw the static cliff-world background on ax."""
    bg = np.zeros((env.NROWS, env.NCOLS))
    for r, c in env.CLIFF:
        bg[r, c] = -2
    bg[env.GOAL[0], env.GOAL[1]] = 2
    ax.imshow(bg, cmap="RdYlGn", vmin=-3, vmax=3, aspect="auto")

    for r, c in env.CLIFF:
        ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="black", zorder=2))
        ax.text(c, r, "Cliff", ha="center", va="center", color="white",
                fontsize=6, zorder=3)
    # Goal
    r, c = env.GOAL
    ax.text(c, r, "GOAL", ha="center", va="center", fontsize=7,
            fontweight="bold", color="darkgreen", zorder=3)
    # Start
    r, c = env.START
    ax.text(c + .02, r - .35, "S", ha="center", va="center", fontsize=7,
            fontweight="bold", color="navy", zorder=3)

    ax.set_xticks(range(env.NCOLS))
    ax.set_yticks(range(env.NROWS))
    ax.set_xticklabels(range(env.NCOLS), fontsize=7)
    ax.set_yticklabels(range(env.NROWS), fontsize=7)


def plot_value_function(V, env, title="Value Function", ax=None, vmin=None):
    """
    Colour-coded heat-map of V with numeric annotations.
    Cliff cells are drawn in black; the goal shows V = 0.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 3.5))
        standalone = True
    else:
        standalone = False

    grid = V.reshape(env.NROWS, env.NCOLS).copy()
    # Mask cliff from colour scale
    mask = np.zeros_like(grid, dtype=bool)
    for r, c in env.CLIFF:
        mask[r, c] = True

    if vmin is None:
        vmin = grid[~mask].min() if (~mask).any() else -1
    vmin = min(vmin, -1e-6)   # TwoSlopeNorm requires vmin < vcenter < vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vmin / 2, vmax=0)

    masked_grid = np.ma.array(grid, mask=mask)
    im = ax.imshow(masked_grid, cmap="RdYlGn", norm=norm, aspect="auto")

    for r in range(env.NROWS):
        for c in range(env.NCOLS):
            state = (r, c)
            if state in env.CLIFF:
                ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1,
                                           color="black", zorder=2))
                ax.text(c, r, "Cliff", ha="center", va="center",
                        color="white", fontsize=6, zorder=3)
            else:
                val = grid[r, c]
                ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                        fontsize=7, zorder=3)

    r, c = env.GOAL
    ax.text(c - .3, r - .35, "G", ha="center", va="center", fontsize=7,
            fontweight="bold", color="darkgreen", zorder=4)
    r, c = env.START
    ax.text(c + .3, r - .35, "S", ha="center", va="center", fontsize=7,
            fontweight="bold", color="navy", zorder=4)

    ax.set_xticks(range(env.NCOLS))
    ax.set_yticks(range(env.NROWS))
    ax.set_xticklabels(range(env.NCOLS), fontsize=7)
    ax.set_yticklabels(range(env.NROWS), fontsize=7)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    if standalone:
        plt.tight_layout()
    return ax


def plot_policy(policy, env, title="Policy", ax=None):
    """
    Visualise a deterministic policy as directional arrows on the grid.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 3.5))
        standalone = True
    else:
        standalone = False

    _base_grid(env, ax)

    for s_idx in range(env.n_states):
        state = env.index_state(s_idx)
        r, c = state
        if state in env.CLIFF or env.is_terminal(state):
            continue
        dr, dc = env.ACTION_DELTAS[policy[s_idx]]
        ax.annotate(
            "", xy=(c + dc * 0.38, r + dr * 0.38), xytext=(c, r),
            arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.8),
            zorder=4,
        )

    ax.set_title(title, fontsize=10)
    if standalone:
        plt.tight_layout()
    return ax


def plot_trajectory(traj, env, title="Trajectory", ax=None):
    """
    Draw a single greedy-episode trajectory on the cliff gridworld.

    The path is shown as a heat-map of visit counts plus directed arrows
    between consecutive states.  Cliff cells stay black; start/goal are labelled.

    Parameters
    ----------
    traj : list of (row, col) states returned by _greedy_trajectory
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 3.5))

    # Visit-count heat-map (excluding cliff)
    counts = np.zeros((env.NROWS, env.NCOLS))
    for s in traj:
        counts[s[0], s[1]] += 1

    bg = np.zeros((env.NROWS, env.NCOLS))
    for r, c in env.CLIFF:
        bg[r, c] = np.nan
    disp = np.where(np.isnan(bg), np.nan, counts)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("black")
    ax.imshow(disp, cmap=cmap, vmin=0, vmax=max(counts.max(), 1),
              aspect="auto", interpolation="nearest")

    # Cliff labels
    for r, c in env.CLIFF:
        ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="black", zorder=2))
        ax.text(c, r, "Cliff", ha="center", va="center",
                color="white", fontsize=6, zorder=3)

    # Path arrows
    for i in range(len(traj) - 1):
        r0, c0 = traj[i]
        r1, c1 = traj[i + 1]
        if (r0, c0) in env.CLIFF or (r1, c1) in env.CLIFF:
            continue
        ax.annotate("",
                    xy=(c1, r1), xytext=(c0, r0),
                    arrowprops=dict(arrowstyle="->", color="crimson",
                                   lw=1.4, shrinkA=5, shrinkB=5),
                    zorder=5)

    # Start / Goal labels
    r, c = env.START
    ax.text(c, r - .35, "S", ha="center", va="center",
            fontsize=7, fontweight="bold", color="navy", zorder=6)
    r, c = env.GOAL
    ax.text(c, r, "G", ha="center", va="center",
            fontsize=7, fontweight="bold", color="darkgreen", zorder=6)

    ax.set_xticks(range(env.NCOLS))
    ax.set_yticks(range(env.NROWS))
    ax.set_xticklabels(range(env.NCOLS), fontsize=7)
    ax.set_yticklabels(range(env.NROWS), fontsize=7)
    steps = len(traj) - 1
    ax.set_title(f"{title}  ({steps} steps)", fontsize=9)
    return ax


def plot_convergence(history, env, iterations, title="Bellman Operator Iterations",
                     figsize=None):
    """
    Show value function snapshots at selected iteration indices.

    Parameters
    ----------
    history    : list of V arrays (output of policy_evaluation / value_iteration)
    iterations : list of iteration indices to display
    """
    iterations = [i for i in iterations if i < len(history)]
    n = len(iterations)
    if figsize is None:
        figsize = (13, 3.0 * n)
    fig, axes = plt.subplots(n, 1, figsize=figsize)
    if n == 1:
        axes = [axes]

    vmin = min(history[-1].min(), -1)
    for ax, it in zip(axes, iterations):
        plot_value_function(history[it], env,
                            title=f"Iteration {it}", ax=ax, vmin=vmin)
    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    return fig


def plot_delta_curve(history, title="Max |ΔV| per Iteration", ax=None):
    """Plot the max-norm Bellman residual across iterations."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    deltas = [np.max(np.abs(history[k+1] - history[k]))
              for k in range(len(history) - 1)]
    ax.semilogy(deltas, color="steelblue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("max |ΔV|  (log scale)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_rmse_curve(history, V_ref, env=None, exclude_cliff=True,
                    title="RMSE vs. converged V", ax=None):
    """
    Plot the root-mean-square error of each V snapshot against a reference V.

    With γ = 1 and uniform step costs, max|ΔV| stays constant until all states
    converge (a step function).  The RMSE against the true V^π decays
    monotonically each iteration and is therefore a more informative convergence
    diagnostic in that regime.

    Parameters
    ----------
    history      : list of V arrays (output of policy_evaluation / value_iteration)
    V_ref        : reference value function (e.g. history[-1] = converged V)
    exclude_cliff: if True and env is provided, cliff states are excluded from RMSE
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    if exclude_cliff and env is not None:
        mask = np.array([env.state_index(s) for s in env.all_states()
                         if s not in env.CLIFF])
    else:
        mask = np.arange(len(V_ref))
    rmses = [np.sqrt(np.mean((V[mask] - V_ref[mask]) ** 2)) for V in history]
    ax.plot(rmses, color="darkorange")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def smooth(x, window=10):
    """Simple moving-average smoothing."""
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_learning_curves(rewards_dict, window=10,
                         title="Learning Curves (smoothed)", ax=None):
    """
    Plot per-episode reward curves (smoothed) for multiple algorithms.

    Parameters
    ----------
    rewards_dict : {label: array_of_rewards}   OR
                   {label: (mean_array, std_array)}  for multi-seed averages
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        standalone = True
    else:
        standalone = False

    for label, data in rewards_dict.items():
        if isinstance(data, tuple):
            mean, std = data
            sm = smooth(mean, window)
            ss = smooth(std,  window)
            xs = np.arange(len(sm))
            ax.plot(xs, sm, label=label)
            ax.fill_between(xs, sm - ss, sm + ss, alpha=0.2)
        else:
            sm = smooth(np.array(data), window)
            ax.plot(np.arange(len(sm)), sm, label=label)

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Sum of rewards (window={window})")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if standalone:
        plt.tight_layout()
    return ax


# ════════════════════════════════════════════════════════════════════════════
# Demo helpers  (called directly from the notebook)
# ════════════════════════════════════════════════════════════════════════════

def demo_environment(env):
    """Visualise the cliff gridworld layout."""
    fig, ax = plt.subplots(figsize=(13, 3.5))
    _base_grid(env, ax)
    ax.set_title("Cliff Gridworld  (S = start, G = goal, black = cliff)", fontsize=11)
    # Add a legend
    legend_elements = [
        mpatches.Patch(color="black",  label="Cliff  (reward −100, reset to S)"),
        mpatches.Patch(color="gold",   label="Goal   (reward −1, episode ends)"),
        mpatches.Patch(color="#6baed6", label="Other  (reward −1)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig


def demo_bellman_operator(env, gamma=1.0):
    """
    Run iterative policy evaluation (Bellman operator) on a fixed policy
    that always terminates:
      row 3 → UP  /  rows 0-2, col < 11 → RIGHT  /  col 11 → DOWN

    Returns (policy, V, history).
    No figures are created; plot from the notebook for full control.
    """
    policy = np.zeros(env.n_states, dtype=int)
    for s_idx in range(env.n_states):
        r, c = env.index_state(s_idx)
        if r < 3:
            policy[s_idx] = env.RIGHT if c < env.NCOLS - 1 else env.DOWN
        else:
            policy[s_idx] = env.UP
    V, history = policy_evaluation(policy, env, gamma)
    print(f"Policy evaluation converged in {len(history) - 1} iterations.")
    return policy, V, history


def demo_policy_iteration(env, gamma=1.0):
    """
    Run Policy Iteration and return a single multi-panel figure
    showing (V, improved π) after each improvement step.

    Returns (policy, V, pi_history, V_history, fig).
    """
    policy, V, pi_history, V_history = policy_iteration(env, gamma)
    n_steps = len(V_history)
    print(f"Policy Iteration converged in {n_steps} evaluation step(s).")

    fig, axes = plt.subplots(n_steps, 2, figsize=(13, 3.5 * n_steps))
    if n_steps == 1:
        axes = axes[np.newaxis, :]

    vmin = min(v.min() for v in V_history)
    for step, (V_k, pi_k) in enumerate(zip(V_history, pi_history[1:])):
        plot_value_function(V_k, env,
                            title=f"Step {step + 1}: Value Function",
                            ax=axes[step, 0], vmin=vmin)
        plot_policy(pi_k, env,
                    title=f"Step {step + 1}: Improved Policy",
                    ax=axes[step, 1])

    fig.suptitle("Policy Iteration", fontsize=13, y=1.01)
    plt.tight_layout()
    return policy, V, pi_history, V_history, fig


def demo_value_iteration(env, gamma=1.0):
    """
    Run Value Iteration and return results.
    No figures are created; plot from the notebook for full control.

    Returns (V, policy, history).
    """
    V, policy, history = value_iteration(env, gamma)
    print(f"Value Iteration converged in {len(history) - 1} iterations.")
    return V, policy, history


def demo_td_evaluation(env, gamma=1.0, alpha=0.1, n_episodes=500):
    """
    Evaluate the optimal policy (from DP) with TD(0).
    No figures are created; plot from the notebook for full control.

    Returns (V_td, V_true, opt_policy).
    """
    opt_policy, V_true, _, _ = policy_iteration(env, gamma)
    V_td, _, _ = td_evaluation(opt_policy, env, gamma=gamma, alpha=alpha,
                                n_episodes=n_episodes, seed=0)
    valid = [env.state_index(s) for s in env.all_states()
             if s not in env.CLIFF and not env.is_terminal(s)]
    rmse = np.sqrt(np.mean((V_td[valid] - V_true[valid]) ** 2))
    print(f"TD(0) RMSE vs. true V after {n_episodes} episodes: {rmse:.3f}")
    return V_td, V_true, opt_policy


def demo_sarsa_qlearning(env, gamma=1.0, alpha=0.5, epsilon=0.1,
                          n_episodes=500, n_seeds=20):
    """
    Run SARSA and Q-learning over multiple seeds.
    No figures are created; plot from the notebook for full control.

    Returns a dict with Q functions, policies, and reward statistics.
    """
    print(f"Running SARSA × {n_seeds} seeds …")
    sarsa_mean, sarsa_std = run_multiple_seeds(
        sarsa, env, n_seeds=n_seeds,
        gamma=gamma, alpha=alpha, epsilon=epsilon, n_episodes=n_episodes)

    print(f"Running Q-learning × {n_seeds} seeds …")
    ql_mean, ql_std = run_multiple_seeds(
        q_learning, env, n_seeds=n_seeds,
        gamma=gamma, alpha=alpha, epsilon=epsilon, n_episodes=n_episodes)

    Q_sarsa, _ = sarsa(env, gamma=gamma, alpha=alpha,
                        epsilon=epsilon, n_episodes=n_episodes, seed=0)
    Q_ql, _    = q_learning(env, gamma=gamma, alpha=alpha,
                              epsilon=epsilon, n_episodes=n_episodes, seed=0)

    return dict(
        Q_sarsa=Q_sarsa, Q_ql=Q_ql,
        pi_sarsa=q_to_policy(Q_sarsa, env),
        pi_ql=q_to_policy(Q_ql, env),
        sarsa_mean=sarsa_mean, sarsa_std=sarsa_std,
        ql_mean=ql_mean,       ql_std=ql_std,
    )
