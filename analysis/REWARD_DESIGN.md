# Reward Design Study: eliminating the "do-nothing" collapse

This document studies *why* the original reward makes the agent collapse to never trading, and
designs a reward whose optimum is **trade-when-there-is-edge** rather than **do nothing**. It
is analysis + design only; the implementation is `growth_reward.py`, proven by the unit tests
in `tests/rl/test_growth_reward.py` (pure arithmetic — no model is trained to verify it).

---

## 1. The diagnosis (measured)

A decomposition of one trading episode under the original reward gave:

| term | value |
|---|---|
| P&L | **+0.235** (positive edge!) |
| risk penalty (−λ·CVaR) | −1.18 |
| cost penalty (−λ·cost) | −0.54 |
| **net** | **−1.49** |

So trading has *genuinely positive expected P&L*, but the penalties dwarf it, making the net
reward of trading negative. PPO did its job perfectly — it found the optimum — but **the
optimum was "stay flat"** (reward ≡ 0). The collapse is a reward-shaping defect, not a
learning-algorithm bug.

There are **three structural reasons** the original reward induces this:

1. **The "do nothing" floor is a free, risk-free zero.** FLAT yields exactly 0 (no P&L, no
   cost, no risk). Trading is a noisy, penalty-laden, *mean-positive* bet. A certain 0 beats a
   noisy +ε unless ε is large — and advantage-normalized PPO is mildly risk-averse, so it
   prefers the certain zero.

2. **Chicken-and-egg / deceptive reward.** Early in training the policy picks *bad* condors, so
   trading's realized reward is genuinely negative. The agent learns "trading is bad" before it
   can learn "good trading is good." It never explores its way out because flat is safe.

3. **Realized P&L is a very high-variance teacher.** A short condor usually expires for a small
   gain and occasionally takes a large loss (left-skew). Training on the *single realized
   terminal draw* gives an extremely noisy signal: the agent often can't tell a good decision
   (that got unlucky) from a bad one. Sample efficiency suffers badly.

---

## 2. Design principles for a reward whose optimum is "trade the edge"

### P1 — Make "do nothing" cost something *when, and only when, edge exists*
The deepest fix: declining a **qualifying, positive-expected-value** condor incurs a small
**opportunity cost**. This removes the free-zero attractor. Crucially it is **self-gating**:
the opportunity cost uses the *existing* condor-selection pipeline, which returns no candidate
in high-vol / news-blackout / no-edge conditions. So:

* calm market with real edge → declining is penalized (the agent must justify sitting out),
* bad market with no qualifying condor → edge = 0 → FLAT is free (we never force a bad trade).

This is exactly the behaviour the spec wants ("don't collapse to do nothing" *and* "don't
over-strictize so no trade is made"): the gate decides *whether* edge exists; the reward only
nudges the agent to take edge that the gate already certified.

### P2 — Teach with *expected* edge, not just realized P&L
Reward the **risk-adjusted expected P&L of the chosen condor** (which the engine already
computes: `candidate.expected_pnl`) as a dense, low-variance signal, blended with the realized
P&L for grounding. "Was this a good *bet*?" is far more learnable than "did this specific bet
*win*?". This is potential-based shaping using the model's own EV estimate — legitimate
because that EV is exactly the quantity the agent is meant to exploit — and it is **decayed to
zero** over training so the final policy is judged on true growth, not the shaping.

### P3 — Use log-wealth (Kelly) growth as the core objective
Make the core per-step reward the **log-wealth increment** `log(1 + pnl/equity)`. This is the
theoretically-grounded objective: maximizing `E[log wealth]` is maximizing long-run growth (the
Kelly criterion). It has three properties that directly fix the collapse:

* **Do-nothing = `log(1) = 0`** — still the neutral baseline, but…
* **A positive-edge bet has positive expected log-growth** (the Kelly theorem). So when edge
  exists, trading is *provably* better than the zero baseline — the optimum is no longer "do
  nothing."
* **Losses are punished concavely** — a 10% loss hurts more than a 10% gain helps, which is the
  correct, built-in risk aversion for a short-gamma book, *without* an ad-hoc CVaR penalty
  large enough to kill the edge.

### P4 — Risk-normalize so the weights are interpretable
Normalize every term by `equity` (or per-trade risk capital) *before* weighting, so a weight of
`1.0` means "this term matters as much as one unit of growth." The original used a fixed
`reference_risk=1000`, which made the weights un-calibrated to the actual signal magnitudes
(hence the 0.235-vs-1.18 imbalance). With normalized terms the soft risk penalty can be small
(e.g. 0.1) and still meaningful, and it never swamps the edge.

### P5 — Keep the hard limit-breach penalty
Breaching a hard risk limit must remain strictly dominated, so a large fixed penalty stays.
This is a safety floor, not a tuning knob.

---

## 3. The new reward (formal definition)

Per step, with `equity` = account equity at step start:

```
step_return = clip(pnl_change / equity, -1 + ε, +∞)
growth      = log(1 + step_return)                              # P3, Kelly core

opportunity = −opp_w · max(0, best_available_edge / equity)     # P1, only if NOT traded
edge_shape  = edge_w · shaping_coef · (chosen_expected_edge / equity)   # P2, decayed
risk_pen    = −risk_w · max(0, incremental_cvar / equity)       # P4, soft
breach_pen  = −breach_penalty   if limit_breached else 0        # P5, hard

reward = growth + opportunity + edge_shape + risk_pen + breach_pen
```

where `opportunity` is applied only on FLAT steps and `edge_shape` only on TRADE steps.

### Provable properties (see the tests)
1. **FLAT, no edge → reward = 0.** We never punish sitting out a bad market.
2. **FLAT, edge available → reward < 0.** The free-zero attractor is gone.
3. **TRADE realizing positive return → reward > 0.**
4. **TRADE realizing a loss → reward < 0 but bounded** (concave log; defined-risk keeps
   `1 + step_return > 0`).
5. **Do-nothing is never optimal when edge exists**: taking a positive-edge trade that realizes
   its EV yields strictly higher reward than declining the same edge — provable deterministically
   because `growth(EV) + edge_shape > −opportunity` whenever EV > 0 and the soft risk penalty is
   below the (opportunity-boosted) edge. This is the calibration invariant, and it is asserted
   by a unit test.

These are arithmetic facts about the reward function, verified without training anything.

---

## 4. Recommended agent upgrades (design only — validate locally)

The reward fix removes the collapse. Two agent-side changes make learning faster and more
robust; they need training to validate, so they are specified here for you to enable and tune.

### A1 — Risk-sensitive advantages via the distributional critic (CVaR-PPO)
The critic already predicts the *distribution* of returns and exposes `cvar()`. Instead of GAE
advantages from the **mean** value `V(s)`, use a risk-sensitive baseline
`V_β(s) = (1−β)·mean + β·CVaR_α`, `β ∈ [0, 1]`. This makes the policy optimize a coherent risk
measure directly — principled risk aversion in the *objective*, not bolted onto the reward.
It is a small change in `PPOAgent.update` (swap the baseline used for the advantage), gated by a
`risk_aversion_beta` config defaulting to 0 (i.e. standard PPO) so it is opt-in.

### A2 — Entropy floor / optimistic strategic init (anti-premature-determinism)
The greedy (deterministic) policy can lock onto FLAT before learning, because the near-zero
mean-head init makes the strategic arg-max essentially random and the agent stops exploring
once advantage normalization shrinks gradients. Two cheap, standard mitigations:

* **Entropy floor:** keep `entropy_coef` from decaying below a small positive value, or add a
  target-entropy term, so the policy never becomes prematurely deterministic.
* **Optimistic strategic init:** initialize the FLAT logit slightly *below* the harvest logits
  so the *starting* exploration has a mild trading prior. This biases only the initial
  exploration, not the converged policy (it is washed out by learning), and it lets the agent
  discover that good trading pays before it gives up. Implement as an optional init flag on
  `GaussianActor`; default off to preserve current behaviour and tests.

### A3 — Train long enough, and judge on the promotion gate
The earlier 25-iteration demo was ~100× too short for a noisy financial task. Use a realistic
budget (hundreds–thousands of updates), decay the edge-shaping to zero over the first ~30–50% of
training, and **let the deflated-Sharpe promotion gate be the arbiter** — it already correctly
rejected the collapsed agent, so it is the honest performance test.

---

## 5. How to use the new reward

`growth_reward.compute_growth_reward(...)` is a drop-in alternative to `reward.compute_reward`
with the same breakdown-style output. Wire it into the environment by passing a
`GrowthRewardConfig` (the env exposes an opt-in hook; the default remains the original reward so
existing behaviour and tests are unchanged). The env already computes the chosen condor's
`expected_pnl` and can compute the best-available edge on flat steps via the same
`select_iron_condor` call it uses for the incremental-CVaR estimate.
