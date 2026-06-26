# Criticality Ranking -- how each module matters for simulation accuracy and training

Every module in ``modules.json`` carries a ``criticality`` record with these fields:

| Field | Meaning |
|---|---|
| ``simulation_accuracy`` | How much this module affects the numerical correctness of the price/distribution simulation (1 = peripheral, 5 = core closed-form math). |
| ``training_accuracy`` | How much this module affects the training loop (gradients, convergence, outer-loop learning) (1 = peripheral, 5 = loss / PPO / reward / backbone). |
| ``overall_criticality`` | ``max(simulation_accuracy, training_accuracy)``. |
| ``role`` | Short tag categorising what the module does. |
| ``rationale`` | Why this scoring (free-form, 1-3 sentences). |
| ``regression_sensitivity`` | What happens on a bug: ``catastrophic`` (silently wrong numbers / real money loss / training divergence), ``high`` (significant accuracy loss / training instability), ``medium`` (noticeable but recoverable), ``low`` (cosmetic / minor). |
| ``fallback_strategy`` | How to mitigate a regression: ``circuit_breaker`` (kill switch), ``theory_check`` (compare to analytical limit), ``mc_fallback`` (substitute direct MC ground truth), ``manual_review`` (operator review), ``none`` (no critical failure mode). |

## Aggregate histograms (current run)

### Simulation accuracy

| Score | Modules |
|---|---:|
| 1 | 1 |
| 2 | 23 |
| 3 | 28 |
| 4 | 28 |
| 5 | 23 |

### Training accuracy

| Score | Modules |
|---|---:|
| 1 | 1 |
| 2 | 31 |
| 3 | 35 |
| 4 | 19 |
| 5 | 17 |

### Regression sensitivity

| Severity | Modules |
|---|---:|
| low | 23 |
| medium | 18 |
| high | 35 |
| catastrophic | 27 |

### Fallback strategy

| Strategy | Modules |
|---|---:|
| none | 19 |
| manual_review | 13 |
| mc_fallback | 13 |
| theory_check | 33 |
| circuit_breaker | 25 |


## Top-criticality modules (must be regression-tested first)

These modules have **catastrophic** regression sensitivity: a bug here either
silently corrupts simulation results, breaks training convergence, or -- for the
execution layer -- can route a real-money order by accident. They should be the
first thing covered by any new regression test, and any code review touching them
must be especially careful.

| Module | Package | Role |
|---|---|---|
| agent.gae | agent | gae_advantage_computation |
| agent.helper_critic | agent | alpha_meta_learner |
| agent.losses | agent | ppo_clip_and_quantile_huber_losses |
| agent.ppo | agent | ppo_mlp_core |
| agent.ppo_transformer | agent | ppo_transformer_core_with_memory |
| backtest.promotion | backtest | history_to_live_promotion_gates |
| calibration.hurst | calibration | hurst_estimator |
| core.market_alpha | core | alpha_to_model_mapping |
| execution.ibkr | execution | ibkr_live_adapter |
| execution.live_guard | execution | typed_phrase_live_trading_arming |
| execution.oms | execution | idempotent_risk_gated_oms |
| models.rbergomi.alpha_calibration | models.rbergomi | alpha_to_rbergomi_params |
| models.rbergomi.kernel | models.rbergomi | volterra_kernel_math |
| models.rbergomi.params | models.rbergomi | rbergomi_params_container |
| models.rbergomi.simulator | models.rbergomi | ground_truth_rbergomi_simulator |
| news.gate | news | news_event_trade_gate |
| pricing.black_scholes | pricing | analytical_bs_pricing_oracle |
| pricing.payoff | pricing | payoff_functions |
| regime.gate | regime | trade_gate_low_vol_only |
| rl.action | rl | parameterized_action_decoder |
| rl.env | rl | gymnasium_pomdp_env |
| rl.growth_reward | rl | growth_optimal_alternative_reward |
| rl.reward | rl | risk_sensitive_reward |
| services.runner | services | broker_factory |
| strategy.risk_supervisor | strategy | hard_risk_supervisor_with_kill_switch |
| surrogate.guardrail | surrogate | mc_fallback_decision |
| surrogate.losses | surrogate | pinball_quantile_loss |


## Why this matters

The criticality ranking is the answer to the question "if I have one hour to
regression-test this codebase, which modules do I test first?". The
simulation_accuracy=5 + training_accuracy=5 modules are the math cores and
the learning algorithm -- the 27 catastrophic modules form the union of
"simulation-critical" and "training-critical" modules.

The full per-module criticality record lives in ``modules.json`` under each
module's ``criticality`` key.
