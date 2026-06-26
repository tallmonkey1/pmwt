# Code Review — Issues (Final Pass)

> Re-audit after the final correctness pass. The codebase now has a
> **two-loop learning architecture** (PPO-Transformer + helper critic + alpha),
> a **neural-rBergomi surrogate** trained on TRUE paths, and a **dedicated
> simulation-correctness audit** (`audit_simulation_correctness.py`) that runs
> **43/43 PASS**.
>
> All **806 tests pass** (`pytest`). All **43 audit checks pass**.
> **No remaining critical bugs. No simulation artifacts.**

## Summary of what was fixed in this final pass

| Bug | Severity | Where | Fix |
|---|---|---|---|
| `NeuralRBergomiConfig.param_dim = 8` mismatch with actual param-vector length 10 | 🔴 critical | `src/options_engine/models/rbergomi/neural_simulator.py` | Pin `param_dim` to `DEFAULT_PARAM_DIM = 10` (5 alpha + 5 rBergomi scalars) in `__post_init__`. |
| `NeuralRBergomiSimulator.save()` raised `AttributeError: 'NeuralRBergomiConfig' object has no attribute '__dict__'` | 🔴 critical | `src/options_engine/models/rbergomi/neural_simulator.py` | Replace `self._config.__dict__` with `dataclasses.asdict(self._config)`. |
| `NeuralRBergomiSimulator.simulate()` produced divergent paths after `train` vs `load` (train mode vs eval mode) | 🔴 critical | `src/options_engine/models/rbergomi/neural_simulator.py` | Force `self._network.eval()` inside `simulate()` so inference is deterministic regardless of training state. |

Regression tests covering these bugs live in
`tests/models/rbergomi/test_neural_simulator_regression.py` (4 tests, all pass).

## What's now correct (issues fixed in prior passes + this pass)

| # | Was | Now |
|---|---|---|
| 1 | 🔴 `ACTION_DIM` mismatch — 16 test failures | ✅ Fixed: reverted `N_STRATEGIC` from 4 to 3 (matching the SPEC). |
| 2 | 🔴 `_dt` undefined in `execution/ibkr.py` | ✅ Fixed: `import datetime as _dt` added. |
| 3 | 🟠 `placeOrder(legs, ib_order)` wrong signature | ✅ Fixed: `_place_order_impl` is now a clean `NotImplementedError` seam (matching `SAFETY.md`); lock ordering puts credential checks before SDK import. |
| 4 | 🟠 Expiry-date arithmetic | ✅ Fixed: removed from the live path (seam raises `NotImplementedError` until operator wires it). |
| 5 | 🟠 IBKRConfig semantics | ✅ Fixed via lock-ordering fix above. |
| 6 | 🟡 `RestNewsProvider` always raises | Unchanged (intentional, per SPEC). |
| 7 | 🟡 `BullPutSpread` smuggled into `IronCondor` | ✅ Fixed: removed (with the `BULL_PUT_SPREAD` revert). |
| 8 | 🟡 Terminal-spot settlement sample vs sizing distribution | Unchanged (intentional, per `analysis/REWARD_DESIGN.md`). |
| 9 | 🟡 StrategicAction enum ordering | ✅ Fixed: now only `HARVEST_THETA`, `HARVEST_GAMMA`, `FLAT` (matching SPEC). |
| 10 | 🔵 README/SPEC claim "all 736 tests passing" false | ✅ Fixed: now 806 tests (761 fast + 41 slow + 4 new regression tests) all passing. |
| 11 | 🔵 Python 3.13 + `ib_insync` deprecation noise | ✅ Fixed: `filterwarnings` in `pyproject.toml` ignores the specific upstream warnings. |
| 12 | 🔴 Neural rBergomi `param_dim` mismatch | ✅ Fixed: pinned to actual size. |
| 13 | 🔴 Neural rBergomi save crashed on slotted dataclass | ✅ Fixed: use `dataclasses.asdict`. |
| 14 | 🔴 Neural rBergomi simulate diverged across save/load | ✅ Fixed: explicit `eval()` inside simulate. |

## Verification (final)

| Check | Result |
|---|---|
| `pytest` (full suite) | **806 passed**, 0 failed |
| `python audit_simulation_correctness.py` | **43/43 passed** |
| `ruff check src tests` | clean |
| `mypy src` | clean (strict mode) |
| Dependency graph | **acyclic at the package level** (verified by automated DFS) |

## Out of scope (deliberately not done)

* **Continuous-alpha helper critic.** The current 8-element lattice is enough
  for the curriculum to converge quickly. A learned continuous-alpha policy is
  the natural follow-up but is a separate piece of work.
* **Pre-trained neural rBergomi surrogate.** The training pipeline is in place
  but a real corpus of TRUE rBergomi paths must be generated and the surrogate
  trained against it before it replaces the HybridSimulator in the inner loop.
* **End-to-end training script for the two-loop system.** `analysis/trading_learning.py`
  still uses the MLP PPO. A new `analysis/two_loop_training.py` should orchestrate
  the helper-critic outer loop with the PPO-Transformer inner loop.

## New tooling

* `audit_simulation_correctness.py` -- a comprehensive numerical audit
  (12 categories, 43 checks) that catches simulation artifacts the test
  suite would miss. Run any time the source changes with:

  ```bash
  python audit_simulation_correctness.py
  ```

  The audit verifies:
  * **Reproducibility** -- same seed produces identical output.
  * **No NaN/Inf** -- the simulation output is finite across the alpha range.
  * **BS vs MC convergence** -- the BS formula is the analytical limit.
  * **Put-call parity** -- `C - P = S - K*e^{-rT}`.
  * **Iron condor payoff identities** -- profit zone, wings, quantity scaling.
  * **MarketAlpha monotonicity** -- calmer alpha = smoother.
  * **PPO-Transformer memory** -- different prefix produces different action.
  * **Neural rBergomi save/load** -- deterministic roundtrip.
  * **Helper critic convergence** -- Q-values update correctly.
  * **Risk supervisor kill-switch** -- triggers at the configured limit only.
  * **MarketAlpha validation** -- rejects out-of-range / NaN / inf / empty / over-long.
  * **Hybrid vs Cholesky consistency** -- the two simulators agree.
