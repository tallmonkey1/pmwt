# Execution Safety Model (read before touching this package)

> **A wrong line here can lose real money.** This package is built *fail-closed*: every
> default is the safe one, and sending a real order requires several independent, deliberate
> conditions to all be true at once. No single mistake — a wrong flag, a leaked credential, a
> typo — can route an order to a live broker.

## The five independent locks on real orders

A live order can only reach a real broker if **all** of the following hold simultaneously:

1. **Operational mode is `ACCOUNT_TRADING`.** History and paper modes physically cannot
   construct a live broker (the factory refuses).
2. **Credentials are present** in the environment (resolved via `SecretRef`, never stored).
3. **`enable_live_trading=True`** is passed explicitly to the live-broker factory.
4. **A typed confirmation token** exactly matches the required phrase
   (`LiveTradingArming.REQUIRED_PHRASE`). A boolean alone is not enough — arming requires
   typing the phrase, which cannot happen by accident or by a default value.
5. **The risk supervisor approves the order** (per-trade/day caps, leverage, margin,
   drawdown kill-switch). The OMS calls it on *every* order, in every mode.

If any lock is not satisfied, the system uses the **simulated broker** (or refuses to start
live trading) and logs the reason. The simulated broker is the default everywhere.

## Idempotency

Every order carries a **client order id**. The OMS rejects a second submission with the same
id (deduplication), so a retry, a reconnect, or a double-call can never create a duplicate
position. This is the standard institutional protection against the most expensive execution
bug: accidental order multiplication.

## Reconciliation

The OMS keeps a local order/position ledger and can **reconcile** it against the broker's
reported state, surfacing any divergence (a fill the broker has that we don't, or vice
versa). Divergence halts new trading until resolved.

## Kill switch

The OMS exposes `flatten_all()`, which the runner invokes when the risk supervisor's
trailing-drawdown kill-switch trips. Flattening is defined-risk (closing condors), and is
always permitted even when *opening* is blocked.

## What is intentionally NOT implemented here

The actual IBKR network/order calls are a single, clearly-marked integration seam
(`IBKRBroker._place_order_impl`). Without credentials *and* arming it raises a clear error
rather than silently doing nothing. Wiring the vendor SDK is the one step that requires the
operator's own account and explicit, audited sign-off — by design.
