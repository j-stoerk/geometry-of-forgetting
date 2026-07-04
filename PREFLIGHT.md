# The Pre-Flight Protocol — decision *before* training, scored after

Standard CL benchmarks score methods after training. This protocol benchmarks
**decisions made before training**: the geometry panel is computed from probes,
the predictions and per-task actions are *committed*, training runs, and then
the decisions — not just the scores — are evaluated. (Roadmap: "standard
benchmark protocols for 'decision before training', not just 'score after
training'.")

## The protocol

| Step | What happens | What is committed |
|---|---|---|
| **P1 Panel** | From small probes only (features + targets per task; no training): pairwise overlap matrix, share density, per-task predicted floor, predicted naive forgetting, OOD screen | the numbers |
| **P2 Commit** | One recommended action per task (share / project / replay / skip / expand), with the predicted outcome of following them | actions + predictions |
| **P3 Train** | Run the stream under (a) naive, (b) the committed actions, (c) fixed strategies for reference | — |
| **P4 Score** | **Prediction fidelity**: rank correlation + relative error of predicted vs realized naive forgetting. **Decision quality**: regret vs the best fixed policy in hindsight, *budget-normalized* (replay events, added rank, fetches are resources, not free). **Safety asymmetry**: false alarms (cost: wasted budget) vs misses (cost: retention) reported separately | — |

Success is not "our method wins" — it is *the panel predicted what happened and
the committed actions were the right calls*, with errors on the cheap side
(false alarm > miss).

## Instantiations in this repository

- **Exact regime** ([`evaluate_preflight_protocol.py`](evaluate_preflight_protocol.py),
  30 random mixed-regime streams): prediction fidelity Spearman ρ = 1.000,
  median relative error 0.0% (the geometry *is* the outcome under A1);
  committed actions beat both zero-budget fixed policies outright and reach
  80% of always-replay's loss at 27% of its replay budget.
- **Domain demo, physical drift**
  ([`demo_process_surrogate.py`](demo_process_surrogate.py), adaptive process
  surrogate; campaigns with new windows, fouling drift, partial-overlap shifts,
  sensor glitches; replay = costed archive fetches): the operational pre-flight
  question *"will this update corrupt a validated operating window?"* catches
  20/20 damaging updates with 5 conservative false alarms; the controller gets
  the best validated-window retention (0.071 RMSE vs replay-all 0.97, naive
  7.9) at half the archive budget. Replay-all *loses* despite full budget —
  rehearsal cannot reject glitches; the decision layer can.
- **Real streams**: the same panel runs on cached ViT features
  (`kaggle_vit_multiseed.py` caches) and LLM micro-caches
  (`kaggle_functional_gate_v2.py`); there the protocol's value inverts — it
  measures how much predictive power survives estimation error (the LR-1e-5
  pythia run is the cautionary instance: pre-flight would have flagged the
  stream as non-forgetting *before* 80 GPU-minutes were spent).

## Reporting template

```
panel:      density=…  max_overlap=…  predicted_naive_forgetting=…  floors=[…]
committed:  task0=share task1=replay …   predicted_outcome=…
realized:   naive=…  committed=…  best_fixed(hindsight)=…(policy)
fidelity:   rho=…  rel_err=…
decisions:  regret=…  budget_used/budget_best=…  false_alarms=…  misses=…
```

One line of the ledger (`interference_ledger.py` → `LedgerRow`) per training
step provides the online continuation of the same panel after flight begins.
