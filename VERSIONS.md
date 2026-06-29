# Artifact versions

All current artifacts are 1.0.0 (the research scripts, curated from the original work into this repo).
Each carries `# artifact-version:` in-file.

| artifact | version | role / what it tests |
|---|---|---|
| `core.py` | 1.0.0 | the prior + machinery: featurize (board→relational graph), exact Syzygy `label`, `load_positions`, `mine_schemas`, `align`/`evaluate` |
| `mine_run.py` | 1.0.0 | mine schemas by support × lift on real positions; which named concepts emerge |
| `pressure_test.py` | 1.0.0 | rediscovery vs a permutation null (survives) + a vocabulary lesion (fails) |
| `ho_test.py` | 1.0.0 | do higher-order ENABLES/CAUSE edges earn their keep? (no; a missing first-order primitive does) |
| `transfer_test.py` | 1.0.0 | KRPvKR → KQPvKQ cross-material transfer (geometry carries, rook-cutting doesn't) |
| `nsv_transfer.py` | 1.0.0 | no-shared-vocabulary R↔Q analogy discovery (structure finds it, ties under symmetry) |
| `separation_test.py` | 1.0.0 | does relational beat flat first-order for transfer? (no — flat transfers better) |
| `krpvkr_pilot.py` | 1.0.0 | compression-optimal vs discrimination-optimal schema selection (exact labels) → null |
| `krpvkr_pilot2.py` | 1.0.0 | stricter non-combining rerun of the above → null again; redundancy is the cause |
| `forcing_content.py` | 1.0.0 | tempo-criticality (stm-flip WDL fraction) per material — the structure measure |

Curated from the original research dir; module `learn_schemas_scaffold` was renamed `core` and the
cross-imports updated. Behaviour unchanged.
