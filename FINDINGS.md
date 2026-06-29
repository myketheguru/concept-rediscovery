# Findings — the honest one-pager

**Question:** does the compression/systematicity objective, on exact-label endgame data, rediscover
named human concepts (Lucena, Philidor, opposition)?

## What we found
1. **Partial rediscovery.** Mining relational schemas by support × predictive lift surfaces structures
   that line up with some named concepts — but not cleanly, and not all. (`mine_run.py`)
2. **The prior is load-bearing (the core result).** Rediscovery **survives a permutation null** (so
   it's tracking real outcome structure, not noise) but **fails a vocabulary lesion** (swap the
   relational vocabulary for a concept-agnostic one and it collapses). *Capability tracks the prior;
   the vocabulary you choose IS the inductive bias.* (`pressure_test.py`)
3. **Higher-order structure earns nothing here.** ENABLES/CAUSE edges add **+0.000**; the only thing
   that helped was a *missing first-order primitive* (rule of the square, **+0.062**). For scoring a
   position, first-order fidelity is the whole game. (`ho_test.py`)
4. **Transfer is geometric, not relational.** KRPvKR → KQPvKQ: king/pawn geometry carries; rook-cutting
   doesn't. And a **flat first-order** model transfers *better* than the relational library at matched
   accuracy — conjunctions overfit piece-specific structure. (`transfer_test.py`, `separation_test.py`)
5. **Relational structure helps *mapping discovery*, not scoring.** With no shared vocabulary, argument
   structure recovers the R↔Q correspondence ~2× better than unary predicates — but ties under a
   structural symmetry (the tiebreak needs another prior). (`nsv_transfer.py`)
6. **The headline hypothesis nulled — twice.** "Compression-optimal schemas differ predictively from
   discrimination-optimal ones" gave a gap of **+0.004**, then **+0.000** on a stricter, non-combining
   rerun. The two objectives select *different* schemas (Jaccard ~0.4) that are **predictively
   interchangeable**; cause = vocabulary redundancy. The missed-by-compression schemas are tempo-coupled
   (on-thesis) but cost nothing. (`krpvkr_pilot.py`, `krpvkr_pilot2.py`)

## The honest ceiling
Compression rediscovers *some* human concepts, but **the prior does much of the work**, higher-order
relations add nothing for scoring, transfer rides first-order geometry, and the cleanest
compression-vs-discrimination claim is null. This is a real, interpretable, bounded result — the value
is the falsifiable experiments and the documented limits, not a grand claim.
