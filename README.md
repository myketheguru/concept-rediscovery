# concept-rediscovery

**Can a compression objective, run on data with exact ground truth, rediscover named human
chess-endgame concepts — Lucena, Philidor, opposition — without being told them?**

This is **Track B** of a research program on the powers and limits of representation. It's the
original experiment — with honest ceilings, including **null results that redirected the question.**

> **Part of one body of curious work.** Companion repo —
> **[representation-dial](https://github.com/myketheguru/representation-dial)** — holds Track **A**
> (foundations: why search is conserved, where a matched representation collapses it) and Track **C**
> (the frontier: representation as the access/search dial). This repo is Track **B**.

> Engineering discipline pointed at foundational science: exact ground truth (Syzygy tablebases),
> every claim a falsifiable experiment, **nulls reported as loudly as wins.**

## The question
Are human endgame concepts just *high-compression structures of outcome data*? Endgames let us test
it, because the outcome is **exact** — Syzygy tablebases give win/draw/loss with zero label noise.
We mine relational schemas from labelled positions by *compression × predictive lift* and ask whether
the named concepts emerge — and, when they do, whether it's the **data** or the **prior** doing the work.

## The tour (run in this order)
The repo is flat (the scripts share a `core` module); the narrative order is:

1. **Setup — the prior.** `core.py` — board → relational graph in a hand-built vocabulary (*the
   vocabulary IS the inductive bias*) + exact Syzygy labels + the mining/alignment machinery.
2. **Mining — rediscovery.** `mine_run.py` — mine schemas by support × lift; see which named concepts
   emerge → *partial rediscovery* (the honest headline).
3. **Pressure tests — the honest core.** `pressure_test.py` — rediscovery **survives a permutation
   null** but **fails a vocabulary lesion**. The prior is load-bearing: *capability tracks the prior.*
4. **What helps, what doesn't.**
   - `ho_test.py` — higher-order (ENABLES/CAUSE) edges earn **nothing**; a missing *first-order*
     primitive (rule of the square) does.
   - `transfer_test.py` — KRPvKR → KQPvKQ: king/pawn geometry transfers, rook-cutting doesn't.
   - `nsv_transfer.py` — with **no shared vocabulary**, structure discovers the R↔Q analogy but ties
     under a symmetry; relational > unary.
   - `separation_test.py` — flat first-order **transfers better** than relational; relational only
     helps mapping-discovery.
5. **The nulls — the ceiling.** `krpvkr_pilot.py` + `krpvkr_pilot2.py` — does compression-optimal schema
   selection differ from discrimination-optimal? → **null, twice** (gap +0.004, then +0.000); cause =
   vocabulary redundancy. `forcing_content.py` — the tempo-criticality measure used there.

See **[FINDINGS.md](FINDINGS.md)** for the one-page honest summary, **[PROJECTS.md](PROJECTS.md)** for
how this fits the program.

## Running it
Python 3.11+, `pip install -r requirements.txt`, plus **data you download yourself** (exact tablebases
+ games — not bundled; see **[SETUP_DATA.md](SETUP_DATA.md)**). Each script self-documents and prints
its verdict; run from the repo root, e.g. `python mine_run.py`.

## About
By **myketheguru** — software engineer (10+ yrs), independent researcher. Versions in
[VERSIONS.md](VERSIONS.md).

## Honest framing
A real, interpretable, **bounded** result — concept-rediscovery with documented limits and honest
nulls, **not a breakthrough**. The thesis: *the relational vocabulary you choose IS the inductive
bias.* Compression rediscovers some human concepts, but the prior does much of the work — and the
headline compression-vs-discrimination hypothesis nulled twice. The ceilings are the result.
