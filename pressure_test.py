# artifact-version: 1.0.0
"""
pressure_test.py  --  try to BREAK the concept-rediscovery claim.

Two adversarial tests, both designed to be able to falsify "the compression
objective rediscovers human concepts":

  (A) LABEL-PERMUTATION NULL. Shuffle the outcomes, re-mine. If the held-out
      lift / accuracy of the "rediscovered" schemas survives shuffling, the
      result was multiple-comparisons overfitting, not signal. Real must beat
      the null by a lot.

  (B) VOCABULARY LESION. Re-run with a concept-AGNOSTIC spatial vocabulary
      (featurize_krpvkr_dumb: raw rank/file buckets + same-file/rank/adjacency,
      no cuts/blocks/supports/ENABLES). If the concepts + accuracy survive the
      lesion, they live in the DATA; if they collapse (or stop separating from
      the null), the rich prior was doing the work -- exactly what the thesis
      predicts (capability tracks the prior). Either way is honest.

Scoring uses the spine evaluate()/align() throughout (faithful; the rich vocab
names roles so alignment is ~identity, but the dumb vocab's symmetric predicates
genuinely need the search). Mining + lift use exact literal containment, so the
support/lift/null numbers are unaffected by alignment ambiguity.

Reads data/krpvkr_cache.csv (build it with build_cache.py). Nothing is rigged:
labels come from Syzygy, the miner never sees outcome names or concept names.
Run: python pressure_test.py [n_shuffles]
"""
import sys, csv, random, statistics, collections
import chess
from core import (
    featurize_krpvkr, featurize_krpvkr_dumb, mine_schemas, evaluate, _literals,
)
from mine_run import pattern_stats, _select

CACHE = "data/krpvkr_cache.csv"


def load_cache():
    train, hold = [], []
    with open(CACHE, newline="") as f:
        for r in csv.DictReader(f):
            b = chess.Board(r["fen"])
            (hold if r["holdout"] == "1" else train).append((b, r["outcome"]))
    return train, hold


def graphs(boards, featurizer):
    return [(featurizer(b), o) for b, o in boards]


def lib_accuracy(library, hold_lg, maj):
    """(coverage, covered_accuracy, full_accuracy) via the spine evaluate().
    full_accuracy commits the library where confident and falls back to the
    majority class otherwise -- directly comparable to baseline and to the null."""
    cov = cor = full = 0
    for g, truth in hold_lg:
        v = evaluate(g, library)[0] if library else "UNCERTAIN"
        pred = maj if v == "UNCERTAIN" else v
        if v != "UNCERTAIN":
            cov += 1; cor += (v == truth)
        full += (pred == truth)
    n = len(hold_lg)
    return cov / n, (cor / cov if cov else 0.0), full / n


def mine_once(train_lg, hold_lg, min_support, max_literals, maj):
    schemas = mine_schemas(train_lg, holdout=hold_lg, min_support=min_support,
                           min_lift=0.10, max_literals=max_literals)
    gen = [s for s in schemas if s.get("generalizes")]
    lib = _select(gen)
    best_hl = max((s.get("holdout_lift", 0) for s in schemas), default=0.0)
    cov, acc, full = lib_accuracy(lib, hold_lg, maj)
    return {"n_schemas": len(schemas), "n_gen": len(gen), "best_holdout_lift": best_hl,
            "lib_size": len(lib), "coverage": cov, "accuracy": acc, "full_accuracy": full,
            "schemas": schemas, "lib": lib}


def permutation_null(train_lg, hold_lg, min_support, max_literals, shuffles, maj):
    """Re-mine with outcomes shuffled within each split (preserves base rates)."""
    rng = random.Random(0)
    accs, best_hls, n_gens = [], [], []
    tr_o = [o for _, o in train_lg]
    ho_o = [o for _, o in hold_lg]
    for _ in range(shuffles):
        a = tr_o[:]; rng.shuffle(a)
        b = ho_o[:]; rng.shuffle(b)
        tr = [(g, a[i]) for i, (g, _) in enumerate(train_lg)]
        ho = [(g, b[i]) for i, (g, _) in enumerate(hold_lg)]
        r = mine_once(tr, ho, min_support, max_literals, maj)
        accs.append(r["full_accuracy"]); best_hls.append(r["best_holdout_lift"]); n_gens.append(r["n_gen"])
    msd = lambda xs: (statistics.mean(xs), statistics.pstdev(xs))
    return {"accuracy": msd(accs), "best_holdout_lift": msd(best_hls), "n_gen": msd(n_gens)}


def main():
    shuffles = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    train_b, hold_b = load_cache()
    base = collections.Counter(o for _, o in train_b)
    nT, nH = len(train_b), len(hold_b)
    maj = base.most_common(1)[0][0]
    maj_acc = sum(o == maj for _, o in hold_b) / nH
    msup = max(20, nT // 40)
    print(f"=== PRESSURE TEST  (train {nT} / holdout {nH}; base {dict(base)}) ===")
    print(f"majority baseline ('{maj}'): holdout accuracy {maj_acc:.3f}  (the number to beat)\n")

    for name, feat in [("RICH vocab (concept-shaped)", featurize_krpvkr),
                       ("DUMB vocab (lesion: spatial primitives only)", featurize_krpvkr_dumb)]:
        train_lg = graphs(train_b, feat)
        hold_lg = graphs(hold_b, feat)
        real = mine_once(train_lg, hold_lg, msup, 4, maj)
        null = permutation_null(train_lg, hold_lg, msup, 4, shuffles, maj)
        na, ns = null["accuracy"]; ba, bs = null["best_holdout_lift"]; ga, gs = null["n_gen"]

        print(f"--- {name} ---")
        print(f"  REAL : gen-schemas={real['n_gen']:5d}  best-holdout-lift={real['best_holdout_lift']:+.2f}  "
              f"lib={real['lib_size']}  coverage={real['coverage']:.2f}  "
              f"full-ACCURACY={real['full_accuracy']:.3f} (covered {real['accuracy']:.3f})")
        print(f"  NULL : gen-schemas={ga:5.0f}±{gs:.0f}  best-holdout-lift={ba:+.2f}±{bs:.2f}  "
              f"          full-ACCURACY={na:.3f}±{ns:.3f}  (label-shuffled x{shuffles})")
        libdist = collections.Counter(s["outcome"] for s in real["lib"])
        print(f"  library outcomes: {dict(libdist)}")
        d_acc = real["full_accuracy"] - na
        sig = d_acc / (ns or 0.01)
        d_lift = real["best_holdout_lift"] - ba
        # two independent reads: end-to-end accuracy AND evaluator-free lift gap
        usable = real["full_accuracy"] > maj_acc and sig > 2.0
        clean_lift = d_lift > 0.30
        verdict = ("SIGNAL: usable (>baseline) AND lift separates from null" if usable and clean_lift
                   else "PARTIAL: lift separates but library not usable by the evaluator" if clean_lift
                   else "FAILS: real below baseline / not separable from shuffled labels")
        print(f"  => {verdict}   [acc {real['full_accuracy']:.3f} vs base {maj_acc:.3f}, "
              f"{sig:+.1f}sigma vs null; lift gap {d_lift:+.2f}]\n")

    # ---- named-concept held-out lift: real vs null (RICH vocab; exact containment)
    print("--- named-concept held-out lift: REAL vs label-shuffled null (RICH vocab) ---")
    hold_lg = graphs(hold_b, featurize_krpvkr)
    ho_lits = [(_literals(g), o) for g, o in hold_lg]
    concepts = {
        "Lucena: rook cuts king + advanced pawn":
            frozenset([("wr_cuts_file", ("WR", "BK")), ("pawn_promoting", ("WP",))]),
        "King blockade -> draw": frozenset([("bk_blocks", ("WP",))]),
        "Philidor: blockade + checking rook": frozenset([("CAUSE_fortress", ("BK", "WP"))]),
        "Rook-pawn + advanced": frozenset([("rook_pawn", ("WP",)), ("pawn_promoting", ("WP",))]),
    }
    rng = random.Random(1)
    ho_o = [o for _, o in ho_lits]
    for cname, sig in concepts.items():
        _, _, _, real_l = pattern_stats(sig, ho_lits)
        nulls = []
        for _ in range(shuffles):
            sh = ho_o[:]; rng.shuffle(sh)
            _, _, _, l = pattern_stats(sig, [(lits, sh[i]) for i, (lits, _) in enumerate(ho_lits)])
            nulls.append(l)
        m, sd = statistics.mean(nulls), statistics.pstdev(nulls)
        flag = "REAL >> null" if real_l - m > 3 * (sd or 0.01) else "not distinguishable"
        print(f"  {cname:42s} real {real_l:+.2f} | null {m:+.2f}±{sd:.2f}  -> {flag}")


if __name__ == "__main__":
    main()
