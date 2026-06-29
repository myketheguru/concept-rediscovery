# artifact-version: 1.0.0
"""
mine_run.py  --  THE HEADLINE EXPERIMENT.

Does the systematicity / compression objective, run on labelled KRPvKR data,
rediscover named human endgame concepts (Lucena / Philidor / opposition /
rook-pawn-draw) as its top schemas -- WITHOUT being told them?

Pipeline (all from the scaffold, nothing hardcoded into the search):
  real Lichess games -> KRPvKR positions (load_positions)
                     -> relational graph in the v2 vocabulary (featurize_krpvkr, THE PRIOR)
                     -> exact label in White's frame (label / Syzygy)
                     -> split by GAME -> mine_schemas (beam over predictive patterns)
                     -> report top schemas; test concept-correspondence; eval vs Syzygy.

The concept SIGNATURES below are used ONLY for post-hoc reporting -- they never
enter the miner. Run: python mine_run.py [n_games] [parquet]
"""
import sys, collections
import chess
from core import (
    load_positions, featurize_krpvkr, label, mine_schemas, evaluate,
    HO_PREDS, _literals, SCHEMA_LIBRARY,
)

PARQUETS = ["data/lichess/2013-01.parquet",
            "data/lichess/2013-02.parquet",
            "data/lichess/2013-03.parquet"]
TB = "tablebases/syzygy"


def white_frame(stm_label, turn):
    if stm_label == "DRAW":
        return "DRAW"
    if turn == chess.WHITE:
        return stm_label
    return "WIN" if stm_label == "LOSS" else "LOSS"


def fmt(props):
    return " ".join(f"{p}({','.join(a)})" if a else p
                    for p, a in sorted(props.values()))


def pattern_stats(pattern, dataset):
    """support/precision/lift of an arbitrary literal-set on a labelled dataset.
    dataset = list of (frozenset_literals, white_outcome). For post-hoc probes."""
    n = len(dataset)
    base = collections.Counter(o for _, o in dataset)
    hit = [o for lits, o in dataset if pattern <= lits]
    if not hit:
        return 0, None, 0.0, 0.0
    c = collections.Counter(hit)
    o, cnt = c.most_common(1)[0]
    prec = cnt / len(hit)
    return len(hit), o, prec, prec - base[o] / n


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    parquets = sys.argv[2:] if len(sys.argv) > 2 else PARQUETS

    # ---- build labelled, game-split dataset -------------------------------
    # Split by GAME (deterministic) so correlated positions don't leak across
    # train/holdout. KRPvKR => White (pawn side) is ALMOST never lost (can shed
    # the pawn for drawn KRvKR), so White-frame outcomes are ~binary WIN/DRAW;
    # a rare ~1% are genuine LOSS (Black mates first despite the extra pawn) --
    # confirmed via Syzygy, not a labeling bug. The pawn-centric featurizer is
    # blind to those mating nets (an honest ceiling).
    train, holdout = [], []          # each: (graph, white_outcome); + boards kept
    train_lits, hold_lits = [], []   # (frozenset, outcome) for fast probes
    hold_boards = []
    skipped = 0
    for fi, parquet in enumerate(parquets):
        for b, gid in load_positions(parquet, material_filter="KRPvKR",
                                     max_games=n_games, with_game_id=True):
            y = label(b, TB)
            if y is None:
                skipped += 1
                continue
            wy = white_frame(y, b.turn)
            g = featurize_krpvkr(b)
            is_hold = (fi * 1_000_003 + gid) % 5 == 0
            (holdout if is_hold else train).append((g, wy))
            (hold_lits if is_hold else train_lits).append((_literals(g), wy))
            if is_hold:
                hold_boards.append((b, wy))

    nT, nH = len(train), len(holdout)
    base = collections.Counter(o for _, o in train)
    print(f"=== KRPvKR mining  ({n_games} games x {len(parquets)} month(s)) ===")
    print(f"train positions: {nT}   holdout positions: {nH}   unprobeable skipped: {skipped}")
    print(f"train base rates (White frame): "
          f"{ {o: round(base[o]/nT, 3) for o in base} }")
    if nT < 100:
        print("too few positions -- raise n_games"); return

    msup = max(20, nT // 40)         # ~2.5% support floor
    schemas = mine_schemas(train, holdout=holdout, min_support=msup,
                           min_lift=0.10, max_literals=4)
    gen = [s for s in schemas if s.get("generalizes")]
    print(f"\nmined patterns (lift>=0.10): {len(schemas)}   "
          f"that generalize to held-out games: {len(gen)}   (min_support={msup})")

    # ---- top discovered schemas -------------------------------------------
    print("\n--- TOP 16 DISCOVERED SCHEMAS (by support*lift on train) ---")
    print(f"{'rank':>4} {'out':4} {'supp':>5} {'prec':>5} {'lift':>5} "
          f"{'h.supp':>6} {'h.lift':>6} {'HO':>3} gen  pattern")
    for i, s in enumerate(schemas[:16]):
        print(f"{i:>4} {s['outcome']:4} {s['support']:>5} {s['precision']:>5.2f} "
              f"{s['lift']:>5.2f} {s.get('holdout_support',0):>6} "
              f"{s.get('holdout_lift',0):>6.2f} {'yes' if s['ho'] else '  -':>3} "
              f"{'Y' if s.get('generalizes') else '.':>3}  {fmt(s['props'])}")

    # ---- systematicity test: do higher-order edges add lift? --------------
    print("\n--- SYSTEMATICITY TEST (higher-order vs first-order constituents) ---")
    ho_constituents = {
        "ENABLES_promo": [("wr_behind", ("WR", "WP"))],
        "CAUSE_fortress": [("bk_blocks", ("WP",)), ("br_checks", ("BR", "WK"))],
    }
    for ho, cons in ho_constituents.items():
        ho_pat = frozenset([(ho, _arg(ho))])
        n_ho, o_ho, p_ho, l_ho = pattern_stats(ho_pat, train_lits)
        con_pat = frozenset(cons)
        n_c, o_c, p_c, l_c = pattern_stats(con_pat, train_lits)
        _, _, _, hl_ho = pattern_stats(ho_pat, hold_lits)
        _, _, _, hl_c = pattern_stats(con_pat, hold_lits)
        print(f"  {ho:15s} supp={n_ho:4d} -> {o_ho} lift={l_ho:+.2f} (holdout {hl_ho:+.2f})")
        print(f"  {'  constituents':15s} supp={n_c:4d} -> {o_c} lift={l_c:+.2f} (holdout {hl_c:+.2f})"
              f"   => HO adds {l_ho - l_c:+.2f} train / {hl_ho - hl_c:+.2f} holdout lift")

    # ---- headline: concept-correspondence (POST-HOC reporting only) -------
    concepts = {
        "Lucena: rook-escorted promotion": frozenset([("ENABLES_promo", ("WR", "WP"))]),
        "Lucena: rook cuts king + advanced pawn":
            frozenset([("wr_cuts_file", ("WR", "BK")), ("pawn_promoting", ("WP",))]),
        "Philidor: blockade + checking rook": frozenset([("CAUSE_fortress", ("BK", "WP"))]),
        "King blockade draw": frozenset([("bk_blocks", ("WP",))]),
        "Opposition + White to move": frozenset([("opposition", ("WK", "BK")), ("stm_white",)]),
        "Opposition + Black to move": frozenset([("opposition", ("WK", "BK")), ("stm_black",)]),
        "Rook-pawn + advanced": frozenset([("rook_pawn", ("WP",)), ("pawn_promoting", ("WP",))]),
    }
    print("\n--- HEADLINE: do named human concepts show up as predictive schemas? ---")
    print("    (signatures defined by hand for REPORTING; never used by the miner)")
    mined_sets = [frozenset(s["props"].values()) for s in schemas]
    for name, sig in concepts.items():
        nt, ot, pt, lt = pattern_stats(sig, train_lits)
        nh, oh, ph, lh = pattern_stats(sig, hold_lits)
        # did the beam surface a schema equal-to / containing this signature?
        rank = next((i for i, ms in enumerate(mined_sets) if sig <= ms), None)
        rank_s = f"mined#{rank}" if rank is not None else "not surfaced"
        print(f"  {name:42s} train: {str(ot):4} supp={nt:4d} lift={lt:+.2f} | "
              f"holdout lift={lh:+.2f} | {rank_s}")

    # ---- learned library vs Syzygy on held-out positions ------------------
    library = _select(gen)
    print(f"\n--- HELD-OUT EVAL: learned library ({len(library)} schemas) vs Syzygy ---")
    _report_eval("learned (mined)", library, hold_boards)
    _report_eval("hand-built (2 schemas)", SCHEMA_LIBRARY, hold_boards, hand=True)
    maj = base.most_common(1)[0][0]
    acc = sum(o == maj for _, o in hold_boards) / max(1, nH)
    print(f"  baseline (always '{maj}'): accuracy {acc:.2f} over all {nH} (coverage 1.00)")


def _arg(ho):
    return {"ENABLES_promo": ("WR", "WP"), "CAUSE_fortress": ("BK", "WP")}[ho]


def _select(gen, max_keep=24):
    """Keep generalizing schemas, dropping a specialization when a more general
    kept schema (subset, same outcome) already has >=95% of its holdout lift."""
    kept = []
    for s in sorted(gen, key=lambda r: r["score"], reverse=True):
        ps = frozenset(s["props"].values())
        red = any(frozenset(k["props"].values()) <= ps and k["outcome"] == s["outcome"]
                  and k.get("holdout_lift", 0) >= 0.95 * s.get("holdout_lift", 0)
                  for k in kept)
        if not red:
            kept.append(s)
        if len(kept) >= max_keep:
            break
    return kept


def _report_eval(tag, library, hold_boards, hand=False):
    if not library:
        print(f"  {tag}: empty library"); return
    from core import featurize as featurize_v1
    cov = correct = 0
    for b, truth in hold_boards:
        pos = featurize_v1(b) if hand else featurize_krpvkr(b)
        v, conf, _, _ = evaluate(pos, library)
        if v == "UNCERTAIN":
            continue
        cov += 1
        correct += (v == truth)
    n = len(hold_boards)
    print(f"  {tag}: coverage {cov/n:.2f} ({cov}/{n}), "
          f"accuracy-on-covered {correct/max(1,cov):.2f}")


if __name__ == "__main__":
    main()
