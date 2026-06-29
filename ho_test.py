# artifact-version: 1.0.0
"""
ho_test.py  --  can higher-order edges EARN THEIR KEEP?

Prior runs found the higher-order (HO) edges weak. But those HO predicates were
straw men: ENABLES_promo/CAUSE_fortress are literally conjunctions of first-order
(FO) predicates already in the vocabulary, so the miner can rebuild them from
their parts -- they cannot add lift by construction.

This test gives the HO hypothesis its best shot:
  * adds GENUINE higher-order predicates not expressible as FO conjunctions:
      - interpose(WR,BR,WK): a relation-of-relations -- WR can block the rank/
        file check that BR aims at WK (the Lucena bridge).
      - square_of_pawn(BK,WP): the rule of the square (lone king controls the
        pawn run) -- a computed geometric fact, not a conjunction of rank/file/
        adjacency primitives.
  * sharp hypothesis (thesis): HO adds no CAPABILITY (all of it is a function of
    the 5 squares) but buys TRACTABILITY -- it compresses an outcome-relevant
    configuration into one literal, reachable under a BOUNDED search. So HO
    should help most at SHALLOW depth and the gap should close as depth grows.
    If FO-only == FO+HO at every depth, systematicity/W_HO buys nothing here.

Method: FO-only vs FO+HO, swept over search depth (max_literals), with a
label-permutation null floor; plus a "room" ceiling (does FO even determine the
outcome?) and an ambiguous-subset test (does HO help exactly where FO fails?).
Reads data/krpvkr_cache.csv. Run: python ho_test.py [n_shuffles]
"""
import sys, csv, random, statistics, collections
import chess
from core import featurize_krpvkr, mine_schemas, evaluate, _literals, HO_PREDS
from mine_run import _select

CACHE = "data/krpvkr_cache.csv"
BASE_HO = set(HO_PREDS)                       # ENABLES_promo, CAUSE_fortress (in base featurizer)
EXTRA_HO = {"interpose", "square_of_pawn"}    # genuine HO, added here
ALL_HO = BASE_HO | EXTRA_HO
F, R = chess.square_file, chess.square_rank


def extra_ho(board):
    g = lambda pt, c: next(iter(board.pieces(pt, c)))
    wk, wr, wp = g(chess.KING, 1), g(chess.ROOK, 1), g(chess.PAWN, 1)
    bk, br = g(chess.KING, 0), g(chess.ROOK, 0)
    btw = lambda a, b, c: min(a, c) < b < max(a, c)
    out = []
    if (R(br) == R(wk) and btw(F(br), F(wr), F(wk))) or \
       (F(br) == F(wk) and btw(R(br), R(wr), R(wk))):
        out.append(("interpose", ("WR", "BR", "WK")))
    promo = chess.square(F(wp), 7)
    slack = 1 if board.turn == chess.BLACK else 0
    if chess.square_distance(bk, promo) <= (7 - R(wp)) + slack:
        out.append(("square_of_pawn", ("BK", "WP")))
    return out


def featurize_sel(board, ho_include):
    """Rich graph with HO predicates restricted to `ho_include` (a set)."""
    g = featurize_krpvkr(board)
    props = {k: v for k, v in g["props"].items()
             if v[0] not in BASE_HO or v[0] in ho_include}
    i = 0
    for pred, args in extra_ho(board):
        if pred in ho_include:
            props[f"x{i}"] = (pred, args); i += 1
    return {"entities": g["entities"], "props": props}


def load_cache():
    train, hold = [], []
    for r in csv.DictReader(open(CACHE, newline="")):
        b = chess.Board(r["fen"])
        (hold if r["holdout"] == "1" else train).append((b, r["outcome"]))
    return train, hold


def lib_accuracy(lib, hold_lg, maj):
    cov = cor = full = 0
    for g, truth in hold_lg:
        v = evaluate(g, lib)[0] if lib else "UNCERTAIN"
        pred = maj if v == "UNCERTAIN" else v
        if v != "UNCERTAIN":
            cov += 1; cor += (v == truth)
        full += (pred == truth)
    n = len(hold_lg)
    return full / n, cov / n


def run(train_lg, hold_lg, msup, depth, maj):
    sch = mine_schemas(train_lg, holdout=hold_lg, min_support=msup, min_lift=0.10, max_literals=depth)
    gen = [s for s in sch if s.get("generalizes")]
    lib = _select(gen)
    acc, cov = lib_accuracy(lib, hold_lg, maj)
    best_hl = max((s.get("holdout_lift", 0) for s in sch), default=0.0)
    return {"acc": acc, "cov": cov, "n_gen": len(gen), "best_hl": best_hl, "lib": lib}


def null_acc(train_lg, hold_lg, msup, depth, maj, shuffles):
    rng = random.Random(0)
    tr_o = [o for _, o in train_lg]; ho_o = [o for _, o in hold_lg]
    accs = []
    for _ in range(shuffles):
        a = tr_o[:]; rng.shuffle(a); b = ho_o[:]; rng.shuffle(b)
        tr = [(g, a[i]) for i, (g, _) in enumerate(train_lg)]
        ho = [(g, b[i]) for i, (g, _) in enumerate(hold_lg)]
        accs.append(run(tr, ho, msup, depth, maj)["acc"])
    return statistics.mean(accs), statistics.pstdev(accs)


def impurity(boards, ho_include):
    """In-sample ceiling: group all positions by FO(+selected HO) signature;
    report % of positions in mixed-outcome groups and the optimistic error if you
    predict each group's majority. ~0 impurity => signature determines outcome."""
    groups = collections.defaultdict(list)
    for b, o in boards:
        groups[frozenset(featurize_sel(b, ho_include)["props"].values())].append(o)
    n = len(boards); inmixed = err = 0
    for outs in groups.values():
        c = collections.Counter(outs)
        if len(c) > 1:
            inmixed += len(outs)
        err += len(outs) - c.most_common(1)[0][1]
    return inmixed / n, err / n


def main():
    shuffles = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    train_b, hold_b = load_cache()
    nT, nH = len(train_b), len(hold_b)
    maj = collections.Counter(o for _, o in train_b).most_common(1)[0][0]
    maj_acc = sum(o == maj for _, o in hold_b) / nH
    msup = max(20, nT // 40)
    pooled = train_b + hold_b
    feats = {"FO": set(), "FO+HO": ALL_HO}
    print(f"=== HIGHER-ORDER EARN-THEIR-KEEP TEST  (train {nT} / holdout {nH}) ===")
    print(f"majority baseline: {maj_acc:.3f}\n")

    # 1) ROOM CEILING: does the FO signature even determine the outcome?
    print("--- room ceiling: outcome impurity given the full signature (in-sample) ---")
    for tag, inc in feats.items():
        mixed, err = impurity(pooled, inc)
        print(f"  {tag:6s}: {mixed*100:5.1f}% of positions in mixed-outcome groups; "
              f"optimistic group-majority error {err*100:4.1f}%")
    print("  (if FO impurity ~ 0, FO already determines outcome -> no room for HO)")

    # 2) DEPTH SWEEP: FO vs FO+HO held-out accuracy across search depth
    print("\n--- depth sweep: held-out accuracy (null floor at depth 4) ---")
    print(f"  {'depth':>5} {'FO acc':>7} {'FO+HO acc':>9} {'delta':>6} {'FO ngen':>8} {'HO ngen':>8}")
    cache_lib = {}
    for d in (2, 3, 4, 5):
        rows = {}
        for tag, inc in feats.items():
            tr = [(featurize_sel(b, inc), o) for b, o in train_b]
            ho = [(featurize_sel(b, inc), o) for b, o in hold_b]
            rows[tag] = run(tr, ho, msup, d, maj)
            cache_lib[(tag, d)] = (tr, ho, rows[tag])
        delta = rows["FO+HO"]["acc"] - rows["FO"]["acc"]
        print(f"  {d:>5} {rows['FO']['acc']:>7.3f} {rows['FO+HO']['acc']:>9.3f} {delta:>+6.3f} "
              f"{rows['FO']['n_gen']:>8} {rows['FO+HO']['n_gen']:>8}")
    # null floor at depth 4
    for tag, inc in feats.items():
        tr, ho, _ = cache_lib[(tag, 4)]
        m, s = null_acc(tr, ho, msup, 4, maj, shuffles)
        print(f"  null({tag}, depth4): {m:.3f}±{s:.3f}")

    # 3) PER-HO ABLATION at depth 4: which single HO predicate (if any) helps?
    print("\n--- per-HO ablation at depth 4 (FO + one HO predicate) ---")
    base_tr = [(featurize_sel(b, set()), o) for b, o in train_b]
    base_ho = [(featurize_sel(b, set()), o) for b, o in hold_b]
    fo_acc = run(base_tr, base_ho, msup, 4, maj)["acc"]
    print(f"  FO-only: {fo_acc:.3f}")
    for h in sorted(ALL_HO):
        tr = [(featurize_sel(b, {h}), o) for b, o in train_b]
        ho = [(featurize_sel(b, {h}), o) for b, o in hold_b]
        a = run(tr, ho, msup, 4, maj)["acc"]
        print(f"  FO+{h:16s}: {a:.3f}   ({a-fo_acc:+.3f})")

    # 4) AMBIGUOUS SUBSET: does HO help exactly where FO fails?
    print("\n--- where FO fails: does FO+HO recover it? (held-out subset) ---")
    tr_fo, ho_fo, r_fo = cache_lib[("FO", 4)]
    tr_h, ho_h, r_h = cache_lib[("FO+HO", 4)]
    fo_lib, ho_lib = r_fo["lib"], r_h["lib"]
    hard = [i for i, (g, o) in enumerate(ho_fo)
            if (evaluate(g, fo_lib)[0] if fo_lib else "UNCERTAIN") != o]
    if hard:
        fo_right = sum((evaluate(ho_fo[i][0], fo_lib)[0] if fo_lib else "UNCERTAIN") == ho_fo[i][1] for i in hard)
        ho_right = sum((evaluate(ho_h[i][0], ho_lib)[0] if ho_lib else "UNCERTAIN") == ho_h[i][1] for i in hard)
        print(f"  FO wrong/uncertain on {len(hard)}/{nH} held-out positions")
        print(f"  of those, FO+HO now correct: {ho_right}/{len(hard)}  "
              f"(FO itself: {fo_right}/{len(hard)})")
    print("\nVERDICT GUIDE: HO earns its keep iff FO+HO beats FO on held-out ABOVE the")
    print("null band, and most at shallow depth (tractability) -- else FO is sufficient.")


if __name__ == "__main__":
    main()
