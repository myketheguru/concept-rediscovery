# artifact-version: 1.0.0
"""krpvkr_pilot2.py -- REVISED single-material rerun (design fix from krpvkr_pilot.py).

Pilot-1 found: selection divergence real, but the held-out accuracy gap was NULL because
a logistic COMBINER reassembles outcome signal from redundant compressive schemas. Fixes:
  (2) NON-COMBINING evaluator: single-best-matching-schema vote (no combiner to rescue
      compression) -- the place a true compression-vs-discrimination gap would surface;
  (3) small libraries B = 4, 8, 16 (redundancy can't compensate);
  (4) tempo-STRATIFIED sampling (oversample positions whose WDL flips with side-to-move),
      so the sharp/forcing structure isn't drowned by generic positions.
Contrast: report single-best (non-combining) vs logistic (combining) side by side.
Run (needs tablebases/syzygy): python krpvkr_pilot2.py
"""
import chess, chess.syzygy, random
from collections import Counter
from core import featurize_krpvkr, _literals
from krpvkr_pilot import (candidates, U_C, U_D, feats, train, acc, jaccard, show,
                          CLS, PIECES, TB_DIR, W, B, P)


def _wclass(bd, tb):
    try:
        wdl = tb.probe_wdl(bd)
    except Exception:
        return None
    wf = wdl if bd.turn == W else -wdl
    return (wf > 0) - (wf < 0)


def gen_stratified(n, tb, seed, keep_noncritical=0.30):
    """Random legal KRPvKR positions, OVERSAMPLING tempo-critical placements (WDL class
    flips with side-to-move). Returns (data, frac_critical). White-frame outcome."""
    rng = random.Random(seed)
    data, crit = [], 0
    while len(data) < n:
        sq = rng.sample(range(64), len(PIECES))
        bd = chess.Board.empty(); ok = True
        for (pt, c), s in zip(PIECES, sq):
            if pt == P and chess.square_rank(s) in (0, 7):
                ok = False; break
            bd.set_piece_at(s, chess.Piece(pt, c))
        if not ok:
            continue
        bw = bd.copy(); bw.turn = W; bb = bd.copy(); bb.turn = B
        is_crit = False
        if bw.is_valid() and bb.is_valid():
            cw, cb = _wclass(bw, tb), _wclass(bb, tb)
            if cw is not None and cb is not None:
                is_crit = (cw != cb)
        if not is_crit and rng.random() > keep_noncritical:
            continue
        bd.turn = W if rng.random() < 0.5 else B
        if not bd.is_valid():
            continue
        try:
            wdl = tb.probe_wdl(bd)
        except Exception:
            continue
        wf = wdl if bd.turn == W else -wdl
        outcome = "WIN" if wf > 0 else "LOSS" if wf < 0 else "DRAW"
        data.append((_literals(featurize_krpvkr(bd)), outcome))
        crit += is_crit
    return data, crit / len(data)


# ---------- the NON-COMBINING evaluator: single best-matching schema decides ----------
def schema_tags(lib, tr_lits, tr_labels):
    """Each schema -> (majority-outcome, train-precision, support) over the train set."""
    tags = []
    for S in lib:
        idx = [i for i, ls in enumerate(tr_lits) if S <= ls]
        if not idx:
            tags.append((None, 0.0, 0)); continue
        c = Counter(tr_labels[i] for i in idx)
        o, cnt = c.most_common(1)[0]
        tags.append((o, cnt / len(idx), len(idx)))
    return tags


def single_best_acc(lib, tags, ho_lits, ho_labels, fallback):
    """Predict each position by the SINGLE matching schema of highest train-precision
    (ties -> larger support). No combiner. Fallback = base-rate majority."""
    ok = 0
    for ls, truth in zip(ho_lits, ho_labels):
        best_o, best_key = None, (-1.0, -1)
        for S, (o, prec, sup) in zip(lib, tags):
            if o is not None and S <= ls and (prec, sup) > best_key:
                best_key = (prec, sup); best_o = o
        ok += ((best_o if best_o is not None else fallback) == truth)
    return ok / len(ho_labels)


def run(n_train=4500, n_hold=2000, min_support=90, seed=0):
    tb = chess.syzygy.open_tablebase(TB_DIR)
    tr, fc_tr = gen_stratified(n_train, tb, seed)
    ho, fc_ho = gen_stratified(n_hold, tb, seed + 999)
    labels = [o for _, o in tr]; N = len(tr); base = Counter(labels)
    maj = base.most_common(1)[0][0]; base_acc = base[maj] / N
    tr_lits = [ls for ls, _ in tr]; ho_lits = [ls for ls, _ in ho]
    ho_labels = [o for _, o in ho]
    print(f"=== KRPvKR REVISED pilot (tempo-stratified) ===")
    print(f"train {N} (frac tempo-critical {fc_tr:.2f}), holdout {len(ho)} ({fc_ho:.2f})")
    print(f"outcome dist {dict(base)}  base-rate acc {base_acc:.3f}")

    pool = candidates(tr, min_support)
    items = list(pool.items())
    print(f"label-blind candidate pool (size>=2, supp>={min_support}): {len(pool)}")
    sc_C = sorted(items, key=lambda kv: U_C(*kv), reverse=True)
    sc_D = sorted(items, key=lambda kv: U_D(kv[0], kv[1], labels, base, N), reverse=True)

    y_tr = [CLS[o] for o in labels]; y_ho = [CLS[o] for o in ho_labels]
    perm = labels[:]; random.Random(7).shuffle(perm); pbase = Counter(perm)
    sc_Dp = sorted(items, key=lambda kv: U_D(kv[0], kv[1], perm, pbase, N), reverse=True)

    print(f"\n{'B':>3s} {'Jacc':>5s} | {'NON-COMBINING single-best acc':^34s} | {'COMBINING logistic acc':^26s}")
    print(f"{'':>3s} {'':>5s} | {'L_C':>7s} {'L_D':>7s} {'gap':>7s} {'D_perm':>7s} | {'L_C':>7s} {'L_D':>7s} {'gap':>7s}")
    for Bn in (4, 8, 16, 32):
        L_C = [p for p, _ in sc_C[:Bn]]
        L_D = [p for p, _ in sc_D[:Bn]]
        L_Dp = [p for p, _ in sc_Dp[:Bn]]
        # non-combining
        sb_C = single_best_acc(L_C, schema_tags(L_C, tr_lits, labels), ho_lits, ho_labels, maj)
        sb_D = single_best_acc(L_D, schema_tags(L_D, tr_lits, labels), ho_lits, ho_labels, maj)
        sb_Dp = single_best_acc(L_Dp, schema_tags(L_Dp, tr_lits, labels), ho_lits, ho_labels, maj)
        # combining (logistic) -- only at B=8,32 to bound runtime
        if Bn in (8, 32):
            FtrC, dimC = feats(tr_lits, L_C); FhoC, _ = feats(ho_lits, L_C)
            FtrD, dimD = feats(tr_lits, L_D); FhoD, _ = feats(ho_lits, L_D)
            lo_C = acc(train(FtrC, y_tr, dimC), FhoC, y_ho)
            lo_D = acc(train(FtrD, y_tr, dimD), FhoD, y_ho)
            lo = f"{lo_C:7.3f} {lo_D:7.3f} {lo_D-lo_C:+7.3f}"
        else:
            lo = f"{'-':>7s} {'-':>7s} {'-':>7s}"
        print(f"{Bn:>3d} {jaccard(L_C,L_D):5.2f} | {sb_C:7.3f} {sb_D:7.3f} {sb_D-sb_C:+7.3f} {sb_Dp:7.3f} | {lo}")

    print("\nselection content @B=16 (tempo-coupling):")
    for nm, sc in (("L_C compression", sc_C), ("L_D discrimination", sc_D)):
        lib = [p for p, _ in sc[:16]]
        stm = sum(any(pr.startswith("stm_") for pr, _ in p) for p in lib)
        ho_ = sum(any(pr.startswith(("ENABLES", "CAUSE")) for pr, _ in p) for p in lib)
        print(f"  {nm:20s}: {stm:2d}/16 contain a side-to-move literal, {ho_} higher-order")
    print("\ntop discriminative schemas MISSED by compression (high U_D, not in L_C@16):")
    setC = set(p for p, _ in sc_C[:16]); k = 0
    for p, s in sc_D:
        if p not in setC:
            print(f"  MI={U_D(p,s,labels,base,N):.3f} supp={len(s):4d}  {show(p)}")
            k += 1
            if k >= 5: break
    print("\nREAD: if single-best gap (L_D-L_C) is large & positive at small B while the logistic")
    print("gap stays ~0, the divergence is REAL but redundancy-masked (combiner washes it).")
    print("If single-best gap is also ~0, the null is robust to the evaluator -> genuine NULL.")


if __name__ == "__main__":
    run()
