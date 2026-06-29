# artifact-version: 1.0.0
"""
transfer_test.py  --  CROSS-MATERIAL TRANSFER: do schemas mined on KRPvKR predict
outcomes on KQPvKQ (rook -> queen)?

This is the analogy/morphism test. We use a PIECE-AGNOSTIC vocabulary: the white
and black long-range pieces are roles WL / BL (rook in KRPvKR, queen in KQPvKQ),
and the shared predicates are computed rook-like (the moves R and Q have in
common) + the pawn/king predicates + square_of_pawn (the first-order primitive
that earned its keep). The analogy mapping R<->Q is thus carried by the agnostic
vocabulary (a prior choice -- "the vocabulary is the inductive bias"); the TEST
is whether the relational SCHEMAS learned on rooks transfer to queens.

What falsifies / confirms:
  * transfer library beats KQPvKQ baseline + permutation null  -> structure transfers.
  * transfer ~ native KQPvKQ-mined library                     -> strong transfer.
  * transfer ~ baseline                                        -> structure was rook-specific.

KQPvKQ positions are random legal (exact Syzygy labels); real games have too few.
Run: python transfer_test.py [n_kqpvkq] [n_shuffles]
"""
import sys, csv, random, statistics, collections
import chess, chess.syzygy
from core import mine_schemas, evaluate, _literals
from mine_run import _select, white_frame, pattern_stats
from core import label as syzygy_label

TB = "tablebases/syzygy"
KRPVKR_CACHE = "data/krpvkr_cache.csv"
F, R = chess.square_file, chess.square_rank


def featurize_endgame(board):
    """Piece-agnostic pawn-endgame vocabulary (KRPvKR or KQPvKQ). WL/BL = white/
    black long-range piece. Shared predicates are rook-like (R&Q common moves)."""
    one = lambda pt, c: (next(iter(board.pieces(pt, c))) if board.pieces(pt, c) else None)
    longp = lambda c: one(chess.ROOK, c) if board.pieces(chess.ROOK, c) else one(chess.QUEEN, c)
    wk, bk, wp = one(chess.KING, 1), one(chess.KING, 0), one(chess.PAWN, 1)
    wl, bl = longp(1), longp(0)            # NB: square 0 (a1) is falsy -- don't use `or`
    props, n = {}, [0]
    def add(p, *a): props[f"p{n[0]}"] = (p, tuple(a)); n[0] += 1

    pf, pr = F(wp), R(wp)
    promo = chess.square(pf, 7)
    add("stm_white") if board.turn == chess.WHITE else add("stm_black")
    add("pawn_promoting", "WP") if pr >= 6 else \
        add("pawn_advanced", "WP") if pr >= 4 else add("pawn_back", "WP")
    if pf in (0, 7): add("rook_pawn", "WP")
    if R(wk) > pr and abs(F(wk) - pf) <= 1:           add("wk_ahead", "WP")
    if abs(F(wk) - pf) <= 1 and abs(R(wk) - pr) <= 1: add("wk_supports", "WP")
    if chess.square_distance(wk, promo) <= 1:         add("wk_on_promo", "WP")
    if R(bk) > pr and abs(F(bk) - pf) <= 1:           add("bk_ahead", "WP")
    if F(bk) == pf and R(bk) >= pr + 1:               add("bk_blocks", "WP")
    if chess.square_distance(bk, promo) <= 1:         add("bk_on_promo", "WP")
    df, dr = abs(F(wk) - F(bk)), abs(R(wk) - R(bk))
    if (df == 0 and dr == 2) or (dr == 0 and df == 2): add("opposition", "WK", "BK")
    if chess.square_distance(wk, bk) == 1:             add("kings_close", "WK", "BK")
    if F(wl) == pf and R(wl) < pr: add("wl_behind", "WL", "WP")
    if F(wl) == pf and R(wl) > pr: add("wl_front", "WL", "WP")
    if F(wl) != F(bk) and (F(bk) - F(wl)) * (pf - F(wl)) < 0: add("wl_cuts_file", "WL", "BK")
    if R(wl) != R(bk) and (R(bk) - R(wl)) * (pr - R(wl)) < 0: add("wl_cuts_rank", "WL", "BK")
    if F(bl) == pf and R(bl) < pr: add("bl_behind", "BL", "WP")
    if F(bl) == pf and R(bl) > pr: add("bl_front", "BL", "WP")
    if (F(bl) == F(wk) or R(bl) == R(wk)) and chess.square_distance(bl, wk) >= 3:
        add("bl_checks", "BL", "WK")
    slack = 1 if board.turn == chess.BLACK else 0
    if chess.square_distance(bk, promo) <= (7 - pr) + slack:
        add("square_of_pawn", "BK", "WP")
    return {"entities": ["WK", "WL", "WP", "BK", "BL"], "props": props}


def krpvkr_dataset():
    tr, ho = [], []
    for r in csv.DictReader(open(KRPVKR_CACHE, newline="")):
        b = chess.Board(r["fen"])
        (ho if r["holdout"] == "1" else tr).append((featurize_endgame(b), r["outcome"]))
    return tr, ho


def kqpvkq_dataset(n, seed):
    tb = chess.syzygy.open_tablebase(TB)
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < n:
        wk, bk, wq, bq, wp = rng.sample(chess.SQUARES, 5)
        if R(wp) in (0, 7):
            continue
        b = chess.Board(None)
        for sq, pc in [(wk, (6, 1)), (bk, (6, 0)), (wq, (5, 1)), (bq, (5, 0)), (wp, (1, 1))]:
            b.set_piece_at(sq, chess.Piece(*pc))
        b.turn = rng.choice([chess.WHITE, chess.BLACK])
        if not b.is_valid():
            continue
        key = b._transposition_key() + (b.turn,)
        if key in seen:
            continue
        seen.add(key)
        y = syzygy_label(b, tb)
        if y is None:
            continue
        out.append((featurize_endgame(b), white_frame(y, b.turn)))
    tb.close()
    return out


def lib_accuracy(lib, data, maj):
    cov = cor = full = 0
    for g, truth in data:
        v = evaluate(g, lib)[0] if lib else "UNCERTAIN"
        pred = maj if v == "UNCERTAIN" else v
        if v != "UNCERTAIN":
            cov += 1; cor += (v == truth)
        full += (pred == truth)
    n = len(data)
    return full / n, cov / n


def mine_lib(train, hold, msup, depth=4):
    sch = mine_schemas(train, holdout=hold, min_support=msup, min_lift=0.10, max_literals=depth)
    return _select([s for s in sch if s.get("generalizes")]), sch


def null_floor(train, hold, msup, maj, shuffles):
    rng = random.Random(0)
    tr_o = [o for _, o in train]; ho_o = [o for _, o in hold]
    accs = []
    for _ in range(shuffles):
        a = tr_o[:]; rng.shuffle(a); b = ho_o[:]; rng.shuffle(b)
        lib, _ = mine_lib([(g, a[i]) for i, (g, _) in enumerate(train)],
                          [(g, b[i]) for i, (g, _) in enumerate(hold)], msup)
        accs.append(lib_accuracy(lib, [(g, b[i]) for i, (g, _) in enumerate(hold)], maj)[0])
    return statistics.mean(accs), statistics.pstdev(accs)


def main():
    n_kq = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    shuffles = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    kr_tr, kr_ho = krpvkr_dataset()
    msup_kr = max(20, len(kr_tr) // 40)
    transfer_lib, _ = mine_lib(kr_tr, kr_ho, msup_kr)
    # sanity: transfer lib should still work on its OWN material
    kr_maj = collections.Counter(o for _, o in kr_tr).most_common(1)[0][0]
    kr_acc, _ = lib_accuracy(transfer_lib, kr_ho, kr_maj)

    kq = kqpvkq_dataset(n_kq, seed=7)
    split = int(len(kq) * 0.6)
    kq_tr, kq_ho = kq[:split], kq[split:]
    kq_base = collections.Counter(o for _, o in kq_tr)
    kq_maj = kq_base.most_common(1)[0][0]
    kq_maj_acc = sum(o == kq_maj for _, o in kq_ho) / len(kq_ho)
    msup_kq = max(20, len(kq_tr) // 40)

    native_lib, native_all = mine_lib(kq_tr, kq_ho, msup_kq)

    transfer_acc, transfer_cov = lib_accuracy(transfer_lib, kq_ho, kq_maj)
    native_acc, native_cov = lib_accuracy(native_lib, kq_ho, kq_maj)
    nfl = null_floor(kq_tr, kq_ho, msup_kq, kq_maj, shuffles)

    print(f"=== CROSS-MATERIAL TRANSFER  KRPvKR -> KQPvKQ ===")
    print(f"KRPvKR train {len(kr_tr)} / holdout {len(kr_ho)};  "
          f"KQPvKQ {len(kq)} random legal (train {len(kq_tr)} / holdout {len(kq_ho)})")
    print(f"KQPvKQ base rates: { {o: round(kq_base[o]/len(kq_tr),3) for o in kq_base} }\n")
    print(f"sanity -- transfer library on its OWN material (KRPvKR holdout): acc {kr_acc:.3f} "
          f"(lib={len(transfer_lib)})\n")
    print(f"--- KQPvKQ held-out accuracy ---")
    print(f"  majority baseline ('{kq_maj}')         : {kq_maj_acc:.3f}")
    print(f"  permutation null (native mining)       : {nfl[0]:.3f}±{nfl[1]:.3f}")
    print(f"  TRANSFER (KRPvKR-mined lib, {len(transfer_lib):2d} sch) : {transfer_acc:.3f}  (coverage {transfer_cov:.2f})")
    print(f"  native KQPvKQ-mined lib  ({len(native_lib):2d} sch)   : {native_acc:.3f}  (coverage {native_cov:.2f})  [ceiling]")
    gap = transfer_acc - max(kq_maj_acc, nfl[0])
    print(f"\n  => transfer vs baseline/null: {gap:+.3f}  "
          f"({'TRANSFERS' if gap > 3*(nfl[1] or .01) and transfer_acc>kq_maj_acc else 'does NOT transfer'})")

    # per-schema lift retention: do the top KRPvKR schemas keep their lift on KQPvKQ?
    print("\n--- per-schema lift retention (top transfer schemas: KRPvKR vs KQPvKQ holdout) ---")
    kr_lits = [(_literals(g), o) for g, o in kr_ho]
    kq_lits = [(_literals(g), o) for g, o in kq_ho]
    for s in transfer_lib[:10]:
        sig = frozenset(s["props"].values())
        _, okr, _, lkr = pattern_stats(sig, kr_lits)
        nkq, okq, _, lkq = pattern_stats(sig, kq_lits)
        pat = " ".join(f"{p}({','.join(a)})" if a else p for p, a in sorted(sig))
        print(f"  KR:{str(okr):4} lift{lkr:+.2f} | KQ:{str(okq):4} lift{lkq:+.2f} supp{nkq:4d} | {pat[:70]}")


if __name__ == "__main__":
    main()
