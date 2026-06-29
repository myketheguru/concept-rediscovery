# artifact-version: 1.0.0
"""
separation_test.py  --  is the relational/morphism level NECESSARY, or merely
EXPRESSIBLE? (the gap the report was missing.)

Test #4 showed first-order suffices for SCORING within a domain, so an in-domain
separation curve would likely be flat. The place separation can appear is
TRANSFER: a flat feature model trained on rooks may not generalize to queens at
all, while the relational library transfers via the role morphism. So we measure
TRANSFER RETENTION (KRPvKR -> KQPvKQ) for three models matched on in-domain
accuracy:

  (1) FLAT / piece-agnostic : logistic regression on the bag of first-order
      predicate features (the role-abstracted vocabulary, NO conjunctions).
  (2) FLAT / piece-specific : logistic regression on per-piece one-hot file/rank
      (rook and queen occupy DIFFERENT feature slots -> no role abstraction).
  (3) RELATIONAL            : the mined schema library + alignment morphism.

Decomposition:
  * (1) vs (3) : does relational structure add transfer GIVEN the abstraction?
                 (necessary vs merely expressible, holding features fixed)
  * (2) vs (3) : does relational+morphism retain where piece-specific-flat collapses?
                 (is the transfer in the abstraction/morphism?)

Run: python separation_test.py [n_kqpvkq]
"""
import sys, csv, math, random, collections
import chess
from transfer_test import (featurize_endgame, mine_lib, lib_accuracy)
from nsv_transfer import kqpvkq_boards

KR_CACHE = "data/krpvkr_cache.csv"
R, F = chess.square_rank, chess.square_file


def kr_boards():
    tr, ho = [], []
    for r in csv.DictReader(open(KR_CACHE, newline="")):
        b = chess.Board(r["fen"])
        (ho if r["holdout"] == "1" else tr).append((b, r["outcome"]))
    return tr, ho


# ---- feature extractors -------------------------------------------------------
def feats_agnostic(board):
    """Bag of first-order predicate names (role-abstracted; no conjunctions)."""
    return [p for p, _ in featurize_endgame(board)["props"].values()]


def feats_specific(board):
    """Per-piece one-hot file/rank. Rook and queen go in DIFFERENT slots, so a
    rook-trained model has no weights for queen slots (no role abstraction)."""
    out = ["stm_w" if board.turn else "stm_b"]
    def slot(name, sq):
        out.append(f"{name}_f{F(sq)}"); out.append(f"{name}_r{R(sq)}")
    one = lambda pt, c: (next(iter(board.pieces(pt, c))) if board.pieces(pt, c) else None)
    slot("WK", one(chess.KING, 1)); slot("BK", one(chess.KING, 0)); slot("WP", one(chess.PAWN, 1))
    for pt, tag in [(chess.ROOK, "R"), (chess.QUEEN, "Q")]:
        w, b = one(pt, 1), one(pt, 0)
        if w is not None: slot(f"WL{tag}", w)
        if b is not None: slot(f"BL{tag}", b)
    return out


# ---- sparse multinomial logistic regression (pure python) ---------------------
def encode(rows, fmap, fit):
    X = []
    for feats in rows:
        idx = []
        for f in feats:
            if f not in fmap:
                if fit:
                    fmap[f] = len(fmap)
                else:
                    continue
            idx.append(fmap[f])
        X.append(idx)
    return X


def train_lr(X, Y, n_feat, K, iters=250, lr=0.5, l2=1e-4):
    W = [[0.0] * n_feat for _ in range(K)]
    b = [0.0] * K
    N = len(X)
    for _ in range(iters):
        gW = [[0.0] * n_feat for _ in range(K)]
        gb = [0.0] * K
        for x, y in zip(X, Y):
            lo = [b[c] + sum(W[c][i] for i in x) for c in range(K)]
            m = max(lo); ex = [math.exp(v - m) for v in lo]; s = sum(ex)
            for c in range(K):
                d = ex[c] / s - (1.0 if c == y else 0.0)
                gb[c] += d
                for i in x:
                    gW[c][i] += d
        for c in range(K):
            b[c] -= lr * gb[c] / N
            Wc, gWc = W[c], gW[c]
            for i in range(n_feat):
                Wc[i] -= lr * (gWc[i] / N + l2 * Wc[i])
    return W, b


def lr_accuracy(W, b, X, Y, K):
    cor = 0
    for x, y in zip(X, Y):
        lo = [b[c] + sum(W[c][i] for i in x) for c in range(K)]
        if max(range(K), key=lambda c: lo[c]) == y:
            cor += 1
    return cor / len(X)


def run_flat(extractor, kr_tr, kr_ho, kq_ho, classes):
    cidx = {c: i for i, c in enumerate(classes)}
    fmap = {}
    Xtr = encode([extractor(b) for b, _ in kr_tr], fmap, fit=True)
    Ytr = [cidx[o] for _, o in kr_tr]
    Xkr = encode([extractor(b) for b, _ in kr_ho], fmap, fit=False)
    Ykr = [cidx[o] for _, o in kr_ho]
    Xkq = encode([extractor(b) for b, _ in kq_ho], fmap, fit=False)
    Ykq = [cidx[o] for _, o in kq_ho]
    W, b = train_lr(Xtr, Ytr, len(fmap), len(classes))
    return lr_accuracy(W, b, Xkr, Ykr, len(classes)), lr_accuracy(W, b, Xkq, Ykq, len(classes))


def main():
    n_kq = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    classes = ["WIN", "DRAW", "LOSS"]
    kr_tr, kr_ho = kr_boards()
    kq = kqpvkq_boards(n_kq, seed=21)
    kr_maj = collections.Counter(o for _, o in kr_tr).most_common(1)[0][0]
    kq_maj = collections.Counter(o for _, o in kq).most_common(1)[0][0]
    kr_base = sum(o == kr_maj for _, o in kr_ho) / len(kr_ho)
    kq_base = sum(o == kq_maj for _, o in kq) / len(kq)

    # relational library (mined on KR), evaluated in-domain and on transfer
    kr_tr_g = [(featurize_endgame(b), o) for b, o in kr_tr]
    kr_ho_g = [(featurize_endgame(b), o) for b, o in kr_ho]
    kq_g = [(featurize_endgame(b), o) for b, o in kq]
    rel_lib, _ = mine_lib(kr_tr_g, kr_ho_g, max(20, len(kr_tr) // 40))
    rel_kr = lib_accuracy(rel_lib, kr_ho_g, kr_maj)[0]
    rel_kq = lib_accuracy(rel_lib, kq_g, kq_maj)[0]

    fa_kr, fa_kq = run_flat(feats_agnostic, kr_tr, kr_ho, kq, classes)
    fs_kr, fs_kq = run_flat(feats_specific, kr_tr, kr_ho, kq, classes)

    def ret(indomain, transfer, ib, tb):
        denom = indomain - ib
        return (transfer - tb) / denom if denom > 1e-6 else 0.0

    print(f"=== SEPARATION: relational vs flat, IN-DOMAIN vs TRANSFER ===")
    print(f"KRPvKR holdout {len(kr_ho)} (base {kr_base:.3f});  "
          f"KQPvKQ {len(kq)} (base {kq_base:.3f})\n")
    print(f"{'model':32s} {'KR in-domain':>12s} {'KQ transfer':>12s} {'retention':>10s}")
    rows = [("FLAT / piece-agnostic (1st-order)", fa_kr, fa_kq),
            ("FLAT / piece-specific (coords)", fs_kr, fs_kq),
            ("RELATIONAL (schemas + morphism)", rel_kr, rel_kq)]
    for name, kr, kq_ in rows:
        print(f"{name:32s} {kr:>12.3f} {kq_:>12.3f} {ret(kr,kq_,kr_base,kq_base):>10.2f}")
    print(f"{'(baseline)':32s} {kr_base:>12.3f} {kq_base:>12.3f}")
    print("\nReading:")
    print(" (1)vs(3): if agnostic-flat transfers ~ relational, conjunctions/morphism are")
    print("          MERELY EXPRESSIBLE for prediction -- the abstraction carries transfer.")
    print(" (2)vs(3): if piece-specific-flat collapses to baseline while relational retains,")
    print("          the transferable structure lives in the role abstraction / morphism.")


if __name__ == "__main__":
    main()
