# artifact-version: 1.0.0
"""krpvkr_pilot.py -- single-material pilot for STUDY_compression_vs_discrimination.md.

Question: holding the vocabulary fixed and mining ONE label-blind candidate pool,
do COMPRESSION-optimal schemas (MDL, label-blind) and DISCRIMINATION-optimal schemas
(mutual information with exact Syzygy WDL) select the same library? Measure divergence
(Jaccard, Kendall-tau) + a held-out 3-class accuracy gap, with a permutation-null.

Vocabulary + literalization reused from core (the fixed PRIOR).
Pure Python (no numpy/sklearn). Run (needs tablebases/syzygy): python krpvkr_pilot.py
"""
import chess, chess.syzygy, random, math
from collections import Counter, defaultdict
from core import featurize_krpvkr, _literals

TB_DIR = "tablebases/syzygy"
W, B, K, Q, R, P = chess.WHITE, chess.BLACK, chess.KING, chess.QUEEN, chess.ROOK, chess.PAWN
PIECES = [(K, W), (R, W), (P, W), (K, B), (R, B)]
CLS = {"WIN": 0, "DRAW": 1, "LOSS": 2}


# ---------- data: random legal KRPvKR positions + exact WDL (White's frame) ----------
def gen(n, tb, seed):
    rng = random.Random(seed)
    data = []
    while len(data) < n:
        sq = rng.sample(range(64), len(PIECES))
        bd = chess.Board.empty(); ok = True
        for (pt, c), s in zip(PIECES, sq):
            if pt == P and chess.square_rank(s) in (0, 7):
                ok = False; break
            bd.set_piece_at(s, chess.Piece(pt, c))
        if not ok:
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
    return data


# ---------- label-blind frequent-itemset candidate pool (mined by SUPPORT only) ----------
def candidates(data, min_support, max_lits=4, beam=900):
    N = len(data)
    lit_sup = defaultdict(set)
    for i, (lits, _) in enumerate(data):
        for l in lits:
            lit_sup[l].add(i)
    freq = [l for l, s in lit_sup.items() if len(s) >= min_support]
    supp = {frozenset([l]): lit_sup[l] for l in freq}
    beamset = list(supp)
    for _ in range(2, max_lits + 1):
        grown = []
        for p in beamset:
            ps = supp[p]
            for l in freq:
                if l in p:
                    continue
                np = p | {l}
                if np in supp:
                    continue
                s = ps & lit_sup[l]
                if len(s) < min_support:
                    continue
                supp[np] = s
                grown.append((len(s), np))           # rank by SUPPORT only (label-blind)
        grown.sort(key=lambda x: x[0], reverse=True)
        beamset = [np for _, np in grown[:beam]]
        if not beamset:
            break
    return {p: s for p, s in supp.items() if len(p) >= 2}   # size>=2 (size-1 never compresses)


# ---------- the two objectives ----------
def U_C(p, s):                                            # MDL / compression gain, LABEL-BLIND
    return len(s) * (len(p) - 1) - len(p)

def U_D(p, s, labels, base, N):                            # mutual information with WDL (bits)
    sup = len(s)
    if sup == 0 or sup == N:
        return 0.0
    pres = Counter(labels[i] for i in s)
    mi = 0.0
    for present, cnts in ((1, pres), (0, {o: base[o] - pres.get(o, 0) for o in base})):
        pf = (sup if present else N - sup) / N
        for o in base:
            j = cnts.get(o, 0) / N
            if j > 0:
                mi += j * math.log2(j / (pf * base[o] / N))
    return mi


# ---------- pure-Python multinomial logistic (sparse binary features) ----------
def feats(litsets, lib):
    out = []
    for ls in litsets:
        act = [k for k, Sset in enumerate(lib) if Sset <= ls]
        act.append(len(lib))                              # bias term
        out.append(act)
    return out, len(lib) + 1

def train(F, y, dim, Kc=3, iters=250, lr=0.5, l2=1e-3):
    Wt = [[0.0] * dim for _ in range(Kc)]; n = len(F)
    for _ in range(iters):
        g = [[0.0] * dim for _ in range(Kc)]
        for act, yi in zip(F, y):
            z = [sum(Wt[c][j] for j in act) for c in range(Kc)]
            m = max(z); ez = [math.exp(v - m) for v in z]; ssum = sum(ez)
            for c in range(Kc):
                gc = ez[c] / ssum - (1.0 if c == yi else 0.0)
                if gc:
                    row = g[c]
                    for j in act:
                        row[j] += gc
        for c in range(Kc):
            for j in range(dim):
                Wt[c][j] -= lr * (g[c][j] / n + l2 * Wt[c][j])
    return Wt

def acc(Wt, F, y, Kc=3):
    ok = 0
    for act, yi in zip(F, y):
        z = [sum(Wt[c][j] for j in act) for c in range(Kc)]
        ok += (z.index(max(z)) == yi)
    return ok / len(y)


def jaccard(a, b):
    A, Bs = set(a), set(b)
    return len(A & Bs) / len(A | Bs) if (A | Bs) else 1.0

def kendall(pairs):
    c = d = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            s = (pairs[i][0] - pairs[j][0]) * (pairs[i][1] - pairs[j][1])
            if s > 0: c += 1
            elif s < 0: d += 1
    return (c - d) / (c + d) if (c + d) else 0.0

def show(p):
    return " & ".join(f"{pr}({','.join(a)})" if a else pr for pr, a in sorted(p))


def run(n_train=4500, n_hold=2000, min_support=110, B_lib=32, seed=0):
    tb = chess.syzygy.open_tablebase(TB_DIR)
    tr = gen(n_train, tb, seed)
    ho = gen(n_hold, tb, seed + 999)
    labels = [o for _, o in tr]; N = len(tr); base = Counter(labels)
    print(f"=== KRPvKR pilot (train {N}, holdout {len(ho)}) ===")
    print("outcome dist (White frame):", dict(base), " base-rate acc =",
          f"{max(base.values())/N:.3f}")

    pool = candidates(tr, min_support)
    print(f"label-blind candidate pool (freq itemsets, size>=2, supp>={min_support}): {len(pool)}")

    items = list(pool.items())
    sc_C = sorted(items, key=lambda kv: U_C(*kv), reverse=True)
    sc_D = sorted(items, key=lambda kv: U_D(kv[0], kv[1], labels, base, N), reverse=True)

    for Bn in (16, B_lib, 64):
        L_C = [p for p, _ in sc_C[:Bn]]
        L_D = [p for p, _ in sc_D[:Bn]]
        print(f"  B={Bn:3d}:  Jaccard(L_C,L_D) = {jaccard(L_C, L_D):.3f}")

    # rank divergence over the whole pool (sample to bound O(M^2))
    rng = random.Random(1)
    samp = items if len(items) <= 700 else rng.sample(items, 700)
    rC = {p: i for i, (p, _) in enumerate(sorted(samp, key=lambda kv: U_C(*kv), reverse=True))}
    rD = {p: i for i, (p, _) in enumerate(sorted(samp, key=lambda kv: U_D(kv[0], kv[1], labels, base, N), reverse=True))}
    tau = kendall([(rC[p], rD[p]) for p, _ in samp])
    print(f"  Kendall tau (U_C vs U_D rankings, m={len(samp)}): {tau:.3f}")

    # held-out 3-class accuracy: L_C vs L_D vs baseline vs permuted-L_D
    L_C = [p for p, _ in sc_C[:B_lib]]
    L_D = [p for p, _ in sc_D[:B_lib]]
    y_tr = [CLS[o] for o in labels]; y_ho = [CLS[o] for _, o in ho]
    ho_lits = [ls for ls, _ in ho]; tr_lits = [ls for ls, _ in tr]
    def heldout_acc(lib):
        Ftr, dim = feats(tr_lits, lib); Fho, _ = feats(ho_lits, lib)
        return acc(train(Ftr, y_tr, dim), Fho, y_ho)
    aC, aD = heldout_acc(L_C), heldout_acc(L_D)

    # permutation null: shuffle labels -> recompute U_D -> reselect L_D_perm
    perm = labels[:]; random.Random(7).shuffle(perm)
    pbase = Counter(perm)
    sc_Dp = sorted(items, key=lambda kv: U_D(kv[0], kv[1], perm, pbase, N), reverse=True)
    L_Dp = [p for p, _ in sc_Dp[:B_lib]]
    aDp = heldout_acc(L_Dp)

    print(f"\nheld-out 3-class accuracy (base-rate {max(base.values())/N:.3f}):")
    print(f"  L_C  (compression-selected) : {aC:.3f}")
    print(f"  L_D  (discrimination-selected): {aD:.3f}   gap over L_C = {aD-aC:+.3f}")
    print(f"  L_D under PERMUTED labels    : {aDp:.3f}   (null: should ~= base-rate)")
    print(f"  Jaccard(L_C, L_D_perm) = {jaccard(L_C, L_Dp):.3f}")

    print("\nTop 'compression-BLIND discriminative' schemas (high U_D, NOT in L_C):")
    setC = set(L_C); k = 0
    for p, s in sc_D:
        if p not in setC:
            print(f"  MI={U_D(p,s,labels,base,N):.3f} supp={len(s):4d} U_C={U_C(p,s):5d}  {show(p)}")
            k += 1
            if k >= 6: break
    print("\nTop 'discrimination-BLIND compressive' schemas (high U_C, low MI):")
    setD = set(L_D); k = 0
    for p, s in sc_C:
        if p not in setD:
            print(f"  U_C={U_C(p,s):6d} supp={len(s):4d} MI={U_D(p,s,labels,base,N):.3f}  {show(p)}")
            k += 1
            if k >= 6: break

    print("\nREAD: low Jaccard + low/!=1 tau + (aD >> aC) => compression misses the discriminative")
    print("structure in KRPvKR (reproduces the prior anecdote). High Jaccard + aD~=aC => null.")


if __name__ == "__main__":
    run()
