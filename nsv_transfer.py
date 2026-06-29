# artifact-version: 1.0.0
"""
nsv_transfer.py  --  NO-SHARED-VOCABULARY cross-material transfer.

The handed transfer (transfer_test.py) gave the matcher the rook<->queen
correspondence for free, via shared predicate names + role-named entities. Here
we take it away: the QUEEN material uses DISJOINT predicate names ('q_' prefix)
and ANONYMIZED, per-position-shuffled entity labels (e0..e4). The matcher sees
opaque tokens; it must DISCOVER, from structure alone:
   phi : schema entities (WK,WL,WP,BK,BL)  ->  position entities (e0..e4)
   rho : schema predicates (rook vocab)    ->  position predicates (queen vocab)
maximizing preserved literals (a consistent relation dictionary, not per-literal
cheating). This is the structure-mapping the thesis rests on (cf.
no_shared_vocabulary.py): structure should recover the analogy, but TIE under
symmetry (e.g. stm_white vs stm_black, cuts_file vs cuts_rank), where a tiebreak
needs another prior (grounding).

Reports: (1) the DISCOVERED dictionary vs the intended one; (2) symmetry ties;
(3) transfer accuracy with the DISCOVERED morphism vs the HANDED one vs baseline
vs native ceiling. Run: python nsv_transfer.py [n_holdout]
"""
import sys, csv, random, collections
from itertools import permutations
import chess, chess.syzygy
from core import mine_schemas, evaluate, _literals
from mine_run import _select, white_frame
from core import label as syzygy_label
from transfer_test import featurize_endgame, krpvkr_dataset, mine_lib, lib_accuracy

TB = "tablebases/syzygy"
R = chess.square_rank


def kqpvkq_boards(n, seed):
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
        out.append((b, white_frame(y, b.turn)))
    tb.close()
    return out


def anonymize(graph, rng):
    """Queen graph -> opaque: predicates get 'q_' prefix, entities shuffled to
    e0..e4. Returns (anon_graph, true_role_of_generic) (truth only for scoring)."""
    ents = list(graph["entities"])
    labels = [f"e{i}" for i in range(len(ents))]
    rng.shuffle(labels)
    emap = dict(zip(ents, labels))                 # role -> generic
    props = {k: ("q_" + p, tuple(emap[a] for a in args))
             for k, (p, args) in graph["props"].items()}
    return {"entities": labels, "props": props}, {v: k for k, v in emap.items()}


def index(graph):
    by_pred = collections.defaultdict(set)
    arity = {}
    for p, args in graph["props"].values():
        by_pred[p].add(tuple(args)); arity[p] = len(args)
    return by_pred, arity


def assign_rho(req, pos_by_pred, pos_arity):
    """Greedy injective rho: schema-pred -> position-pred (same arity) maximizing
    matched translated tuples. Returns (score, rho)."""
    cands = []
    for spred, tuples in req.items():
        ar = len(tuples[0])
        for qpred, qset in pos_by_pred.items():
            if pos_arity[qpred] != ar:
                continue
            cnt = sum(t in qset for t in tuples)
            if cnt:
                cands.append((cnt, spred, qpred))
    cands.sort(reverse=True)
    used_s, used_q, rho, total = set(), set(), {}, 0
    for cnt, s, q in cands:
        if s in used_s or q in used_q:
            continue
        rho[s] = q; used_s.add(s); used_q.add(q); total += cnt
    return total, rho


def best_match(schema, pos_by_pred, pos_arity, pos_entities):
    """Discover (phi, rho) maximizing preserved literals. Returns (score, rho)."""
    se = schema["entities"]
    slits = list(schema["props"].values())
    best_score, best_rho = -1, {}
    for perm in permutations(pos_entities, len(se)):
        phi = dict(zip(se, perm))
        req = collections.defaultdict(list)
        for pred, args in slits:
            req[pred].append(tuple(phi[a] for a in args))
        score, rho = assign_rho(req, pos_by_pred, pos_arity)
        if score > best_score:
            best_score, best_rho = score, rho
    return best_score, best_rho


def is_structured(sch):
    """Enough relational structure to constrain a morphism: >=2 literals and at
    least one binary (relational) predicate -- not a bag of swappable unaries."""
    return len(sch["props"]) >= 2 and any(len(a) >= 2 for _, a in sch["props"].values())


def discovered_eval(schema_lib, anon_positions, threshold=0.6):
    """Apply rook schemas to anonymized queen positions via discovered morphism.
    Prediction uses the best-matching schema. The relation dictionary is tallied
    ONLY from FULL matches (conf==1.0) of STRUCTURED schemas -- the applications
    where the discovered morphism is actually pinned by structure."""
    preds = []
    dictionary = collections.defaultdict(collections.Counter)   # rook pred -> Counter(queen pred)
    for g in anon_positions:
        by_pred, arity = index(g)
        ents = g["entities"]
        best = ("UNCERTAIN", -1.0)
        for sch in schema_lib:
            score, rho = best_match(sch, by_pred, arity, ents)
            conf = score / len(sch["props"])
            if conf > best[1]:
                best = (sch["outcome"], conf)
            if conf >= 0.999 and is_structured(sch):       # trustworthy dictionary only
                for s, q in rho.items():
                    dictionary[s][q] += 1
        preds.append(best[0] if best[1] >= threshold else "UNCERTAIN")
    return preds, dictionary


def main():
    nH = int(sys.argv[1]) if len(sys.argv) > 1 else 800

    # rook library (mined on real KRPvKR games, agnostic/rook vocab)
    kr_tr, kr_ho = krpvkr_dataset()
    rook_lib, _ = mine_lib(kr_tr, kr_ho, max(20, len(kr_tr) // 40))

    # queen positions (exact labels), featurized two ways from the SAME boards
    boards = kqpvkq_boards(nH, seed=11)
    shared = [(featurize_endgame(b), o) for b, o in boards]      # handed correspondence
    rng = random.Random(99)
    anon = [(anonymize(featurize_endgame(b), rng)[0], o) for b, o in boards]

    base = collections.Counter(o for _, o in shared)
    maj = base.most_common(1)[0][0]
    maj_acc = sum(o == maj for _, o in shared) / nH

    # native ceiling (mine on queen, shared vocab) -- split these same positions
    split = nH // 2
    native_lib, _ = mine_lib(shared[:split], shared[split:], max(20, split // 40))

    # accuracies on the SAME held-out queen positions
    handed_acc, handed_cov = lib_accuracy(rook_lib, shared, maj)
    native_acc, native_cov = lib_accuracy(native_lib, shared[split:], maj)

    print(f"=== NO-SHARED-VOCABULARY TRANSFER  KRPvKR -> KQPvKQ ===")
    print(f"rook library: {len(rook_lib)} schemas;  queen holdout: {nH} (base {dict(base)})\n")

    structured = [s for s in rook_lib if is_structured(s)]
    # discovered-morphism transfer: full library, and structured-only
    full_acc = lambda preds: sum((p if p != "UNCERTAIN" else maj) == o
                                 for p, (_, o) in zip(preds, anon)) / nH
    dpreds, dictionary = discovered_eval(rook_lib, [g for g, _ in anon])
    spreds, _ = discovered_eval(structured, [g for g, _ in anon])
    cov = sum(p != "UNCERTAIN" for p in dpreds)
    scov = sum(p != "UNCERTAIN" for p in spreds)

    print("--- KQPvKQ held-out accuracy ---")
    print(f"  majority baseline ('{maj}')                 : {maj_acc:.3f}")
    print(f"  HANDED correspondence (shared vocab, {len(rook_lib)} sch): {handed_acc:.3f}  (cov {handed_cov:.2f})")
    print(f"  DISCOVERED morphism, FULL lib ({len(rook_lib)} sch)      : {full_acc(dpreds):.3f}  (cov {cov/nH:.2f})  <- trivial schemas overfit")
    print(f"  DISCOVERED morphism, STRUCTURED ({len(structured)} sch)    : {full_acc(spreds):.3f}  (cov {scov/nH:.2f})")
    print(f"  native KQPvKQ-mined (ceiling)               : {native_acc:.3f}  (cov {native_cov:.2f})")

    # discovered dictionary vs intended (intended: rook pred X -> 'q_'+X), split by arity
    rook_arity = {p: len(a) for s in rook_lib for p, a in s["props"].values()}
    print("\n--- DISCOVERED relation dictionary (from full structural matches) ---")
    rec = {1: [0, 0], 2: [0, 0]}                       # arity -> [correct, total]
    for spred in sorted(dictionary, key=lambda s: -sum(dictionary[s].values())):
        c = dictionary[spred]
        (q1, n1) = c.most_common(1)[0]
        n2 = c.most_common(2)[1][1] if len(c) > 1 else 0
        intended = "q_" + spred
        ar = 2 if rook_arity.get(spred, 1) >= 2 else 1
        ok = (q1 == intended)
        rec[ar][0] += ok * n1; rec[ar][1] += sum(c.values())
        tie = "  <TIE>" if n2 >= 0.8 * n1 else ""
        print(f"  [{'binary' if ar==2 else 'unary '}] {spred:20s} -> {q1:22s} {n1:4d}x "
              f"{'OK' if ok else 'XX(want '+intended+')'}{tie}")
    for ar, lab in [(2, "binary/relational"), (1, "unary")]:
        cc, tt = rec[ar]
        print(f"  {lab:18s} dictionary recovery: {cc}/{tt} = {cc/max(1,tt)*100:3.0f}% to the intended twin")
    print("\nReading: if DISCOVERED ~ HANDED and the dictionary recovers the structural")
    print("twins, structure-alone recovered the analogy; ties (stm/cuts symmetry) are")
    print("where a non-structural prior (grounding) would be needed to break them.")


if __name__ == "__main__":
    main()
