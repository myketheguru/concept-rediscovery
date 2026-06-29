# artifact-version: 1.0.0
"""
core.py  --  START HERE.

Goal: learn the schema library from labelled games instead of hand-writing it.
See CLAUDE.md for the full thesis and milestone spec.

This file is a SPINE: the featurizer + alignment eval WORK and run on real boards
(demo at the bottom). The data/label/mining stages are stubbed with TODOs and the
algorithm described inline.

Design discipline (CLAUDE.md): discovered-not-handed; weight higher-order structure;
let it fail and report the ceiling; THE FEATURIZER VOCABULARY IS THE PRIOR.
"""
from itertools import permutations

try:
    import chess, chess.pgn
    HAVE_CHESS = True
except ImportError:
    HAVE_CHESS = False

W_HO = 3  # systematicity weight on higher-order relations


# ============================================================================
# 1. FEATURIZER  --  board -> relational graph.  *** THE PRIOR LIVES HERE ***
# ============================================================================
# v1 is intentionally crude and white-pawn-centric. Refining these predicate
# definitions -- and ADDING higher-order ENABLES/CAUSE edges -- is the core
# scientific work. When an eval is wrong, suspect THIS first, not the search.

def featurize(board):
    """Return {'entities':[...], 'props':{id:(pred,(args...))}}.
    TODO(core): color symmetry; passed-pawn detection; attacker/defender roles;
                'shelters'/'cuts' geometry; higher-order ENABLES/CAUSE edges
                (without these the eval has NO systematicity signal)."""
    ents, props = [], {}
    n = [0]
    def add(pred, *args):
        props[f"p{n[0]}"] = (pred, tuple(args)); n[0] += 1

    name = {}
    for sq in chess.SQUARES:
        pc = board.piece_at(sq)
        if pc:
            nm = f"{'w' if pc.color else 'b'}{pc.symbol().upper()}{chess.square_name(sq)}"
            name[sq] = nm; ents.append(nm)

    wk = [s for s, p in name.items() if board.piece_at(s).piece_type == chess.KING
          and board.piece_at(s).color]
    wr = [s for s, p in name.items() if board.piece_at(s).piece_type == chess.ROOK
          and board.piece_at(s).color]
    for s in chess.SQUARES:
        pc = board.piece_at(s)
        if pc and pc.piece_type == chess.PAWN and pc.color == chess.WHITE:
            P, r, f = name[s], chess.square_rank(s), chess.square_file(s)
            if r >= 4:        add("near_promo", P)      # 5th rank+ (ranks are 0-indexed)
            if f in (0, 7):   add("rook_pawn", P)       # the feature Lucena ignores
            for k in wk:                                 # king ahead of own pawn
                if chess.square_rank(k) > r and abs(chess.square_file(k) - f) <= 1:
                    add("in_front", name[k], P)
            # TODO: shelters(rook, king) bridge geometry; ENABLES(shelters, in_front)
    return {"entities": ents, "props": props}


# ----------------------------------------------------------------------------
# 1b. FEATURIZER v2 -- KRPvKR endgame vocabulary.  *** THE PRIOR FOR MINING ***
# ----------------------------------------------------------------------------
# The richer vocabulary the milestone calls for. Predicates are GENERAL geometry
# (king-in-front, rook-behind, kings-in-opposition, rook-cuts-king, ...), NEVER
# named after a human concept -- the experiment is whether the miner ASSEMBLES
# them into the named concepts, scoring by outcome-lift alone. Two predicates
# are tagged higher-order (ENABLES_/CAUSE_): they summarise a CONFIGURATION of
# first-order facts. We emit the first-order parts too, so the miner can reveal
# whether the higher-order edge adds lift over its constituents (the falsifiable
# systematicity test) rather than us assuming it via W_HO.
#
# Entities are CANONICAL ROLES (one per piece: WK WR WP BK BR). For a single
# fixed material this is reading the board, not handing an analogy -- and it
# makes schema->position alignment the identity, so subgraph matching reduces to
# literal-set containment (mine_schemas exploits this). The morphism SEARCH is
# exercised by the cross-domain toys / by transfer to other materials, both out
# of scope here. White is always the pawn side; outcomes are taken in White's
# frame by the driver.
HO_PREDS = {"ENABLES_promo", "CAUSE_fortress"}

def featurize_krpvkr(board):
    """KRPvKR board -> role-named relational graph. Assumes exactly WK,WR,WP vs
    BK,BR (guaranteed by load_positions(material_filter='KRPvKR'))."""
    F, R = chess.square_file, chess.square_rank
    wk = next(iter(board.pieces(chess.KING, chess.WHITE)))
    bk = next(iter(board.pieces(chess.KING, chess.BLACK)))
    wr = next(iter(board.pieces(chess.ROOK, chess.WHITE)))
    br = next(iter(board.pieces(chess.ROOK, chess.BLACK)))
    wp = next(iter(board.pieces(chess.PAWN, chess.WHITE)))

    props, n = {}, [0]
    def add(pred, *args):
        props[f"p{n[0]}"] = (pred, tuple(args)); n[0] += 1

    pf, pr = F(wp), R(wp)
    promo = chess.square(pf, 7)                       # white promotes on rank 8

    # side to move (0-ary) -- lets tempo concepts (opposition) be expressible
    add("stm_white") if board.turn == chess.WHITE else add("stm_black")

    # pawn: advancement bucket + rook-pawn flag
    add("pawn_promoting", "WP") if pr >= 6 else \
        add("pawn_advanced", "WP") if pr >= 4 else add("pawn_back", "WP")
    if pf in (0, 7):
        add("rook_pawn", "WP")

    # white king vs its pawn / the promotion square
    if R(wk) > pr and abs(F(wk) - pf) <= 1:           add("wk_ahead", "WP")
    if abs(F(wk) - pf) <= 1 and abs(R(wk) - pr) <= 1: add("wk_supports", "WP")
    if chess.square_distance(wk, promo) <= 1:         add("wk_on_promo", "WP")

    # black (defending) king vs the pawn
    if R(bk) > pr and abs(F(bk) - pf) <= 1:           add("bk_ahead", "WP")
    if F(bk) == pf and R(bk) >= pr + 1:               add("bk_blocks", "WP")
    if chess.square_distance(bk, promo) <= 1:         add("bk_on_promo", "WP")

    # kings: direct opposition / proximity
    df, dr = abs(F(wk) - F(bk)), abs(R(wk) - R(bk))
    if (df == 0 and dr == 2) or (dr == 0 and df == 2): add("opposition", "WK", "BK")
    if chess.square_distance(wk, bk) == 1:             add("kings_close", "WK", "BK")

    # white rook vs pawn / cutting the black king off
    if F(wr) == pf and R(wr) < pr: add("wr_behind", "WR", "WP")
    if F(wr) == pf and R(wr) > pr: add("wr_front", "WR", "WP")
    if F(wr) != F(bk) and (F(bk) - F(wr)) * (pf - F(wr)) < 0: add("wr_cuts_file", "WR", "BK")
    if R(wr) != R(bk) and (R(bk) - R(wr)) * (pr - R(wr)) < 0: add("wr_cuts_rank", "WR", "BK")

    # black rook: behind / in front of pawn / checking distance from white king
    if F(br) == pf and R(br) < pr: add("br_behind", "BR", "WP")
    if F(br) == pf and R(br) > pr: add("br_front", "BR", "WP")
    br_checks = (F(br) == F(wk) or R(br) == R(wk)) and chess.square_distance(br, wk) >= 3
    if br_checks:                                  add("br_checks", "BR", "WK")
    if chess.square_distance(br, promo) <= 2:      add("br_guards_promo", "BR", "WP")

    # ---- higher-order: configurations that ENABLE / CAUSE an outcome ----
    behind = F(wr) == pf and R(wr) < pr
    path_clear = not (F(bk) == pf and R(bk) > pr)  # defending king not on the file ahead
    if behind and pr >= 4 and path_clear:
        add("ENABLES_promo", "WR", "WP")           # rook-escorted promotion (Lucena engine)
    if F(bk) == pf and R(bk) >= pr + 1 and br_checks and pf not in (0, 7):
        add("CAUSE_fortress", "BK", "WP")          # blockade + checking rook (Philidor engine)

    return {"entities": ["WK", "WR", "WP", "BK", "BR"], "props": props}


def featurize_krpvkr_dumb(board):
    """NEGATIVE-CONTROL vocabulary (for the pressure test). Generic, concept-
    AGNOSTIC spatial primitives only: per-piece coarse rank/file buckets +
    pairwise same-file / same-rank / adjacency / proximity. NO concept-shaped
    relations (no cuts / blocks / supports / ahead / behind, no ENABLES/CAUSE).

    The named concepts are still EXPRESSIBLE here, just as LARGER conjunctions
    (e.g. rook-behind-pawn = same_file(WR,WP) + WR low-rank + WP high-rank). So
    this asks the honest question: does the rediscovery live in the DATA, or did
    the rich featurizer pre-chunk the concepts into the prior? The thesis
    predicts the richer prior buys fidelity -- i.e. this control should do worse.
    Same roles (reading the board is not concept knowledge)."""
    F, R = chess.square_file, chess.square_rank
    sq = {"WK": next(iter(board.pieces(chess.KING, chess.WHITE))),
          "WR": next(iter(board.pieces(chess.ROOK, chess.WHITE))),
          "WP": next(iter(board.pieces(chess.PAWN, chess.WHITE))),
          "BK": next(iter(board.pieces(chess.KING, chess.BLACK))),
          "BR": next(iter(board.pieces(chess.ROOK, chess.BLACK)))}
    props, n = {}, [0]
    def add(pred, *args):
        props[f"p{n[0]}"] = (pred, tuple(args)); n[0] += 1

    add("stm_white") if board.turn == chess.WHITE else add("stm_black")
    rbucket = lambda r: ("r_lo", "r_lo", "r_2", "r_2", "r_4", "r_4", "r_hi", "r_hi")[r]
    fbucket = lambda f: ("f_e", "f_e", "f_m", "f_m", "f_m", "f_m", "f_e", "f_e")[f]
    for name, s in sq.items():
        add(f"rank_{rbucket(R(s))}", name)
        add(f"file_{fbucket(F(s))}", name)
    roles = ["WK", "WR", "WP", "BK", "BR"]
    for i in range(len(roles)):
        for j in range(i + 1, len(roles)):
            a, b = roles[i], roles[j]
            sa, sb = sq[a], sq[b]
            if F(sa) == F(sb): add("same_file", a, b)
            if R(sa) == R(sb): add("same_rank", a, b)
            d = chess.square_distance(sa, sb)
            if d == 1:   add("adjacent", a, b)
            elif d <= 2: add("near", a, b)
    return {"entities": roles, "props": props}


# ============================================================================
# 2. ALIGNMENT + EVAL  --  reused from chess_eval_alignment.py (the spine).
#    Discovers the schema->position injection by search; scores preserved,
#    higher-order-weighted structure. Allows PARTIAL matches.
# ============================================================================
def _pw(args, own): return W_HO if any(a in own for a in args) else 1

def align(schema, pos):
    sp, se = schema["props"], schema["entities"]
    pp, pe = pos["props"], pos["entities"]
    Bidx = {(pred,) + tuple(a): pid for pid, (pred, a) in pp.items()}
    total = sum(_pw(a, sp) for _, a in sp.values()) or 1
    deficit = max(0, len(se) - len(pe))
    pool = list(pe) + [f"__none{i}" for i in range(deficit)]
    best_w, best = -1, ({}, [])
    n = 0
    for perm in permutations(pool, len(se)):
        n += 1
        phi = {s: e for s, e in zip(se, perm) if not str(e).startswith("__none")}
        match, changed = {}, True
        while changed:
            changed = False
            for sid, (pred, args) in sp.items():
                if sid in match: continue
                mp, ok = [], True
                for a in args:
                    if a in phi:     mp.append(phi[a])
                    elif a in match: mp.append(match[a])
                    else:            ok = False; break
                if not ok: continue
                bid = Bidx.get((pred,) + tuple(mp))
                if bid is not None and bid not in match.values():
                    match[sid] = bid; changed = True
        w = sum(_pw(sp[sid][1], sp) for sid in match)
        if w > best_w:
            best_w, best = w, (phi, [sp[sid][0] for sid in match])
    return best_w, total, n, best[0], best[1]

def evaluate(pos, library, threshold=0.6):
    rows = []
    for sch in library:
        w, total, _, phi, preds = align(sch, pos)
        rows.append((w / total, w, sch, phi, preds))
    rows.sort(key=lambda r: (round(r[0], 6), r[1]), reverse=True)
    conf, w, sch, phi, preds = rows[0]
    return (sch["outcome"] if conf >= threshold else "UNCERTAIN"), conf, sch, preds


# ============================================================================
# 3. DATA  --  stream labelled positions from real Lichess games.
# ============================================================================
# Source: the OFFICIAL Lichess games database, mirrored on Hugging Face as
# parquet -- `Lichess/standard-chess-games` (lichess.org itself is firewall-
# blocked here; HF is reachable). One month per file; 2013 months are smallest.
# Fetch with `python fetch_data.py 2013 1` (see data/README.md). We read parquet
# directly rather than PGN -- same data, no decompression dance.

_MOVE_NUM = __import__("re").compile(r"\d+\.(\.\.)?")
_RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}

def _sans(movetext):
    """Yield SAN tokens from Lichess movetext (drops move numbers / result)."""
    for tok in movetext.split():
        if tok in _RESULTS or _MOVE_NUM.fullmatch(tok):
            continue
        yield tok

def material(board):
    """Canonical-ish material signature, e.g. 'KRPvKR' (white v black, by value).
    Use to filter to one endgame -- alignment is NP-hard, so keep graphs small."""
    order = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    side = lambda c: "".join(chess.piece_symbol(pt).upper() * len(board.pieces(pt, c))
                             for pt in order)
    return f"{side(chess.WHITE)}v{side(chess.BLACK)}"

def load_positions(path, max_men=5, material_filter=None, dedup=True,
                   max_games=None, with_game_id=False):
    """Stream a Lichess monthly parquet; replay each game; yield every board with
    <= max_men pieces (optionally only `material_filter`, e.g. 'KRPvKR').

    Yields fresh chess.Board copies (or (board, game_index) if with_game_id --
    split train/holdout by GAME so correlated positions don't leak across the
    split). `dedup` drops repeat FENs so consecutive plies / transpositions don't
    swamp the sample. Labels are NOT attached here -- call label(board, tb)
    separately. Illegal SAN aborts that one game (Lichess data is clean, but be
    safe)."""
    import pyarrow.parquet as pq
    seen, games = set(), 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=1000, columns=["movetext"]):
        for movetext in batch.column("movetext").to_pylist():
            if max_games is not None and games >= max_games:
                return
            gid = games
            games += 1
            b = chess.Board()
            for san in _sans(movetext):
                try:
                    b.push_san(san)
                except ValueError:
                    break
                if chess.popcount(b.occupied) > max_men:
                    continue
                if material_filter and material(b) != material_filter:
                    continue
                if dedup:
                    key = b._transposition_key()
                    if key in seen:
                        continue
                    seen.add(key)
                yield (b.copy(stack=False), gid) if with_game_id else b.copy(stack=False)


# ============================================================================
# 4. LABELS  --  exact outcome via Syzygy tablebases.
# ============================================================================
# Implemented. Probes Syzygy WDL and returns the EXACT theoretical result from
# the side-to-move's perspective. Download the capture-closure first (see
# tablebases/README -- for KRPvKR you need ~12 small <=5-man .rtbw files, not
# the whole 1 GB set). Only LEGAL positions probe: if the side NOT to move is in
# check the probe walks an illegal king capture and dies -- always is_valid().

_TB_CACHE = {}

def _open_tb(tb_path):
    """Cache open tablebases by path -- opening per-call is wasteful."""
    import chess.syzygy
    tb = _TB_CACHE.get(tb_path)
    if tb is None:
        tb = chess.syzygy.open_tablebase(tb_path)
        _TB_CACHE[tb_path] = tb
    return tb

def label(board, tb_path):
    """Exact WIN / DRAW / LOSS from the SIDE-TO-MOVE's perspective, via Syzygy.

    tb_path may be a directory path or an already-open tablebase object.
    Returns None when the position cannot be probed (over the piece limit, a
    sub-table outside the downloaded closure, or an illegal board). We use the
    50-move-AGNOSTIC theoretical value: cursed wins / blessed losses (|wdl|==1)
    count as WIN / LOSS, not DRAW -- that is the exact game-theoretic outcome,
    which is what makes concept-rediscovery meaningful. The harness converts
    this side-to-move label into White's frame for comparison.
    Fallbacks (PGN %eval, game result) belong upstream in load_positions, not
    here -- this function is the exact-label path on purpose."""
    import chess.syzygy
    if board.is_checkmate():
        return "LOSS"            # side to move is mated
    if board.is_game_over():
        return "DRAW"            # stalemate / insufficient material / etc.
    if not board.is_valid():
        return None              # illegal: probe would walk a king capture
    tb = tb_path if hasattr(tb_path, "probe_wdl") else _open_tb(tb_path)
    try:
        wdl = tb.probe_wdl(board)
    except (chess.syzygy.MissingTableError, KeyError):
        return None              # outside the downloaded closure
    return "WIN" if wdl > 0 else "LOSS" if wdl < 0 else "DRAW"


# ============================================================================
# 5. MINE  --  the experiment: discover schemas from labelled data.
# ============================================================================
# Beam search over small relational patterns (sets of literals over the role
# variables), scored by predictive lift, validated on held-out GAMES.
#
# Connectivity: for a SINGLE fixed material every literal describes the same
# five-piece scene, so requiring shared-entity connectivity is the wrong
# constraint -- it would reject genuine multi-piece concepts (Lucena needs
# pawn+king+rook+enemy-king relations whose binary literals don't all share an
# entity). So patterns are free literal-sets by default (require_connected=False);
# the support/lift/beam bounds keep the space tractable. The `_connected` gate
# remains for multi-material mining, where it is meaningful.
#
# Fixed-material trick: because entities are canonical roles, a pattern S matches
# a position iff S's literals are a SUBSET of the position's literals. So the
# support set of S is the INTERSECTION of its literals' support sets -- frequent
# predictive itemset mining, exact and fast, no isomorphism search. (For multi-
# material mining you'd restore the morphism search via align(); see featurizer
# v2 note.) score(S) = support(S) * lift(S) is the MDL/compression reading: how
# many positions one predictive structure explains above the base rate.

from collections import Counter, defaultdict

def _literals(graph):
    return frozenset(graph["props"].values())          # {(pred, args), ...}

def _connected(pattern):
    """Pattern is relationally connected: its >=1-ary literals share entities in
    one component. 0-ary literals (stm_*) are global context, always allowed."""
    lits = [a for _, a in pattern if a]
    if not lits:
        return True
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for args in lits:
        for a in args[1:]:
            parent[find(args[0])] = find(a)
    roots = {find(a) for args in lits for a in args}
    return len(roots) == 1

def mine_schemas(labelled_graphs, min_support=40, min_lift=0.12,
                 max_literals=4, beam_width=600, holdout=None,
                 holdout_keep=0.5, ho_preds=HO_PREDS, require_connected=False):
    """Discover predictive relational schemas. `labelled_graphs` is a list of
    (graph, outcome). Returns schema dicts (evaluate-compatible) ranked by
    score = support * lift, each carrying train + held-out stats and an `ho`
    flag (contains a higher-order predicate). Nothing about the OUTCOME or the
    concept names enters the search -- patterns are grown and scored purely by
    support and outcome-lift, so the experiment can fail."""
    tx = [(_literals(g), o) for g, o in labelled_graphs]
    N = len(tx)
    base = Counter(o for _, o in tx)
    base_rate = {o: base[o] / N for o in base}

    lit_sup = defaultdict(set)                          # literal -> {tx indices}
    for i, (lits, _) in enumerate(tx):
        for l in lits:
            lit_sup[l].add(i)
    freq = [l for l, s in lit_sup.items() if len(s) >= min_support]

    def stats(supp):
        c = Counter(tx[i][1] for i in supp)
        o, cnt = c.most_common(1)[0]
        return o, cnt / len(supp), cnt / len(supp) - base_rate.get(o, 0), c

    supp_of = {frozenset([l]): lit_sup[l] for l in freq}
    beam = list(supp_of)
    for _ in range(2, max_literals + 1):               # grow one literal at a time
        scored = []
        for p in beam:
            ps = supp_of[p]
            for l in freq:
                if l in p:
                    continue
                np = p | {l}
                if np in supp_of:
                    continue
                if require_connected and not _connected(np):
                    continue
                s = ps & lit_sup[l]
                if len(s) < min_support:
                    continue
                supp_of[np] = s
                _, _, lift, _ = stats(s)
                scored.append((len(s) * max(lift, 0.0), np))
        scored.sort(key=lambda x: x[0], reverse=True)
        beam = [np for _, np in scored[:beam_width]]
        if not beam:
            break

    hgraphs = [(_literals(g), o) for g, o in holdout] if holdout else None
    hN = len(hgraphs) if hgraphs else 0
    hbase = Counter(o for _, o in hgraphs) if hgraphs else Counter()

    out = []
    for p, s in supp_of.items():
        o, prec, lift, counts = stats(s)
        if lift < min_lift:
            continue
        rec = {"outcome": o, "support": len(s), "precision": prec, "lift": lift,
               "score": len(s) * lift, "ho": any(pr in ho_preds for pr, _ in p),
               "counts": dict(counts), "_p": p}
        if hgraphs is not None:
            hs = [i for i, (lits, _) in enumerate(hgraphs) if p <= lits]
            if hs:
                hprec = sum(hgraphs[i][1] == o for i in hs) / len(hs)
                hl = hprec - hbase[o] / hN
                rec.update(holdout_support=len(hs), holdout_precision=hprec,
                           holdout_lift=hl,
                           generalizes=len(hs) >= max(5, min_support // 4)
                                       and hl >= holdout_keep * lift)
            else:
                rec.update(holdout_support=0, holdout_precision=0.0,
                           holdout_lift=0.0, generalizes=False)
        out.append(rec)

    out.sort(key=lambda r: r["score"], reverse=True)
    for k, r in enumerate(out):                         # -> evaluate-compatible schema
        lits = sorted(r.pop("_p"))
        r["name"] = f"M{k}_{r['outcome']}"
        r["entities"] = sorted({a for _, args in lits for a in args})
        r["props"] = {str(j): lit for j, lit in enumerate(lits)}
    return out


# ============================================================================
# DEMO  --  proves the spine runs on real boards (hand-built library for now).
# ============================================================================
SCHEMA_LIBRARY = [
    {"name": "WinningKingPawn", "outcome": "WIN", "entities": ["aK", "aP"],
     "props": {"1": ("in_front", ("aK", "aP")), "2": ("near_promo", ("aP",))}},
    {"name": "RookPawnDraw", "outcome": "DRAW", "entities": ["aK", "aP"],
     "props": {"1": ("in_front", ("aK", "aP")), "2": ("near_promo", ("aP",)),
               "3": ("rook_pawn", ("aP",))}},
]

if __name__ == "__main__":
    if not HAVE_CHESS:
        print("pip install chess  # then re-run the demo"); raise SystemExit
    demos = [
        ("central pawn, king in front", "8/8/4K3/4P3/8/8/8/7k w - - 0 1", "WIN"),
        ("ROOK pawn, king in front",    "8/8/K7/P7/8/8/8/7k w - - 0 1",    "DRAW"),
    ]
    for label_txt, fen, truth in demos:
        pos = featurize(chess.Board(fen))
        v, conf, sch, preds = evaluate(pos, SCHEMA_LIBRARY)
        print(f"\n{label_txt}")
        print(f"  graph props : {[p[0] for p in pos['props'].values()]}")
        print(f"  eval        : {v}  (conf {conf:.2f}) via '{sch['name']}'  matched {preds}")
        print(f"  truth       : {truth}  -> {'OK' if v == truth else 'WRONG'}")
    print("\nNote: first-order vocabulary only. The rook-pawn case resolves ONLY because")
    print("'rook_pawn' is in the vocabulary -- the prior. Add ENABLES/CAUSE edges and a")
    print("learned library next (see mine_schemas + CLAUDE.md).")
