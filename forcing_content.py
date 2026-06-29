# artifact-version: 1.0.0
"""forcing_content.py -- the INDEPENDENT VARIABLE for the compression-vs-discrimination
study (STUDY_compression_vs_discrimination.md). Forcing/tempo content of a material =
the fraction of legal positions whose WDL outcome class (win/draw/loss, in White's
frame) FLIPS when the side to move is flipped -- i.e. zugzwang/tempo-criticality.
Computed EXACTLY from Syzygy, no engine. Feasibility result: spans 0.007 (KQvK) ->
0.651 (KQPvKQ) across the 9 on-disk materials. Run: python forcing_content.py"""
import chess, chess.syzygy, random

TB_DIR = "tablebases/syzygy"
W, B, K, Q, R, P = chess.WHITE, chess.BLACK, chess.KING, chess.QUEEN, chess.ROOK, chess.PAWN


def _rand_board(pieces, rng):
    sq = rng.sample(range(64), len(pieces))
    bd = chess.Board.empty()
    for (pt, c), s in zip(pieces, sq):
        if pt == P and chess.square_rank(s) in (0, 7):
            return None
        bd.set_piece_at(s, chess.Piece(pt, c))
    return bd


def _wclass(bd, tb):
    try:
        wdl = tb.probe_wdl(bd)
    except Exception:
        return None
    wf = wdl if bd.turn == W else -wdl
    return (wf > 0) - (wf < 0)              # +1 win / 0 draw / -1 loss, White's frame


def forcing(pieces, tb, n=3000, seed=0):
    rng = random.Random(seed)
    flips = tot = tries = 0
    while tot < n and tries < n * 80:
        tries += 1
        bd = _rand_board(pieces, rng)
        if bd is None:
            continue
        bw = bd.copy(); bw.turn = W; bb = bd.copy(); bb.turn = B
        if not bw.is_valid() or not bb.is_valid():
            continue
        cw = _wclass(bw, tb); cb = _wclass(bb, tb)
        if cw is None or cb is None:
            continue
        tot += 1
        flips += (cw != cb)
    return (flips / tot if tot else 0.0), tot


MATERIALS = [
    ("KRvK",   [(K, W), (R, W), (K, B)]),
    ("KQvK",   [(K, W), (Q, W), (K, B)]),
    ("KPvK",   [(K, W), (P, W), (K, B)]),
    ("KQvKP",  [(K, W), (Q, W), (K, B), (P, B)]),
    ("KRvKP",  [(K, W), (R, W), (K, B), (P, B)]),
    ("KQvKQ",  [(K, W), (Q, W), (K, B), (Q, B)]),
    ("KRvKR",  [(K, W), (R, W), (K, B), (R, B)]),
    ("KRPvKR", [(K, W), (R, W), (P, W), (K, B), (R, B)]),
    ("KQPvKQ", [(K, W), (Q, W), (P, W), (K, B), (Q, B)]),
]


def main():
    tb = chess.syzygy.open_tablebase(TB_DIR)
    print(f"{'material':9s} {'forcing content (stm-flip WDL frac)':36s} {'n':>6s}")
    res = []
    for name, pc in MATERIALS:
        f, t = forcing(pc, tb); res.append((name, f))
        print(f"{name:9s} {f:6.3f}  {'#'*int(f*60):30s} {t:>6d}")
    fs = [f for _, f in res]
    print(f"\nrange: {min(fs):.3f} .. {max(fs):.3f}  (spread {max(fs)-min(fs):.3f})")


if __name__ == "__main__":
    main()
