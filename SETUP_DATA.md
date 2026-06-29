# Data setup

The experiments need **exact ground truth** and **real games**, neither bundled here (tablebases are
large; games are huge). You download them once into two gitignored folders.

## 1. Syzygy tablebases (exact win/draw/loss labels) — required
- What: the **3-4-5-man Syzygy WDL** tables (`.rtbw` files, ~1 GB total). ≤5 men covers every endgame
  used here (KRPvKR, KQPvKQ, and their sub-tables).
- Where to put them: `tablebases/syzygy/` in the repo root (the scripts use `TB_DIR = "tablebases/syzygy"`).
- How: download the standard 3-4-5-man Syzygy WDL set from any public Syzygy mirror (search
  "Syzygy 3-4-5 tablebases download"; the Lichess tablebase server and the standard chess-tablebase
  hosts distribute them). `python-chess` reads them via `chess.syzygy.open_tablebase`.
- Check: `python -c "import chess.syzygy as s; tb=s.open_tablebase('tablebases/syzygy'); print('ok')"`.

## 2. Lichess games (for the mining corpus) — required for mining/transfer
- What: one or more monthly files from the **Lichess open database**. The mining scripts expect
  parquet at `data/lichess/YYYY-MM.parquet` (e.g. `2013-01.parquet`; early-2013 months are smallest).
- Two sources:
  - **Parquet (what the code reads):** the Hugging Face mirror `Lichess/standard-chess-games`
    (download a month, save as `data/lichess/2013-01.parquet`). `load_positions` in `core.py` reads
    the `movetext` column.
  - **PGN (alternative):** `database.lichess.org` monthly PGN — you'd convert/adapt the loader.
- Note: filter hard (the code keeps only ≤5-man / KRPvKR positions); a single small month is plenty.

## Folder layout after setup
```
concept-rediscovery/
  tablebases/syzygy/   <- .rtbw files (gitignored)
  data/lichess/        <- YYYY-MM.parquet (gitignored)
  core.py, mine_run.py, ...
```

## Which scripts need which data
- **Need tablebases only** (they generate positions + label): `krpvkr_pilot.py`, `krpvkr_pilot2.py`,
  `forcing_content.py`, `transfer_test.py`, `nsv_transfer.py`, `separation_test.py`.
- **Need tablebases + Lichess games**: `mine_run.py`, `pressure_test.py`, `ho_test.py` (they mine from
  real game positions).

Scripts that can't find their data will error at the tablebase/parquet open — that's expected; install
the data above first.
