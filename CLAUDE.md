# CLAUDE.md — SEC Analyzer

## Project Overview

SEC Analyzer is a terminal CLI tool that discovers US microcap stocks ($20-100M market cap), downloads their SEC filings from EDGAR, and runs two-tier LLM analysis to find hidden gems with improving fundamentals. Entry point: `python cli.py`. All 14 modules live in a flat directory structure — no packages, no subdirectories. After analysis, a feedback loop automatically collects price outcomes, computes alpha vs IWC benchmark, and calibrates signal weights via walk-forward validation.

## Architecture

```
cli.py  →  engine.py  →  scraper.py        (universe building, filing discovery, downloads)
                      →  llm_analysis.py    (Tier 1 screening + Tier 2 deep analysis)
                      →  forensics.py       (linguistic forensics: hedging, confidence, specificity)
                      →  filing_signals.py  (10 research-backed signals with academic citations)
                      →  insider_tracker.py (Form 4 insider buys/sells, SC 13D/G tracking)
                      →  inflection_detector.py (XBRL financial inflection points)
                      →  catalyst_extractor.py  (pattern-based turnaround catalyst extraction)
                      →  market_data.py     (price data, benchmark alpha, short interest, 13F, peer comparison)
                      →  calibration.py     (feedback loop: walk-forward validation, multi-window optimization, confidence scoring)
                      →  database.py        (SQLite adapter v2, parallel to JSON storage)
                      →  analyzer.py        (legacy batch analysis module)
                      →  monitor.py         (daily EDGAR monitor for new filings)
```

## Key Conventions

- **Logging**: Every module uses `logger = logging.getLogger(__name__)`. CLI sets level via `--verbose`.
- **Dual storage**: JSON files in `scraper_data/` (primary) + SQLite via `database.py` (parallel). Engine syncs both.
- **Rich UI**: CLI uses `rich` for tables/progress but falls back gracefully if not installed (`HAS_RICH` flag).
- **`--json` flag**: Every CLI command that displays data supports `--json` for machine-readable output.
- **Thread safety**: Analysis uses `ThreadPoolExecutor` with explicit locks (`_t1_lock`, `_t2_lock`).
- **Rate limiting**: `RateLimiter` class at 299 req/min for both FMP and EDGAR APIs.
- **Config persistence**: API keys stored in `config.json` via `engine.persist_key()`.

## CPS Scoring (5 dimensions, default weights)

| Dimension              | Default Weight | Max Raw |
|------------------------|---------------|---------|
| catalyst_strength      | 25            | 25      |
| improvement_trajectory | 20            | 20      |
| insider_conviction     | 20            | 20      |
| turnaround_indicators  | 20            | 20      |
| filing_recency         | 15            | 15      |

Weights are overridden by calibration results when available (`calibration.py`).

## Research Signal Weights (10 signals)

See `filing_signals.py` `SIGNAL_WEIGHTS` dict. Primary (60%): mda_similarity 20%, lm_sentiment 15%, risk_factor_changes 13%, positive_similarity 12%. Secondary (25%): sentiment_delta 10%, legal_proceedings 8%, uncertainty_language 7%. Tertiary (15%): filing_timeliness 5%, document_complexity 5%, insider_pattern 5%.

## LLM Models

Defined in `llm_analysis.py` `MODELS` dict:
- **Tier 1** (default): `gpt-4o-mini` — $0.15/$0.60 per 1M tokens
- **Tier 2** (default): `claude-haiku-4-5-20251001` — $1.00/$5.00 per 1M tokens
- **Tier 2** (optional): `claude-sonnet-4-5-20250929` — $3.00/$15.00
- **Tier 3**: `claude-opus-4-6` — $5.00/$25.00, supports extended thinking

## Data Paths

- `scraper_data/` — universe JSON, filings index, downloaded filings
- `results/` — per-filing T1/T2 JSON result files + signal vector snapshots (`*_signals_*.json`)
- `prepared/` — local analysis bundles: `t1/`, `t2/`, `texts/`, `t1_results/`, `t2_results/`, `manifest.json`
- `config.json` — API keys, watchlist, calibrated weights
- `sec_data/text_cache/` — cached cleaned filing text
- `sec_data/market_data/` — backtest results cache + benchmark cache (`benchmark_cache.json`)
- `sec_data/calibration/` — active weights (`active_weights.json`) + calibration history
- `sec_data/sec_analyzer.db` — SQLite database (schema v2: includes signal_vector/alpha columns)

---

## Documentation Update Mandate

**When modifying this project, Claude MUST update `README.md` if any change affects:**

1. **CLI commands or flags** — Any change to argparse definitions in `cli.py`
2. **Modules** — Adding, removing, or renaming any `.py` file
3. **Scoring logic** — CPS weights in `engine.py`, signal weights in `filing_signals.py`
4. **API keys or configuration** — Environment variables, `config.json` schema
5. **Data storage paths** — Directory structure, file naming conventions
6. **LLM models or costs** — `MODELS` dict in `llm_analysis.py`
7. **Dependencies** — `requirements.txt` changes
8. **Rate limits** — FMP or EDGAR rate limit constants in `scraper.py`

### How to Update README.md

- Preserve the existing section structure (do not reorganize sections)
- When updating the Module Reference table, verify line counts with `wc -l *.py`
- When updating CLI commands, verify against the argparse block in `cli.py` (lines 1042-1150)
- When updating costs, verify against `MODELS` dict in `llm_analysis.py` (lines 25-60)
- When updating CPS weights, verify against `_DEFAULT_WEIGHTS` in `engine.py` (line 239)
- When updating signal weights, verify against `SIGNAL_WEIGHTS` in `filing_signals.py` (lines 160-176)
- When updating calibration signals, verify against `SIGNAL_DEFINITIONS` in `calibration.py` (lines 48-95)
