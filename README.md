# SEC Analyzer

**Terminal CLI tool for discovering hidden-gem microcap stocks through SEC filing analysis.**

Automatically screens US microcap companies ($20-100M market cap), downloads their SEC filings from EDGAR, and runs two-tier LLM analysis — fast bulk screening with GPT-4o-mini followed by deep forensic analysis with Claude — to surface companies with improving fundamentals, authentic management tone, and turnaround catalysts the market may be missing.

### Key Features

- **Automated universe building** from FMP screener (NYSE, NASDAQ, AMEX, OTC)
- **Multi-form filing coverage**: 10-K, 10-Q, 8-K, DEF 14A, SC 13D/G
- **Two-tier LLM analysis**: Tier 1 screens all filings cheaply; Tier 2 goes deep on top candidates
- **10 research-backed signals** with academic citations (Lazy Prices, Loughran-McDonald, etc.)
- **Forensic language analysis**: hedging ratios, confidence scoring, specificity metrics
- **Insider tracking**: Form 4 transactions, cluster buying detection, activist investor monitoring
- **Financial inflection detection**: XBRL-based revenue acceleration, profitability crossings, margin trends
- **Turnaround catalyst extraction**: Pattern-based detection across 8 catalyst categories
- **Calibration feedback loop**: Correlates signals with actual price outcomes to optimize weights
- **Benchmark alpha tracking**: Computes excess returns vs IWC micro-cap ETF across 30/60/90/180d windows
- **Walk-forward validation**: Temporal cross-validation to detect overfitting in weight calibration
- **Confidence scoring**: Empirical confidence intervals on gem scores based on historical backtest data
- **Auto outcome collection**: Automatically captures price outcomes and signal snapshots after analysis
- **Rich terminal UI** with tables, progress bars, and color — graceful fallback without `rich`

---

## Quick Start

There are two analysis workflows: **local** (recommended, uses Claude Code as the LLM — no per-call API cost) and **API-based** (sends filings to OpenAI/Anthropic APIs).

### Local workflow (recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set SEC_USER_AGENT (required by SEC EDGAR — use your email)
export SEC_USER_AGENT="yourname@example.com"

# 3. Set up API keys — only FMP required for local workflow
python cli.py setup

# 4. Build universe and download filings
python cli.py scan

# 5. Prepare analysis bundles (no API calls)
python cli.py prepare

# 6. Ask Claude Code to analyze the bundles in prepared/t1/ and prepared/t2/
#    Then import the results:
python cli.py import-analysis

# 7. View top picks
python cli.py gems
```

### API-based workflow (alternative)

Requires OpenAI and/or Anthropic API keys. Sends filing text to external LLMs.

```bash
python cli.py setup      # Set all API keys
python cli.py scan
python cli.py analyze    # Uses GPT-4o-mini (T1) + Claude Haiku (T2)
python cli.py gems
```

---

## Configuration

### API Keys

| Key | Env Variable | Required | Purpose |
|-----|-------------|----------|---------|
| FMP | `FMP_API_KEY` | Yes | Universe building via company screener. Requires Starter plan ($22/mo)+ |
| Anthropic | `ANTHROPIC_API_KEY` | API workflow only | Tier 2 deep analysis (Claude Haiku/Sonnet/Opus). Not needed for local Claude Code workflow |
| OpenAI | `OPENAI_API_KEY` | API workflow only | Tier 1 screening (GPT-4o-mini). Not needed for local Claude Code workflow |

Keys can be set via:
- `python cli.py setup` (interactive, saves to `config.json`)
- Environment variables (take precedence over config file)

### Other Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SEC_USER_AGENT` | *(none — must be set)* | **Required** by SEC EDGAR. Set to your email address |
| `SEC_DATA_DIR` | `sec_data` | Override base data directory |
| `TEXT_CACHE_DIR` | `sec_data/text_cache` | Override text cache location |
| `MARKET_DATA_DIR` | `sec_data/market_data` | Override market data cache location |

### config.json

Persisted automatically by `cli.py setup`. Schema:

```json
{
  "fmp_api_key": "...",
  "anthropic_api_key": "...",
  "openai_api_key": "...",
  "watchlist": ["TICKER1", "TICKER2"],
  "calibrated_weights": { ... }
}
```

---

## CLI Command Reference

All commands support `--json` for machine-readable output. Use `--verbose` / `-v` for debug logging.

### `setup`

Interactive API key configuration.

```bash
python cli.py setup
```

### `status`

Show pipeline state summary: universe size, filing counts, analysis progress, API key status.

```bash
python cli.py status [--json]
```

### `scan`

Run the scraper pipeline: build universe, find filings on EDGAR, download them, fetch insider data.

```bash
python cli.py scan [--step universe find download insider]
                   [--min-cap MIN] [--max-cap MAX]
                   [--max-companies N] [--max-downloads N]
                   [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--step` | all four | Run specific steps only |
| `--min-cap` | 20000000 | Minimum market cap in dollars |
| `--max-cap` | 100000000 | Maximum market cap in dollars |
| `--max-companies` | unlimited | Limit companies to scan |
| `--max-downloads` | unlimited | Limit filing downloads |

### `analyze`

Run Tier 1 + Tier 2 LLM analysis on downloaded filings. After analysis completes, automatically collects price outcomes for all analyzed filings and captures signal snapshots for future calibration.

```bash
python cli.py analyze [--t1-workers N] [--t2-workers N]
                      [--model MODEL] [--tier2-pct PCT]
                      [--effort LEVEL] [--reanalyze]
                      [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--t1-workers` | 6 | Tier 1 parallel workers |
| `--t2-workers` | 3 | Tier 2 parallel workers |
| `--model` | haiku | T2 model: `haiku`, `sonnet`, `opus`, `gpt4o-mini` |
| `--tier2-pct` | 0.25 | Top % of companies for Tier 2 (0.25 = top 25%) |
| `--effort` | none | Analysis effort: `low`, `medium`, `high`, `max` (affects thinking budget) |
| `--reanalyze` | false | Re-analyze already processed filings |

### `prepare`

Prepare analysis bundles for local Claude Code analysis (no API keys needed for the analysis step). Creates `prepared/` directory with T1 and/or T2 bundles containing all context needed for analysis.

```bash
python cli.py prepare                    # Auto-detect T1 or T2
python cli.py prepare --tier 1           # Force T1 preparation
python cli.py prepare --tier 2           # Force T2 preparation
python cli.py prepare --tier2-pct 0.10   # Top 10% for T2
python cli.py prepare --reanalyze        # Re-prepare already processed
python cli.py prepare --json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--tier` | auto | Force specific tier: `1` or `2` |
| `--tier2-pct` | 0.25 | Top % of companies for T2 (0.25 = top 25%) |
| `--reanalyze` | false | Re-prepare already processed filings |

### `import-analysis`

Import analysis results produced by Claude Code back into the system. Reads JSON from `prepared/t1_results/` and/or `prepared/t2_results/`.

```bash
python cli.py import-analysis            # Auto-detect from available results
python cli.py import-analysis --tier 1   # Import T1 only
python cli.py import-analysis --tier 2   # Import T2 only
python cli.py import-analysis --json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--tier` | auto | Import specific tier: `1` or `2` |

### `gems`

Show top gem candidates ranked by recency-weighted gem score.

```bash
python cli.py gems [--limit N] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--limit` | all | Maximum results to show (omit to show all) |

### `potential`

Show companies ranked by Company Potential Score (CPS).

```bash
python cli.py potential [--limit N] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--limit` | 30 | Maximum results to show |

### `company`

Deep dive on a single company: filings, scores, T2 analysis details.

```bash
python cli.py company TICKER [--json]
```

### `backtest`

Run price backtesting on analyzed filings. Shows 30/60/90/180-day returns with alpha vs IWC benchmark.

```bash
python cli.py backtest [--clear] [--all-filings] [--json]
```

| Flag | Description |
|------|-------------|
| `--clear` | Clear cached backtest results first |
| `--all-filings` | Include T1-only filings (not just T2) |

### `calibrate`

Run feedback loop calibration: correlate signals with actual price returns, suggest optimized weights. Supports walk-forward validation and multi-horizon analysis.

```bash
python cli.py calibrate [--window DAYS] [--apply] [--multi-window] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--window` | 90 | Return window in days |
| `--apply` | false | Auto-apply suggested weights |
| `--multi-window` | false | Analyze all return windows (30/60/90/180d) with signal performance matrix |

### `outcomes`

Collect and display price outcome status for analyzed filings. Shows which filings have mature return windows, pending windows, and next maturation dates.

```bash
python cli.py outcomes [--refresh] [--pending] [--json]
```

| Flag | Description |
|------|-------------|
| `--refresh` | Re-fetch all price data |
| `--pending` | Show only filings with immature windows |

### `portfolio`

Portfolio-level risk analysis for top gem picks. Shows position sizing, sector concentration, diversification metrics, and signal similarity between picks.

```bash
python cli.py portfolio [--top 10] [--max-sector 0.35] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--top` | 10 | Number of top picks to include |
| `--max-sector` | 0.35 | Maximum portfolio allocation to any one sector |

### `watchlist`

Manage a personal watchlist of tickers.

```bash
python cli.py watchlist [--add TICKER] [--remove TICKER] [--json]
```

### `export`

Export gem results to CSV.

```bash
python cli.py export [--output PATH]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `sec_data/export.csv` | Output file path |

### `backup`

Create a zip backup of the `sec_data/` directory.

```bash
python cli.py backup [--output PATH]
```

### Global Flags

| Flag | Description |
|------|-------------|
| `--json` | Output raw JSON (available on most commands) |
| `--verbose`, `-v` | Enable debug logging |

---

## Architecture Overview

```
                         ┌─────────────┐
                         │   cli.py    │  Terminal interface (16 commands)
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │  engine.py  │  Orchestration + auto outcome collection
                         └──────┬──────┘
                                │
       ┌────────────────────────┼────────────────────────┐
       │                        │                        │
┌──────▼──────┐          ┌──────▼──────┐          ┌──────▼──────┐
│ scraper.py  │          │llm_analysis │          │ database.py │
│ FMP + EDGAR │          │  T1 + T2    │          │ SQLite v2   │
└──────┬──────┘          └──────┬──────┘          └─────────────┘
       │                        │
       │    ┌───────────────────┼───────────────────┐
       │    │         │         │         │         │
       │  forensics  insider  inflection signals  catalysts
       │    │         │         │         │         │
       │    │         │         │         │         │
┌──────▼────▼─┐  ┌────▼──┐  ┌──▼───┐  ┌─▼──────┐  │
│ market_data │  │insider│  │infle-│  │filing_ │  │
│ + benchmark │  │tracker│  │ction │  │signals │  │
│ + alpha     │  └───────┘  └──────┘  └────────┘  │
└──────┬──────┘                              ┌─────▼──────┐
       │                                     │catalyst_   │
       │     ┌───────────────┐               │extractor   │
       └────►│calibration.py │               └────────────┘
             │walk-forward   │
             │multi-window   │
             │confidence     │
             └───────────────┘
```

### Two-Tier Analysis System

| Tier | Model (default) | Purpose | Cost | Context |
|------|----------------|---------|------|---------|
| Tier 1 | GPT-4o-mini | Fast bulk screening of all filings | $0.15/$0.60 per 1M tokens | 128K |
| Tier 2 | Claude Haiku 4.5 | Deep forensic analysis on top candidates | $1.00/$5.00 per 1M tokens | 200K |

**Alternative models** (via `--model`):

| Model | CLI Flag | Tier | Cost (input/output per 1M) | Context | Notes |
|-------|----------|------|---------------------------|---------|-------|
| Claude Sonnet 4.5 | `sonnet` | Tier 2 | $3.00 / $15.00 | 200K | Higher quality |
| Claude Opus 4.6 | `opus` | Tier 3 | $5.00 / $25.00 | 200K | Supports extended thinking + effort levels |

Tier 2 analysis integrates 7 data sources per filing: filing text, forensic metrics, filing-over-filing diff, insider activity, XBRL financial inflections, 10 research-backed signals, and turnaround catalyst extraction.

**Local analysis** (via `prepare` + `import-analysis`): Uses Claude Code itself as the LLM — T1 via Haiku subagents, T2 via Opus 4.6 with enhanced deep signal analysis (narrative cross-checking, omission analysis, information asymmetry detection, contrarian thesis). Zero additional API cost.

---

## Module Reference

| Module | Lines | Purpose |
|--------|------:|---------|
| `engine.py` | 2,628 | Orchestration: pipeline control, CPS computation, analysis coordination, prepare/import for local analysis, portfolio risk analysis, auto outcome collection with auto-calibration trigger, signal snapshots |
| `llm_analysis.py` | 2,037 | LLM integration: model config, API clients, T1/T2 prompts (including Opus deep analysis), JSON parsing, cost tracking |
| `scraper.py` | 1,949 | FMP universe building, EDGAR filing discovery (CIK + EFTS), downloading, text extraction |
| `filing_signals.py` | 1,904 | 10 research-backed signals + amendment/restatement detection + revenue concentration analysis |
| `calibration.py` | 1,521 | Feedback loop: 31 signals, temporal weighting, walk-forward validation, multi-window optimization, confidence scoring |
| `cli.py` | 1,262 | Terminal CLI: argparse definitions, Rich UI, command handlers (16 subcommands including `portfolio`) |
| `catalyst_extractor.py` | 1,200 | Pattern-based turnaround catalyst extraction across 8 categories (no LLM) |
| `market_data.py` | 1,220 | Price data (yfinance), benchmark returns (IWC), alpha computation, enhanced transcript analysis, short interest, 13F, peer comparison |
| `inflection_detector.py` | 1,085 | XBRL financial inflection points: revenue acceleration, profitability crossings, margin trends, FCF, DSO, debt/EBITDA, interest coverage, SBC%, deferred revenue |
| `analyzer.py` | 842 | Legacy batch analysis: flag detection, rule-based scoring |
| `database.py` | 769 | SQLite adapter: schema v2 with signal_vector/alpha columns, parallel storage to JSON |
| `insider_tracker.py` | 745 | Form 4 insider transactions, SC 13D/G tracking, accumulation scoring, cluster detection |
| `monitor.py` | 561 | Daily EDGAR monitor: EFTS polling for new filings, auto-analysis trigger |
| `forensics.py` | 547 | Linguistic forensics: hedging ratio, confidence score, specificity, readability (no LLM) |
| **Total** | **18,270** | |

---

## Scoring Systems

### Company Potential Score (CPS)

CPS ranks companies on a 0-100 scale across 5 weighted dimensions. Weights default to the values below but can be calibrated via `python cli.py calibrate --apply`.

| Dimension | Default Weight | Max Raw | What It Measures |
|-----------|---------------|---------|-----------------|
| Catalyst Strength | 25 | 25 | Best T1 composite score + gem potential level |
| Improvement Trajectory | 20 | 20 | Score delta between newest and oldest filings |
| Insider Conviction | 20 | 20 | Accumulation score, cluster buying, activist holders |
| Turnaround Indicators | 20 | 20 | Turnaround score, research signals, diff signals, SC 13D presence |
| Filing Recency | 15 | 15 | Days since newest filing + filing count + form type coverage |

Each raw component is normalized to its weight's fraction, then summed to produce CPS 0-100.

### Research Signal Score (10 signals)

Composite score from `filing_signals.py`, weighted by academic evidence strength:

**Primary Signals (60% combined weight):**

| # | Signal | Weight | Source | Key Finding |
|---|--------|--------|--------|-------------|
| 1 | MD&A Similarity | 20% | Lazy Prices (Cohen, Malloy, Nguyen — JF 2020) | 188bps/month alpha; high similarity = positive |
| 2 | LM Sentiment | 15% | Loughran & McDonald (JF 2011) | R-squared 9.75%; finance-specific word lists |
| 3 | Risk Factor Changes | 13% | Lazy Prices + Kirtac & Germano (2024) | "Especially informative for future returns" |
| 4 | Positive Similarity | 12% | Positive Similarity of Company Filings (2022) | Uncorrelated to FF5; low similarity = bullish |

**Secondary Signals (25% combined weight):**

| # | Signal | Weight | Source | Key Finding |
|---|--------|--------|--------|-------------|
| 5 | Sentiment Delta | 10% | Brown & Tucker (2011); Bochkay (2014) | Period-over-period sentiment change |
| 6 | Legal Proceedings | 8% | Offutt & Xie (2025) | Alpha unexplained by FF5+momentum |
| 7 | Uncertainty Language | 7% | LM uncertainty word list | Predicts IPO returns, volatility |

**Tertiary Signals (15% combined weight):**

| # | Signal | Weight | Source | Key Finding |
|---|--------|--------|--------|-------------|
| 8 | Filing Timeliness | 5% | Duarte-Silva et al. (2013) | Late filers = performance deterioration |
| 9 | Document Complexity | 5% | Li (2008); Lo et al. (2017) | Opacity = negative signal |
| 10 | Insider Pattern | 5% | Ozlen & Batumoglu (2025) | 70-80% alpha pre-disclosure; cluster signals useful |

Each signal returns 0-100 (>60 positive, 40-60 neutral, <40 negative). Composite uses evidence-strength-weighted average, re-normalized for signals with available data.

### Tier 1 Scoring

8 dimensions scored 1-10 by LLM, multiplied by 10 for composite 0-100:
- **Tone**: confidence, transparency, consistency, operational_focus, capital_discipline
- **Financial**: liquidity_health, growth_quality, profitability_trend

Specialized prompts for: 10-K/10-Q (standard), SC 13D/G (ownership/activist), 8-K (events), DEF 14A (governance).

### Tier 2 Scoring

Deep analysis producing: `final_gem_score` (0-100), `conviction_level` (high/medium/low), `recommendation` (strong_buy/buy/hold/avoid), plus structured assessments of management authenticity, competitive position, capital allocation, turnaround phase, and research signal confirmation.

**Extracted T2 sub-scores** (indexed on filing record for signal-level backtesting):
- `mgmt_authenticity_score` (1-10) — Management tone authenticity
- `competitive_position_score` (1-10) — Competitive moat assessment
- `capital_allocation_score` (1-10) — Capital deployment quality
- `turnaround_conviction` — Overall turnaround conviction level
- `catalyst_stacking_count` (0-8) — Number of active catalyst categories
- `revenue_momentum` — Revenue trajectory direction
- `profitability_trajectory` — Profitability trend direction
- `dilution_risk` — Equity dilution risk assessment

### Calibration Signal Definitions (31 signals)

The calibration engine (`calibration.py`) tracks 31 signals in `SIGNAL_DEFINITIONS`, grouped by source:
- **CPS components** (5): catalyst_strength, improvement_trajectory, insider_conviction, turnaround_indicators, filing_recency
- **T1 outputs** (2): best_t1_score, gem_potential_numeric
- **Insider signals** (4): accumulation_score, cluster_score, unique_buyers, has_activist
- **Filing-level** (10): turnaround_score, research_signal_score, diff_signal_numeric, filing_count, has_8k, has_proxy, has_ownership, days_since_filing, has_amendment, revenue_concentration_score
- **T2 outputs** (2): final_gem_score, conviction_numeric
- **T2 sub-scores** (4): mgmt_authenticity_score, competitive_position_score, capital_allocation_score, catalyst_stacking_count
- **Opus deep signals** (4): has_contrarian_thesis, asymmetry_level_numeric, narrative_contradictions, non_obvious_catalyst_count

### Feedback Loop & Backtesting

The system builds an automated feedback loop to validate its predictions:

1. **Outcome Collection**: After each analysis run, `auto_collect_outcomes()` fetches price data for all analyzed filings and records `baseline_price` at analysis time (survivor bias protection)
2. **Benchmark Alpha**: Returns are compared against IWC (iShares Micro-Cap ETF) across 30/60/90/180-day windows, computing excess alpha
3. **Signal Snapshots**: Full signal vectors are captured at analysis time and stored as `results/TICKER_signals_DATE.json`
4. **Walk-Forward Validation**: Temporal cross-validation prevents overfitting — trains on older data, tests on newer (engages at N >= 15 filings)
5. **Multi-Window Optimization**: Different signals predict at different horizons; `--multi-window` blends weights across all 4 windows
6. **Confidence Scoring**: Gem scores include empirical confidence intervals from historical backtest data by score bucket
7. **Temporal Signal Weighting**: Recent signals weigh more than old ones (exponential decay, 180-day half-life) in correlation analysis
8. **Auto-Calibration Trigger**: When `auto_collect_outcomes()` detects sufficient new data (>=15 companies with returns, >=5 new since last calibration or >=30 days elapsed), it automatically runs full calibration

**Bootstrap timeline** (fresh install):

| Milestone | When | What Unlocks |
|-----------|------|-------------|
| First `analyze` | Day 1 | Baseline prices captured, signal snapshots saved |
| 30d windows mature | Day 30+ | First calibration possible (if N >= 10) |
| N >= 15 filings | Varies | Walk-forward validation engages |
| 90d windows mature | Day 90+ | Primary evaluation window |
| 180d windows mature | Day 180+ | Full multi-window optimization |

---

## Data Storage

```
sec_analyzer/
├── config.json                    # API keys, watchlist, calibrated weights
├── results/                       # Per-filing analysis results
│   ├── TICKER_tier1_DATE.json
│   ├── TICKER_tier2_DATE.json
│   └── TICKER_signals_DATE.json   # Signal vector snapshots
├── prepared/                      # Local analysis bundles (Claude Code workflow)
│   ├── manifest.json              # Bundle manifest with counts
│   ├── t1/                        # T1 analysis bundles
│   ├── t2/                        # T2 analysis bundles (with all context)
│   ├── texts/                     # Pre-truncated filing text (≤80K chars)
│   ├── t1_results/                # Claude Code writes T1 results here
│   └── t2_results/                # Claude Code writes T2 results here
├── scraper_data/                  # Scraper state and filings
│   ├── company_universe.json      # FMP universe snapshot
│   ├── filings_index.json         # Filing metadata + analysis state
│   └── filings/                   # Downloaded filing HTML files
│       └── TICKER_FORM_DATE.html
├── sec_data/
│   ├── text_cache/                # Pre-extracted clean text from filings
│   ├── market_data/               # Backtest results and benchmark cache
│   │   ├── backtest_results.json  # Cached backtest results with alpha
│   │   └── benchmark_cache.json   # IWC/IWM benchmark return cache
│   ├── calibration/               # Calibration history and active weights
│   │   ├── active_weights.json    # Currently applied CPS weights
│   │   └── history/               # Past calibration reports
│   ├── sec_analyzer.db            # SQLite database (schema v2)
│   └── export.csv                 # CSV export output
└── sec_data_backup_TIMESTAMP.zip  # Backup archives
```

---

## Cost and Rate Limits

### LLM Costs per Filing (estimates)

| Tier | Model | Avg Input Tokens | Avg Output Tokens | Est. Cost/Filing |
|------|-------|-----------------|-------------------|-----------------|
| Tier 1 | GPT-4o-mini | ~30,000 | ~1,500 | ~$0.005 |
| Tier 2 | Claude Haiku 4.5 | ~35,000 | ~3,000 | ~$0.050 |
| Tier 2 | Claude Sonnet 4.5 | ~35,000 | ~3,000 | ~$0.150 |
| Tier 2 | Claude Opus 4.6 | ~35,000 | ~3,000 | ~$0.250 |

**Example run** (200 filings, top 25% to Tier 2):
- Tier 1: 200 filings x $0.005 = ~$1.00
- Tier 2: 50 filings x $0.05 = ~$2.50
- **Total: ~$3.50** (with default Haiku)

### Rate Limits

| API | Limit | Configured As |
|-----|-------|--------------|
| FMP | 300 req/min (Starter plan) | 299 req/min (`scraper.py`) |
| SEC EDGAR | 10 req/sec (600/min) | 299 req/min (`scraper.py`, conservative) |

EDGAR endpoints are free — only require `User-Agent` header with contact email.

---

## Workflows

### First-Time Full Scan

```bash
python cli.py setup                          # Set API keys
python cli.py scan                           # Build universe + find + download + insider
python cli.py analyze                        # Run T1 + T2 analysis
python cli.py gems                           # View top picks
python cli.py company TICKER                 # Deep dive on interesting company
```

### Incremental Update

```bash
python cli.py scan --step find download      # Find and download new filings only
python cli.py analyze                        # Only analyzes unprocessed filings
python cli.py gems --json > latest.json      # Export results
```

### Calibration Loop

```bash
python cli.py outcomes                       # Collect price outcomes for all analyzed filings
python cli.py backtest                       # Get price returns with alpha vs benchmark
python cli.py backtest --all-filings         # Include T1-only filings too
python cli.py calibrate                      # Correlate signals with returns
python cli.py calibrate --multi-window       # Analyze all return horizons (30/60/90/180d)
python cli.py calibrate --apply              # Apply optimized weights
python cli.py potential                      # View re-ranked companies with new weights
```

### Local Analysis Workflow (Claude Code — No API Keys)

Use Claude Code itself as the LLM backend, eliminating external API costs for the analysis step. The workflow is: **prepare** (gather all non-LLM data) → **analyze** (Claude Code reads data, produces results) → **import** (feed results back into the system).

```bash
# Step 1: Prepare bundles (gathers all non-LLM context)
python cli.py prepare                    # Auto-detect T1 or T2
python cli.py prepare --tier 1           # T1 only
python cli.py prepare --tier 2           # T2 only (requires T1 results)

# Step 2: Ask Claude Code to analyze the bundles
#   T1: Haiku subagents read prepared/t1/*.json + filing text, write to prepared/t1_results/
#   T2: Opus reads prepared/t2/*.json + filing text, produces deep analysis with
#       opus_deep_signals (narrative cross-check, omission analysis, contrarian thesis, etc.)
#   Results are written to prepared/t2_results/

# Step 3: Import results back into the system
python cli.py import-analysis            # Auto-detect from available results
python cli.py import-analysis --tier 1   # Import T1 only
python cli.py import-analysis --tier 2   # Import T2 only

# Step 4: View results (same as API workflow)
python cli.py gems
python cli.py company TICKER
```

**T2 bundles** include Opus-specific deep signal instructions that guide Claude Code to perform:
- Narrative vs. numbers cross-checking (verify management claims against financial data)
- Omission analysis (what's NOT being said)
- Information asymmetry detection (what management knows that the market doesn't)
- Second-order implications (if X then Y reasoning)
- Language evolution patterns (hedging/specificity/confidence trends across filings)
- Contrarian signal detection (where consensus might be wrong)
- Footnote mining (buried disclosures)

These Opus-specific signals are indexed for backtesting: `has_contrarian_thesis`, `asymmetry_level`, `narrative_contradictions`, `non_obvious_catalyst_count`.

### Custom Parameters

```bash
# Smaller universe, higher quality analysis
python cli.py scan --min-cap 50000000 --max-cap 100000000 --max-companies 100
python cli.py analyze --model opus --tier2-pct 0.5 --effort high --t2-workers 2

# Quick screen only (no deep analysis)
python cli.py scan --step universe find download
python cli.py analyze --t1-workers 10
python cli.py potential --limit 50 --json
```

---

## Dependencies

```
requests>=2.28.0        # HTTP client for FMP + EDGAR APIs
beautifulsoup4>=4.12.0  # HTML parsing for filing text extraction
anthropic>=0.39.0       # Claude API client (Tier 2 analysis)
openai>=1.0.0           # OpenAI API client (Tier 1 screening)
yfinance>=0.2.0         # Price data for backtesting
rich>=13.0.0            # Terminal UI (optional — graceful fallback)
```
