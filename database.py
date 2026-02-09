"""
SEC Analyzer - SQLite Database Adapter (#24)
=============================================
Provides SQLite-backed storage for filings index, universe, and analysis results.
Works alongside the existing JSON file system for backward compatibility.
Gives filtering, pagination, and crash resilience for free.

Usage:
    db = SECDatabase(data_dir="sec_data")
    db.import_from_json(universe, filings_index)  # One-time migration
    db.upsert_filing({...})
    results = db.query_filings(ticker="ACME", analyzed=True)
"""

import os
import json
import sqlite3
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Universe: companies tracked
CREATE TABLE IF NOT EXISTS universe (
    ticker TEXT PRIMARY KEY,
    company_name TEXT,
    cik TEXT,
    exchange TEXT,
    market_cap REAL,
    sector TEXT,
    industry TEXT,
    scanned_form_types TEXT,  -- JSON array
    added_at TEXT,
    extra TEXT  -- JSON blob for any additional fields
);

-- Filings index: all known filings
CREATE TABLE IF NOT EXISTS filings (
    accession_number TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    company_name TEXT,
    cik TEXT,
    form_type TEXT,
    filing_date TEXT,
    filing_url TEXT,
    downloaded INTEGER DEFAULT 0,
    local_path TEXT,
    download_retries INTEGER DEFAULT 0,
    analyzed INTEGER DEFAULT 0,
    llm_analyzed INTEGER DEFAULT 0,
    tier2_analyzed INTEGER DEFAULT 0,
    score REAL,
    tier1_score REAL,
    final_gem_score REAL,
    gem_potential TEXT,
    conviction TEXT,
    recommendation TEXT,
    diff_signal TEXT,
    insider_signal TEXT,
    inflection_signal TEXT,
    composite_score REAL,
    turnaround_score REAL,
    research_signal_score REAL,
    selected_for_tier2 INTEGER DEFAULT 0,
    result_file TEXT,
    tier1_result_file TEXT,
    tier2_result_file TEXT,
    analyzed_at TEXT,
    exchange TEXT,
    extra TEXT  -- JSON blob for additional fields
);

-- Insider data per company
CREATE TABLE IF NOT EXISTS insider_data (
    ticker TEXT PRIMARY KEY,
    data TEXT NOT NULL,  -- Full JSON blob
    fetched_at TEXT
);

-- Watchlist
CREATE TABLE IF NOT EXISTS watchlist (
    ticker TEXT PRIMARY KEY,
    added_at TEXT
);

-- Backtest results cache
CREATE TABLE IF NOT EXISTS backtest_results (
    ticker TEXT,
    filing_date TEXT,
    result TEXT,  -- JSON blob
    computed_at TEXT,
    PRIMARY KEY (ticker, filing_date)
);

-- Analysis progress (crash recovery)
CREATE TABLE IF NOT EXISTS analysis_progress (
    key TEXT PRIMARY KEY,
    value TEXT,  -- JSON blob
    updated_at TEXT
);

-- Cost tracking
CREATE TABLE IF NOT EXISTS cost_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    estimated_cost REAL,
    ticker TEXT,
    analysis_tier INTEGER,
    logged_at TEXT
);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_filings_ticker ON filings(ticker);
CREATE INDEX IF NOT EXISTS idx_filings_form_type ON filings(form_type);
CREATE INDEX IF NOT EXISTS idx_filings_date ON filings(filing_date);
CREATE INDEX IF NOT EXISTS idx_filings_downloaded ON filings(downloaded);
CREATE INDEX IF NOT EXISTS idx_filings_analyzed ON filings(llm_analyzed);
CREATE INDEX IF NOT EXISTS idx_filings_tier2 ON filings(tier2_analyzed);
CREATE INDEX IF NOT EXISTS idx_filings_gem_score ON filings(final_gem_score);
CREATE INDEX IF NOT EXISTS idx_filings_composite ON filings(composite_score);
CREATE INDEX IF NOT EXISTS idx_universe_exchange ON universe(exchange);
CREATE INDEX IF NOT EXISTS idx_cost_log_ticker ON cost_log(ticker);
"""

# Fields that map directly to columns (not stored in 'extra')
FILING_COLUMNS = {
    'accession_number', 'ticker', 'company_name', 'cik', 'form_type',
    'filing_date', 'filing_url', 'downloaded', 'local_path', 'download_retries',
    'analyzed', 'llm_analyzed', 'tier2_analyzed', 'score', 'tier1_score',
    'final_gem_score', 'gem_potential', 'conviction', 'recommendation',
    'diff_signal', 'insider_signal', 'inflection_signal', 'composite_score',
    'turnaround_score', 'research_signal_score', 'selected_for_tier2',
    'result_file', 'tier1_result_file', 'tier2_result_file', 'analyzed_at',
    'exchange', 'extra',
}

UNIVERSE_COLUMNS = {
    'ticker', 'company_name', 'cik', 'exchange', 'market_cap', 'sector',
    'industry', 'scanned_form_types', 'added_at', 'extra',
}


class SECDatabase:
    """SQLite-backed storage with JSON backward compatibility."""

    def __init__(self, data_dir: str = "sec_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.db_path = os.path.join(data_dir, "sec_analyzer.db")
        self._init_db()

    def _init_db(self):
        """Create schema if needed."""
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)
            # Set schema version
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION))
            )
            conn.commit()
        logger.info(f"SQLite database initialized at {self.db_path}")

    @contextmanager
    def _conn(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # IMPORT / EXPORT (JSON ↔ SQLite)
    # ------------------------------------------------------------------

    def import_from_json(self, universe: List[Dict], filings_index: List[Dict]) -> Dict:
        """One-time migration from JSON to SQLite. Returns import stats."""
        stats = {'universe': 0, 'filings': 0, 'skipped': 0}

        with self._conn() as conn:
            # Import universe
            for c in universe:
                try:
                    self._upsert_universe_row(conn, c)
                    stats['universe'] += 1
                except Exception as e:
                    logger.debug(f"Skip universe {c.get('ticker')}: {e}")
                    stats['skipped'] += 1

            # Import filings
            for f in filings_index:
                try:
                    self._upsert_filing_row(conn, f)
                    stats['filings'] += 1
                except Exception as e:
                    logger.debug(f"Skip filing {f.get('accession_number')}: {e}")
                    stats['skipped'] += 1

            conn.commit()

        logger.info(f"SQLite import: {stats}")
        return stats

    def export_to_json(self) -> Tuple[List[Dict], List[Dict]]:
        """Export SQLite data back to JSON format for backward compat."""
        with self._conn() as conn:
            universe = [self._row_to_universe_dict(r)
                       for r in conn.execute("SELECT * FROM universe").fetchall()]
            filings = [self._row_to_filing_dict(r)
                      for r in conn.execute("SELECT * FROM filings ORDER BY filing_date DESC").fetchall()]
        return universe, filings

    # ------------------------------------------------------------------
    # UNIVERSE OPERATIONS
    # ------------------------------------------------------------------

    def upsert_company(self, company: Dict):
        with self._conn() as conn:
            self._upsert_universe_row(conn, company)
            conn.commit()

    def upsert_companies(self, companies: List[Dict]):
        with self._conn() as conn:
            for c in companies:
                self._upsert_universe_row(conn, c)
            conn.commit()

    def get_universe(self, exchange: str = "", page: int = 1,
                     per_page: int = 50) -> Tuple[List[Dict], int]:
        with self._conn() as conn:
            where = []
            params = []
            if exchange:
                where.append("exchange = ?")
                params.append(exchange)

            where_sql = f"WHERE {' AND '.join(where)}" if where else ""
            total = conn.execute(
                f"SELECT COUNT(*) FROM universe {where_sql}", params
            ).fetchone()[0]

            offset = (page - 1) * per_page
            rows = conn.execute(
                f"SELECT * FROM universe {where_sql} ORDER BY ticker LIMIT ? OFFSET ?",
                params + [per_page, offset]
            ).fetchall()

            return [self._row_to_universe_dict(r) for r in rows], total

    def get_company(self, ticker: str) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM universe WHERE ticker = ?", (ticker.upper(),)
            ).fetchone()
            return self._row_to_universe_dict(row) if row else None

    # ------------------------------------------------------------------
    # FILING OPERATIONS
    # ------------------------------------------------------------------

    def upsert_filing(self, filing: Dict):
        with self._conn() as conn:
            self._upsert_filing_row(conn, filing)
            conn.commit()

    def upsert_filings(self, filings: List[Dict]):
        with self._conn() as conn:
            for f in filings:
                self._upsert_filing_row(conn, f)
            conn.commit()

    def get_filing(self, accession_number: str) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM filings WHERE accession_number = ?",
                (accession_number,)
            ).fetchone()
            return self._row_to_filing_dict(row) if row else None

    def query_filings(self, ticker: str = "", form_type: str = "",
                      downloaded: Optional[bool] = None,
                      analyzed: Optional[bool] = None,
                      tier2: Optional[bool] = None,
                      page: int = 1, per_page: int = 50,
                      sort_by: str = "filing_date",
                      sort_dir: str = "DESC") -> Tuple[List[Dict], int]:
        """Flexible filing query with filters and pagination."""
        with self._conn() as conn:
            where = []
            params = []

            if ticker:
                where.append("ticker = ?")
                params.append(ticker.upper())
            if form_type:
                where.append("form_type = ?")
                params.append(form_type)
            if downloaded is not None:
                where.append("downloaded = ?")
                params.append(1 if downloaded else 0)
            if analyzed is not None:
                where.append("llm_analyzed = ?")
                params.append(1 if analyzed else 0)
            if tier2 is not None:
                where.append("tier2_analyzed = ?")
                params.append(1 if tier2 else 0)

            where_sql = f"WHERE {' AND '.join(where)}" if where else ""

            # Validate sort column
            valid_sorts = {'filing_date', 'ticker', 'composite_score',
                          'final_gem_score', 'tier1_score', 'form_type'}
            if sort_by not in valid_sorts:
                sort_by = 'filing_date'
            if sort_dir.upper() not in ('ASC', 'DESC'):
                sort_dir = 'DESC'

            total = conn.execute(
                f"SELECT COUNT(*) FROM filings {where_sql}", params
            ).fetchone()[0]

            offset = (page - 1) * per_page
            rows = conn.execute(
                f"SELECT * FROM filings {where_sql} ORDER BY {sort_by} {sort_dir} LIMIT ? OFFSET ?",
                params + [per_page, offset]
            ).fetchall()

            return [self._row_to_filing_dict(r) for r in rows], total

    def get_filings_for_ticker(self, ticker: str) -> List[Dict]:
        """Get all filings for a company."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM filings WHERE ticker = ? ORDER BY filing_date DESC",
                (ticker.upper(),)
            ).fetchall()
            return [self._row_to_filing_dict(r) for r in rows]

    def get_unanalyzed_filings(self, max_retries: int = 3) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM filings WHERE downloaded = 1 AND llm_analyzed = 0 "
                "AND download_retries < ? ORDER BY filing_date DESC",
                (max_retries,)
            ).fetchall()
            return [self._row_to_filing_dict(r) for r in rows]

    def get_top_filings(self, score_field: str = "composite_score",
                        limit: int = 100) -> List[Dict]:
        valid = {'composite_score', 'final_gem_score', 'tier1_score'}
        if score_field not in valid:
            score_field = 'composite_score'
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM filings WHERE {score_field} IS NOT NULL "
                f"ORDER BY {score_field} DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [self._row_to_filing_dict(r) for r in rows]

    def count_filings(self, **filters) -> int:
        with self._conn() as conn:
            where = []
            params = []
            for k, v in filters.items():
                if k in FILING_COLUMNS:
                    where.append(f"{k} = ?")
                    params.append(v)
            where_sql = f"WHERE {' AND '.join(where)}" if where else ""
            return conn.execute(
                f"SELECT COUNT(*) FROM filings {where_sql}", params
            ).fetchone()[0]

    def get_filing_stats(self) -> Dict:
        """Get aggregated statistics for the dashboard."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
            downloaded = conn.execute("SELECT COUNT(*) FROM filings WHERE downloaded=1").fetchone()[0]
            analyzed = conn.execute("SELECT COUNT(*) FROM filings WHERE llm_analyzed=1").fetchone()[0]
            tier2 = conn.execute("SELECT COUNT(*) FROM filings WHERE tier2_analyzed=1").fetchone()[0]
            tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM filings").fetchone()[0]

            # Form type breakdown
            form_types = {}
            for row in conn.execute(
                "SELECT form_type, COUNT(*) as cnt FROM filings GROUP BY form_type ORDER BY cnt DESC"
            ).fetchall():
                form_types[row['form_type']] = row['cnt']

            return {
                'total': total,
                'downloaded': downloaded,
                'analyzed': analyzed,
                'tier2_analyzed': tier2,
                'unique_tickers': tickers,
                'form_types': form_types,
            }

    # ------------------------------------------------------------------
    # INSIDER DATA
    # ------------------------------------------------------------------

    def save_insider_data(self, ticker: str, data: Dict):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO insider_data (ticker, data, fetched_at) VALUES (?, ?, ?)",
                (ticker.upper(), json.dumps(data), datetime.now().isoformat())
            )
            conn.commit()

    def get_insider_data(self, ticker: str) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT data FROM insider_data WHERE ticker = ?", (ticker.upper(),)
            ).fetchone()
            return json.loads(row['data']) if row else None

    # ------------------------------------------------------------------
    # WATCHLIST
    # ------------------------------------------------------------------

    def get_watchlist(self) -> List[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT ticker FROM watchlist ORDER BY added_at").fetchall()
            return [r['ticker'] for r in rows]

    def add_to_watchlist(self, ticker: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO watchlist (ticker, added_at) VALUES (?, ?)",
                (ticker.upper(), datetime.now().isoformat())
            )
            conn.commit()

    def remove_from_watchlist(self, ticker: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))
            conn.commit()

    # ------------------------------------------------------------------
    # BACKTEST RESULTS
    # ------------------------------------------------------------------

    def save_backtest(self, ticker: str, filing_date: str, result: Dict):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO backtest_results (ticker, filing_date, result, computed_at) "
                "VALUES (?, ?, ?, ?)",
                (ticker, filing_date, json.dumps(result), datetime.now().isoformat())
            )
            conn.commit()

    def get_backtests(self, limit: int = 200) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM backtest_results ORDER BY computed_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            results = []
            for r in rows:
                d = json.loads(r['result'])
                d['ticker'] = r['ticker']
                d['filing_date'] = r['filing_date']
                results.append(d)
            return results

    # ------------------------------------------------------------------
    # COST TRACKING
    # ------------------------------------------------------------------

    def log_cost(self, model: str, input_tokens: int, output_tokens: int,
                 estimated_cost: float, ticker: str = "", tier: int = 0):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO cost_log (model, input_tokens, output_tokens, estimated_cost, "
                "ticker, analysis_tier, logged_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (model, input_tokens, output_tokens, estimated_cost,
                 ticker, tier, datetime.now().isoformat())
            )
            conn.commit()

    def get_cost_summary(self) -> Dict:
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COALESCE(SUM(estimated_cost), 0) as total, "
                "COALESCE(SUM(input_tokens), 0) as inp, "
                "COALESCE(SUM(output_tokens), 0) as outp, "
                "COUNT(*) as calls FROM cost_log"
            ).fetchone()

            by_model = {}
            for r in conn.execute(
                "SELECT model, SUM(estimated_cost) as cost, COUNT(*) as calls, "
                "SUM(input_tokens) as inp, SUM(output_tokens) as outp "
                "FROM cost_log GROUP BY model"
            ).fetchall():
                by_model[r['model']] = {
                    'cost': round(r['cost'], 4),
                    'calls': r['calls'],
                    'input_tokens': r['inp'],
                    'output_tokens': r['outp'],
                }

            by_tier = {}
            for r in conn.execute(
                "SELECT analysis_tier, SUM(estimated_cost) as cost, COUNT(*) as calls "
                "FROM cost_log WHERE analysis_tier > 0 GROUP BY analysis_tier"
            ).fetchall():
                by_tier[f"tier{r['analysis_tier']}"] = {
                    'cost': round(r['cost'], 4),
                    'calls': r['calls'],
                }

            return {
                'total_cost': round(total['total'], 4),
                'total_input_tokens': total['inp'],
                'total_output_tokens': total['outp'],
                'total_calls': total['calls'],
                'by_model': by_model,
                'by_tier': by_tier,
            }

    # ------------------------------------------------------------------
    # ANALYSIS PROGRESS (crash recovery)
    # ------------------------------------------------------------------

    def save_progress(self, key: str, value: Any):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO analysis_progress (key, value, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(value), datetime.now().isoformat())
            )
            conn.commit()

    def load_progress(self, key: str) -> Optional[Any]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM analysis_progress WHERE key = ?", (key,)
            ).fetchone()
            return json.loads(row['value']) if row else None

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _upsert_universe_row(self, conn, company: Dict):
        ticker = company.get('ticker', '').upper()
        if not ticker:
            return

        extra_fields = {k: v for k, v in company.items() if k not in UNIVERSE_COLUMNS}
        sft = company.get('scanned_form_types')
        if isinstance(sft, list):
            sft = json.dumps(sft)

        conn.execute("""
            INSERT OR REPLACE INTO universe
            (ticker, company_name, cik, exchange, market_cap, sector, industry,
             scanned_form_types, added_at, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker,
            company.get('company_name', ''),
            company.get('cik', ''),
            company.get('exchange', ''),
            company.get('market_cap'),
            company.get('sector', ''),
            company.get('industry', ''),
            sft,
            company.get('added_at', datetime.now().isoformat()),
            json.dumps(extra_fields) if extra_fields else None,
        ))

    def _upsert_filing_row(self, conn, filing: Dict):
        acc = filing.get('accession_number', '')
        if not acc:
            return

        extra_fields = {k: v for k, v in filing.items() if k not in FILING_COLUMNS}

        conn.execute("""
            INSERT OR REPLACE INTO filings
            (accession_number, ticker, company_name, cik, form_type, filing_date,
             filing_url, downloaded, local_path, download_retries,
             analyzed, llm_analyzed, tier2_analyzed,
             score, tier1_score, final_gem_score, gem_potential,
             conviction, recommendation, diff_signal, insider_signal,
             inflection_signal, composite_score, turnaround_score,
             research_signal_score, selected_for_tier2,
             result_file, tier1_result_file, tier2_result_file,
             analyzed_at, exchange, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            acc,
            filing.get('ticker', '').upper(),
            filing.get('company_name', ''),
            filing.get('cik', ''),
            filing.get('form_type', ''),
            filing.get('filing_date', ''),
            filing.get('filing_url', ''),
            1 if filing.get('downloaded') else 0,
            filing.get('local_path', ''),
            filing.get('download_retries', 0),
            1 if filing.get('analyzed') else 0,
            1 if filing.get('llm_analyzed') else 0,
            1 if filing.get('tier2_analyzed') else 0,
            filing.get('score'),
            filing.get('tier1_score'),
            filing.get('final_gem_score'),
            filing.get('gem_potential'),
            filing.get('conviction'),
            filing.get('recommendation'),
            filing.get('diff_signal'),
            filing.get('insider_signal'),
            filing.get('inflection_signal'),
            filing.get('composite_score'),
            filing.get('turnaround_score'),
            filing.get('research_signal_score'),
            1 if filing.get('selected_for_tier2') else 0,
            filing.get('result_file', ''),
            filing.get('tier1_result_file', ''),
            filing.get('tier2_result_file', ''),
            filing.get('analyzed_at', ''),
            filing.get('exchange', ''),
            json.dumps(extra_fields) if extra_fields else None,
        ))

    def _row_to_filing_dict(self, row) -> Dict:
        """Convert SQLite Row to dict, merging extra fields."""
        if not row:
            return {}
        d = dict(row)
        # Convert int booleans back
        for bk in ('downloaded', 'analyzed', 'llm_analyzed', 'tier2_analyzed', 'selected_for_tier2'):
            if bk in d:
                d[bk] = bool(d[bk])
        # Merge extra
        extra = d.pop('extra', None)
        if extra:
            try:
                d.update(json.loads(extra))
            except (json.JSONDecodeError, TypeError):
                pass
        return d

    def _row_to_universe_dict(self, row) -> Dict:
        """Convert SQLite Row to dict, merging extra fields."""
        if not row:
            return {}
        d = dict(row)
        # Parse scanned_form_types JSON
        sft = d.get('scanned_form_types')
        if sft and isinstance(sft, str):
            try:
                d['scanned_form_types'] = json.loads(sft)
            except (json.JSONDecodeError, TypeError):
                pass
        # Merge extra
        extra = d.pop('extra', None)
        if extra:
            try:
                d.update(json.loads(extra))
            except (json.JSONDecodeError, TypeError):
                pass
        return d

    # ------------------------------------------------------------------
    # SYNC: keep JSON and SQLite in sync
    # ------------------------------------------------------------------

    def sync_from_scraper(self, scraper):
        """Sync current scraper state into SQLite (call after pipeline steps)."""
        try:
            self.upsert_companies(scraper.universe)
            self.upsert_filings(scraper.filings_index)
            logger.debug("SQLite sync complete")
        except Exception as e:
            logger.warning(f"SQLite sync failed: {e}")
