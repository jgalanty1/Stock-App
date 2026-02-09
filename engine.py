"""
SEC Analyzer — Orchestration Engine
====================================
All pipeline, analysis, and result functions extracted from app.py.
No Flask dependency. Called by cli.py.
"""

import os
import json
import time
import logging
import traceback
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG PERSISTENCE
# ============================================================================
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_FOLDER, exist_ok=True)

_PERSISTED_KEYS = {
    'FMP_API_KEY': 'fmp_api_key',
    'OPENAI_API_KEY': 'openai_api_key',
    'ANTHROPIC_API_KEY': 'anthropic_api_key',
}


def load_config() -> dict:
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return {}


def save_config(cfg: dict):
    tmp = CONFIG_FILE + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, CONFIG_FILE)
    except IOError as e:
        logger.error(f"Config save failed: {e}")


def persist_key(env_var: str, value: str):
    os.environ[env_var] = value
    cfg_key = _PERSISTED_KEYS.get(env_var)
    if cfg_key:
        cfg = load_config()
        cfg[cfg_key] = value
        save_config(cfg)


def mask_key(key: str) -> str:
    if not key or len(key) < 10:
        return '••••' if key else ''
    return key[:6] + '•' * 8 + key[-4:]


def restore_keys():
    cfg = load_config()
    restored = []
    for env_var, cfg_key in _PERSISTED_KEYS.items():
        val = cfg.get(cfg_key, '')
        if val and not os.environ.get(env_var):
            os.environ[env_var] = val
            restored.append(env_var)
    return restored


def get_key_status() -> Dict:
    """Return status of all API keys."""
    return {
        'fmp': {
            'set': bool(os.environ.get('FMP_API_KEY')),
            'masked': mask_key(os.environ.get('FMP_API_KEY', '')),
        },
        'anthropic': {
            'set': bool(os.environ.get('ANTHROPIC_API_KEY')),
            'masked': mask_key(os.environ.get('ANTHROPIC_API_KEY', '')),
        },
        'openai': {
            'set': bool(os.environ.get('OPENAI_API_KEY')),
            'masked': mask_key(os.environ.get('OPENAI_API_KEY', '')),
        },
    }


# ============================================================================
# SCRAPER / DB SINGLETONS
# ============================================================================
_scraper = None
_db = None


def get_scraper():
    global _scraper
    if _scraper is None:
        from scraper import SECScraper
        _scraper = SECScraper()
    return _scraper


def get_db():
    global _db
    if _db is None:
        try:
            from database import SECDatabase
            _db = SECDatabase(data_dir=os.environ.get("SEC_DATA_DIR", "sec_data"))
            scraper = get_scraper()
            if scraper.universe or scraper.filings_index:
                _db.import_from_json(scraper.universe, scraper.filings_index)
        except Exception as e:
            logger.warning(f"SQLite init failed: {e}")
    return _db


def sync_db():
    try:
        db = get_db()
        if db:
            db.sync_from_scraper(get_scraper())
    except Exception:
        pass


# ============================================================================
# STATUS / STATS
# ============================================================================
def get_status() -> Dict:
    """Get summary of current pipeline state."""
    scraper = get_scraper()
    stats = scraper.get_stats()

    # Count analysis states
    t1_done = sum(1 for f in scraper.filings_index if f.get('llm_analyzed'))
    t2_done = sum(1 for f in scraper.filings_index if f.get('tier2_analyzed'))
    downloaded = sum(1 for f in scraper.filings_index if f.get('downloaded'))

    return {
        'universe_count': len(scraper.universe),
        'filings_total': len(scraper.filings_index),
        'filings_downloaded': downloaded,
        'tier1_analyzed': t1_done,
        'tier2_analyzed': t2_done,
        'keys': get_key_status(),
        **stats,
    }


# ============================================================================
# PIPELINE: SCAN (Universe → Find → Download → Insider)
# ============================================================================
def run_pipeline(steps=None, min_market_cap=None, max_market_cap=None,
                 max_companies=None, max_downloads=None,
                 progress_fn: Callable = None):
    """
    Run the scraper pipeline.
    steps: list of 'universe', 'find', 'download', 'insider'
    progress_fn: callback(step, message, current, total)
    """
    if steps is None:
        steps = ['universe', 'find', 'download', 'insider']

    scraper = get_scraper()

    def _progress(step, msg, current=0, total=0):
        if progress_fn:
            progress_fn(step, msg, current, total)
        else:
            logger.info(f"[{step}] {msg}")

    if 'universe' in steps:
        fmp_key = os.environ.get("FMP_API_KEY", "")
        if not fmp_key:
            raise RuntimeError("FMP_API_KEY not set. Run: python cli.py setup")

        _progress('universe', f"Building universe (${int((min_market_cap or 20_000_000)/1e6)}-${int((max_market_cap or 100_000_000)/1e6)}M)...")
        count = scraper.step1_build_universe(
            fmp_api_key=fmp_key,
            progress_callback=lambda msg: _progress('universe', msg),
            min_market_cap=min_market_cap,
            max_market_cap=max_market_cap,
        )
        _progress('universe', f"Universe: {count} companies")

    if 'find' in steps:
        _progress('find', "Finding filings on EDGAR...")
        count = scraper.step2_find_filings(
            max_companies=max_companies,
            progress_callback=lambda c, t, tk, n: _progress('find', f"{c}/{t} — {tk} ({n} filings)", c, t),
        )
        _progress('find', f"Found {count} filings")

    if 'download' in steps:
        _progress('download', "Downloading filings...")
        count = scraper.step3_download_filings(
            max_downloads=max_downloads,
            progress_callback=lambda c, t, tk, s: _progress('download', f"{c}/{t} — {tk} {'✓' if s else '✗'}", c, t),
        )
        _progress('download', f"Downloaded {count} filings")

    if 'insider' in steps or 'download' in steps:
        _progress('insider', "Fetching Form 4 insider data...")
        count = scraper.step4_fetch_insider_data(
            progress_callback=lambda c, t, tk, s: _progress('insider', f"{c}/{t} — {tk}", c, t),
        )
        _progress('insider', f"Insider data for {count} companies")

    sync_db()
    return get_status()


# ============================================================================
# CPS COMPUTATION
# ============================================================================
def compute_company_potential_scores(scraper, tier1_results) -> List[Dict]:
    """
    Compute CPS for each company. Extracted from app.py _compute_company_potential_scores.
    Returns list sorted by CPS desc.
    """
    _DEFAULT_WEIGHTS = {
        'catalyst_strength': 25, 'improvement_trajectory': 20,
        'insider_conviction': 20, 'turnaround_indicators': 20,
        'filing_recency': 15,
    }
    try:
        from calibration import CalibrationEngine
        engine = CalibrationEngine(scraper, [], data_dir=scraper.base_dir)
        cal_weights = engine.get_current_weights()
        weights = cal_weights if cal_weights else _DEFAULT_WEIGHTS
    except Exception:
        weights = _DEFAULT_WEIGHTS

    _ORIG_MAX = {
        'catalyst_strength': 25, 'improvement_trajectory': 20,
        'insider_conviction': 20, 'turnaround_indicators': 20,
        'filing_recency': 15,
    }

    t1_by_ticker = defaultdict(list)
    for r in tier1_results:
        t = r.get('ticker', '')
        if t:
            t1_by_ticker[t].append(r)

    all_filings_by_ticker = defaultdict(list)
    for f in scraper.filings_index:
        if f.get('downloaded') and f.get('ticker'):
            all_filings_by_ticker[f['ticker']].append(f)

    company_lookup = {}
    for c in scraper.universe:
        tk = c.get('ticker', '')
        if tk:
            company_lookup[tk] = c

    results = []

    for ticker, t1_filings in t1_by_ticker.items():
        if not t1_filings:
            continue
        company = company_lookup.get(ticker, {})
        company_name = (t1_filings[0].get('company_name', '') or
                        company.get('company_name', ''))
        t1_filings.sort(key=lambda x: x.get('filing_date', ''), reverse=True)
        newest = t1_filings[0]
        oldest = t1_filings[-1]

        # 1. Catalyst strength (25 max)
        best_t1 = max(r.get('composite_score', 0) or 0 for r in t1_filings)
        catalyst_pts = min(25, round(best_t1 * 0.25))
        gem_levels = [r.get('gem_potential', 'low') for r in t1_filings]
        if 'high' in gem_levels:
            catalyst_pts = min(25, catalyst_pts + 5)
        elif 'medium' in gem_levels:
            catalyst_pts = min(25, catalyst_pts + 2)

        # 2. Improvement trajectory (20 max)
        improvement_pts = 10
        if len(t1_filings) >= 2:
            delta = (newest.get('composite_score', 0) or 0) - (oldest.get('composite_score', 0) or 0)
            improvement_pts = max(0, min(20, round(10 + delta * (10 / 30))))

        # 3. Insider conviction (20 max)
        insider_pts = 5
        ins = company.get('insider_data', {}) or {}
        if ins and ins.get('signal') not in ('error', 'no_data', None, ''):
            acc_score = ins.get('accumulation_score', 50)
            insider_pts = max(0, min(15, round((acc_score - 30) * 15 / 70)))
            if ins.get('cluster_score', 0) >= 3:
                insider_pts = min(20, insider_pts + 3)
            if ins.get('unique_buyers', 0) >= 3:
                insider_pts = min(20, insider_pts + 2)
            has_activist = ins.get('has_activist_holder', False)
            if not has_activist:
                has_activist = any(
                    f.get('form_type', '').startswith('SC 13D')
                    for f in all_filings_by_ticker.get(ticker, [])
                )
            if has_activist:
                insider_pts = min(20, insider_pts + 5)

        # 4. Turnaround indicators (20 max)
        turnaround_pts = 5
        for fl in all_filings_by_ticker.get(ticker, []):
            ts = fl.get('turnaround_score')
            if ts is not None:
                turnaround_pts = max(turnaround_pts, round(ts * 0.10))
            rs = fl.get('research_signal_score')
            if rs is not None and rs >= 65:
                turnaround_pts = min(20, turnaround_pts + 3)
            ds = fl.get('diff_signal', '')
            if ds in ('strongly_improving', 'improving'):
                turnaround_pts = min(20, turnaround_pts + 4)
            elif ds == 'slightly_improving':
                turnaround_pts = min(20, turnaround_pts + 2)
        ownership_forms = [f for f in all_filings_by_ticker.get(ticker, [])
                           if f.get('form_type', '').startswith('SC 13D')]
        if ownership_forms:
            turnaround_pts = min(20, turnaround_pts + 4)

        # 5. Filing recency & coverage (15 max)
        recency_pts = 5
        try:
            newest_date = datetime.strptime(newest.get('filing_date', '2020-01-01'), '%Y-%m-%d')
            days_old = (datetime.now() - newest_date).days
            if days_old <= 90:
                recency_pts = 10
            elif days_old <= 180:
                recency_pts = 7
            elif days_old <= 365:
                recency_pts = 4
            else:
                recency_pts = 2
        except Exception:
            pass
        filing_count = len(all_filings_by_ticker.get(ticker, []))
        if filing_count >= 4:
            recency_pts = min(15, recency_pts + 4)
        elif filing_count >= 2:
            recency_pts = min(15, recency_pts + 2)
        has_periodic = any(f.get('form_type', '') in ('10-K', '10-Q')
                          for f in all_filings_by_ticker.get(ticker, []))
        has_ownership = any(f.get('form_type', '').startswith('SC 13')
                           for f in all_filings_by_ticker.get(ticker, []))
        if has_periodic and has_ownership:
            recency_pts = min(15, recency_pts + 3)

        # Total CPS
        raw_components = {
            'catalyst_strength': catalyst_pts,
            'improvement_trajectory': improvement_pts,
            'insider_conviction': insider_pts,
            'turnaround_indicators': turnaround_pts,
            'filing_recency': recency_pts,
        }
        cps = 0
        weighted_components = {}
        for comp_name, raw_pts in raw_components.items():
            orig_max = _ORIG_MAX.get(comp_name, 20)
            fraction = min(1.0, raw_pts / orig_max) if orig_max > 0 else 0
            w = weights.get(comp_name, _DEFAULT_WEIGHTS.get(comp_name, 15))
            weighted_pts = round(fraction * w, 1)
            weighted_components[comp_name] = weighted_pts
            cps += weighted_pts
        cps = max(0, min(100, round(cps)))

        results.append({
            'ticker': ticker,
            'company_name': company_name,
            'cps': cps,
            'components': weighted_components,
            'raw_components': raw_components,
            'weights_used': dict(weights),
            'best_t1_score': best_t1,
            'filing_count': filing_count,
            't1_count': len(t1_filings),
            'has_activist': bool(ownership_forms),
            'insider_signal': ins.get('signal', '') if ins else '',
            'accumulation_score': ins.get('accumulation_score', 50) if ins else None,
            'improvement_delta': round(
                (newest.get('composite_score', 0) or 0) -
                (oldest.get('composite_score', 0) or 0), 1
            ) if len(t1_filings) >= 2 else 0,
            'newest_date': newest.get('filing_date', ''),
            'gem_potential': newest.get('gem_potential', ''),
            'all_filings': [
                {
                    'accession_number': f.get('accession_number'),
                    'filing_date': f.get('filing_date'),
                    'form_type': f.get('form_type'),
                    'local_path': f.get('local_path'),
                    'company_name': f.get('company_name', company_name),
                    'tier1_score': f.get('tier1_score'),
                    'tier2_analyzed': f.get('tier2_analyzed', False),
                    'final_gem_score': f.get('final_gem_score'),
                }
                for f in all_filings_by_ticker.get(ticker, [])
            ],
        })

    results.sort(key=lambda x: x['cps'], reverse=True)
    return results


# ============================================================================
# LLM ANALYSIS — TIER 1 + TIER 2
# ============================================================================

# Shared progress state for CLI display
analysis_status = {
    "running": False,
    "tier": 0,
    "progress": 0,
    "total": 0,
    "message": "",
    "completed_tier1": 0,
    "failed_tier1": 0,
    "completed_tier2": 0,
    "failed_tier2": 0,
    "estimated_cost_usd": 0.0,
    "errors": [],
}


def run_analysis(tier2_pct=0.25, reanalyze=False, tier2_model='',
                 effort='', tier1_workers=6, tier2_workers=3,
                 progress_fn: Callable = None) -> Dict:
    """
    Run full Tier 1 + Tier 2 analysis.
    This is the main analysis orchestrator, extracted from app.py run_analysis().
    Returns status dict with results summary.
    """
    global analysis_status
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from scraper import extract_text_from_html, get_cached_text, _get_session, _edgar_limiter
    from llm_analysis import analyze_filing_tier1, analyze_filing_tier2
    from forensics import analyze_language, format_forensics_for_prompt, diff_filings
    from insider_tracker import (
        fetch_insider_transactions, fetch_institutional_filings,
        analyze_insider_activity, format_insider_for_prompt,
    )
    from inflection_detector import (
        fetch_xbrl_financials, extract_financial_metrics,
        detect_inflections, analyze_filing_timing, format_inflection_for_prompt,
    )

    scraper = get_scraper()

    analysis_status = {
        "running": True, "tier": 1, "progress": 0, "total": 0,
        "message": "Starting...", "completed_tier1": 0, "failed_tier1": 0,
        "completed_tier2": 0, "failed_tier2": 0,
        "estimated_cost_usd": 0.0, "errors": [],
    }

    def _progress(msg):
        analysis_status["message"] = msg
        if progress_fn:
            progress_fn(analysis_status)
        else:
            logger.info(msg)

    # Get filings
    if reanalyze:
        filings_to_analyze = [f for f in scraper.filings_index if f.get('downloaded')]
    else:
        filings_to_analyze = [
            f for f in scraper.filings_index
            if f.get('downloaded') and not f.get('llm_analyzed')
        ]

    if not filings_to_analyze:
        _progress("No filings to analyze")
        analysis_status["running"] = False
        return analysis_status

    # Resolve T2 model config
    t2_model_name = tier2_model or "claude-haiku-4-5-20251001"
    thinking_budget = None
    if effort == 'high':
        thinking_budget = 10000
    elif effort == 'medium':
        thinking_budget = 5000

    # ================================================================
    # TIER 1 — PARALLEL
    # ================================================================
    total = len(filings_to_analyze)
    analysis_status["total"] = total
    _progress(f"Tier 1: analyzing {total} filings [{tier1_workers}x parallel]...")

    _t1_lock = threading.Lock()
    TIER1_WORKERS = min(tier1_workers, total)

    def _analyze_one_t1(filing_idx_tuple):
        idx, filing = filing_idx_tuple
        ticker = filing.get('ticker', 'UNKNOWN')
        filepath = filing.get('local_path', '')
        if not filepath or not os.path.exists(filepath):
            return (idx, filing, None, f"{ticker} T1: file not found")
        try:
            content = get_cached_text(filepath)
            if not content:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if filepath.endswith('.html') or filepath.endswith('.htm'):
                    content = extract_text_from_html(content)
            result = analyze_filing_tier1(
                ticker=ticker,
                company_name=filing.get('company_name', ''),
                filing_text=content,
                form_type=filing.get('form_type', '10-K'),
            )
            if result:
                result['accession_number'] = filing.get('accession_number')
                result['filing_date'] = filing.get('filing_date')
                result['local_path'] = filepath
                result['company_name'] = filing.get('company_name', '')
                return (idx, filing, result, None)
            return (idx, filing, None, f"{ticker} T1: returned None")
        except Exception as e:
            return (idx, filing, None, f"{ticker} T1: {type(e).__name__}: {str(e)[:200]}")

    tier1_results = []
    completed_count = 0

    with ThreadPoolExecutor(max_workers=TIER1_WORKERS) as executor:
        futures = {executor.submit(_analyze_one_t1, (i, f)): i
                   for i, f in enumerate(filings_to_analyze)}
        for future in as_completed(futures):
            idx, filing, result, error = future.result()
            ticker = filing.get('ticker', 'UNKNOWN')
            completed_count += 1

            with _t1_lock:
                if result:
                    tier1_results.append(result)
                    analysis_status["completed_tier1"] += 1
                    analysis_status["estimated_cost_usd"] += round(
                        80000 * 0.15 / 1_000_000 + 2000 * 0.60 / 1_000_000, 4)
                    result_filename = f"{ticker}_tier1_{filing.get('filing_date', '')}.json"
                    result_path = os.path.join(RESULTS_FOLDER, result_filename)
                    with open(result_path, 'w') as rf:
                        json.dump(result, rf, indent=2)
                    filing['llm_analyzed'] = True
                    filing['tier1_score'] = result.get('composite_score')
                    filing['gem_potential'] = result.get('gem_potential')
                    filing['tier1_result_file'] = result_filename
                else:
                    analysis_status["failed_tier1"] += 1
                    if error:
                        analysis_status["errors"].append(error)
                    logger.error(f"T1 error: {error}")

                analysis_status["progress"] = completed_count
                ok = analysis_status["completed_tier1"]
                fail = analysis_status["failed_tier1"]
                _progress(f"Tier 1: {completed_count}/{total} — {ticker} (✅{ok} ❌{fail}) [{TIER1_WORKERS}x]")

            if completed_count % 25 == 0:
                scraper._save_state()

    scraper._save_state()

    # ================================================================
    # CPS + TIER 2 SELECTION
    # ================================================================
    if not tier1_results:
        analysis_status["tier"] = 2
        _progress("Tier 1 produced 0 results — skipping Tier 2")
        analysis_status["running"] = False
        return analysis_status

    # Include previously analyzed T1 results
    all_t1 = list(tier1_results)
    for f in scraper.filings_index:
        if f.get('llm_analyzed') and f.get('tier1_score') is not None:
            if not any(r.get('accession_number') == f.get('accession_number') for r in all_t1):
                all_t1.append({
                    'ticker': f.get('ticker', ''),
                    'company_name': f.get('company_name', ''),
                    'composite_score': f.get('tier1_score'),
                    'gem_potential': f.get('gem_potential', ''),
                    'accession_number': f.get('accession_number'),
                    'filing_date': f.get('filing_date'),
                    'local_path': f.get('local_path', ''),
                    'form_type': f.get('form_type', ''),
                })

    company_scores = compute_company_potential_scores(scraper, all_t1)
    _progress(f"CPS computed for {len(company_scores)} companies")

    # Select top companies by CPS percentile
    company_count = max(1, int(len(company_scores) * tier2_pct))
    top_companies = set(c['ticker'] for c in company_scores[:company_count])
    _progress(f"Tier 2 selecting top {company_count} of {len(company_scores)} companies")

    # Build T2 candidates: ALL filings from selected companies
    tier2_candidates = []
    for comp in company_scores[:company_count]:
        ticker = comp['ticker']
        for f_info in comp.get('all_filings', []):
            if f_info.get('local_path') and os.path.exists(f_info.get('local_path', '')):
                tier2_candidates.append({
                    'ticker': ticker,
                    'company_name': comp['company_name'],
                    'accession_number': f_info.get('accession_number'),
                    'filing_date': f_info.get('filing_date'),
                    'composite_score': f_info.get('tier1_score') or 0,
                    'form_type': f_info.get('form_type'),
                    'local_path': f_info.get('local_path'),
                    'cps': comp['cps'],
                })

    tier2_candidates.sort(key=lambda x: (x.get('ticker', ''), x.get('filing_date', '')), reverse=True)

    # ================================================================
    # TIER 2 — PRE-FETCH + PARALLEL LLM
    # ================================================================
    analysis_status["tier"] = 2
    analysis_status["total"] = len(tier2_candidates)
    analysis_status["progress"] = 0

    # Build lookups
    ticker_to_cik = {c.get('ticker', ''): c.get('cik', '') for c in scraper.universe if c.get('ticker')}
    all_filings_by_ticker = defaultdict(list)
    for f in scraper.filings_index:
        t = f.get('ticker', '')
        if t:
            all_filings_by_ticker[t].append(f)
    for t in all_filings_by_ticker:
        all_filings_by_ticker[t].sort(key=lambda x: x.get('filing_date', ''), reverse=True)

    edgar_session = _get_session()
    _insider_cache = {}
    _xbrl_cache = {}
    _market_cache = {}

    # --- PRE-FETCH PHASE ---
    unique_tickers = list(set(t.get('ticker', '') for t in tier2_candidates if t.get('ticker')))
    _progress(f"Pre-fetching EDGAR + market data for {len(unique_tickers)} tickers...")

    def _prefetch_ticker_data(ticker):
        result = {'ticker': ticker, 'insider': None, 'xbrl': None, 'market': {}}
        cik = ticker_to_cik.get(ticker, '')
        cached_insider = scraper.get_insider_for_ticker(ticker)
        if cached_insider and cached_insider.get('signal') not in ('error', None):
            result['insider'] = cached_insider
        elif cik:
            try:
                txns = fetch_insider_transactions(cik, edgar_session, _edgar_limiter)
                inst_filings = fetch_institutional_filings(
                    cik, edgar_session, _edgar_limiter, ticker=ticker,
                    company_name=next((c.get('company_name', '') for c in scraper.universe
                                       if c.get('ticker') == ticker), ''))
                result['insider'] = analyze_insider_activity(txns, inst_filings)
            except Exception:
                pass
        if cik:
            try:
                xbrl_facts = fetch_xbrl_financials(cik, edgar_session, _edgar_limiter)
                if xbrl_facts:
                    metrics = extract_financial_metrics(xbrl_facts)
                    result['xbrl'] = detect_inflections(metrics)
            except Exception:
                pass
        try:
            from market_data import (
                fetch_short_interest, compute_peer_comparison,
                fetch_institutional_holders, analyze_institutional_changes,
                fetch_latest_transcript, extract_transcript_signals,
            )
            mkt = {}
            try:
                si = fetch_short_interest(ticker)
                if si:
                    mkt['short_interest'] = si
            except Exception:
                pass
            try:
                peers = compute_peer_comparison(ticker)
                if peers:
                    mkt['peer_comparison'] = peers
            except Exception:
                pass
            try:
                holders = fetch_institutional_holders(ticker)
                if holders:
                    mkt['institutional'] = analyze_institutional_changes(holders)
            except Exception:
                pass
            try:
                transcript = fetch_latest_transcript(ticker)
                if transcript and transcript.get('content'):
                    mkt['transcript'] = extract_transcript_signals(transcript)
            except Exception:
                pass
            result['market'] = mkt
        except ImportError:
            pass
        return result

    PREFETCH_WORKERS = min(6, len(unique_tickers))
    prefetch_done = 0
    with ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as pf_executor:
        pf_futures = {pf_executor.submit(_prefetch_ticker_data, tk): tk for tk in unique_tickers}
        for fut in as_completed(pf_futures):
            try:
                pf_result = fut.result()
                tk = pf_result['ticker']
                if pf_result['insider']:
                    _insider_cache[tk] = pf_result['insider']
                if pf_result['xbrl']:
                    _xbrl_cache[tk] = pf_result['xbrl']
                _market_cache[tk] = pf_result.get('market', {})
            except Exception:
                pass
            prefetch_done += 1
            _progress(f"Pre-fetch: {prefetch_done}/{len(unique_tickers)} tickers [{PREFETCH_WORKERS}x]")

    # --- T2 PARALLEL LLM ---
    _t2_lock = threading.Lock()
    _t2_start_time = time.time()
    _t2_filing_times = []
    TIER2_WORKERS = max(1, min(tier2_workers, len(tier2_candidates)))
    _pending_t2 = []

    def _save_t2_result(t2_result, ctx):
        tk = ctx['ticker']
        t1_result = ctx['t1_result']
        diff_data = ctx.get('diff_data')
        insider_result = ctx.get('insider_result')
        inflection_result = ctx.get('inflection_result')
        research_signals_result = ctx.get('research_signals_result')
        catalyst_result = ctx.get('catalyst_result')
        current_forensics = ctx.get('current_forensics')
        _filing_start = ctx.get('filing_start', time.time())

        if t2_result:
            t2_result['tier1_score'] = t1_result.get('composite_score')
            t2_result['accession_number'] = t1_result.get('accession_number')
            t2_result['filing_date'] = t1_result.get('filing_date')
            if current_forensics:
                t2_result['forensics'] = current_forensics
            if diff_data and 'error' not in (diff_data or {}):
                prior_filing_info = ctx.get('prior_filing_info') or {}
                t2_result['filing_diff'] = {
                    'prior_filing_date': prior_filing_info.get('filing_date', ''),
                    'overall_signal': diff_data.get('overall_signal', ''),
                    'positive_shifts': diff_data.get('positive_shifts', 0),
                    'negative_shifts': diff_data.get('negative_shifts', 0),
                }
            if insider_result:
                t2_result['insider_activity'] = {
                    'accumulation_score': insider_result.get('accumulation_score', 50),
                    'signal': insider_result.get('signal', 'no_data'),
                    'unique_buyers': insider_result.get('unique_buyers', 0),
                    'cluster_score': insider_result.get('cluster_score', 0),
                    'has_activist_holder': insider_result.get('has_activist_holder', False),
                }
            if inflection_result:
                t2_result['financial_inflections'] = {
                    'inflection_score': inflection_result.get('inflection_score', 50),
                    'signal': inflection_result.get('signal', 'neutral'),
                }
            if research_signals_result:
                t2_result['research_signals'] = {
                    'research_signal_score': research_signals_result.get('research_signal_score', 50),
                    'research_signal': research_signals_result.get('research_signal', 'neutral'),
                }
            if catalyst_result:
                t2_result['turnaround_catalysts'] = {
                    'turnaround_score': catalyst_result.get('turnaround_score', 50),
                    'turnaround_signal': catalyst_result.get('turnaround_signal', ''),
                    'turnaround_phase': catalyst_result.get('turnaround_phase', ''),
                }

            with _t2_lock:
                analysis_status["completed_tier2"] += 1
                analysis_status["estimated_cost_usd"] += round(
                    200000 * 3.0 / 1_000_000 + 4000 * 15.0 / 1_000_000, 4)
                result_filename = f"{tk}_tier2_{t1_result.get('filing_date', '')}.json"
                result_path = os.path.join(RESULTS_FOLDER, result_filename)
                with open(result_path, 'w') as rf:
                    json.dump(t2_result, rf, indent=2)
                for f in scraper.filings_index:
                    if f.get('accession_number') == t1_result.get('accession_number'):
                        f['tier2_analyzed'] = True
                        raw_score = t2_result.get('final_gem_score')
                        try:
                            f['final_gem_score'] = int(float(raw_score)) if raw_score is not None else None
                        except (ValueError, TypeError):
                            f['final_gem_score'] = None
                        f['conviction'] = t2_result.get('conviction_level')
                        f['recommendation'] = t2_result.get('recommendation')
                        f['tier2_result_file'] = result_filename
                        f['has_prior_diff'] = diff_data is not None and 'error' not in (diff_data or {})
                        f['diff_signal'] = diff_data.get('overall_signal', '') if diff_data else ''
                        f['insider_signal'] = insider_result.get('signal', '') if insider_result else ''
                        f['accumulation_score'] = insider_result.get('accumulation_score', 50) if insider_result else None
                        f['inflection_signal'] = inflection_result.get('signal', '') if inflection_result else ''
                        f['inflection_score'] = inflection_result.get('inflection_score', 50) if inflection_result else None
                        f['research_signal_score'] = research_signals_result.get('research_signal_score', 50) if research_signals_result else None
                        f['research_signal'] = research_signals_result.get('research_signal', '') if research_signals_result else ''
                        f['turnaround_score'] = catalyst_result.get('turnaround_score', 50) if catalyst_result else None
                        f['turnaround_signal'] = catalyst_result.get('turnaround_signal', '') if catalyst_result else ''
                        f['turnaround_phase'] = catalyst_result.get('turnaround_phase', '') if catalyst_result else ''
                        break
                scraper._save_state()
                analysis_status["tier2_last_completed"] = tk
                _filing_elapsed = time.time() - _filing_start
                _t2_filing_times.append(_filing_elapsed)
        else:
            with _t2_lock:
                analysis_status["failed_tier2"] += 1
                analysis_status["errors"].append(f"{tk} T2: LLM returned None")

    def _drain_t2_futures(wait_all=False):
        nonlocal _pending_t2
        still_pending = []
        for fut, ctx in _pending_t2:
            if wait_all or fut.done():
                try:
                    t2_result = fut.result(timeout=300)
                    _save_t2_result(t2_result, ctx)
                except Exception as e:
                    with _t2_lock:
                        analysis_status["failed_tier2"] += 1
                        analysis_status["errors"].append(f"{ctx['ticker']} T2 LLM: {e}")
            else:
                still_pending.append((fut, ctx))
        _pending_t2 = still_pending

    t2_executor = ThreadPoolExecutor(max_workers=TIER2_WORKERS)
    _progress(f"Tier 2: {len(tier2_candidates)} filings [{TIER2_WORKERS}x parallel LLM]...")

    for i, t1_result in enumerate(tier2_candidates):
        ticker = t1_result.get('ticker', 'UNKNOWN')

        # Skip already done
        if not reanalyze:
            acc = t1_result.get('accession_number', '')
            if any(f.get('accession_number') == acc and f.get('tier2_analyzed')
                   for f in scraper.filings_index):
                analysis_status["progress"] = i + 1
                continue

        analysis_status["progress"] = i + 1
        _filing_start = time.time()

        filepath = t1_result.get('local_path', '')
        if not filepath or not os.path.exists(filepath):
            analysis_status["failed_tier2"] += 1
            continue

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if filepath.endswith('.html') or filepath.endswith('.htm'):
                content = extract_text_from_html(content)

            # Forensics
            current_forensics = analyze_language(content)

            # Prior diff
            diff_data = None
            prior_filing_info = None
            current_form = t1_result.get('form_type', '')
            current_acc = t1_result.get('accession_number', '')
            _PERIODIC = {'10-K', '10-Q'}
            _OWN_13D = {'SC 13D', 'SC 13D/A'}
            _OWN_13G = {'SC 13G', 'SC 13G/A'}

            def _same_family(ft1, ft2):
                if ft1 in _PERIODIC and ft2 in _PERIODIC: return True
                if ft1 in _OWN_13D and ft2 in _OWN_13D: return True
                if ft1 in _OWN_13G and ft2 in _OWN_13G: return True
                return ft1 == ft2

            for pf in all_filings_by_ticker.get(ticker, []):
                pf_acc = pf.get('accession_number', '')
                pf_date = pf.get('filing_date', '')
                cur_date = t1_result.get('filing_date', '')
                pf_form = pf.get('form_type', '')
                if (pf_acc != current_acc and pf_date < cur_date
                        and _same_family(current_form, pf_form)
                        and pf.get('downloaded') and pf.get('local_path')
                        and os.path.exists(pf.get('local_path', ''))):
                    prior_filing_info = pf
                    break

            prior_text = None
            if prior_filing_info:
                try:
                    pf_path = prior_filing_info['local_path']
                    with open(pf_path, 'r', encoding='utf-8', errors='ignore') as pf:
                        prior_content = pf.read()
                    if pf_path.endswith('.html') or pf_path.endswith('.htm'):
                        prior_content = extract_text_from_html(prior_content)
                    prior_text = prior_content
                    prior_forensics = analyze_language(prior_content)
                    diff_data = diff_filings(prior_forensics, current_forensics)
                except Exception:
                    pass

            forensic_prompt = format_forensics_for_prompt(current_forensics, diff_data)

            # Insider (from cache)
            insider_result = _insider_cache.get(ticker)
            if not insider_result:
                cached_insider = scraper.get_insider_for_ticker(ticker)
                if cached_insider and cached_insider.get('signal') not in ('error', None):
                    insider_result = cached_insider
            insider_prompt = format_insider_for_prompt(insider_result) if insider_result else ""

            # XBRL (from cache)
            inflection_result = _xbrl_cache.get(ticker)
            inflection_prompt = ""
            if inflection_result:
                try:
                    filing_timing = analyze_filing_timing(
                        t1_result.get('filing_date', ''), t1_result.get('form_type', '10-K'))
                    inflection_prompt = format_inflection_for_prompt(inflection_result, filing_timing)
                except Exception:
                    pass

            # Research signals
            research_signals_prompt = ""
            research_signals_result = None
            try:
                from filing_signals import compute_all_signals, format_signals_for_prompt
                insider_signal_data = None
                if insider_result:
                    insider_signal_data = {
                        'summary': {
                            'net_purchases': (insider_result.get('open_market_buys', 0) -
                                              insider_result.get('open_market_sells', 0)),
                            'buyer_count': insider_result.get('unique_buyers', 0),
                            'seller_count': insider_result.get('unique_sellers', 0),
                            'executive_buyers': 0,
                            'cluster_buy': insider_result.get('cluster_score', 0) >= 2,
                            'total_purchase_value': insider_result.get('total_buy_value', 0),
                        },
                    }
                sigs = compute_all_signals(content, prior_text, insider_signal_data,
                                           t1_result.get('filing_date', ''),
                                           t1_result.get('form_type', '10-K'))
                research_signals_result = sigs
                research_signals_prompt = format_signals_for_prompt(sigs)
            except Exception:
                pass

            # Catalysts
            catalyst_prompt = ""
            catalyst_result = None
            try:
                from catalyst_extractor import extract_catalysts, format_catalysts_for_prompt
                catalyst_result = extract_catalysts(content, t1_result.get('form_type', '10-K'))
                catalyst_prompt = format_catalysts_for_prompt(catalyst_result)
            except Exception:
                pass

            # Cross-reference context
            cross_ref_context = ""
            try:
                other_filings = [f for f in all_filings_by_ticker.get(ticker, [])
                                 if f.get('accession_number') != current_acc
                                 and f.get('tier1_score') is not None]
                if other_filings:
                    cross_ref_context = f"Other filings for {ticker}: "
                    for of in other_filings[:3]:
                        cross_ref_context += (
                            f"{of.get('form_type','')} ({of.get('filing_date','')}): "
                            f"T1 score={of.get('tier1_score','?')}, "
                            f"gem={of.get('gem_potential','?')}; "
                        )
            except Exception:
                pass

            # Market data (from cache)
            peer_comparison = ""
            short_interest = ""
            insider_timing = ""
            transcript_signals = ""
            market_data_context = ""
            mkt = _market_cache.get(ticker, {})
            try:
                from market_data import (
                    is_short_squeeze_candidate, format_peer_comparison_for_prompt,
                    correlate_insider_timing,
                )
                if mkt.get('short_interest'):
                    si = mkt['short_interest']
                    short_interest = f"Short interest: {si.get('short_pct_float',0):.1f}% of float"
                    try:
                        sq = is_short_squeeze_candidate(si, insider_result or {}, t1_result.get('cps', 0))
                        if sq.get('is_candidate'):
                            short_interest += f"\n⚠ SQUEEZE CANDIDATE (score {sq['score']})"
                    except Exception:
                        pass
                if mkt.get('peer_comparison'):
                    peer_comparison = format_peer_comparison_for_prompt(mkt['peer_comparison'])
                if mkt.get('institutional'):
                    inst = mkt['institutional']
                    market_data_context = f"Institutional: {inst.get('signal', '?')}"
                if insider_result:
                    try:
                        timing = correlate_insider_timing(insider_result, t1_result.get('filing_date', ''))
                        if timing.get('timing_signal') != 'neutral':
                            insider_timing = f"Insider timing: {timing['timing_signal']} (score {timing['timing_score']})"
                    except Exception:
                        pass
                if mkt.get('transcript'):
                    tsigs = mkt['transcript']
                    transcript_signals = (
                        f"Earnings call: {tsigs.get('positive_count',0)} positive, "
                        f"{tsigs.get('negative_count',0)} negative"
                    )
            except ImportError:
                pass

            # Submit LLM call to thread pool
            _t2_ctx = {
                'ticker': ticker, 'idx': i, 't1_result': dict(t1_result),
                'diff_data': diff_data, 'prior_filing_info': prior_filing_info,
                'insider_result': insider_result, 'inflection_result': inflection_result,
                'research_signals_result': research_signals_result,
                'catalyst_result': catalyst_result,
                'current_forensics': current_forensics,
                'filing_start': _filing_start,
            }
            _t2_future = t2_executor.submit(
                analyze_filing_tier2,
                ticker=ticker,
                company_name=t1_result.get('company_name', ''),
                filing_text=content,
                form_type=t1_result.get('form_type', '10-K'),
                model=t2_model_name,
                forensic_data=forensic_prompt,
                insider_data=insider_prompt,
                inflection_data=inflection_prompt,
                research_signals_data=research_signals_prompt,
                catalyst_data=catalyst_prompt,
                thinking_budget=thinking_budget,
                effort=effort,
                cross_ref_context=cross_ref_context,
                peer_comparison=peer_comparison,
                short_interest=short_interest,
                insider_timing=insider_timing,
                transcript_signals=transcript_signals,
                market_data_context=market_data_context,
            )
            _pending_t2.append((_t2_future, _t2_ctx))
            _drain_t2_futures(wait_all=False)

            ok = analysis_status["completed_tier2"]
            fail = analysis_status["failed_tier2"]
            _progress(f"Tier 2: {i+1}/{len(tier2_candidates)} — {ticker} (✅{ok} ❌{fail}) [{TIER2_WORKERS}x]")

        except Exception as e:
            analysis_status["failed_tier2"] += 1
            analysis_status["errors"].append(f"{ticker} T2: {type(e).__name__}: {str(e)[:200]}")

    # Drain remaining
    if _pending_t2:
        _progress(f"Tier 2: waiting for {len(_pending_t2)} remaining LLM calls...")
        _drain_t2_futures(wait_all=True)

    t2_executor.shutdown(wait=True)
    scraper._save_state()
    sync_db()

    elapsed = time.time() - _t2_start_time
    elapsed_str = f"{int(elapsed//60)}m{int(elapsed%60):02d}s"
    _progress(
        f"✅ Complete — T1: {analysis_status['completed_tier1']} OK / "
        f"{analysis_status['failed_tier1']} failed — "
        f"T2: {analysis_status['completed_tier2']} OK / "
        f"{analysis_status['failed_tier2']} failed — ⏱ {elapsed_str}"
    )
    analysis_status["running"] = False
    return analysis_status


# ============================================================================
# RESULTS — GEMS
# ============================================================================
def get_gems() -> List[Dict]:
    """Get top gem candidates, one per ticker, recency-weighted."""
    scraper = get_scraper()
    ticker_filings = defaultdict(list)
    for f in scraper.filings_index:
        if f.get('tier2_analyzed') and f.get('final_gem_score') is not None:
            ticker_filings[f.get('ticker', '')].append(f)

    stocks = []
    for ticker, filings in ticker_filings.items():
        if not filings:
            continue
        filings.sort(key=lambda x: x.get('filing_date', ''), reverse=True)
        latest = filings[0]
        company = next((c for c in scraper.universe
                        if c.get('ticker', '').upper() == ticker.upper()), None)
        company_name = latest.get('company_name') or (company or {}).get('company_name', '')

        # Recency-weighted gem score
        w_sum, w_total = 0.0, 0.0
        for i, fl in enumerate(filings):
            w = 0.5 ** i
            w_sum += (fl.get('final_gem_score', 0) or 0) * w
            w_total += w
        gem_score = round(w_sum / w_total, 1) if w_total else 0

        # Qualitative fields from most recent
        conviction = recommendation = diff_signal = insider_signal = ''
        for fl in filings:
            if fl.get('conviction') and not conviction: conviction = fl['conviction']
            if fl.get('recommendation') and not recommendation: recommendation = fl['recommendation']
            if fl.get('diff_signal') and not diff_signal: diff_signal = fl['diff_signal']
            if fl.get('insider_signal') and not insider_signal: insider_signal = fl['insider_signal']

        # One-liner from result file
        one_liner = ''
        for fl in filings:
            rf = fl.get('tier2_result_file', '')
            if rf:
                rpath = os.path.join(RESULTS_FOLDER, rf)
                if os.path.exists(rpath):
                    try:
                        with open(rpath) as fp:
                            data = json.load(fp)
                        if data.get('one_liner'):
                            one_liner = data['one_liner']
                            break
                    except Exception:
                        pass

        # Sentiment trend
        sentiment = 'stable'
        if len(filings) >= 2:
            delta = (filings[0].get('final_gem_score', 0) or 0) - (filings[-1].get('final_gem_score', 0) or 0)
            if delta >= 15: sentiment = 'improving'
            elif delta >= 5: sentiment = 'slightly_improving'
            elif delta <= -15: sentiment = 'deteriorating'
            elif delta <= -5: sentiment = 'slightly_deteriorating'

        form_types = sorted(set(fl.get('form_type', '') for fl in filings if fl.get('form_type')))

        stocks.append({
            'ticker': ticker,
            'company_name': company_name,
            'gem_score': gem_score,
            'conviction': conviction,
            'recommendation': recommendation,
            'one_liner': one_liner,
            'sentiment': sentiment,
            'diff_signal': diff_signal,
            'insider_signal': insider_signal,
            'form_types': form_types,
            'filing_count': len(filings),
            'latest_date': filings[0].get('filing_date', ''),
        })

    stocks.sort(key=lambda x: x['gem_score'], reverse=True)
    return stocks


# ============================================================================
# RESULTS — POTENTIAL (CPS)
# ============================================================================
def get_potential() -> List[Dict]:
    """Get companies ranked by CPS."""
    scraper = get_scraper()
    # Gather T1 results
    all_t1 = []
    for f in scraper.filings_index:
        if f.get('llm_analyzed') and f.get('tier1_score') is not None:
            all_t1.append({
                'ticker': f.get('ticker', ''),
                'company_name': f.get('company_name', ''),
                'composite_score': f.get('tier1_score'),
                'gem_potential': f.get('gem_potential', ''),
                'accession_number': f.get('accession_number'),
                'filing_date': f.get('filing_date'),
                'local_path': f.get('local_path', ''),
                'form_type': f.get('form_type', ''),
            })
    if not all_t1:
        return []
    return compute_company_potential_scores(scraper, all_t1)


# ============================================================================
# RESULTS — COMPANY DETAIL
# ============================================================================
def get_company_detail(ticker: str) -> Dict:
    """Get detailed info for one company."""
    scraper = get_scraper()
    ticker = ticker.upper()
    company = next((c for c in scraper.universe if c.get('ticker', '').upper() == ticker), None)
    filings = [f for f in scraper.filings_index if f.get('ticker', '').upper() == ticker]
    filings.sort(key=lambda x: x.get('filing_date', ''), reverse=True)

    # Load T2 result files
    t2_results = []
    for f in filings:
        rf = f.get('tier2_result_file', '')
        if rf:
            rpath = os.path.join(RESULTS_FOLDER, rf)
            if os.path.exists(rpath):
                try:
                    with open(rpath) as fp:
                        t2_results.append(json.load(fp))
                except Exception:
                    pass

    return {
        'ticker': ticker,
        'company': company or {},
        'filings': filings,
        't2_results': t2_results,
    }


# ============================================================================
# BACKTEST
# ============================================================================
def run_backtest(clear_cache=False) -> Dict:
    """Run price backtesting on all T2 analyzed filings."""
    scraper = get_scraper()

    if clear_cache:
        data_dir = os.environ.get("MARKET_DATA_DIR", "sec_data/market_data")
        cache_file = os.path.join(data_dir, "backtest_results.json")
        if os.path.exists(cache_file):
            os.remove(cache_file)
        try:
            from market_data import _price_cache
            _price_cache.clear()
        except (ImportError, AttributeError):
            pass

    # Build CPS lookup
    potential = get_potential()
    cps_lookup = {c.get('ticker', ''): c.get('cps', 0) for c in potential}

    filings = [
        {
            'ticker': f.get('ticker'),
            'filing_date': f.get('filing_date'),
            'final_gem_score': f.get('final_gem_score'),
            'conviction': f.get('conviction', ''),
            'cps': cps_lookup.get(f.get('ticker', ''), 0),
        }
        for f in scraper.filings_index
        if f.get('tier2_analyzed') and f.get('filing_date')
    ]

    if not filings:
        return {'error': 'No Tier 2 analyzed filings to backtest', 'total': 0}

    from market_data import batch_backtest, compute_backtest_stats
    results = batch_backtest(filings)
    stats = compute_backtest_stats(results)

    # Flatten stats
    flat = dict(stats)
    by_w = stats.get('by_window', {})
    for window in ['30', '60', '90']:
        w = by_w.get(window, {})
        flat[f'avg_{window}d'] = w.get('avg_return', 0)
        flat[f'win_rate_{window}d'] = w.get('win_rate', 0)

    errors = [r for r in results if r.get('error')]
    with_data = [r for r in results if r.get('windows') and len(r['windows']) > 0]
    flat['error_count'] = len(errors)
    flat['with_price_data'] = len(with_data)

    results_with_90 = [r for r in results if '90' in r.get('windows', {})]
    if results_with_90:
        best = max(results_with_90, key=lambda r: r['windows']['90']['change_pct'])
        worst = min(results_with_90, key=lambda r: r['windows']['90']['change_pct'])
        flat['best_ticker'] = best.get('ticker', '?')
        flat['best_return'] = best['windows']['90']['change_pct']
        flat['worst_ticker'] = worst.get('ticker', '?')
        flat['worst_return'] = worst['windows']['90']['change_pct']

    individual = []
    for r in results[:100]:
        w = r.get('windows', {})
        individual.append({
            'ticker': r.get('ticker', '?'),
            'filing_date': r.get('filing_date', ''),
            'gem_score': r.get('final_gem_score') or r.get('gem_score'),
            'return_30d': w.get('30', {}).get('change_pct'),
            'return_60d': w.get('60', {}).get('change_pct'),
            'return_90d': w.get('90', {}).get('change_pct'),
            'hit': r.get('hit', False),
            'error': r.get('error', ''),
        })

    return {'total': len(results), 'stats': flat, 'results': individual}


# ============================================================================
# CALIBRATION
# ============================================================================
def run_calibration(return_window=90, auto_apply=False) -> Dict:
    """Run feedback loop calibration."""
    scraper = get_scraper()
    potential = get_potential()
    from calibration import CalibrationEngine
    engine = CalibrationEngine(scraper, potential, data_dir=scraper.base_dir)
    report = engine.run_full_calibration(return_window=return_window)
    if auto_apply and report.get('weight_optimization', {}).get('suggested_weights'):
        engine.apply_weights(report['weight_optimization']['suggested_weights'])
        report['weights_applied'] = True
    return report


# ============================================================================
# WATCHLIST
# ============================================================================
def get_watchlist() -> List[str]:
    cfg = load_config()
    return cfg.get('watchlist', [])


def add_to_watchlist(ticker: str):
    cfg = load_config()
    wl = cfg.get('watchlist', [])
    ticker = ticker.upper()
    if ticker not in wl:
        wl.append(ticker)
    cfg['watchlist'] = wl
    save_config(cfg)


def remove_from_watchlist(ticker: str):
    cfg = load_config()
    wl = cfg.get('watchlist', [])
    ticker = ticker.upper()
    cfg['watchlist'] = [t for t in wl if t != ticker]
    save_config(cfg)


# ============================================================================
# EXPORT & BACKUP
# ============================================================================
def export_csv(output_path: str = None) -> str:
    """Export gems + potential to CSV."""
    import csv
    if output_path is None:
        output_path = os.path.join('sec_data', 'export.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    gems = get_gems()
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Ticker', 'Company', 'Gem Score', 'Conviction',
                         'Recommendation', 'Sentiment', 'Insider', 'Diff',
                         'One-Liner', 'Filings', 'Latest Date'])
        for g in gems:
            writer.writerow([
                g['ticker'], g['company_name'], g['gem_score'],
                g['conviction'], g['recommendation'], g['sentiment'],
                g['insider_signal'], g['diff_signal'], g['one_liner'],
                g['filing_count'], g['latest_date'],
            ])
    return output_path


def backup_data(output_path: str = None) -> str:
    """Zip sec_data/ for safekeeping."""
    import zipfile
    if output_path is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"sec_data_backup_{ts}.zip"

    data_dir = os.environ.get("SEC_DATA_DIR", "sec_data")
    if not os.path.exists(data_dir):
        raise RuntimeError(f"Data directory not found: {data_dir}")

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, os.path.dirname(data_dir))
                zf.write(filepath, arcname)
    return output_path


# Initialize on import
restore_keys()
