"""
SEC Filing Daily Monitor
=========================
Runs once per day to detect new 10-K/10-Q filings from the company universe.

Strategy:
  - Uses SEC EDGAR EFTS (full-text search) API to find ALL new 10-K/10-Q
    filings from the past day in a single request (efficient, ~1-2 API calls)
  - Filters results against the saved company universe (by CIK)
  - Downloads new filings automatically
  - Optionally triggers auto-analysis on new filings

This avoids polling each company individually (which could be 500+ requests).

Scheduling:
  - Uses a background thread with a simple sleep loop
  - Configurable run time (default: 6:00 AM daily)
  - Persists state (last_checked, history) to disk
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

MONITOR_STATE_FILE = "monitor_state.json"
MONITOR_LOG_FILE = "monitor_log.json"

# EDGAR EFTS full-text search endpoint (free, no key needed)
EFTS_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"


# ============================================================================
# EDGAR EFTS API — Efficient new-filing detection
# ============================================================================

def search_new_filings_efts(start_date: str, end_date: str,
                            form_types: List[str] = None,
                            session=None) -> List[Dict]:
    """
    Use SEC EDGAR Full-Text Search (EFTS) API to find all filings
    of given types filed between start_date and end_date.

    This is MUCH more efficient than polling per-company — one request
    gets all filings filed that day.
    """
    from scraper import _get_session, _safe_request, SEC_REQUEST_DELAY

    if session is None:
        session = _get_session()
    if form_types is None:
        form_types = ["10-K", "10-Q"]

    forms_str = ",".join(form_types)
    all_filings = []
    page_from = 0
    page_size = 100
    max_pages = 20  # Safety limit

    while page_from < max_pages * page_size:
        params = {
            "q": "*",
            "forms": forms_str,
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "from": page_from,
            "size": page_size,
        }

        data = _safe_request(session, EFTS_SEARCH_URL, params=params,
                             delay=SEC_REQUEST_DELAY)

        if not data or not isinstance(data, dict):
            logger.warning(f"EFTS search returned no data (from={page_from})")
            break

        hits = data.get("hits", {}).get("hits", [])
        total = data.get("hits", {}).get("total", {})
        total_count = total.get("value", 0) if isinstance(total, dict) else total

        for hit in hits:
            source = hit.get("_source", {})
            filing = {
                "cik": str(source.get("entity_id", "")).zfill(10),
                "company_name": (source.get("display_names", [""])[0]
                                 if source.get("display_names")
                                 else source.get("entity_name", "")),
                "form_type": source.get("form_type", ""),
                "filing_date": source.get("file_date", ""),
                "accession_number": source.get("file_num", hit.get("_id", "")),
            }

            # Extract accession from _id if available
            hit_id = hit.get("_id", "")
            if hit_id and ":" in hit_id:
                filing["accession_number"] = hit_id.split(":")[0]
            elif hit_id:
                filing["accession_number"] = hit_id

            all_filings.append(filing)

        logger.info(f"EFTS page from={page_from}: {len(hits)} hits (total: {total_count})")

        page_from += page_size
        if page_from >= total_count or len(hits) < page_size:
            break

    logger.info(f"EFTS search found {len(all_filings)} filings "
                f"from {start_date} to {end_date}")
    return all_filings


def filter_to_universe(filings: List[Dict], universe: List[Dict]) -> List[Dict]:
    """Filter EFTS results to only companies in our universe (match on CIK)."""
    # Build CIK lookup
    cik_to_company = {}
    for c in universe:
        cik = c.get("cik", "").zfill(10)
        if cik:
            cik_to_company[cik] = c

    matched = []
    for filing in filings:
        cik = filing.get("cik", "").zfill(10)
        if cik in cik_to_company:
            company = cik_to_company[cik]
            filing["ticker"] = company.get("ticker", "")
            filing["exchange"] = company.get("exchange", "")
            filing["market_cap"] = company.get("market_cap")
            matched.append(filing)

    logger.info(f"Filtered to {len(matched)} filings matching universe "
                f"(out of {len(filings)} total)")
    return matched


def resolve_filing_urls(filings: List[Dict], session=None) -> List[Dict]:
    """
    For each matched filing, query EDGAR submissions API to get
    the download URL for the primary document.
    """
    from scraper import _get_session, _safe_request, SEC_REQUEST_DELAY

    if session is None:
        session = _get_session()

    resolved = []
    # Group by CIK to avoid duplicate API calls
    by_cik = {}
    for f in filings:
        by_cik.setdefault(f["cik"], []).append(f)

    for cik, cik_filings in by_cik.items():
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        data = _safe_request(session, url, delay=SEC_REQUEST_DELAY)

        if not data:
            logger.warning(f"Could not fetch submissions for CIK {cik}")
            continue

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        for filing in cik_filings:
            target_date = filing.get("filing_date", "")
            target_form = filing.get("form_type", "")

            for i in range(len(forms)):
                if (forms[i] == target_form and
                        i < len(dates) and dates[i] == target_date):
                    accession = accessions[i] if i < len(accessions) else ""
                    primary_doc = primary_docs[i] if i < len(primary_docs) else ""
                    accession_clean = accession.replace("-", "")

                    filing["accession_number"] = accession
                    filing["primary_document"] = primary_doc
                    filing["filing_url"] = (
                        f"https://www.sec.gov/Archives/edgar/data/"
                        f"{cik.lstrip('0')}/{accession_clean}/{primary_doc}"
                    )
                    resolved.append(filing)
                    break

    logger.info(f"Resolved download URLs for {len(resolved)}/{len(filings)} filings")
    return resolved


# ============================================================================
# MONITOR STATE
# ============================================================================

class MonitorState:
    """Persists monitor state and run log to disk."""

    def __init__(self, data_dir: str = "scraper_data"):
        self.data_dir = data_dir
        self.state_path = os.path.join(data_dir, MONITOR_STATE_FILE)
        self.log_path = os.path.join(data_dir, MONITOR_LOG_FILE)
        os.makedirs(data_dir, exist_ok=True)
        self.state = self._load_state()
        self.log = self._load_log()

    def _load_state(self) -> Dict:
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                return json.load(f)
        return {
            "enabled": False,
            "run_time": "06:00",
            "last_checked": None,
            "last_run_at": None,
            "last_run_status": None,
            "last_run_message": "",
            "auto_analyze": False,
            "use_llm_for_auto": False,
            "total_filings_found": 0,
            "total_runs": 0,
        }

    def _load_log(self) -> List[Dict]:
        if os.path.exists(self.log_path):
            with open(self.log_path) as f:
                data = json.load(f)
                return data[-90:] if len(data) > 90 else data
        return []

    def save(self):
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)
        with open(self.log_path, "w") as f:
            json.dump(self.log[-90:], f, indent=2)

    def add_log_entry(self, entry: Dict):
        entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log.append(entry)
        self.save()


# ============================================================================
# DAILY CHECK — Core logic
# ============================================================================

def run_daily_check(scraper, monitor_state: MonitorState,
                    auto_analyze: bool = False,
                    use_llm: bool = False) -> Dict:
    """
    Run a single daily check for new filings.
    Returns a summary dict.
    """
    from scraper import _get_session, download_filing, extract_text_from_html

    summary = {
        "status": "success",
        "checked_range": None,
        "new_filings_found": 0,
        "new_filings_downloaded": 0,
        "new_filings_analyzed": 0,
        "filings": [],
        "errors": [],
    }

    # Date range
    if monitor_state.state.get("last_checked"):
        start_date = monitor_state.state["last_checked"]
    else:
        start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

    end_date = datetime.now().strftime("%Y-%m-%d")
    summary["checked_range"] = f"{start_date} to {end_date}"

    logger.info(f"Daily monitor: checking {start_date} to {end_date}")

    if not scraper.universe:
        summary["status"] = "error"
        summary["errors"].append("No company universe loaded.")
        return summary

    session = _get_session()

    # 1) Find all new 10-K/10-Q via EFTS
    try:
        all_new = search_new_filings_efts(start_date, end_date, session=session)
    except Exception as e:
        summary["status"] = "error"
        summary["errors"].append(f"EFTS search failed: {str(e)}")
        return summary

    # 2) Filter to universe
    matched = filter_to_universe(all_new, scraper.universe)

    if not matched:
        summary["status"] = "no_new"
        _update_state_after_run(monitor_state, end_date, summary)
        return summary

    # 3) Resolve download URLs
    resolved = resolve_filing_urls(matched, session=session)
    summary["new_filings_found"] = len(resolved)

    # 4) Download (skip duplicates)
    existing_accessions = {f.get("accession_number") for f in scraper.filings_index}

    for filing in resolved:
        accession = filing.get("accession_number", "")
        if accession in existing_accessions:
            continue

        path = download_filing(filing, scraper.filings_dir, session)
        if path:
            filing["downloaded"] = True
            filing["local_path"] = path
            filing["analyzed"] = False
            filing["discovered_by"] = "daily_monitor"
            filing["discovered_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            scraper.filings_index.append(filing)
            existing_accessions.add(accession)
            summary["new_filings_downloaded"] += 1
            summary["filings"].append({
                "ticker": filing.get("ticker", ""),
                "company_name": filing.get("company_name", ""),
                "form_type": filing.get("form_type", ""),
                "filing_date": filing.get("filing_date", ""),
                "exchange": filing.get("exchange", ""),
            })
        else:
            summary["errors"].append(
                f"Download failed: {filing.get('ticker', '')} {filing.get('form_type', '')}"
            )

    scraper._save_state()

    # 5) Auto-analyze if enabled
    if auto_analyze and summary["new_filings_downloaded"] > 0:
        summary["new_filings_analyzed"] = _auto_analyze_new(
            scraper, use_llm, summary["errors"]
        )

    _update_state_after_run(monitor_state, end_date, summary)

    logger.info(
        f"Daily check complete: {summary['new_filings_found']} found, "
        f"{summary['new_filings_downloaded']} downloaded, "
        f"{summary['new_filings_analyzed']} analyzed"
    )
    return summary


def _update_state_after_run(monitor_state, end_date, summary):
    """Update monitor state and log after a run."""
    monitor_state.state["last_checked"] = end_date
    monitor_state.state["last_run_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    monitor_state.state["last_run_status"] = summary["status"]
    monitor_state.state["last_run_message"] = (
        f"Found {summary['new_filings_found']} new, "
        f"downloaded {summary['new_filings_downloaded']}, "
        f"analyzed {summary['new_filings_analyzed']}"
    )
    monitor_state.state["total_filings_found"] += summary["new_filings_found"]
    monitor_state.state["total_runs"] += 1
    monitor_state.save()

    monitor_state.add_log_entry({
        "date": end_date,
        "status": summary["status"],
        "new_found": summary["new_filings_found"],
        "downloaded": summary["new_filings_downloaded"],
        "analyzed": summary["new_filings_analyzed"],
        "errors": len(summary["errors"]),
        "filings": summary["filings"],
    })


def _auto_analyze_new(scraper, use_llm, errors_list) -> int:
    """Analyze filings discovered by the daily monitor that haven't been analyzed."""
    from scraper import extract_text_from_html
    from analyzer import SECFilingAnalyzer

    analyzer = SECFilingAnalyzer(use_llm=use_llm)
    analyzed = 0

    for filing in scraper.filings_index:
        if (filing.get("discovered_by") == "daily_monitor" and
                filing.get("downloaded") and not filing.get("analyzed")):

            filepath = filing.get("local_path", "")
            if not filepath or not os.path.exists(filepath):
                continue

            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                if filepath.endswith(('.html', '.htm')):
                    content = extract_text_from_html(content)

                if not content or len(content) < 200:
                    continue

                results = analyzer.analyze_filing(
                    ticker=filing.get("ticker", "UNKNOWN"),
                    company_name=filing.get("company_name", "Unknown"),
                    current_text=content,
                )

                # Save results
                result_filename = (
                    f"{filing.get('ticker', 'UNK')}_"
                    f"{filing.get('form_type', '').replace('/', '-')}_"
                    f"{filing.get('filing_date', '')}.json"
                )
                results_dir = "results"
                os.makedirs(results_dir, exist_ok=True)
                with open(os.path.join(results_dir, result_filename), 'w') as rf:
                    json.dump(results, rf, indent=2)

                scraper.mark_analyzed(filing.get("accession_number", ""), {
                    "final_score": results.get("final_score"),
                    "risk_rating": results.get("risk_rating"),
                    "red_flag_count": results.get("score_breakdown", {}).get("red_flag_count", 0),
                    "yellow_flag_count": results.get("score_breakdown", {}).get("yellow_flag_count", 0),
                    "green_flag_count": results.get("score_breakdown", {}).get("green_flag_count", 0),
                    "sentiment_trajectory": results.get("sentiment_trajectory"),
                    "key_concerns": results.get("key_concerns", []),
                    "result_file": result_filename,
                })
                analyzed += 1

            except Exception as e:
                errors_list.append(
                    f"Analysis failed for {filing.get('ticker', '')}: {str(e)}"
                )

    return analyzed


# ============================================================================
# BACKGROUND SCHEDULER
# ============================================================================

class DailyScheduler:
    """
    Background thread that runs the daily check at a configured time.
    Simple sleep-loop approach — no heavy dependencies.
    """

    def __init__(self, scraper, monitor_state: MonitorState):
        self.scraper = scraper
        self.monitor_state = monitor_state
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._last_check_result: Optional[Dict] = None

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def last_result(self) -> Optional[Dict]:
        return self._last_check_result

    def start(self):
        if self.is_running:
            return
        self._stop_event.clear()
        self._running = True
        self.monitor_state.state["enabled"] = True
        self.monitor_state.save()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Daily monitor scheduler started")

    def stop(self):
        self._stop_event.set()
        self._running = False
        self.monitor_state.state["enabled"] = False
        self.monitor_state.save()
        logger.info("Daily monitor scheduler stopped")

    def run_now(self) -> Dict:
        """Trigger an immediate check."""
        result = run_daily_check(
            self.scraper,
            self.monitor_state,
            auto_analyze=self.monitor_state.state.get("auto_analyze", False),
            use_llm=self.monitor_state.state.get("use_llm_for_auto", False),
        )
        self._last_check_result = result
        return result

    def _run_loop(self):
        logger.info(f"Scheduler loop started — run time: "
                    f"{self.monitor_state.state.get('run_time', '06:00')}")

        while not self._stop_event.is_set():
            try:
                now = datetime.now()
                run_time_str = self.monitor_state.state.get("run_time", "06:00")

                try:
                    run_hour, run_minute = map(int, run_time_str.split(":"))
                except ValueError:
                    run_hour, run_minute = 6, 0

                # Within 2-minute window of configured time
                if (now.hour == run_hour and
                        run_minute <= now.minute < run_minute + 2):

                    last_run = self.monitor_state.state.get("last_run_at", "")
                    today = now.strftime("%Y-%m-%d")

                    if not last_run or not last_run.startswith(today):
                        logger.info(f"Scheduled run at {now.strftime('%H:%M')}")
                        try:
                            result = run_daily_check(
                                self.scraper,
                                self.monitor_state,
                                auto_analyze=self.monitor_state.state.get("auto_analyze", False),
                                use_llm=self.monitor_state.state.get("use_llm_for_auto", False),
                            )
                            self._last_check_result = result
                        except Exception as e:
                            logger.error(f"Daily check failed: {e}")
                            self.monitor_state.state["last_run_status"] = "error"
                            self.monitor_state.state["last_run_message"] = str(e)
                            self.monitor_state.save()

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")

            # Check every 60 seconds
            self._stop_event.wait(timeout=60)
