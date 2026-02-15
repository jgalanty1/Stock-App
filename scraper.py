"""
SEC Filing Auto-Scraper
========================
Discovers US microcap companies ($20-100M market cap) on NYSE, NASDAQ,
AMEX, OTCQX, and OTCQB, then downloads their latest 10-K/10-Q filings
from SEC EDGAR.

Universe source:
  FMP Company Screener (tries stable endpoint, falls back to legacy v3)
  - Requires FMP Starter plan ($22/mo) or higher
  - Hard server-side market cap filter: $20M-$100M
  - No proxies, no float estimates, no fallbacks

Rate limits:
  - FMP: 299 requests/min (Starter plan allows 300/min)
  - SEC EDGAR: 299 requests/min (SEC allows 10/sec = 600/min, we stay well under)

All EDGAR endpoints are free, no key needed (just User-Agent header).
"""

import os
import json
import time
import re
import logging
import threading
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Set
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    ""  # Must be set — SEC blocks generic agents
)
if not SEC_USER_AGENT:
    logger.error(
        "SEC_USER_AGENT not configured. Set via 'python cli.py setup' or SEC_USER_AGENT env var."
    )
    raise ValueError("SEC_USER_AGENT required for EDGAR access")

# Hard market cap bounds (USD). No results outside this range.
MIN_MARKET_CAP = 20_000_000    # $20M
MAX_MARKET_CAP = 100_000_000   # $100M

# Filing types to download
# CIK-based types are found via company's CIK submissions endpoint
# (includes both company-filed AND third-party filings like SC 13D)
# EFTS-based types use full-text search as a SUPPLEMENT to catch any missed filings
# Note: SEC uses two names for the same forms (both normalized to "SC" prefix):
#   Old (HTML, pre-Dec 2024): SC 13D, SC 13D/A, SC 13G, SC 13G/A
#   New (XML, post-Dec 2024): SCHEDULE 13D, SCHEDULE 13D/A, SCHEDULE 13G, SCHEDULE 13G/A
TARGET_FORM_TYPES_CIK = [
    "10-K", "10-Q",
    "8-K",                          # Event filings — most immediate catalysts (#2)
    "DEF 14A",                      # Proxy statements — governance signals (#6)
    "SC 13D", "SC 13D/A",          # Activist investors
    "SC 13G", "SC 13G/A",          # Passive large holders
]
TARGET_FORM_TYPES_EFTS = ["SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A", "13F-HR"]  # Supplementary

# How far back to look for filings (days)
FILING_LOOKBACK_DAYS = 365
# SC 13D/G filings are filed infrequently — use longer lookback
OWNERSHIP_LOOKBACK_DAYS = 730  # 2 years
# 8-K event filings — shorter lookback, they're time-sensitive
EVENT_LOOKBACK_DAYS = 180
EVENT_FORM_TYPES = {"8-K", "8-K/A"}
# Proxy statements — annual, use standard lookback
PROXY_FORM_TYPES = {"DEF 14A", "DEFA14A"}
OWNERSHIP_FORM_PREFIXES = ("SC 13D", "SC 13G", "SCHEDULE 13")


def _normalize_form_type(form_type: str) -> str:
    """Normalize SEC form type names to canonical short form.
    SEC uses both 'SC 13D' and 'SCHEDULE 13D' interchangeably."""
    ft = form_type.strip().upper()
    ft = ft.replace("SCHEDULE 13D/A", "SC 13D/A")
    ft = ft.replace("SCHEDULE 13D", "SC 13D")
    ft = ft.replace("SCHEDULE 13G/A", "SC 13G/A")
    ft = ft.replace("SCHEDULE 13G", "SC 13G")
    return ft

# ------------------------------------------------------------------
# RATE LIMITING — 299 requests per minute max for both FMP and EDGAR
# ------------------------------------------------------------------
RATE_LIMIT_PER_MIN = 299

# Storage paths — relative to CWD at import time (assumed to be project root)
DATA_DIR = os.path.abspath("scraper_data")
UNIVERSE_FILE = "company_universe.json"
FILINGS_DIR = "filings"
FILINGS_INDEX = "filings_index.json"


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    def __init__(self, max_per_minute: int = RATE_LIMIT_PER_MIN):
        self.min_interval = 60.0 / max_per_minute
        self._next_slot = 0.0
        self._call_count = 0
        self._window_start = time.monotonic()
        self._lock = threading.Lock()

    def wait(self):
        # Token-bucket: claim a time slot inside the lock, sleep outside
        with self._lock:
            now = time.monotonic()
            if now < self._next_slot:
                sleep_until = self._next_slot
            else:
                sleep_until = now
            self._next_slot = sleep_until + self.min_interval
            self._call_count += 1
            count = self._call_count
        # Sleep outside lock so multiple threads can wait concurrently
        delay = sleep_until - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        if count % 100 == 0:
            window = time.monotonic() - self._window_start
            rate = count / (window / 60) if window > 0 else 0
            logger.info(f"Rate limiter: {count} calls, {rate:.0f}/min actual")

    @property
    def stats(self) -> Dict[str, Any]:
        window = time.monotonic() - self._window_start
        return {
            "total_calls": self._call_count,
            "elapsed_seconds": round(window, 1),
            "actual_rate_per_min": round(self._call_count / (window / 60), 1) if window > 0 else 0,
        }


_fmp_limiter = RateLimiter(RATE_LIMIT_PER_MIN)
_edgar_limiter = RateLimiter(RATE_LIMIT_PER_MIN)


# ============================================================================
# HTTP HELPERS
# ============================================================================

_session = None
_session_lock = threading.Lock()

def _get_session():
    """Singleton session for the main thread. Reuses TCP connections."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                s = requests.Session()
                s.headers.update({
                    "User-Agent": SEC_USER_AGENT,
                    "Accept-Encoding": "gzip, deflate",
                })
                _session = s
    return _session


_thread_local = threading.local()

def _get_thread_session():
    """Per-thread session for worker threads. Each thread gets its own session."""
    session = getattr(_thread_local, 'session', None)
    if session is None:
        session = requests.Session()
        session.headers.update({
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        })
        _thread_local.session = session
    return session


def _safe_request(session, url: str, params: dict = None,
                  limiter: RateLimiter = None, retries: int = 3) -> Optional[Any]:
    if limiter is None:
        limiter = _edgar_limiter

    for attempt in range(retries):
        try:
            limiter.wait()
            resp = session.get(url, params=params, timeout=30)

            if resp.status_code == 200:
                content_type = resp.headers.get("Content-Type", "")
                if "json" in content_type:
                    return resp.json()
                # Not JSON content type — return as text
                return resp.text

            elif resp.status_code == 429:
                wait = min(2 ** (attempt + 2), 30)
                logger.warning(f"Rate limited (429) on {url}, waiting {wait}s...")
                time.sleep(wait)
                continue

            elif resp.status_code == 404:
                logger.debug(f"Not found (404): {url}")
                return None

            elif resp.status_code in (401, 402, 403):
                body_text = ""
                try:
                    body = resp.json()
                    body_text = json.dumps(body)
                except Exception:
                    body_text = resp.text[:500]
                msg = f"HTTP {resp.status_code} from {url}\nResponse: {body_text}"
                logger.error(msg)
                raise PermissionError(msg)

            else:
                body_text = ""
                try:
                    body_text = resp.text[:500]
                except Exception:
                    pass
                logger.warning(f"HTTP {resp.status_code} for {url}: {body_text}")
                if attempt < retries - 1:
                    time.sleep(2)

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on {url} (attempt {attempt + 1}/{retries})")
            time.sleep(2)
        except PermissionError:
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            if attempt < retries - 1:
                time.sleep(2)

    return None


# ============================================================================
# FMP API DIAGNOSTIC TEST
# ============================================================================

def test_fmp_api(fmp_api_key: str) -> Dict[str, Any]:
    """
    Run diagnostics against FMP to verify key validity and screener access.
    Returns detailed results for display to user.
    """
    results = {
        "key_provided": bool(fmp_api_key),
        "key_length": len(fmp_api_key) if fmp_api_key else 0,
        "tests": [],
        "working_endpoint": None,
        "sample_count": 0,
        "error": None,
    }

    if not fmp_api_key:
        results["error"] = "No API key provided"
        return results

    session = _get_session()

    # Test 1: Stable screener endpoint
    test1 = {"name": "Stable endpoint (/stable/company-screener)", "status": "pending"}
    try:
        url = "https://financialmodelingprep.com/stable/company-screener"
        params = {
            "marketCapMoreThan": MIN_MARKET_CAP,
            "marketCapLowerThan": MAX_MARKET_CAP,
            "country": "US",
            "isActivelyTrading": "true",
            "isEtf": "false",
            "isFund": "false",
            "limit": 5,
            "apikey": fmp_api_key,
        }
        _fmp_limiter.wait()
        resp = session.get(url, params=params, timeout=15)
        test1["http_status"] = resp.status_code
        test1["response_preview"] = resp.text[:800].replace(fmp_api_key, 'REDACTED')

        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                test1["status"] = "PASS"
                test1["count"] = len(data)
                test1["sample"] = {k: data[0].get(k) for k in
                    ["symbol", "companyName", "marketCap", "exchangeShortName", "exchange"]
                    if k in data[0]} if data else None
                results["working_endpoint"] = "stable"
                results["sample_count"] = len(data)
            elif isinstance(data, list) and len(data) == 0:
                test1["status"] = "WARN"
                test1["detail"] = "Returned empty list — endpoint works but no matching data"
            elif isinstance(data, dict):
                test1["status"] = "FAIL"
                test1["detail"] = f"API error: {data}"
            else:
                test1["status"] = "WARN"
                test1["detail"] = f"Unexpected type: {type(data).__name__}"
        elif resp.status_code in (401, 402, 403):
            test1["status"] = "FAIL"
            test1["detail"] = f"Access denied (HTTP {resp.status_code})"
        else:
            test1["status"] = "FAIL"
            test1["detail"] = f"HTTP {resp.status_code}"
    except Exception as e:
        test1["status"] = "ERROR"
        test1["detail"] = str(e)
    results["tests"].append(test1)

    # Test 2: Legacy v3 screener endpoint
    test2 = {"name": "Legacy endpoint (/api/v3/stock-screener)", "status": "pending"}
    try:
        url = "https://financialmodelingprep.com/api/v3/stock-screener"
        params = {
            "marketCapMoreThan": MIN_MARKET_CAP,
            "marketCapLowerThan": MAX_MARKET_CAP,
            "country": "US",
            "isActivelyTrading": "true",
            "isEtf": "false",
            "limit": 5,
            "apikey": fmp_api_key,
        }
        _fmp_limiter.wait()
        resp = session.get(url, params=params, timeout=15)
        test2["http_status"] = resp.status_code
        test2["response_preview"] = resp.text[:800].replace(fmp_api_key, 'REDACTED')

        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                test2["status"] = "PASS"
                test2["count"] = len(data)
                test2["sample"] = {k: data[0].get(k) for k in
                    ["symbol", "companyName", "marketCap", "exchangeShortName", "exchange"]
                    if k in data[0]} if data else None
                if not results["working_endpoint"]:
                    results["working_endpoint"] = "legacy"
                    results["sample_count"] = len(data)
            elif isinstance(data, list) and len(data) == 0:
                test2["status"] = "WARN"
                test2["detail"] = "Returned empty list"
            elif isinstance(data, dict):
                test2["status"] = "FAIL"
                test2["detail"] = f"API error: {data}"
            else:
                test2["status"] = "WARN"
        elif resp.status_code in (401, 402, 403):
            test2["status"] = "FAIL"
            test2["detail"] = f"Access denied (HTTP {resp.status_code})"
        else:
            test2["status"] = "FAIL"
            test2["detail"] = f"HTTP {resp.status_code}"
    except Exception as e:
        test2["status"] = "ERROR"
        test2["detail"] = str(e)
    results["tests"].append(test2)

    # Test 3: Basic profile (to verify key validity independent of plan)
    test3 = {"name": "Profile check (key validity)", "status": "pending"}
    try:
        url = "https://financialmodelingprep.com/stable/profile"
        params = {"symbol": "AAPL", "apikey": fmp_api_key}
        _fmp_limiter.wait()
        resp = session.get(url, params=params, timeout=15)
        test3["http_status"] = resp.status_code
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                test3["status"] = "PASS"
                test3["detail"] = "API key is VALID"
            elif isinstance(data, dict) and ("Error" in data or "error" in data):
                test3["status"] = "FAIL"
                test3["detail"] = f"Key invalid: {data}"
            else:
                test3["status"] = "PASS"
                test3["detail"] = "Key appears valid"
        elif resp.status_code in (401, 403):
            test3["status"] = "FAIL"
            test3["detail"] = "API key is INVALID"
        else:
            test3["status"] = "WARN"
            test3["detail"] = f"HTTP {resp.status_code}"
    except Exception as e:
        test3["status"] = "ERROR"
        test3["detail"] = str(e)
    results["tests"].append(test3)

    # Summary
    if not results["working_endpoint"]:
        key_valid = any(t["status"] == "PASS" for t in results["tests"] if "profile" in t["name"].lower())
        if key_valid:
            results["error"] = (
                "Your API key is valid, but the screener endpoint is NOT accessible. "
                "Your FMP plan does not include the stock screener. "
                "You need the Starter plan ($22/mo) or higher. "
                "Upgrade: https://site.financialmodelingprep.com/developer/docs/pricing"
            )
        else:
            results["error"] = (
                "Could not connect to FMP. Your API key may be invalid or there's "
                "a network issue. Check: https://financialmodelingprep.com/dashboard"
            )

    return results


# ============================================================================
# STEP 1: BUILD COMPANY UNIVERSE (FMP SCREENER)
# ============================================================================

def build_universe_fmp(fmp_api_key: str,
                       progress_callback=None,
                       min_market_cap: int = None,
                       max_market_cap: int = None) -> List[Dict[str, Any]]:
    """
    Use FMP Company Screener to find US companies within a market cap range.

    Strategy:
    1. Try the stable endpoint first (/stable/company-screener)
    2. If that fails, try the legacy v3 endpoint (/api/v3/stock-screener)
    3. Do NOT filter by exchange at the API level — FMP may not support
       OTC exchange codes. Filter by country=US only, keep all exchanges.
    4. The screener does server-side market cap filtering.
    """
    min_cap = int(min_market_cap) if min_market_cap is not None else MIN_MARKET_CAP
    max_cap = int(max_market_cap) if max_market_cap is not None else MAX_MARKET_CAP
    session = _get_session()
    all_companies = []
    working_endpoint = None

    endpoints = [
        ("stable", "https://financialmodelingprep.com/stable/company-screener"),
        ("legacy", "https://financialmodelingprep.com/api/v3/stock-screener"),
    ]

    for ep_name, ep_url in endpoints:
        if progress_callback:
            progress_callback(f"Trying FMP {ep_name} screener...")

        logger.info(f"Trying FMP {ep_name}: {ep_url}")

        params = {
            "marketCapMoreThan": min_cap,
            "marketCapLowerThan": max_cap,
            "country": "US",
            "isActivelyTrading": "true",
            "isEtf": "false",
            "limit": 10000,
            "apikey": fmp_api_key,
        }
        if ep_name == "stable":
            params["isFund"] = "false"

        try:
            data = _safe_request(session, ep_url, params=params, limiter=_fmp_limiter)
        except PermissionError as e:
            logger.warning(f"FMP {ep_name} access denied: {e}")
            if progress_callback:
                progress_callback(f"FMP {ep_name}: access denied, trying next...")
            continue

        if data and isinstance(data, list) and len(data) > 0:
            working_endpoint = ep_name
            logger.info(f"FMP {ep_name}: {len(data)} results")
            if progress_callback:
                progress_callback(f"FMP {ep_name}: {len(data)} raw results")

            for company in data:
                mkt_cap = company.get("marketCap", 0) or 0
                if not (min_cap <= mkt_cap <= max_cap):
                    continue

                symbol = company.get("symbol", "")
                name = company.get("companyName", "")
                exchange = company.get("exchangeShortName",
                           company.get("exchange", ""))

                if any(c in symbol for c in ["-", ".", "^", "+"]):
                    continue
                if any(kw in name.lower() for kw in
                       ["warrant", "unit ", "rights", " right",
                        "acquisition corp", "blank check"]):
                    continue

                all_companies.append({
                    "ticker": symbol,
                    "company_name": name,
                    "exchange": exchange,
                    "market_cap": mkt_cap,
                    "sector": company.get("sector", ""),
                    "industry": company.get("industry", ""),
                    "price": company.get("price", 0),
                    "volume": company.get("volume", 0),
                    "country": company.get("country", "US"),
                    "is_actively_trading": True,
                    "source": f"fmp_{ep_name}",
                })
            break
        elif data and isinstance(data, dict):
            err_msg = data.get("Error Message",
                      data.get("message",
                      data.get("error", str(data))))
            logger.warning(f"FMP {ep_name} error: {err_msg}")
            if progress_callback:
                progress_callback(f"FMP {ep_name}: {err_msg}")
            continue
        else:
            logger.warning(f"FMP {ep_name}: empty response")
            if progress_callback:
                progress_callback(f"FMP {ep_name}: empty, trying next...")
            continue

    if not working_endpoint:
        raise RuntimeError(
            "BOTH FMP screener endpoints failed.\n\n"
            "Possible causes:\n"
            "1. Invalid API key — check at https://financialmodelingprep.com/dashboard\n"
            "2. Plan doesn't include screener — need Starter ($22/mo)+\n"
            "3. FMP is temporarily down\n\n"
            "Use 'Test API' button for exact error details.\n"
            "Upgrade: https://site.financialmodelingprep.com/developer/docs/pricing"
        )

    # Deduplicate
    seen = set()
    unique = []
    for c in all_companies:
        t = c["ticker"]
        if t and t not in seen:
            seen.add(t)
            unique.append(c)

    # Log exchange distribution
    exc = {}
    for c in unique:
        ex = c.get("exchange", "Unknown")
        exc[ex] = exc.get(ex, 0) + 1
    logger.info(f"FMP universe: {len(unique)} companies, exchanges: {exc}")

    if progress_callback:
        min_m = int(min_cap / 1_000_000)
        max_m = int(max_cap / 1_000_000)
        progress_callback(
            f"Found {len(unique)} US microcaps (${min_m}-${max_m}M). "
            f"Exchanges: {', '.join(f'{k}:{v}' for k,v in sorted(exc.items()))}"
        )

    return unique


# ============================================================================
# CIK RESOLUTION
# ============================================================================

def resolve_ciks(companies: List[Dict], session=None,
                 progress_callback=None) -> List[Dict]:
    if session is None:
        session = _get_session()

    if progress_callback:
        progress_callback("Downloading CIK mapping from SEC EDGAR...")

    url = "https://www.sec.gov/files/company_tickers.json"
    data = _safe_request(session, url, limiter=_edgar_limiter)

    if not data:
        raise RuntimeError("Failed to download CIK mapping from SEC EDGAR.")

    ticker_to_cik = {}
    for key, entry in data.items():
        ticker = entry.get("ticker", "").upper()
        cik = str(entry.get("cik_str", "")).zfill(10)
        if ticker:
            ticker_to_cik[ticker] = cik

    resolved = 0
    for company in companies:
        ticker = company["ticker"].upper()
        cik = ticker_to_cik.get(ticker)
        if cik:
            company["cik"] = cik
            resolved += 1

    with_cik = [c for c in companies if c.get("cik")]
    without_cik = [c for c in companies if not c.get("cik")]

    if without_cik:
        shown = [c['ticker'] for c in without_cik[:20]]
        omitted = len(without_cik) - len(shown)
        msg = f"Dropped {len(without_cik)} without CIK: {shown}"
        if omitted > 0:
            msg += f" (and {omitted} more)"
        logger.warning(msg)

    logger.info(f"CIK: {resolved}/{len(companies)} resolved, {len(with_cik)} ready")
    return with_cik


# ============================================================================
# STEP 2: FIND FILINGS ON EDGAR
# ============================================================================

def get_company_filings(cik: str, session=None,
                        form_types: List[str] = None) -> List[Dict]:
    if session is None:
        session = _get_session()
    if form_types is None:
        form_types = TARGET_FORM_TYPES_CIK

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = _safe_request(session, url, limiter=_edgar_limiter)
    if not data:
        return []

    filings = []
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    descriptions = recent.get("primaryDocDescription", [])

    cutoff = (datetime.now() - timedelta(days=FILING_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    ownership_cutoff = (datetime.now() - timedelta(days=OWNERSHIP_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    event_cutoff = (datetime.now() - timedelta(days=EVENT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    for i in range(len(forms)):
        form_type = forms[i]
        filing_date = dates[i] if i < len(dates) else ""
        accession = accessions[i] if i < len(accessions) else ""
        primary_doc = primary_docs[i] if i < len(primary_docs) else ""
        description = descriptions[i] if i < len(descriptions) else ""

        # Normalize: SEC uses both "SC 13D" and "SCHEDULE 13D"
        normalized_form = _normalize_form_type(form_type)

        # Use per-type lookback windows
        is_ownership = any(normalized_form.startswith(p) for p in OWNERSHIP_FORM_PREFIXES)
        is_event = normalized_form in EVENT_FORM_TYPES
        effective_cutoff = (ownership_cutoff if is_ownership else
                            event_cutoff if is_event else cutoff)

        if normalized_form in form_types and filing_date >= effective_cutoff:
            accession_clean = accession.replace("-", "")
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik.lstrip('0')}/{accession_clean}/{primary_doc}"
            )
            filings.append({
                "form_type": normalized_form,  # Canonical name (SC 13D not SCHEDULE 13D)
                "filing_date": filing_date,
                "accession_number": accession,
                "primary_document": primary_doc,
                "description": description,
                "filing_url": filing_url,
                "cik": cik,
            })

    return filings


def _efts_search(session, query: str, form_types: List[str],
                 start_date: str, end_date: str) -> Dict:
    """
    Call EDGAR EFTS full-text search API via GET.
    IMPORTANT: EFTS creates AND filters when multiple form types are comma-separated.
    So we search each form type separately and merge results.
    Returns merged response dict with hits from all form types.
    """
    url = "https://efts.sec.gov/LATEST/search-index"

    all_hits = []
    seen_ids = set()

    for form_type in form_types:
        params = {
            "q": query,
            "forms": form_type,  # Single form type to avoid AND filter bug
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
        }

        _edgar_limiter.wait()

        try:
            resp = session.get(url, params=params, timeout=30)
            logger.info(
                f"EFTS GET query={query} form={form_type} "
                f"→ status={resp.status_code} len={len(resp.text)}"
            )

            if resp.status_code != 200:
                logger.warning(f"EFTS GET failed: HTTP {resp.status_code}")
                continue

            try:
                data = resp.json()
            except json.JSONDecodeError as e:
                logger.error(f"EFTS response not valid JSON for form={form_type}: {e}")
                continue
            hits_obj = data.get('hits', {})
            if isinstance(hits_obj, dict):
                hit_list = hits_obj.get('hits', [])
                total = hits_obj.get('total', {})
                logger.info(
                    f"EFTS form={form_type}: total={total}, "
                    f"returned={len(hit_list)}"
                )
                for hit in hit_list:
                    hit_id = hit.get('_id', '')
                    if hit_id not in seen_ids:
                        seen_ids.add(hit_id)
                        all_hits.append(hit)

        except Exception as e:
            logger.error(f"EFTS search exception for form={form_type}: {e}")

    # Return merged results in standard EFTS format
    return {
        "hits": {
            "total": {"value": len(all_hits)},
            "hits": all_hits,
        }
    }


def search_filings_about_company(ticker: str, company_name: str, cik: str,
                                  session=None,
                                  form_types: List[str] = None,
                                  lookback_days: int = OWNERSHIP_LOOKBACK_DAYS) -> List[Dict]:
    """
    Search EDGAR EFTS for filings filed ABOUT a company by third parties.
    SC 13D/G filings are filed by the investor, not the target company,
    so they don't appear in the company's CIK submissions endpoint.
    We use EDGAR's full-text search to find them.
    """
    if session is None:
        session = _get_session()
    if form_types is None:
        form_types = TARGET_FORM_TYPES_EFTS
    if not form_types:
        return []

    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")

    filings = []

    # SC 13D filings reference the target company by NAME, not ticker.
    # Diagnostic proved: company name finds SC 13D, ticker does NOT.
    # Search company name FIRST, ticker as fallback.
    search_terms = []
    if company_name and len(company_name) > 3:
        # Use first two significant words (keeps punctuation like comma)
        words = [w for w in company_name.split() if w.upper() not in
                 ('INC', 'INC.', 'CORP', 'CORP.', 'LTD', 'LLC', 'CO', 'CO.',
                  'THE', 'GROUP', 'HOLDINGS', 'INTERNATIONAL')]
        if len(words) >= 2:
            search_terms.append(f'"{words[0]} {words[1]}"')
        elif words:
            search_terms.append(f'"{words[0]}"')
    # Ticker as fallback (rarely works for SC 13D but cheap to try)
    search_terms.append(f'"{ticker}"')

    seen_accessions = set()

    for query in search_terms:
        data = _efts_search(session, query, form_types, cutoff, today)
        if not data or not isinstance(data, dict):
            logger.warning(f"EFTS: no data returned for query={query}")
            continue

        # Parse hits — handle multiple possible response structures
        hits = []
        if 'hits' in data:
            hits_obj = data['hits']
            if isinstance(hits_obj, dict):
                hits = hits_obj.get('hits', [])
            elif isinstance(hits_obj, list):
                hits = hits_obj
        elif 'filings' in data:
            # Alternative response format
            hits = data['filings']

        logger.info(f"EFTS search for {query} forms={form_types}: {len(hits)} results")

        for hit in hits:
            # Handle both nested (_source) and flat response formats
            if isinstance(hit, dict) and '_source' in hit:
                src = hit['_source']
                raw_id = hit.get('_id', '')
            else:
                src = hit
                raw_id = hit.get('accession_number', '') or hit.get('accessionNo', '')

            # Extract accession number from _id (format: "accession:filename" or just accession)
            if ':' in raw_id:
                accession = raw_id.split(':')[0]
            else:
                accession = raw_id

            if not accession or accession in seen_accessions:
                continue
            seen_accessions.add(accession)

            # Get form type — try multiple field names, normalize
            form_type_raw = (src.get('form_type', '') or src.get('formType', '')
                             or src.get('root_form', '') or src.get('form', '') or '')
            form_type = _normalize_form_type(form_type_raw) if form_type_raw else ''
            if form_type and form_type not in form_types:
                continue

            # Get filing date — try multiple field names
            filing_date = (
                src.get('file_date', '') or src.get('filedAt', '') or
                src.get('filing_date', '') or ''
            )[:10]

            # Get filer info
            cik_filer = str(
                src.get('entity_id', '') or src.get('cik', '') or ''
            ).lstrip('0') or '0'
            filer_name = (
                src.get('entity_name', '') or src.get('companyName', '') or ''
            )

            # Get filing URL if available
            filing_url = src.get('file_url', '') or src.get('filingUrl', '') or ''

            # Build index URL for resolving primary document during download
            accession_clean = accession.replace('-', '')
            index_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_filer}/{accession_clean}/"
            )

            logger.info(
                f"EFTS hit: {form_type} by {filer_name} on {filing_date} "
                f"acc={accession}"
            )

            filings.append({
                "form_type": form_type,
                "filing_date": filing_date,
                "accession_number": accession,
                "primary_document": "",
                "description": filer_name,
                "filing_url": filing_url,
                "filing_index_url": index_url,
                "cik": cik,
                "filer_cik": cik_filer,
                "filer_name": filer_name,
                "is_efts_result": True,
            })

    logger.info(f"EFTS total: {len(filings)} unique ownership filings for {ticker}")
    return filings


# ============================================================================
# STEP 3: DOWNLOAD FILINGS
# ============================================================================

def download_filing(filing: Dict, output_dir: str,
                    session=None) -> Optional[str]:
    if session is None:
        session = _get_session()

    ticker = re.sub(r'[^\w.-]', '_', filing.get("ticker", "UNKNOWN"))
    form_type = re.sub(r'[^\w.-]', '_', filing["form_type"].replace("/", "-").replace(" ", "-"))
    date = filing["filing_date"]

    filename = f"{ticker}_{form_type}_{date}.html"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 100:
        return filepath

    url = filing.get("filing_url", "")

    # Try direct URL first
    content = None
    if url:
        content = _safe_request(session, url, limiter=_edgar_limiter)

    # If direct URL failed (404, wrong CIK, etc.), resolve via accession
    # Make a copy to avoid mutating the shared filing dict
    filing = dict(filing)
    if not content or not isinstance(content, str) or len(content) <= 100:
        resolved_url = _resolve_filing_url(filing, session)
        if resolved_url and resolved_url != url:
            filing["filing_url"] = resolved_url  # Cache for future use
            content = _safe_request(session, resolved_url, limiter=_edgar_limiter)

    if content and isinstance(content, str) and len(content) > 100:
        # If it's XML (new SCHEDULE 13D format), extract text
        if content.strip().startswith('<?xml') or '<XML>' in content[:500]:
            content = _extract_text_from_xml(content)

        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, "w", encoding="utf-8", errors="replace") as f:
            f.write(content)
        return filepath
    return None


def _extract_filer_cik_from_accession(accession: str) -> str:
    """
    Extract filer CIK from accession number.
    Accession format: 0001234567-YY-NNNNNN (first 10 digits = filer CIK).
    """
    clean = accession.replace("-", "")
    if len(clean) >= 10:
        return clean[:10]
    return ""


def _resolve_filing_url(filing: Dict, session) -> Optional[str]:
    """
    Resolve the primary document URL for any filing by fetching the
    filing index page from EDGAR. Works for both CIK-sourced and EFTS filings.

    Key insight: SC 13D filings are filed by INVESTORS, not the company.
    The EDGAR archive path uses the FILER's CIK, which differs from the
    company CIK. We extract the filer CIK from the accession number.
    """
    accession = filing.get("accession_number", "")
    if not accession:
        return None

    # Get filer CIK: prefer explicit, otherwise extract from accession
    filer_cik = filing.get("filer_cik", "")
    if not filer_cik:
        filer_cik = _extract_filer_cik_from_accession(accession)
    if not filer_cik:
        return None

    accession_clean = accession.replace("-", "")
    base_url = f"https://www.sec.gov/Archives/edgar/data/{filer_cik.lstrip('0')}/{accession_clean}/"

    # Try JSON index first (most reliable)
    index_url = base_url + "index.json"
    data = _safe_request(session, index_url, limiter=_edgar_limiter)

    if data and isinstance(data, dict):
        items = data.get("directory", {}).get("item", [])
        # Find the primary document: prefer HTM/HTML > TXT > XML
        best_html = None
        best_html_size = 0
        best_xml = None
        best_xml_size = 0
        for item in items:
            name = item.get("name", "")
            size = item.get("size", 0)
            # Skip index files
            if "index" in name.lower() or name.startswith("R"):
                continue
            sz = int(size) if isinstance(size, (int, str)) and str(size).isdigit() else 0
            if name.endswith((".htm", ".html", ".txt")):
                if sz > best_html_size:
                    best_html = name
                    best_html_size = sz
            elif name.endswith(".xml") and not name.startswith("primary_doc"):
                if sz > best_xml_size:
                    best_xml = name
                    best_xml_size = sz

        # Prefer HTML; fall back to XML for new SCHEDULE 13D format
        best = best_html or best_xml
        if best:
            url = base_url + best
            filing["primary_document"] = best
            doc_size = best_html_size if best_html else best_xml_size
            logger.info(f"Resolved filing URL: {url} (document size: {doc_size} bytes)")
            return url

    # Fallback: try HTML index page and parse it
    html = _safe_request(session, base_url, limiter=_edgar_limiter)
    if html and isinstance(html, str):
        links = re.findall(r'href="([^"]+\.(?:htm[l]?|txt|xml))"', html, re.IGNORECASE)
        doc_links = [l for l in links if "index" not in l.lower() and not l.startswith("R")]
        if doc_links:
            url = base_url + doc_links[0] if not doc_links[0].startswith("http") else doc_links[0]
            logger.info(f"Resolved filing URL (HTML fallback): {url}")
            return url

    logger.warning(f"Could not resolve primary document for accession {accession}")
    return None


def _extract_text_from_xml(xml_content: str) -> str:
    """
    Extract readable text from SEC XML filings (SCHEDULE 13D/G format).
    Falls back to stripping tags if XML parsing fails.
    """
    try:
        import xml.etree.ElementTree as ET
        # Try to find the XML within potential HTML wrapper
        xml_start = xml_content.find('<?xml')
        if xml_start > 0:
            xml_content = xml_content[xml_start:]

        root = ET.fromstring(xml_content)
        # Collect all text content recursively
        texts = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                texts.append(f"{tag}: {elem.text.strip()}")
            if elem.tail and elem.tail.strip():
                texts.append(elem.tail.strip())
        if texts:
            return '\n'.join(texts)
    except Exception as e:
        logger.debug(f"XML parse failed, falling back to tag stripping: {e}")

    # Fallback: strip all XML/HTML tags
    clean = re.sub(r'<[^>]+>', ' ', xml_content)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


# ============================================================================
# TEXT CACHING (#22) — Pre-extract and cache cleaned text from filings
# ============================================================================

_TEXT_CACHE_DIR = None

def _get_text_cache_dir():
    global _TEXT_CACHE_DIR
    if _TEXT_CACHE_DIR is None:
        _TEXT_CACHE_DIR = os.environ.get("TEXT_CACHE_DIR", "sec_data/text_cache")
    os.makedirs(_TEXT_CACHE_DIR, exist_ok=True)
    return _TEXT_CACHE_DIR


def _cache_filing_text(filepath: str) -> Optional[str]:
    """Extract and cache cleaned text from a filing. Returns cache path."""
    if not filepath or not os.path.exists(filepath):
        return None

    cache_dir = _get_text_cache_dir()
    cache_name = os.path.basename(filepath).rsplit('.', 1)[0] + '.txt'
    cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 50:
        return cache_path

    try:
        file_size = os.path.getsize(filepath)
        if file_size > 50_000_000:
            logger.warning("Large file: %s (%d bytes)", filepath, file_size)

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        if filepath.endswith(('.html', '.htm')):
            content = extract_text_from_html(content)
        elif content.strip().startswith('<?xml') or '<XML>' in content[:500]:
            content = _extract_text_from_xml(content)

        if content and len(content) > 50:
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return cache_path
    except Exception as e:
        logger.debug(f"Text cache failed for {filepath}: {e}")

    return None


def get_cached_text(filepath: str) -> Optional[str]:
    """Get cached cleaned text for a filing, or extract on-the-fly."""
    cache_dir = _get_text_cache_dir()
    cache_name = os.path.basename(filepath).rsplit('.', 1)[0] + '.txt'
    cache_path = os.path.join(cache_dir, cache_name)

    # Try cache first
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 50:
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()

    # Cache miss — extract and cache
    _cache_filing_text(filepath)
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()

    # Final fallback — extract without caching
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        if filepath.endswith(('.html', '.htm')):
            return extract_text_from_html(content)
        return content

    return None


# ============================================================================
# TEXT EXTRACTION
# ============================================================================

def extract_text_from_html(html_content: str) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)
    except ImportError:
        text = re.sub(r"<style[^>]*>.*?</style>", "", html_content,
                       flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<script[^>]*>.*?</script>", "", text,
                       flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class SECScraper:
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.filings_dir = os.path.join(data_dir, FILINGS_DIR)
        self.universe_path = os.path.join(data_dir, UNIVERSE_FILE)
        self.index_path = os.path.join(data_dir, FILINGS_INDEX)
        self.universe: List[Dict] = []
        self.filings_index: List[Dict] = []
        self._save_lock = threading.Lock()
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.filings_dir, exist_ok=True)
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.universe_path):
            try:
                with open(self.universe_path) as f:
                    self.universe = json.load(f)
                logger.info(f"Loaded universe: {len(self.universe)} companies")
            except json.JSONDecodeError as e:
                logger.error(f"Corrupt universe file {self.universe_path}: {e}")
                self.universe = []
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path) as f:
                    self.filings_index = json.load(f)
                logger.info(f"Loaded filings index: {len(self.filings_index)} filings")
            except json.JSONDecodeError as e:
                logger.error(f"Corrupt filings index {self.index_path}: {e}")
                self.filings_index = []

    def _save_state(self):
        with self._save_lock:
            # Atomic write: write to tmp file then rename
            tmp_universe = self.universe_path + ".tmp"
            with open(tmp_universe, "w") as f:
                json.dump(self.universe, f, indent=2)
            os.replace(tmp_universe, self.universe_path)

            tmp_index = self.index_path + ".tmp"
            with open(tmp_index, "w") as f:
                json.dump(self.filings_index, f, indent=2)
            os.replace(tmp_index, self.index_path)

    def step1_build_universe(self, fmp_api_key: str = None,
                              progress_callback=None,
                              min_market_cap: int = None,
                              max_market_cap: int = None,
                              cancel_event=None) -> int:
        if not fmp_api_key:
            raise RuntimeError(
                "FMP API key is required. The Company Screener endpoint "
                "requires the Starter plan ($22/mo) or higher. "
                "Sign up at: https://financialmodelingprep.com/register"
            )

        if progress_callback:
            progress_callback("Querying FMP screener...")

        # Clear old filings — universe is changing so old filings are stale
        old_count = len(self.filings_index)
        if old_count > 0:
            if progress_callback:
                progress_callback(f"Clearing {old_count} filings from previous universe...")
            self.filings_index = []
            logger.info(f"Cleared {old_count} stale filings (universe rebuild)")

        self.universe = build_universe_fmp(fmp_api_key, progress_callback,
                                          min_market_cap=min_market_cap,
                                          max_market_cap=max_market_cap)

        if not self.universe:
            raise RuntimeError(
                "FMP screener returned 0 companies.\n"
                "1. Need Starter plan ($22/mo) — screener not on free plan\n"
                "2. API key may be invalid\n"
                "3. FMP may be down\n\n"
                "Use 'Test API' button for diagnostics."
            )

        if progress_callback:
            progress_callback(f"Found {len(self.universe)} companies. Resolving CIKs...")

        session = _get_session()
        self.universe = resolve_ciks(self.universe, session, progress_callback)
        self._save_state()
        logger.info(f"Universe built: {len(self.universe)} companies with CIK")
        return len(self.universe)

    def step2_find_filings(self, max_companies: int = None,
                            progress_callback=None,
                            cancel_event=None) -> int:
        if not self.universe:
            raise RuntimeError("No companies in universe. Run Step 1 first.")

        session = _get_session()
        companies = self.universe[:max_companies] if max_companies else self.universe
        total = len(companies)
        found = 0
        target_set = set(TARGET_FORM_TYPES_CIK)
        efts_types = set(TARGET_FORM_TYPES_EFTS)

        # Build set of existing accession numbers for dedup
        existing_accessions = {f.get("accession_number") for f in self.filings_index}

        # Migration v1: companies indexed by old code lack scanned_form_types.
        indexed_ciks = {f.get("cik") for f in self.filings_index}
        for company in companies:
            if (company["cik"] in indexed_ciks
                    and not company.get("scanned_form_types")):
                company["scanned_form_types"] = ["10-K", "10-Q"]

        # Migration v7: added 8-K + DEF 14A form types.
        # Migration v8: added 13F-HR to EFTS search
        SCAN_VERSION = 8
        for company in companies:
            if company.get("scan_version", 0) < SCAN_VERSION:
                prev = set(company.get("scanned_form_types", []))
                # Remove types that need re-scan (new types + normalized ownership)
                clear_types = efts_types | {"8-K", "8-K/A", "DEF 14A", "DEFA14A", "13F-HR"}
                cleaned = prev - clear_types
                if cleaned != prev:
                    company["scanned_form_types"] = sorted(cleaned)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Separate companies that need scanning from those already done
        to_scan = []
        for company in companies:
            prev_scanned = set(company.get("scanned_form_types", []))
            missing_types = target_set - prev_scanned
            if missing_types:
                to_scan.append((company, missing_types))

        done_count = total - len(to_scan)
        lock = threading.Lock()

        def _search_one(company, missing_types):
            session = _get_thread_session()
            cik = company["cik"]
            ticker = company["ticker"]
            company_name = company.get("company_name", "")

            new_filings = []

            # ---- Primary: CIK submissions search (finds ALL form types) ----
            cik_filings = get_company_filings(cik, session,
                                               form_types=list(missing_types))
            new_filings.extend(cik_filings)
            logger.info(f"{ticker}: CIK search for {missing_types} → {len(cik_filings)}")

            # ---- Supplementary: EFTS company name search for SC 13D ----
            missing_efts = missing_types & efts_types
            if missing_efts and company_name:
                cik_accessions = {f.get("accession_number") for f in cik_filings}
                efts_filings = search_filings_about_company(
                    ticker, company_name, cik, session,
                    form_types=list(missing_efts))
                efts_new = 0
                for ef in efts_filings:
                    if ef.get("accession_number") not in cik_accessions:
                        new_filings.append(ef)
                        efts_new += 1
                if efts_new:
                    logger.info(f"{ticker}: EFTS supplementary found {efts_new} extra SC 13D")

            return company, new_filings

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for company, missing_types in to_scan:
                if cancel_event and cancel_event.is_set():
                    break
                futures[executor.submit(_search_one, company, missing_types)] = company

            for future in as_completed(futures):
                try:
                    company, new_filings = future.result()
                except Exception as e:
                    logger.error("Future failed: %s", e)
                    continue
                ticker = company["ticker"]
                with lock:
                    done_count += 1
                    added = 0
                    for filing in new_filings:
                        if filing.get("accession_number") in existing_accessions:
                            continue
                        filing["ticker"] = ticker
                        filing["company_name"] = company.get("company_name", "")
                        filing["exchange"] = company.get("exchange", "")
                        filing["market_cap"] = company.get("market_cap")
                        filing["downloaded"] = False
                        filing["local_path"] = None
                        filing["analyzed"] = False
                        self.filings_index.append(filing)
                        existing_accessions.add(filing.get("accession_number"))
                        found += 1
                        added += 1

                    company["scanned_form_types"] = sorted(target_set)
                    company["scan_version"] = SCAN_VERSION

                    if progress_callback:
                        progress_callback(done_count, total, ticker, added)
                    if done_count % 50 == 0:
                        self._save_state()

        self._save_state()
        logger.info(f"Filing search: {found} new filings across {total} companies")
        return found

    def step3_download_filings(self, max_downloads: int = None,
                                progress_callback=None,
                                cancel_event=None) -> int:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Dedup: remove duplicate accession numbers before downloading (#10)
        seen_acc = set()
        deduped = []
        for f in self.filings_index:
            acc = f.get("accession_number", "")
            if acc and acc in seen_acc:
                logger.debug(f"Dedup: removing duplicate accession {acc}")
                continue
            if acc:
                seen_acc.add(acc)
            deduped.append(f)
        if len(deduped) < len(self.filings_index):
            logger.info(f"Dedup: removed {len(self.filings_index) - len(deduped)} duplicates")
            self.filings_index = deduped

        # Pending = not downloaded + failed retries under limit
        pending = [f for f in self.filings_index
                   if not f.get("downloaded") and f.get("download_retries", 0) < 3]
        if max_downloads:
            pending = pending[:max_downloads]

        total = len(pending)
        downloaded = 0
        done_count = 0
        lock = threading.Lock()

        def _download_one(filing):
            session = _get_thread_session()
            path = download_filing(filing, self.filings_dir, session)
            if path:
                # Text cache: pre-extract and cache cleaned text (#22)
                try:
                    _cache_filing_text(path)
                except Exception as e:
                    logger.debug(f"Text cache failed for {path}: {e}")
            return filing, path

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for filing in pending:
                if cancel_event and cancel_event.is_set():
                    break
                futures[executor.submit(_download_one, filing)] = filing

            for future in as_completed(futures):
                try:
                    filing, path = future.result()
                except Exception as e:
                    logger.error("Future failed: %s", e)
                    continue
                with lock:
                    done_count += 1
                    if path:
                        filing["downloaded"] = True
                        filing["local_path"] = path
                        filing["download_retries"] = 0
                        downloaded += 1
                    else:
                        filing["download_retries"] = filing.get("download_retries", 0) + 1
                        logger.info(
                            f"Download failed for {filing.get('ticker','?')} "
                            f"{filing.get('form_type','?')} (retry {filing['download_retries']}/3)"
                        )
                    if progress_callback:
                        progress_callback(done_count, total, filing.get("ticker", ""),
                                        path is not None)
                    if done_count % 25 == 0:
                        self._save_state()

        self._save_state()
        logger.info(f"Downloaded {downloaded}/{total} filings")
        return downloaded

    # ------------------------------------------------------------------
    # STEP 4: PRE-FETCH FORM 4 INSIDER DATA (aggregate per company)
    # ------------------------------------------------------------------

    def step4_fetch_insider_data(self, progress_callback=None,
                                  cancel_event=None,
                                  company_tickers: Set[str] = None) -> int:
        """
        For each company in universe that has a CIK, fetch Form 4 transactions
        and SC 13D/G metadata from EDGAR, compute aggregate insider signal,
        and store on the company record.  This data is later used by Tier 2
        analysis and surfaced in the Filings tab.

        If company_tickers is provided, only fetch for those tickers (e.g. the
        subset that was actually scanned).
        """
        from insider_tracker import (
            fetch_insider_transactions, fetch_institutional_filings,
            analyze_insider_activity,
        )

        from concurrent.futures import ThreadPoolExecutor, as_completed

        companies = [c for c in self.universe if c.get('cik')]
        if company_tickers:
            upper_tickers = {t.upper() for t in company_tickers}
            companies = [c for c in companies
                         if c.get('ticker', '').upper() in upper_tickers]

        # Split into already-done and pending
        pending = [c for c in companies if not c.get('insider_data')]
        total = len(companies)
        fetched = 0
        done_count = total - len(pending)
        lock = threading.Lock()

        def _fetch_one(company):
            session = _get_thread_session()
            ticker = company.get('ticker', '')
            cik = company['cik']
            try:
                txns = fetch_insider_transactions(cik, session, _edgar_limiter)
                inst = fetch_institutional_filings(cik, session, _edgar_limiter,
                                                    ticker=ticker,
                                                    company_name=company.get('company_name', ''))
                result = analyze_insider_activity(txns, inst)
                data = {
                    'signal': result.get('signal', 'no_data'),
                    'accumulation_score': result.get('accumulation_score', 50),
                    'open_market_buys': result.get('open_market_buys', 0),
                    'open_market_sells': result.get('open_market_sells', 0),
                    'total_buy_value': result.get('total_buy_value', 0),
                    'total_sell_value': result.get('total_sell_value', 0),
                    'net_value': result.get('net_value', 0),
                    'unique_buyers': result.get('unique_buyers', 0),
                    'unique_sellers': result.get('unique_sellers', 0),
                    'cluster_score': result.get('cluster_score', 0),
                    'has_activist_holder': result.get('has_activist_holder', False),
                    'has_passive_large_holder': result.get('has_passive_large_holder', False),
                    'institutional_filing_count': result.get('institutional_filing_count', 0),
                    'transaction_count': result.get('transaction_count', 0),
                    'notable_transactions': result.get('notable_transactions', [])[:5],
                    'summary': result.get('summary', ''),
                }
                return company, ticker, data, True
            except Exception as e:
                logger.warning(f"{ticker} (CIK {cik}): insider fetch failed: {e}")
                data = {
                    'signal': 'error',
                    'accumulation_score': 50,
                    'error': str(e),
                }
                return company, ticker, data, False

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for company in pending:
                if cancel_event and cancel_event.is_set():
                    break
                futures[executor.submit(_fetch_one, company)] = company

            for future in as_completed(futures):
                try:
                    company, ticker, data, success = future.result()
                except Exception as e:
                    logger.error("Future failed: %s", e)
                    continue
                with lock:
                    company['insider_data'] = data
                    if success:
                        fetched += 1
                    done_count += 1
                    if progress_callback:
                        progress_callback(done_count, total, ticker, True)
                    if done_count % 25 == 0:
                        self._save_state()

        self._save_state()
        logger.info(f"Insider data fetched for {fetched}/{total} companies")
        return fetched

    def get_insider_for_ticker(self, ticker: str) -> Optional[Dict]:
        """Return cached insider data for a ticker."""
        for c in self.universe:
            if c.get('ticker', '').upper() == ticker.upper():
                return c.get('insider_data')
        return None

    def get_stats(self) -> Dict[str, Any]:
        total_companies = len(self.universe)
        total_filings = len(self.filings_index)
        downloaded = sum(1 for f in self.filings_index if f.get("downloaded"))
        analyzed = sum(1 for f in self.filings_index if f.get("analyzed"))

        by_exchange = {}
        for c in self.universe:
            ex = c.get("exchange", "Unknown")
            by_exchange[ex] = by_exchange.get(ex, 0) + 1

        by_form = {}
        for f in self.filings_index:
            ft = f.get("form_type", "Unknown")
            by_form[ft] = by_form.get(ft, 0) + 1

        cap_ranges = {"<$20M": 0, "$20-40M": 0, "$40-60M": 0, "$60-80M": 0, "$80-100M": 0}
        for c in self.universe:
            mc = c.get("market_cap", 0) or 0
            if mc < 20_000_000:
                cap_ranges["<$20M"] += 1
            elif mc < 40_000_000:
                cap_ranges["$20-40M"] += 1
            elif mc < 60_000_000:
                cap_ranges["$40-60M"] += 1
            elif mc < 80_000_000:
                cap_ranges["$60-80M"] += 1
            else:
                cap_ranges["$80-100M"] += 1

        insider_fetched = sum(1 for c in self.universe
                              if c.get('insider_data') and c['insider_data'].get('signal') != 'error')

        return {
            "total_companies": total_companies,
            "total_filings": total_filings,
            "filings_downloaded": downloaded,
            "filings_pending_download": total_filings - downloaded,
            "filings_analyzed": analyzed,
            "filings_pending_analysis": downloaded - analyzed,
            "insider_data_fetched": insider_fetched,
            "companies_by_exchange": by_exchange,
            "filings_by_type": by_form,
            "market_cap_distribution": cap_ranges,
            "rate_limiter_fmp": _fmp_limiter.stats,
            "rate_limiter_edgar": _edgar_limiter.stats,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def get_unanalyzed_filings(self) -> List[Dict]:
        return [f for f in self.filings_index
                if f.get("downloaded") and not f.get("analyzed")]

    def mark_analyzed(self, accession_number: str, result_summary: dict = None):
        for f in self.filings_index:
            if f.get("accession_number") == accession_number:
                f["analyzed"] = True
                if result_summary:
                    f["score"] = result_summary.get("final_score")
                    f["risk_rating"] = result_summary.get("risk_rating")
                    f["red_flags"] = result_summary.get("red_flag_count", 0)
                    f["yellow_flags"] = result_summary.get("yellow_flag_count", 0)
                    f["green_flags"] = result_summary.get("green_flag_count", 0)
                    f["sentiment_trajectory"] = result_summary.get("sentiment_trajectory")
                    f["key_concerns"] = result_summary.get("key_concerns", [])
                    f["result_file"] = result_summary.get("result_file")
                break
        self._save_state()

    # ------------------------------------------------------------------
    # SMART RE-ANALYSIS DETECTION (#8)
    # ------------------------------------------------------------------

    def detect_new_filings(self) -> List[Dict]:
        """
        Detect companies that have new filings since their last analysis.
        Returns list of {ticker, company_name, new_filings: [...], last_analyzed}
        """
        # Group filings by ticker
        from collections import defaultdict
        by_ticker = defaultdict(list)
        for f in self.filings_index:
            tk = f.get('ticker', '')
            if tk:
                by_ticker[tk].append(f)

        needs_reanalysis = []
        for ticker, filings in by_ticker.items():
            # Find latest analysis date
            analyzed = [f for f in filings if f.get('llm_analyzed') or f.get('tier2_analyzed')]
            if not analyzed:
                continue  # Never analyzed — will be picked up in normal flow

            latest_analysis = max(
                f.get('analyzed_at', f.get('filing_date', ''))
                for f in analyzed
            )

            # Find filings downloaded but not analyzed after latest_analysis
            new_filings = [
                f for f in filings
                if f.get('downloaded') and not f.get('llm_analyzed')
                and f.get('filing_date', '') > latest_analysis
            ]

            if new_filings:
                company = next(
                    (c for c in self.universe
                     if c.get('ticker', '').upper() == ticker.upper()),
                    {}
                )
                needs_reanalysis.append({
                    'ticker': ticker,
                    'company_name': company.get('company_name', ''),
                    'new_filing_count': len(new_filings),
                    'new_filings': [
                        {'form_type': f.get('form_type'), 'filing_date': f.get('filing_date')}
                        for f in new_filings
                    ],
                    'last_analyzed': latest_analysis,
                })

        needs_reanalysis.sort(key=lambda x: x['new_filing_count'], reverse=True)
        logger.info(f"Smart re-analysis: {len(needs_reanalysis)} companies have new filings")
        return needs_reanalysis

    def get_results_ranked(self, sort_by: str = "score_asc",
                           risk_filter: str = "",
                           exchange_filter: str = "",
                           form_type_filter: str = "",
                           tier_filter: str = "") -> List[Dict]:
        """Return all analyzed filings with unified tier info."""
        results = []
        for f in self.filings_index:
            # Determine analysis tier(s) for this filing
            has_batch = f.get("analyzed", False)
            has_t1 = f.get("llm_analyzed", False)
            has_t2 = f.get("tier2_analyzed", False)

            if not (has_batch or has_t1 or has_t2):
                continue

            # Determine best tier and unified score
            if has_t2:
                tier = "tier2"
                score = f.get("final_gem_score") if f.get("final_gem_score") is not None else f.get("tier1_score", 50)
            elif has_t1:
                tier = "tier1"
                score = f.get("tier1_score", 50)
            else:
                tier = "batch"
                score = f.get("score", 50)

            entry = {
                "ticker": f.get("ticker", ""),
                "company_name": f.get("company_name", ""),
                "form_type": f.get("form_type", ""),
                "filing_date": f.get("filing_date", ""),
                "exchange": f.get("exchange", ""),
                "accession_number": f.get("accession_number", ""),
                "tier": tier,
                "score": score,
                # Batch analysis fields
                "risk_rating": f.get("risk_rating", ""),
                "red_flags": f.get("red_flag_count", 0) or 0,
                "yellow_flags": f.get("yellow_flag_count", 0) or 0,
                "green_flags": f.get("green_flag_count", 0) or 0,
                "sentiment_trajectory": f.get("sentiment_trajectory", ""),
                "key_concerns": f.get("key_concerns", []),
                "result_file": f.get("result_file", ""),
                # LLM Tier 1 fields
                "tier1_score": f.get("tier1_score"),
                "gem_potential": f.get("gem_potential", ""),
                "tier1_result_file": f.get("tier1_result_file", ""),
                # LLM Tier 2 fields
                "tier2_analyzed": has_t2,
                "final_gem_score": f.get("final_gem_score"),
                "conviction": f.get("conviction", ""),
                "recommendation": f.get("recommendation", ""),
                "tier2_result_file": f.get("tier2_result_file", ""),
                # Diff fields
                "has_prior_diff": f.get("has_prior_diff", False),
                "diff_signal": f.get("diff_signal", ""),
            }
            results.append(entry)

        # Filters
        if risk_filter:
            results = [r for r in results if r.get("risk_rating", "") == risk_filter]
        if exchange_filter:
            results = [r for r in results if r.get("exchange", "") == exchange_filter]
        if form_type_filter:
            results = [r for r in results if r.get("form_type", "") == form_type_filter]
        if tier_filter:
            results = [r for r in results if r.get("tier", "") == tier_filter]

        # Sort
        if sort_by == "score_asc":
            results.sort(key=lambda x: x.get("score") if x.get("score") is not None else 100)
        elif sort_by == "score_desc":
            results.sort(key=lambda x: x.get("score") if x.get("score") is not None else 0, reverse=True)
        elif sort_by == "red_flags":
            results.sort(key=lambda x: x.get("red_flags", 0), reverse=True)
        elif sort_by == "ticker":
            results.sort(key=lambda x: x.get("ticker", ""))
        elif sort_by == "date":
            results.sort(key=lambda x: x.get("filing_date", ""), reverse=True)
        elif sort_by == "gem_score_desc":
            results.sort(key=lambda x: x.get("final_gem_score") or x.get("tier1_score") or 0, reverse=True)
        return results

    @staticmethod
    def _normalize_score(score):
        """Normalize any score to 0-100 scale."""
        if score is None:
            return None
        try:
            s = float(score)
        except (TypeError, ValueError):
            return None
        if s <= 10:
            logger.warning("Score in 0-10 range: %s", score)
        return round(min(100, max(0, s)), 1)

    def _get_filing_score(self, f):
        """Get the best available score for a filing, normalized to 0-100."""
        # Prefer tier2 > tier1 > batch
        if f.get('final_gem_score') is not None:
            return self._normalize_score(f['final_gem_score']), 'tier2'
        if f.get('tier1_score') is not None:
            return self._normalize_score(f['tier1_score']), 'tier1'
        if f.get('score') is not None:
            return self._normalize_score(f['score']), 'batch'
        return None, None

    def get_stocks_ranked(self, sort_by: str = "score_desc",
                          exchange_filter: str = "",
                          min_score: float = 0) -> List[Dict]:
        """
        Aggregate filings by ticker into one entry per stock.
        Computes recency-weighted composite score across all filings.
        """
        from collections import defaultdict

        # Group analyzed filings by ticker
        ticker_filings = defaultdict(list)
        for f in self.filings_index:
            score, tier = self._get_filing_score(f)
            if score is None:
                continue
            ticker_filings[f.get('ticker', '')].append({
                'filing_date': f.get('filing_date', ''),
                'form_type': f.get('form_type', ''),
                'score': score,
                'tier': tier,
                'accession_number': f.get('accession_number', ''),
                'final_gem_score': self._normalize_score(f.get('final_gem_score')),
                'tier1_score': self._normalize_score(f.get('tier1_score')),
                'batch_score': self._normalize_score(f.get('score')),
                'conviction': f.get('conviction', ''),
                'recommendation': f.get('recommendation', ''),
                'gem_potential': f.get('gem_potential', ''),
                'diff_signal': f.get('diff_signal', ''),
                'has_prior_diff': f.get('has_prior_diff', False),
                'insider_signal': f.get('insider_signal', ''),
                'accumulation_score': f.get('accumulation_score'),
                'inflection_signal': f.get('inflection_signal', ''),
                'inflection_score': f.get('inflection_score'),
                'risk_rating': f.get('risk_rating', ''),
                'tier2_result_file': f.get('tier2_result_file', ''),
                'tier1_result_file': f.get('tier1_result_file', ''),
                'result_file': f.get('result_file', ''),
                'company_name': f.get('company_name', ''),
                'exchange': f.get('exchange', ''),
            })

        stocks = []
        for ticker, filings in ticker_filings.items():
            if not filings:
                continue

            # Sort by date descending (newest first)
            filings.sort(key=lambda x: x['filing_date'], reverse=True)
            latest = filings[0]

            # Company info
            company = next((c for c in self.universe
                            if c.get('ticker', '').upper() == ticker.upper()), None)
            company_name = latest.get('company_name') or (company or {}).get('company_name', '')
            exchange = latest.get('exchange') or (company or {}).get('exchange', '')

            # ---- Recency-weighted composite score ----
            # Exponential decay: most recent = weight 1.0, each prior filing halves
            weighted_sum = 0
            weight_total = 0
            for i, fl in enumerate(filings):
                weight = 0.5 ** i  # 1.0, 0.5, 0.25, 0.125, ...
                weighted_sum += fl['score'] * weight
                weight_total += weight
            weighted_score = round(weighted_sum / weight_total, 1) if weight_total > 0 else 50

            # ---- Best tier achieved ----
            tiers = [fl['tier'] for fl in filings]
            if 'tier2' in tiers:
                best_tier = 'tier2'
            elif 'tier1' in tiers:
                best_tier = 'tier1'
            else:
                best_tier = 'batch'

            # ---- Trajectory from score trend ----
            trajectory = 'stable'
            if len(filings) >= 2:
                oldest_score = filings[-1]['score']
                newest_score = filings[0]['score']
                change = newest_score - oldest_score
                if change >= 15:
                    trajectory = 'strongly_improving'
                elif change >= 5:
                    trajectory = 'improving'
                elif change <= -15:
                    trajectory = 'strongly_deteriorating'
                elif change <= -5:
                    trajectory = 'deteriorating'

            # ---- Latest filing's qualitative fields ----
            conviction = ''
            recommendation = ''
            gem_potential = ''
            diff_signal = ''
            insider_signal = ''
            accumulation_score = None
            inflection_signal = ''
            inflection_score = None
            one_liner = ''
            for fl in filings:
                if fl.get('conviction') and not conviction:
                    conviction = fl['conviction']
                if fl.get('recommendation') and not recommendation:
                    recommendation = fl['recommendation']
                if fl.get('gem_potential') and not gem_potential:
                    gem_potential = fl['gem_potential']
                if fl.get('diff_signal') and not diff_signal:
                    diff_signal = fl['diff_signal']
                if fl.get('insider_signal') and not insider_signal:
                    insider_signal = fl['insider_signal']
                if fl.get('accumulation_score') is not None and accumulation_score is None:
                    accumulation_score = fl['accumulation_score']
                if fl.get('inflection_signal') and not inflection_signal:
                    inflection_signal = fl['inflection_signal']
                if fl.get('inflection_score') is not None and inflection_score is None:
                    inflection_score = fl['inflection_score']

            # Fallback: get insider signal from Step 4 cached data if no filing has it
            if not insider_signal and company:
                ins = company.get('insider_data')
                if ins and ins.get('signal') not in ('error', 'no_data', None, ''):
                    insider_signal = ins['signal']
                    if accumulation_score is None:
                        accumulation_score = ins.get('accumulation_score', 50)

            entry = {
                'ticker': ticker,
                'company_name': company_name,
                'exchange': exchange,
                'market_cap': (company or {}).get('market_cap'),
                'sector': (company or {}).get('sector', ''),
                'score': weighted_score,
                'best_tier': best_tier,
                'filing_count': len(filings),
                'latest_filing_date': filings[0]['filing_date'],
                'latest_form_type': filings[0]['form_type'],
                'latest_score': filings[0]['score'],
                'trajectory': trajectory,
                'conviction': conviction,
                'recommendation': recommendation,
                'gem_potential': gem_potential,
                'diff_signal': diff_signal,
                'insider_signal': insider_signal,
                'accumulation_score': accumulation_score,
                'inflection_signal': inflection_signal,
                'inflection_score': inflection_score,
                'filings': filings,  # all filing entries for detail view
            }
            stocks.append(entry)

        # ---- Filters ----
        if exchange_filter:
            stocks = [s for s in stocks if s.get('exchange', '') == exchange_filter]
        if min_score:
            stocks = [s for s in stocks if s['score'] >= float(min_score)]

        # ---- Sort ----
        if sort_by == 'score_desc':
            stocks.sort(key=lambda x: x['score'], reverse=True)
        elif sort_by == 'score_asc':
            stocks.sort(key=lambda x: x['score'])
        elif sort_by == 'ticker':
            stocks.sort(key=lambda x: x['ticker'])
        elif sort_by == 'date':
            stocks.sort(key=lambda x: x['latest_filing_date'], reverse=True)
        elif sort_by == 'trajectory':
            traj_order = {'strongly_improving': 5, 'improving': 4, 'stable': 3,
                          'deteriorating': 2, 'strongly_deteriorating': 1}
            stocks.sort(key=lambda x: traj_order.get(x['trajectory'], 3), reverse=True)
        elif sort_by == 'filing_count':
            stocks.sort(key=lambda x: x['filing_count'], reverse=True)

        return stocks

    def clear_universe(self):
        logger.info("Clearing universe data")
        self.universe = []
        self.filings_index = []
        self._save_state()
