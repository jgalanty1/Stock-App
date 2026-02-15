"""
Market Data Module — External data integrations for catalyst detection
=====================================================================
Provides:
  1. Price tracking & backtesting validation
  2. Short interest data (FINRA)
  3. 13F Institutional ownership changes
  4. Peer/sector comparison data
  5. Earnings call transcript integration
  6. Sector/industry classification

All functions degrade gracefully if APIs unavailable.
"""

import os
import json
import logging
import re
import statistics
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# Cache directory for market data
MARKET_DATA_DIR = os.environ.get("MARKET_DATA_DIR", "sec_data/market_data")

# Backtesting windows (days after filing)
BACKTEST_WINDOWS = (30, 60, 90, 180)  # Fix 46: immutable tuple


def _load_config_key(key_name: str) -> str:
    """Load API key from config.json."""
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            return config.get(key_name, '')
    except Exception:
        pass
    return ''


def _get_fmp_key() -> str:
    """Get FMP API key from environment or config (not cached at import time)."""
    return os.environ.get('FMP_API_KEY') or _load_config_key('fmp_api_key')


# ============================================================================
# HTTP HELPERS
# ============================================================================

_session = None
_session_lock = threading.Lock()


def _get_session():
    global _session
    with _session_lock:
        if _session is None:
            import requests
            _session = requests.Session()
            ua = os.environ.get("SEC_USER_AGENT", "")
            if not ua:
                logger.warning("SEC_USER_AGENT not set — EDGAR requests may be blocked")
            _session.headers.update({
                "User-Agent": ua,
                "Accept-Encoding": "gzip, deflate",
            })
    return _session


def _fmp_request(endpoint: str, params: dict = None) -> Optional[Any]:
    """Make FMP API request with error handling."""
    fmp_key = _get_fmp_key()
    if not fmp_key:
        logger.debug("FMP_API_KEY not set, skipping market data request")
        return None

    session = _get_session()
    url = f"{FMP_BASE_URL}/{endpoint}"
    if params is None:
        params = {}
    params["apikey"] = fmp_key

    try:
        resp = session.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            try:
                return resp.json()
            except json.JSONDecodeError as e:
                logger.warning(f"FMP invalid JSON response from {url}: {e}")
                return None
        logger.warning(f"FMP {resp.status_code}: {url}")
    except Exception as e:
        logger.warning(f"FMP request failed: {e}")
    return None


# ============================================================================
# 1. PRICE TRACKING & BACKTESTING (Engine integration priority #1)
# ============================================================================

# Price cache to avoid redundant fetches within a session
_price_cache = {}
_cache_lock = threading.Lock()


def fetch_price_history(ticker: str, days: int = 730) -> Optional[List[Dict]]:
    """Fetch daily price history for backtesting.
    Tries FMP API first, falls back to yfinance (free, no key needed).
    Returns list of {date, open, close, high, low, volume} newest first."""

    # Check session cache
    cache_key = f"{ticker}_{days}"
    with _cache_lock:
        if cache_key in _price_cache:
            return _price_cache[cache_key]

    result = None
    source = "unknown"

    # Try FMP first (if key is set)
    if _get_fmp_key():
        try:
            data = _fmp_request(f"historical-price-full/{ticker}", {
                "timeseries": days,
            })
            if data and isinstance(data, dict) and "historical" in data:
                result = data["historical"]  # Newest first
                source = "fmp"
                logger.debug(f"Price data for {ticker}: {len(result)} days from FMP")
        except Exception as e:
            logger.debug(f"FMP price fetch failed for {ticker}: {e}")

    # Fallback to yfinance (free, no API key)
    if not result:
        try:
            import yfinance as yf
            end_date = datetime.now()  # Note: uses local timezone
            start_date = end_date - timedelta(days=days)
            tk = yf.Ticker(ticker)
            df = tk.history(start=start_date.strftime('%Y-%m-%d'),
                           end=end_date.strftime('%Y-%m-%d'))
            if df is not None and len(df) > 0:
                result = []
                # Normalize column names to handle case variations (fix 37)
                col_map = {c.lower(): c for c in df.columns}
                for idx, row in df.iterrows():
                    result.append({
                        "date": idx.strftime('%Y-%m-%d'),
                        "open": round(float(row.get(col_map.get('open', 'Open'), 0)), 4),
                        "close": round(float(row.get(col_map.get('close', 'Close'), 0)), 4),
                        "high": round(float(row.get(col_map.get('high', 'High'), 0)), 4),
                        "low": round(float(row.get(col_map.get('low', 'Low'), 0)), 4),
                        "volume": int(row.get(col_map.get('volume', 'Volume'), 0)),
                    })
                result.sort(key=lambda x: x['date'], reverse=True)  # Newest first
                source = "yfinance"
                logger.debug(f"Price data for {ticker}: {len(result)} days from yfinance")
        except ImportError:
            logger.warning(
                "yfinance not installed. Install with: pip install yfinance. "
                "Alternatively, set FMP_API_KEY for price data."
            )
        except Exception as e:
            logger.debug(f"yfinance price fetch failed for {ticker}: {e}")

    if result:
        with _cache_lock:
            _price_cache[cache_key] = result
        # Store source alongside cache for later retrieval
        with _cache_lock:
            _price_cache[f"{cache_key}_source"] = source
    return result


def clear_price_cache():
    """Clear the in-memory price cache."""
    global _price_cache
    with _cache_lock:
        _price_cache.clear()


def get_price_at_date(ticker: str, target_date: str,
                      history: List[Dict] = None) -> Optional[float]:
    """Get closing price on or near a specific date."""
    if history is None or len(history) == 0:
        history = fetch_price_history(ticker, 365)
    if history is None or len(history) == 0:
        return None

    # Find closest date (prices might not exist on weekends/holidays)
    target = target_date[:10]
    for entry in history:
        if entry.get("date", "") <= target:
            return entry.get("close") or entry.get("adjClose")
    return None


def compute_backtest(ticker: str, filing_date: str,
                     gem_score: float = 0, cps: float = 0,
                     conviction: str = '') -> Dict:
    """
    Compute price changes at 30/60/90/180 days after a filing date.
    Returns {ticker, filing_date, gem_score, cps, windows: {30: +12.5%, ...}, hit: bool}
    """
    result = {
        "ticker": ticker,
        "filing_date": filing_date,
        "gem_score": gem_score,
        "cps": cps,
        "conviction": conviction,
        "windows": {},
        "hit": False,   # True if any window shows >15% gain
        "computed_at": datetime.now().isoformat(),
    }

    history = fetch_price_history(ticker, 730)
    if not history:
        result["error"] = "no_price_data"
        result["error_detail"] = (
            "Could not fetch price history. "
            "Install yfinance (pip install yfinance) or set FMP_API_KEY."
        )
        return result

    # Track which source actually provided the data
    source_key = f"{ticker}_730_source"
    with _cache_lock:
        result["price_source"] = _price_cache.get(source_key, "unknown")
    result["price_points"] = len(history)

    base_price = get_price_at_date(ticker, filing_date, history)
    if not base_price or base_price <= 0:
        result["error"] = "no_base_price"
        result["error_detail"] = (
            f"No price data found on/before {filing_date}. "
            f"History range: {history[-1].get('date','?')} to {history[0].get('date','?')}"
        )
        return result

    result["base_price"] = base_price

    try:
        base_dt = datetime.strptime(filing_date[:10], "%Y-%m-%d")
    except ValueError:
        result["error"] = "invalid_date"
        return result

    for window in BACKTEST_WINDOWS:
        future_date = (base_dt + timedelta(days=window)).strftime("%Y-%m-%d")
        future_price = get_price_at_date(ticker, future_date, history)
        if future_price and future_price > 0:
            change_pct = round((future_price - base_price) / base_price * 100, 2)
            result["windows"][str(window)] = {
                "days": window,
                "price": future_price,
                "change_pct": change_pct,
                "date": future_date,
            }
            if change_pct >= 15:
                result["hit"] = True

    # Compute benchmark returns and alpha
    try:
        bench = compute_benchmark_return(filing_date)
        if bench.get("windows"):
            result["benchmark"] = bench.get("benchmark", "IWC")
            result["benchmark_return"] = {}
            for w_key, w_data in result["windows"].items():
                bench_w = bench["windows"].get(w_key, {})
                bench_change = bench_w.get("change_pct")
                if bench_change is not None:
                    result["benchmark_return"][w_key] = bench_change
                    alpha = round(w_data["change_pct"] - bench_change, 2)
                    w_data["benchmark_return"] = bench_change
                    w_data["alpha"] = alpha
                    result[f"alpha_{w_key}d"] = alpha
    except Exception as e:
        logger.debug(f"Benchmark computation failed for {ticker}: {e}")

    return result


# Benchmark cache for alpha computation
_benchmark_cache = {}
_benchmark_cache_loaded = False
_bench_lock = threading.Lock()
_BENCHMARK_CACHE_FILE = os.path.join(
    os.environ.get("MARKET_DATA_DIR", "sec_data/market_data"),
    "benchmark_cache.json",
)


def compute_benchmark_return(filing_date: str, benchmark: str = "IWC") -> Dict:
    """
    Compute benchmark (IWC micro-cap ETF) returns for same windows as stock backtests.
    Falls back to IWM (Russell 2000) if IWC unavailable.
    Returns {benchmark, filing_date, windows: {30: {change_pct, ...}, ...}}
    """
    global _benchmark_cache, _benchmark_cache_loaded

    cache_key = f"{benchmark}_{filing_date}"
    with _bench_lock:
        if cache_key in _benchmark_cache:
            return _benchmark_cache[cache_key]

    # Load file cache on first access (fix 27)
    with _bench_lock:
        if not _benchmark_cache_loaded:
            _benchmark_cache_loaded = True
            cache_dir = os.path.dirname(_BENCHMARK_CACHE_FILE)
            os.makedirs(cache_dir, exist_ok=True)
            if os.path.exists(_BENCHMARK_CACHE_FILE):
                try:
                    with open(_BENCHMARK_CACHE_FILE) as f:
                        _benchmark_cache = json.load(f)
                    if cache_key in _benchmark_cache:
                        return _benchmark_cache[cache_key]
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Benchmark cache load failed: {e}")
                    _benchmark_cache = {}
                except Exception:
                    pass

    result = {
        "benchmark": benchmark,
        "filing_date": filing_date,
        "windows": {},
    }

    # Try IWC first, fallback to IWM
    for ticker in ([benchmark, "IWM"] if benchmark == "IWC" else [benchmark]):
        history = fetch_price_history(ticker, 730)
        if not history:
            continue

        base_price = get_price_at_date(ticker, filing_date, history)
        if not base_price or base_price <= 0:
            continue

        result["benchmark"] = ticker
        result["base_price"] = base_price

        try:
            base_dt = datetime.strptime(filing_date[:10], "%Y-%m-%d")
        except ValueError:
            break

        for window in BACKTEST_WINDOWS:
            future_date = (base_dt + timedelta(days=window)).strftime("%Y-%m-%d")
            future_price = get_price_at_date(ticker, future_date, history)
            if future_price and future_price > 0:
                change_pct = round((future_price - base_price) / base_price * 100, 2)
                result["windows"][str(window)] = {
                    "days": window,
                    "price": future_price,
                    "change_pct": change_pct,
                    "date": future_date,
                }
        break  # Got data, stop trying fallbacks

    # Cache result
    with _bench_lock:
        _benchmark_cache[cache_key] = result
    try:
        # Atomic write via tmp + os.replace
        tmp_file = _BENCHMARK_CACHE_FILE + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(_benchmark_cache, f)
        os.replace(tmp_file, _BENCHMARK_CACHE_FILE)
    except Exception:
        pass

    return result


def batch_backtest(filings: List[Dict], data_dir: str = None,
                   db_sync: bool = False) -> List[Dict]:
    """Run backtesting on a list of filings. Caches results."""
    if data_dir is None:
        data_dir = MARKET_DATA_DIR
    cache_file = os.path.join(data_dir, "backtest_results.json")
    os.makedirs(data_dir, exist_ok=True)

    # Load cache
    cached = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                data = json.load(f)
            cached = {r["ticker"] + "_" + r["filing_date"]: r
                      for r in data}
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Backtest cache corrupted, rebuilding: {e}")
            cached = {}
        except Exception:
            cached = {}

    results = []
    total = len(filings)
    errors = 0

    # Create DB connection once before loop (not per-filing)
    _db = None
    if db_sync:
        try:
            from database import SECDatabase
            _db = SECDatabase(data_dir=os.environ.get("SEC_DATA_DIR", "sec_data"))
        except Exception as e:
            logger.debug(f"DB connection failed, will skip sync: {e}")

    for i, filing in enumerate(filings):
        ticker = filing.get("ticker", "")
        fdate = filing.get("filing_date", "")
        key = f"{ticker}_{fdate}"

        if key in cached and not cached[key].get("error"):
            results.append(cached[key])
            continue

        result = compute_backtest(
            ticker, fdate,
            gem_score=filing.get("final_gem_score") or filing.get("gem_score", 0),
            cps=filing.get("cps", 0),
            conviction=filing.get("conviction", ''),
        )
        if result.get("error"):
            errors += 1
        cached[key] = result
        results.append(result)

        # Sync to database if requested
        if _db and not result.get("error"):
            try:
                _db.save_backtest(ticker, fdate, result)
            except Exception as e:
                logger.debug(f"DB sync failed for {ticker}: {e}")

        # Log progress every 10 filings
        if (i + 1) % 10 == 0 or i == total - 1:
            logger.info(f"Backtest progress: {i+1}/{total} "
                        f"({errors} errors so far)")

        # TODO: Use RateLimiter from scraper.py for proper coordination
        time.sleep(0.3)  # Simple rate limit between backtest fetches

    # Save cache
    try:
        with open(cache_file, "w") as f:
            json.dump(list(cached.values()), f)
    except Exception as e:
        logger.warning(f"Failed to save backtest cache: {e}")

    return results


def compute_backtest_stats(results: List[Dict]) -> Dict:
    """Aggregate backtesting statistics for model validation."""
    stats = {
        "total_tested": len(results),
        "with_price_data": 0,
        "by_window": {},
        "by_score_bucket": {},
        "hit_rate": 0,  # % of picks that gained >15% in any window
    }

    for window in BACKTEST_WINDOWS:
        w_key = str(window)
        gains = []
        alphas = []
        for r in results:
            w = r.get("windows", {}).get(w_key)
            if w:
                gains.append(w["change_pct"])
                if w.get("alpha") is not None:
                    alphas.append(w["alpha"])
        if gains:
            w_stats = {
                "count": len(gains),
                "avg_return": round(sum(gains) / len(gains), 2),
                "median_return": round(statistics.median(gains), 2),
                "win_rate": round(sum(1 for g in gains if g > 0) / len(gains) * 100, 1),
                "big_win_rate": round(sum(1 for g in gains if g >= 15) / len(gains) * 100, 1),
                "max_gain": round(max(gains), 2) if gains else 0,
                "max_loss": round(min(gains), 2) if gains else 0,
            }
            if alphas:
                w_stats["avg_alpha"] = round(sum(alphas) / len(alphas), 2)
                w_stats["alpha_win_rate"] = round(
                    sum(1 for a in alphas if a > 0) / len(alphas) * 100, 1)
            stats["by_window"][w_key] = w_stats

    # Score bucket analysis
    for bucket_name, min_score, max_score in [
        ("high_cps_70+", 70, 101), ("mid_cps_50-70", 50, 70),
        ("low_cps_30-50", 30, 50), ("bottom_cps_0-30", 0, 30),
    ]:
        bucket_results = [r for r in results
                          if min_score <= (r.get("cps") or 0) < max_score]
        if bucket_results:
            gains_90 = [r["windows"]["90"]["change_pct"]
                        for r in bucket_results
                        if "90" in r.get("windows", {})]
            stats["by_score_bucket"][bucket_name] = {
                "count": len(bucket_results),
                "avg_90d_return": round(sum(gains_90) / len(gains_90), 2) if gains_90 else None,
                "hit_rate": round(sum(1 for r in bucket_results if r.get("hit"))
                                  / len(bucket_results) * 100, 1),
            }

    stats["with_price_data"] = sum(1 for r in results if not r.get("error"))
    hits = sum(1 for r in results if r.get("hit"))
    if stats["with_price_data"]:
        stats["hit_rate"] = round(hits / stats["with_price_data"] * 100, 1)

    return stats


# ============================================================================
# 2. SHORT INTEREST DATA (CPS signal #13)
# ============================================================================

def fetch_short_interest(ticker: str) -> Optional[Dict]:
    """
    Fetch short interest data. Uses FMP as primary source.
    Returns {short_interest, short_ratio, short_pct_float, days_to_cover}
    """
    # Try FMP short interest endpoint
    data = _fmp_request(f"historical/short-interest/{ticker}", {
        "limit": 5,
    })
    if data and isinstance(data, list) and len(data) > 0:
        latest = data[0]
        return {
            "ticker": ticker,
            "date": latest.get("date", ""),
            "short_interest": latest.get("shortInterest", 0),
            "short_ratio": latest.get("shortRatio", 0),
            "short_pct_float": latest.get("shortPercentFloat", 0),
            "days_to_cover": latest.get("daysToCover", 0),
            "source": "fmp",
            "fetched_at": datetime.now().isoformat(),
        }

    # Fallback: try key-metrics endpoint which sometimes has liquidity data
    # Note: shortTermCoverageRatios is a liquidity metric, not short interest.
    # We include it as a supplementary liquidity signal, not for short squeeze analysis.
    metrics = _fmp_request(f"key-metrics/{ticker}", {"limit": 1})
    if metrics and isinstance(metrics, list) and len(metrics) > 0:
        m = metrics[0]
        liquidity_ratio = m.get("shortTermCoverageRatios")
        if liquidity_ratio:
            return {
                "ticker": ticker,
                "date": "",
                "short_interest": 0,
                "short_ratio": 0,
                "short_pct_float": 0,
                "days_to_cover": 0,
                "liquidity_ratio": liquidity_ratio,
                "source": "fmp_metrics",
                "fetched_at": datetime.now().isoformat(),
            }

    return None


def is_short_squeeze_candidate(short_data: Dict, insider_data: Dict = None,
                                cps: float = 0) -> Dict:
    """
    Evaluate short squeeze potential.
    High short interest + insider buying + improving fundamentals = squeeze setup.
    """
    result = {"is_candidate": False, "score": 0, "signals": []}

    si_pct = short_data.get("short_pct_float", 0) or 0
    days_cover = short_data.get("days_to_cover", 0) or 0

    if si_pct >= 15:
        result["score"] += 30
        result["signals"].append(f"High short interest: {si_pct:.1f}% of float")
    elif si_pct >= 10:
        result["score"] += 15
        result["signals"].append(f"Elevated short interest: {si_pct:.1f}% of float")

    if days_cover >= 5:
        result["score"] += 20
        result["signals"].append(f"High days to cover: {days_cover:.1f}")

    if insider_data and insider_data.get("signal") in ("accumulation", "strong_accumulation"):
        result["score"] += 25
        result["signals"].append(f"Insider accumulation: {insider_data.get('signal')}")

    if cps >= 60:
        result["score"] += 25
        result["signals"].append(f"Strong catalyst potential: CPS={cps}")

    result["is_candidate"] = result["score"] >= 50
    return result


# ============================================================================
# 3. 13F INSTITUTIONAL OWNERSHIP (CPS signal #14)
# ============================================================================

def fetch_institutional_holders(ticker: str) -> Optional[List[Dict]]:
    """Fetch institutional holders from FMP."""
    data = _fmp_request(f"institutional-holder/{ticker}")
    if data and isinstance(data, list):
        return [{
            "holder": h.get("holder", ""),
            "shares": h.get("shares", 0),
            "date_reported": h.get("dateReported", ""),
            "change": h.get("change", 0),
            "change_pct": h.get("changePercentage", 0),
        } for h in data[:20]]  # Top 20
    return None


def analyze_institutional_changes(holders: List[Dict]) -> Dict:
    """Analyze institutional ownership patterns."""
    if not holders:
        return {"signal": "no_data"}

    total_change = sum(h.get("change", 0) for h in holders)
    buyers = sum(1 for h in holders if h.get("change", 0) > 0)
    sellers = sum(1 for h in holders if h.get("change", 0) < 0)
    big_movers = [h for h in holders if abs(h.get("change_pct", 0)) >= 20]

    signal = "neutral"
    if buyers > sellers * 2:
        signal = "institutional_accumulation"
    elif sellers > buyers * 2:
        signal = "institutional_distribution"
    elif buyers > sellers:
        signal = "slight_accumulation"

    return {
        "signal": signal,
        "total_holders": len(holders),
        "net_buyers": buyers,
        "net_sellers": sellers,
        "net_change_shares": total_change,
        "big_movers": big_movers[:5],
    }


# ============================================================================
# 4. PEER COMPARISON (CPS dimension #5)
# ============================================================================

def fetch_sector_peers(ticker: str) -> Optional[List[str]]:
    """Get sector peers for comparison."""
    data = _fmp_request(f"stock_peers", {"symbol": ticker})
    if data and isinstance(data, list) and len(data) > 0:
        peers = data[0].get("peersList", [])
        return peers[:10] if peers else None
    return None


def fetch_sector_metrics(ticker: str) -> Optional[Dict]:
    """Fetch company profile including sector/industry and key ratios."""
    data = _fmp_request(f"profile/{ticker}")
    if data and isinstance(data, list) and len(data) > 0:
        p = data[0]
        return {
            "sector": p.get("sector", ""),
            "industry": p.get("industry", ""),
            "market_cap": p.get("mktCap", 0),
            "pe_ratio": p.get("peRatio"),
            "price_to_book": p.get("priceToBook"),
            "beta": p.get("beta"),
            "price": p.get("price"),
            "avg_volume": p.get("volAvg"),
            "description": p.get("description", "")[:300],
        }
    return None


def compute_peer_comparison(ticker: str, peers: List[str] = None) -> Optional[Dict]:
    """Compare company metrics against sector peers."""
    company = fetch_sector_metrics(ticker)
    if not company:
        return None

    if not peers:
        peers = fetch_sector_peers(ticker)
    if not peers:
        return {"company": company, "peers": [], "comparison": "no_peers"}

    peer_metrics = []
    for p in peers[:5]:  # Limit API calls
        m = fetch_sector_metrics(p)
        if m:
            peer_metrics.append({"ticker": p, **m})
        # TODO: Coordinate with RateLimiter from scraper.py for shared FMP rate limit
        time.sleep(0.2)

    if not peer_metrics:
        return {"company": company, "peers": [], "comparison": "no_peer_data"}

    # Compare key ratios
    peer_pes = [p["pe_ratio"] for p in peer_metrics
                if p.get("pe_ratio") is not None and p["pe_ratio"] > 0]
    peer_ptbs = [p["price_to_book"] for p in peer_metrics
                 if p.get("price_to_book") is not None and p["price_to_book"] > 0]

    comparison = {
        "company": company,
        "peer_count": len(peer_metrics),
        "sector": company.get("sector", ""),
        "industry": company.get("industry", ""),
    }

    if peer_pes and company.get("pe_ratio"):
        avg_pe = sum(peer_pes) / len(peer_pes)
        if abs(avg_pe) < 0.01:
            comparison["pe_vs_peers"] = None
            comparison["pe_discount"] = 0
        else:
            comparison["pe_vs_peers"] = round(company["pe_ratio"] / avg_pe, 2)
            comparison["pe_discount"] = round((1 - company["pe_ratio"] / avg_pe) * 100, 1)

    if peer_ptbs and company.get("price_to_book"):
        avg_ptb = sum(peer_ptbs) / len(peer_ptbs)
        comparison["ptb_vs_peers"] = round(company["price_to_book"] / avg_ptb, 2) if avg_ptb else None
        comparison["ptb_discount"] = round((1 - company["price_to_book"] / avg_ptb) * 100, 1) if avg_ptb else 0

    return comparison


def format_peer_comparison_for_prompt(comparison: Dict) -> str:
    """Format peer comparison data for LLM context."""
    if not comparison or comparison.get("comparison") == "no_peers":
        return ""

    company = comparison.get("company", {})
    lines = [
        f"SECTOR: {comparison.get('sector', 'Unknown')} / {comparison.get('industry', 'Unknown')}",
    ]

    if comparison.get("pe_discount") is not None:
        disc = comparison["pe_discount"]
        direction = "discount" if disc > 0 else "premium"
        lines.append(f"P/E vs peers: {abs(disc):.1f}% {direction} "
                      f"(company: {company.get('pe_ratio', 'N/A')}, "
                      f"peer avg implied)")

    if comparison.get("ptb_discount") is not None:
        disc = comparison["ptb_discount"]
        direction = "discount" if disc > 0 else "premium"
        lines.append(f"P/B vs peers: {abs(disc):.1f}% {direction} "
                      f"(company: {company.get('price_to_book', 'N/A')})")

    return "\n".join(lines)


# ============================================================================
# 5. EARNINGS CALL TRANSCRIPTS (CPS signal #12)
# ============================================================================

def fetch_earnings_transcript(ticker: str, year: int = None,
                               quarter: int = None) -> Optional[Dict]:
    """Fetch earnings call transcript from FMP."""
    if year is None:
        year = datetime.now().year
    if quarter is None:
        quarter = max(1, (datetime.now().month - 1) // 3)

    data = _fmp_request(f"earning_call_transcript/{ticker}", {
        "year": year,
        "quarter": quarter,
    })
    if data and isinstance(data, list) and len(data) > 0:
        t = data[0]
        return {
            "ticker": ticker,
            "date": t.get("date", ""),
            "year": year,
            "quarter": quarter,
            "content": t.get("content", ""),
            "source": "fmp",
        }
    return None


def fetch_latest_transcript(ticker: str) -> Optional[Dict]:
    """Try to fetch the most recent earnings call transcript."""
    now = datetime.now()
    year = now.year
    quarter = max(1, (now.month - 1) // 3)

    # Try current quarter, then previous quarters
    for _ in range(4):
        t = fetch_earnings_transcript(ticker, year, quarter)
        if t and t.get("content"):
            return t
        quarter -= 1
        if quarter < 1:
            quarter = 4
            year -= 1
        time.sleep(0.2)

    return None


def extract_transcript_signals(transcript: Dict) -> Dict:
    """
    Extract key signals from earnings call transcript text.
    Produces both categorical signals and a numeric 0-100 transcript_score.
    """
    content = transcript.get("content", "")
    if not content:
        return {"signals": [], "summary": "", "transcript_score": 50}

    content_lower = content.lower()
    words = content.split()
    word_count = len(words)

    signals = []
    score_adj = 0  # adjustments from neutral 50

    # ---- Guidance language ----
    positive_guidance = ["raised guidance", "increased guidance", "raised our",
                         "above expectations", "ahead of plan", "accelerat",
                         "record revenue", "record quarter", "record earnings",
                         "exceeded expectations", "outperformed", "beat consensus",
                         "upside surprise", "stronger than expected"]
    negative_guidance = ["lowered guidance", "reduced guidance", "below expectations",
                         "headwinds", "challenging environment", "restructur",
                         "missed expectations", "shortfall", "deteriorat",
                         "weaker than expected", "downside", "going concern"]

    for phrase in positive_guidance:
        if phrase in content_lower:
            signals.append({"type": "positive_guidance", "phrase": phrase})
            score_adj += 3
    for phrase in negative_guidance:
        if phrase in content_lower:
            signals.append({"type": "negative_guidance", "phrase": phrase})
            score_adj -= 3

    # ---- Catalyst keywords ----
    catalyst_phrases = ["strategic alternatives", "exploring options",
                        "acquisition", "merger", "buyback", "share repurchase",
                        "special dividend", "spin-off", "spinoff",
                        "activist", "board seats", "management change",
                        "strategic review", "privatization", "going private"]
    for phrase in catalyst_phrases:
        if phrase in content_lower:
            signals.append({"type": "catalyst", "phrase": phrase})
            score_adj += 2

    # ---- Loughran-McDonald sentiment scoring ----
    # Subset of high-signal LM positive/negative words for speed
    LM_POSITIVE = {
        "achieve", "attain", "benefit", "efficient", "enhance", "excellent",
        "favorable", "gain", "improve", "leader", "opportunity", "outperform",
        "positive", "profit", "profitab", "progress", "strength", "strong",
        "succeed", "superior", "surpass", "upturn", "uptick",
    }
    LM_NEGATIVE = {
        "adverse", "against", "breach", "burden", "closure", "decline",
        "default", "deficit", "delay", "deteriorat", "difficult", "disappoint",
        "downturn", "impair", "inability", "insolven", "lawsuit", "layoff",
        "litigat", "loss", "negativ", "penalty", "problem", "risk",
        "shortage", "slowdown", "terminat", "threat", "unfavorab", "weak",
    }

    pos_count = 0
    neg_count = 0
    for w in words:
        wl = w.lower().strip(".,;:!?()\"'")
        for p in LM_POSITIVE:
            if wl.startswith(p):
                pos_count += 1
                break
        for n in LM_NEGATIVE:
            if wl.startswith(n):
                neg_count += 1
                break

    total_sentiment_words = pos_count + neg_count
    if total_sentiment_words > 0:
        sentiment_ratio = (pos_count - neg_count) / total_sentiment_words
    else:
        sentiment_ratio = 0

    # Scale: sentiment_ratio ranges roughly -1 to +1
    lm_sentiment_score = max(0, min(100, 50 + sentiment_ratio * 30))
    score_adj += int(sentiment_ratio * 10)

    # ---- Management tone analysis ----
    # Confidence indicators
    confidence_phrases = ["confident", "conviction", "committed to", "well positioned",
                          "on track", "executing well", "ahead of schedule", "momentum"]
    uncertainty_phrases = ["uncertain", "unpredictable", "volatility", "cautious",
                           "prudent", "conservative estimate", "visibility is limited",
                           "difficult to predict"]

    confidence_count = sum(1 for p in confidence_phrases if p in content_lower)
    uncertainty_count = sum(1 for p in uncertainty_phrases if p in content_lower)

    if confidence_count > uncertainty_count + 2:
        signals.append({"type": "tone", "phrase": f"management_confident ({confidence_count} vs {uncertainty_count})"})
        score_adj += 4
    elif uncertainty_count > confidence_count + 2:
        signals.append({"type": "tone", "phrase": f"management_uncertain ({uncertainty_count} vs {confidence_count})"})
        score_adj -= 4

    # ---- Forward-looking statements density ----
    forward_phrases = ["expect", "anticipate", "project", "forecast", "outlook",
                       "looking ahead", "going forward", "next quarter", "next year",
                       "pipeline", "backlog", "order book"]
    forward_count = sum(1 for p in forward_phrases if p in content_lower)
    forward_density = forward_count / max(1, word_count) * 1000
    # High forward density suggests management willing to provide visibility
    if forward_density > 5:
        score_adj += 2

    # ---- Numeric guidance extraction ----
    # Look for specific numeric targets in guidance
    guidance_patterns = [
        r'(?:revenue|sales)\s+(?:of|between|range)\s+\$[\d,.]+',
        r'(?:eps|earnings per share)\s+(?:of|between|range)\s+\$[\d,.]+',
        r'(?:margin|margins)\s+(?:of|to|between)\s+[\d,.]+%',
        r'(?:growth|increase)\s+(?:of|between)\s+[\d,.]+%',
    ]
    numeric_guidance = []
    for pat in guidance_patterns:
        matches = re.findall(pat, content_lower)
        numeric_guidance.extend(matches[:3])

    if numeric_guidance:
        signals.append({"type": "numeric_guidance", "phrase": "; ".join(numeric_guidance[:5])})
        score_adj += 2  # Specificity is positive

    # ---- Compute final transcript score ----
    transcript_score = max(0, min(100, 50 + score_adj))

    positive_count = sum(1 for s in signals if s["type"] == "positive_guidance")
    negative_count = sum(1 for s in signals if s["type"] == "negative_guidance")
    catalyst_count = sum(1 for s in signals if s["type"] == "catalyst")

    return {
        "signals": signals,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "catalyst_count": catalyst_count,
        "word_count": word_count,
        "transcript_score": transcript_score,
        "lm_sentiment_score": round(lm_sentiment_score, 1),
        "sentiment_ratio": round(sentiment_ratio, 3),
        "lm_positive_words": pos_count,
        "lm_negative_words": neg_count,
        "confidence_count": confidence_count,
        "uncertainty_count": uncertainty_count,
        "forward_density": round(forward_density, 2),
        "numeric_guidance_found": len(numeric_guidance),
    }


# ============================================================================
# 6. QUANTITATIVE PRE-FILTER (CPS signal #11)
# ============================================================================

def fetch_quant_screen_data(ticker: str) -> Optional[Dict]:
    """Fetch key quantitative metrics for cheap pre-screening."""
    # Key metrics endpoint has what we need
    metrics = _fmp_request(f"key-metrics-ttm/{ticker}")
    ratios = _fmp_request(f"ratios-ttm/{ticker}")

    result = {"ticker": ticker}

    if metrics and isinstance(metrics, list) and len(metrics) > 0:
        m = metrics[0]
        result.update({
            "pe_ratio": m.get("peRatioTTM"),
            "pb_ratio": m.get("pbRatioTTM"),
            "ev_ebitda": m.get("enterpriseValueOverEBITDATTM"),
            "roe": m.get("roeTTM"),
            "debt_equity": m.get("debtEquityRatioTTM"),
            "revenue_per_share": m.get("revenuePerShareTTM"),
            "book_value_ps": m.get("bookValuePerShareTTM"),
            "market_cap": m.get("marketCapTTM"),
        })

    if ratios and isinstance(ratios, list) and len(ratios) > 0:
        r = ratios[0]
        result.update({
            "gross_margin": r.get("grossProfitMarginTTM"),
            "operating_margin": r.get("operatingProfitMarginTTM"),
            "net_margin": r.get("netProfitMarginTTM"),
            "current_ratio": r.get("currentRatioTTM"),
            "quick_ratio": r.get("quickRatioTTM"),
        })

    return result if len(result) > 1 else None


def quant_prefilter_score(data: Dict) -> Tuple[float, List[str]]:
    """
    Quick quantitative score (0-100) for pre-filtering before LLM.
    Focuses on value + quality signals that predict catalysts.
    Returns (score, reasons).
    """
    score = 50  # Neutral baseline
    reasons = []

    pb = data.get("pb_ratio")
    if pb is not None and pb > 0:
        if pb < 1.0:
            score += 15
            reasons.append(f"Trading below book value (P/B={pb:.2f})")
        elif pb < 1.5:
            score += 8
            reasons.append(f"Low P/B ratio ({pb:.2f})")
        elif pb > 5:
            score -= 5

    pe = data.get("pe_ratio")
    if pe is not None:
        if 0 < pe < 10:
            score += 10
            reasons.append(f"Low P/E ({pe:.1f})")
        elif 10 <= pe < 15:
            score += 5
        elif pe < 0:
            score -= 5  # Negative earnings

    ev_ebitda = data.get("ev_ebitda")
    if ev_ebitda is not None and ev_ebitda > 0:
        if ev_ebitda < 6:
            score += 10
            reasons.append(f"Cheap EV/EBITDA ({ev_ebitda:.1f})")
        elif ev_ebitda < 10:
            score += 5

    gm = data.get("gross_margin")
    if gm is not None:
        if gm > 0.4:
            score += 5
            reasons.append(f"Strong gross margin ({gm:.0%})")
        elif gm < 0.2:
            score -= 5

    cr = data.get("current_ratio")
    if cr is not None:
        if cr > 2.0:
            score += 5
            reasons.append(f"Strong balance sheet (CR={cr:.1f})")
        elif cr < 1.0:
            score -= 10
            reasons.append(f"Liquidity concern (CR={cr:.1f})")

    roe = data.get("roe")
    if roe is not None:
        if roe > 0.15:
            score += 5
        elif roe < 0:
            score -= 5

    return max(0, min(100, score)), reasons


# ============================================================================
# 7. INSIDER TIMING CORRELATION (CPS dimension #4)
# ============================================================================

def correlate_insider_timing(insider_data: Dict, filing_date: str) -> Dict:
    """
    Analyze timing relationship between insider buys and filing dates.
    Insiders buying right before/after a filing is a much stronger signal.
    """
    result = {
        "timing_signal": "neutral",
        "timing_score": 0,  # -100 to +100
        "buys_near_filing": 0,
        "sells_near_filing": 0,
        "details": [],
    }

    transactions = insider_data.get("transactions", [])
    if not transactions or not filing_date:
        return result

    try:
        filing_dt = datetime.strptime(filing_date[:10], "%Y-%m-%d")
    except ValueError:
        return result

    near_buys = 0
    near_sells = 0
    very_near_buys = 0  # Within 14 days

    for txn in transactions:
        txn_date = txn.get("date", "") or txn.get("transaction_date", "")
        if not txn_date:
            continue
        try:
            txn_dt = datetime.strptime(txn_date[:10], "%Y-%m-%d")
        except ValueError:
            continue

        days_diff = (txn_dt - filing_dt).days
        is_buy = txn.get("is_purchase", False) or txn.get("type", "") == "P"

        # Within 30 days before or after filing
        if -30 <= days_diff <= 30:
            if is_buy:
                near_buys += 1
                if -14 <= days_diff <= 14:
                    very_near_buys += 1
                result["details"].append({
                    "date": txn_date,
                    "days_from_filing": days_diff,
                    "action": "buy",
                    "insider": txn.get("name", txn.get("insider_name", "")),
                    "shares": txn.get("shares", 0),
                    "value": txn.get("value", 0),
                })
            else:
                near_sells += 1
                result["details"].append({
                    "date": txn_date,
                    "days_from_filing": days_diff,
                    "action": "sell",
                    "insider": txn.get("name", txn.get("insider_name", "")),
                })

    result["buys_near_filing"] = near_buys
    result["sells_near_filing"] = near_sells

    # Score: +100 means insiders heavily buying around filing
    if near_buys > 0 and near_sells == 0:
        result["timing_score"] = min(100, 30 + near_buys * 15 + very_near_buys * 20)
        result["timing_signal"] = "strong_buy_timing" if very_near_buys >= 2 else "buy_timing"
    elif near_buys > near_sells:
        result["timing_score"] = min(80, near_buys * 10 - near_sells * 5)
        result["timing_signal"] = "mixed_positive"
    elif near_sells > near_buys * 2:
        result["timing_score"] = max(-100, -(near_sells * 15))
        result["timing_signal"] = "sell_timing"
    elif near_sells > 0:
        result["timing_score"] = max(-50, -(near_sells * 10) + near_buys * 5)
        result["timing_signal"] = "mixed_negative"

    return result


# ============================================================================
# AGGREGATE: Fetch all market data for a ticker
# ============================================================================

def fetch_all_market_data(ticker: str, filing_date: str = "",
                          insider_data: Dict = None) -> Dict:
    """Fetch all available market data for a ticker. Used to enrich CPS."""
    result = {
        "ticker": ticker,
        "fetched_at": datetime.now().isoformat(),
    }

    # Short interest
    try:
        si = fetch_short_interest(ticker)
        if si:
            result["short_interest"] = si
    except Exception as e:
        logger.debug(f"{ticker}: short interest fetch failed: {e}")

    # Institutional holders
    try:
        holders = fetch_institutional_holders(ticker)
        if holders:
            result["institutional"] = analyze_institutional_changes(holders)
    except Exception as e:
        logger.debug(f"{ticker}: institutional data fetch failed: {e}")

    # Peer comparison
    try:
        peers = compute_peer_comparison(ticker)
        if peers:
            result["peer_comparison"] = peers
    except Exception as e:
        logger.debug(f"{ticker}: peer comparison failed: {e}")

    # Insider timing correlation
    if insider_data and filing_date:
        try:
            timing = correlate_insider_timing(insider_data, filing_date)
            result["insider_timing"] = timing
        except Exception as e:
            logger.debug(f"{ticker}: insider timing failed: {e}")

    # Quant screen
    try:
        quant = fetch_quant_screen_data(ticker)
        if quant:
            score, reasons = quant_prefilter_score(quant)
            result["quant_screen"] = {
                "data": quant,
                "score": score,
                "reasons": reasons,
            }
    except Exception as e:
        logger.debug(f"{ticker}: quant screen failed: {e}")

    return result
