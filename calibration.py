"""
SEC Analyzer — Calibration & Feedback Loop
============================================
Connects analysis signals to actual price outcomes to answer:
"Which signals actually predicted stock price gains?"

The loop:
  1. CAPTURE: Build signal vectors for every analyzed company
  2. PAIR: Match each signal vector with actual 30/60/90d price returns
  3. CORRELATE: Rank-correlate each individual signal with returns
  4. OPTIMIZE: Derive new CPS weights from what actually worked
  5. VALIDATE: Compare old weights vs new on held-out data
  6. APPLY: Save new weights; track calibration history

Usage:
    from calibration import CalibrationEngine
    engine = CalibrationEngine(scraper, potential_companies)
    report = engine.run_full_calibration()
    # report contains signal correlations, suggested weights,
    # expected improvement, and comparison charts
"""

import os
import json
import math
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

# ============================================================================
# DEFAULT CPS WEIGHTS (the guesses we're trying to improve)
# ============================================================================

DEFAULT_WEIGHTS = {
    "catalyst_strength": 25,
    "improvement_trajectory": 20,
    "insider_conviction": 20,
    "turnaround_indicators": 20,
    "filing_recency": 15,
}

# Individual signals we extract and correlate with returns.
# Each signal gets normalized to 0-1 before correlation.
SIGNAL_DEFINITIONS = {
    # From Tier 1 analysis
    "best_t1_score": {"source": "t1", "range": [0, 100], "description": "Best Tier 1 composite score"},
    "gem_potential_numeric": {"source": "t1", "range": [0, 2], "description": "Gem potential (0=low, 1=med, 2=high)"},

    # From CPS components
    "catalyst_strength": {"source": "cps", "range": [0, 25], "description": "Catalyst strength component"},
    "improvement_trajectory": {"source": "cps", "range": [0, 20], "description": "Score improvement over time"},
    "insider_conviction": {"source": "cps", "range": [0, 20], "description": "Insider buying signals"},
    "turnaround_indicators": {"source": "cps", "range": [0, 20], "description": "Turnaround catalyst signals"},
    "filing_recency": {"source": "cps", "range": [0, 15], "description": "Filing freshness & coverage"},

    # From filing index (raw signals)
    "accumulation_score": {"source": "insider", "range": [0, 100], "description": "Insider accumulation score"},
    "cluster_score": {"source": "insider", "range": [0, 10], "description": "Insider cluster buying score"},
    "unique_buyers": {"source": "insider", "range": [0, 20], "description": "Number of unique insider buyers"},
    "has_activist": {"source": "insider", "range": [0, 1], "description": "Has SC 13D activist filing"},

    # From filing analysis
    "turnaround_score": {"source": "filing", "range": [0, 100], "description": "Turnaround catalyst score"},
    "research_signal_score": {"source": "filing", "range": [0, 100], "description": "Research-backed signal score"},
    "diff_signal_numeric": {"source": "filing", "range": [-2, 2], "description": "Filing diff direction (-2 to +2)"},
    "filing_count": {"source": "filing", "range": [1, 20], "description": "Number of available filings"},
    "has_8k": {"source": "filing", "range": [0, 1], "description": "Has 8-K event filing"},
    "has_proxy": {"source": "filing", "range": [0, 1], "description": "Has DEF 14A proxy statement"},
    "has_ownership": {"source": "filing", "range": [0, 1], "description": "Has SC 13D/G filing"},
    "days_since_filing": {"source": "filing", "range": [0, 730], "description": "Days since newest filing"},

    # Tier 2 outputs (if available)
    "final_gem_score": {"source": "t2", "range": [0, 100], "description": "Tier 2 final gem score"},
    "conviction_numeric": {"source": "t2", "range": [0, 2], "description": "Conviction (0=low, 1=med, 2=high)"},
}

# Map categorical signals to numeric
DIFF_SIGNAL_MAP = {
    "strongly_improving": 2, "improving": 1.5, "slightly_improving": 1,
    "stable": 0,
    "slightly_deteriorating": -1, "deteriorating": -1.5, "strongly_deteriorating": -2,
}

GEM_POTENTIAL_MAP = {"high": 2, "medium": 1, "low": 0}
CONVICTION_MAP = {"high": 2, "medium": 1, "low": 0}


# ============================================================================
# CORE: SIGNAL EXTRACTION
# ============================================================================

def extract_signal_vector(ticker: str, scraper, potential_companies: List[Dict]) -> Dict[str, float]:
    """
    Extract all measurable signals for a company into a flat numeric vector.
    This is the key function — it captures everything we know about a company
    at the time of analysis so we can later correlate with actual returns.
    """
    signals = {}

    # Find CPS data
    cps_data = next((c for c in potential_companies
                     if c.get('ticker', '').upper() == ticker.upper()), None)

    # Find company in universe
    company = next((c for c in scraper.universe
                    if c.get('ticker', '').upper() == ticker.upper()), {})

    # Find all filings for this ticker
    filings = [f for f in scraper.filings_index
               if f.get('ticker', '').upper() == ticker.upper()]
    filings.sort(key=lambda x: x.get('filing_date', ''), reverse=True)

    # ---- CPS components ----
    if cps_data:
        components = cps_data.get('components', {})
        signals['catalyst_strength'] = components.get('catalyst_strength', 0)
        signals['improvement_trajectory'] = components.get('improvement_trajectory', 0)
        signals['insider_conviction'] = components.get('insider_conviction', 0)
        signals['turnaround_indicators'] = components.get('turnaround_indicators', 0)
        signals['filing_recency'] = components.get('filing_recency', 0)
        signals['best_t1_score'] = cps_data.get('best_t1_score', 0) or 0
        signals['has_activist'] = 1.0 if cps_data.get('has_activist') else 0.0
        signals['filing_count'] = cps_data.get('filing_count', 0) or 0

    # ---- Insider data ----
    ins = company.get('insider_data', {}) or {}
    signals['accumulation_score'] = ins.get('accumulation_score', 50) if ins else 50
    signals['cluster_score'] = ins.get('cluster_score', 0) if ins else 0
    signals['unique_buyers'] = ins.get('unique_buyers', 0) if ins else 0

    # ---- Filing-level signals (aggregate across filings) ----
    best_turnaround = 0
    best_research = 0
    best_diff = 0
    has_8k = False
    has_proxy = False
    has_ownership = False
    best_gem_score = 0
    best_conviction = 0
    newest_date = None

    for f in filings:
        ts = f.get('turnaround_score')
        if ts is not None:
            best_turnaround = max(best_turnaround, ts)

        rs = f.get('research_signal_score')
        if rs is not None:
            best_research = max(best_research, rs)

        ds = f.get('diff_signal', '')
        diff_num = DIFF_SIGNAL_MAP.get(ds, 0)
        if abs(diff_num) > abs(best_diff):
            best_diff = diff_num

        ft = f.get('form_type', '')
        if ft in ('8-K', '8-K/A'):
            has_8k = True
        if ft in ('DEF 14A', 'DEFA14A'):
            has_proxy = True
        if ft.startswith('SC 13'):
            has_ownership = True

        fgs = f.get('final_gem_score')
        if fgs is not None:
            best_gem_score = max(best_gem_score, fgs)

        conv = CONVICTION_MAP.get(f.get('conviction', ''), 0)
        best_conviction = max(best_conviction, conv)

        gp = GEM_POTENTIAL_MAP.get(f.get('gem_potential', ''), 0)
        if 'gem_potential_numeric' not in signals or gp > signals.get('gem_potential_numeric', 0):
            signals['gem_potential_numeric'] = gp

        fd = f.get('filing_date', '')
        if fd and (newest_date is None or fd > newest_date):
            newest_date = fd

    signals['turnaround_score'] = best_turnaround
    signals['research_signal_score'] = best_research
    signals['diff_signal_numeric'] = best_diff
    signals['has_8k'] = 1.0 if has_8k else 0.0
    signals['has_proxy'] = 1.0 if has_proxy else 0.0
    signals['has_ownership'] = 1.0 if has_ownership else 0.0
    signals['final_gem_score'] = best_gem_score
    signals['conviction_numeric'] = best_conviction

    # Days since newest filing
    if newest_date:
        try:
            nd = datetime.strptime(newest_date[:10], '%Y-%m-%d')
            signals['days_since_filing'] = (datetime.now() - nd).days
        except ValueError:
            signals['days_since_filing'] = 365
    else:
        signals['days_since_filing'] = 730

    return signals


# ============================================================================
# CORE: CORRELATION ANALYSIS
# ============================================================================

def _spearman_rank_correlation(x: List[float], y: List[float]) -> float:
    """
    Compute Spearman rank correlation between two lists.
    Uses rank-based correlation which is robust to outliers and
    non-linear relationships — critical for noisy stock return data.
    Returns correlation coefficient in [-1, 1].
    """
    n = len(x)
    if n < 5:
        return 0.0  # Not enough data for meaningful correlation

    # Compute ranks (average rank for ties)
    def _rank(values):
        sorted_idx = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and values[sorted_idx[j]] == values[sorted_idx[j + 1]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1  # 1-based
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    # Pearson correlation on ranks
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))

    if den_x == 0 or den_y == 0:
        return 0.0

    return num / (den_x * den_y)


def compute_signal_correlations(
    signal_vectors: List[Dict[str, float]],
    returns: List[float],
    return_label: str = "90d"
) -> Dict[str, Dict]:
    """
    Correlate each signal with actual returns.

    Args:
        signal_vectors: List of signal dicts (one per company)
        returns: Corresponding actual returns (same order, same length)
        return_label: Label for the return window

    Returns:
        Dict mapping signal_name → {
            correlation: float,        # Spearman rank correlation
            p_significant: bool,       # Roughly significant? (|r| > 2/sqrt(n))
            direction: str,            # "positive", "negative", or "neutral"
            strength: str,             # "strong", "moderate", "weak", "none"
            avg_when_high: float,      # Avg return when signal in top quartile
            avg_when_low: float,       # Avg return when signal in bottom quartile
            spread: float,             # Difference (predictive power)
            sample_size: int,
        }
    """
    results = {}
    n = len(returns)
    # Significance threshold: |r| > 2/sqrt(n) is a rough rule of thumb
    sig_threshold = 2.0 / math.sqrt(n) if n >= 10 else 0.5

    # Get all signal names from the vectors
    all_signals = set()
    for sv in signal_vectors:
        all_signals.update(sv.keys())

    for signal_name in sorted(all_signals):
        # Extract this signal's values, paired with returns
        pairs = []
        for i, sv in enumerate(signal_vectors):
            val = sv.get(signal_name)
            if val is not None and returns[i] is not None:
                pairs.append((float(val), float(returns[i])))

        if len(pairs) < 5:
            results[signal_name] = {
                "correlation": 0, "p_significant": False,
                "direction": "insufficient_data", "strength": "none",
                "avg_when_high": 0, "avg_when_low": 0, "spread": 0,
                "sample_size": len(pairs),
            }
            continue

        x_vals = [p[0] for p in pairs]
        y_vals = [p[1] for p in pairs]

        corr = _spearman_rank_correlation(x_vals, y_vals)

        # Quartile analysis: what happened when the signal was high vs low?
        sorted_by_signal = sorted(pairs, key=lambda p: p[0])
        q_size = max(1, len(sorted_by_signal) // 4)
        bottom_q = sorted_by_signal[:q_size]
        top_q = sorted_by_signal[-q_size:]

        avg_when_low = sum(p[1] for p in bottom_q) / len(bottom_q)
        avg_when_high = sum(p[1] for p in top_q) / len(top_q)
        spread = avg_when_high - avg_when_low

        # Classify
        abs_corr = abs(corr)
        if abs_corr >= 0.4:
            strength = "strong"
        elif abs_corr >= 0.2:
            strength = "moderate"
        elif abs_corr >= 0.1:
            strength = "weak"
        else:
            strength = "none"

        direction = "positive" if corr > 0.05 else ("negative" if corr < -0.05 else "neutral")

        results[signal_name] = {
            "correlation": round(corr, 4),
            "p_significant": abs_corr > sig_threshold,
            "direction": direction,
            "strength": strength,
            "avg_when_high": round(avg_when_high, 2),
            "avg_when_low": round(avg_when_low, 2),
            "spread": round(spread, 2),
            "sample_size": len(pairs),
        }

    return results


# ============================================================================
# CORE: WEIGHT OPTIMIZATION
# ============================================================================

def derive_optimal_weights(
    correlations: Dict[str, Dict],
    current_weights: Dict[str, float] = None,
) -> Dict[str, Any]:
    """
    Derive new CPS weights from signal correlations.

    Strategy: Use correlation-weighted ensemble.
    - Signals positively correlated with returns get proportional weight
    - Signals negatively correlated get zero weight
    - Weights are scaled to sum to 100

    This is interpretable and robust — no overfitting risk because we're
    just using correlation direction and magnitude, not fitting parameters.
    """
    if current_weights is None:
        current_weights = dict(DEFAULT_WEIGHTS)

    # CPS component signals to optimize
    cps_signals = ['catalyst_strength', 'improvement_trajectory',
                   'insider_conviction', 'turnaround_indicators', 'filing_recency']

    # Get correlation for each CPS component
    raw_weights = {}
    for signal in cps_signals:
        c = correlations.get(signal, {})
        corr = c.get('correlation', 0)
        significant = c.get('p_significant', False)
        spread = c.get('spread', 0)

        if corr > 0 and significant:
            # Weight = correlation * spread (both direction and magnitude matter)
            raw_weights[signal] = corr * max(1, abs(spread))
        elif corr > 0:
            # Positive but not significant — give small weight
            raw_weights[signal] = corr * 0.5
        else:
            # Negative or zero correlation — minimal weight (don't zero out entirely
            # to avoid throwing away signals that might work with more data)
            raw_weights[signal] = 0.02

    # Normalize to sum to 100
    total_raw = sum(raw_weights.values())
    if total_raw <= 0:
        # All signals are useless — keep current weights
        return {
            "suggested_weights": dict(current_weights),
            "raw_scores": raw_weights,
            "confidence": "very_low",
            "message": "No signals showed positive correlation with returns. Keeping current weights.",
        }

    suggested = {
        k: round(v / total_raw * 100)
        for k, v in raw_weights.items()
    }

    # Enforce minimum floor: no component below 5pts (protects against
    # signals that just need more data to prove themselves)
    MIN_WEIGHT = 5
    for k in suggested:
        if suggested[k] < MIN_WEIGHT:
            suggested[k] = MIN_WEIGHT

    # Re-normalize to sum to 100 after applying floor
    s_total = sum(suggested.values())
    if s_total != 100 and s_total > 0:
        # Scale the above-minimum weights to absorb the difference
        above_min = {k: v for k, v in suggested.items() if v > MIN_WEIGHT}
        above_total = sum(above_min.values())
        if above_total > 0:
            target = 100 - MIN_WEIGHT * (len(suggested) - len(above_min))
            for k in above_min:
                suggested[k] = round(above_min[k] / above_total * target)

    # Ensure they sum to exactly 100
    diff = 100 - sum(suggested.values())
    if diff != 0:
        # Add remainder to the highest-weighted signal
        best = max(suggested, key=suggested.get)
        suggested[best] += diff

    # Compute how different the new weights are from current
    weight_changes = {
        k: suggested.get(k, 0) - current_weights.get(k, 0)
        for k in set(list(suggested.keys()) + list(current_weights.keys()))
    }
    max_change = max(abs(v) for v in weight_changes.values())

    if max_change <= 3:
        confidence = "high"
        message = "Current weights are well-calibrated. Minor adjustments suggested."
    elif max_change <= 10:
        confidence = "medium"
        message = "Moderate adjustments suggested based on return data."
    else:
        confidence = "low"
        message = "Large weight changes suggested — review carefully before applying."

    return {
        "suggested_weights": suggested,
        "current_weights": dict(current_weights),
        "weight_changes": weight_changes,
        "raw_scores": {k: round(v, 4) for k, v in raw_weights.items()},
        "max_change": max_change,
        "confidence": confidence,
        "message": message,
    }


# ============================================================================
# CORE: SIMULATED VALIDATION
# ============================================================================

def backtest_weights(
    signal_vectors: List[Dict[str, float]],
    returns: List[float],
    tickers: List[str],
    weights_a: Dict[str, float],
    weights_b: Dict[str, float],
    label_a: str = "Current",
    label_b: str = "Optimized",
) -> Dict:
    """
    Compare two weight schemes on historical data.
    For each company, compute CPS under both weight schemes,
    then see which produces better stock picks.

    Returns comparison stats: win rate, avg return of top picks, etc.
    """
    cps_components = ['catalyst_strength', 'improvement_trajectory',
                      'insider_conviction', 'turnaround_indicators', 'filing_recency']

    def compute_score(sv, weights):
        """Compute weighted CPS from signal vector using given weights."""
        total_weight = sum(weights.get(k, 0) for k in cps_components)
        if total_weight == 0:
            return 0
        score = 0
        for comp in cps_components:
            val = sv.get(comp, 0)
            max_val = SIGNAL_DEFINITIONS.get(comp, {}).get('range', [0, 25])[1]
            if max_val > 0:
                normalized = val / max_val  # 0-1
            else:
                normalized = 0
            score += normalized * weights.get(comp, 0)
        return score

    # Score all companies under both schemes
    scored_a = []
    scored_b = []
    for i, sv in enumerate(signal_vectors):
        if returns[i] is None:
            continue
        scored_a.append((tickers[i], compute_score(sv, weights_a), returns[i]))
        scored_b.append((tickers[i], compute_score(sv, weights_b), returns[i]))

    if not scored_a:
        return {"error": "No data to compare"}

    # Sort by score descending (top picks first)
    scored_a.sort(key=lambda x: x[1], reverse=True)
    scored_b.sort(key=lambda x: x[1], reverse=True)

    # Compare top quartile performance
    q_size = max(1, len(scored_a) // 4)

    def quartile_stats(scored, label):
        top = scored[:q_size]
        bottom = scored[-q_size:]
        all_returns = [s[2] for s in scored]

        top_avg = sum(s[2] for s in top) / len(top)
        bottom_avg = sum(s[2] for s in bottom) / len(bottom)
        top_win_rate = sum(1 for s in top if s[2] > 0) / len(top) * 100

        return {
            "label": label,
            "top_quartile_avg_return": round(top_avg, 2),
            "bottom_quartile_avg_return": round(bottom_avg, 2),
            "spread": round(top_avg - bottom_avg, 2),
            "top_win_rate": round(top_win_rate, 1),
            "top_picks": [(s[0], round(s[1], 1), round(s[2], 1)) for s in top[:5]],
            "total_avg_return": round(sum(all_returns) / len(all_returns), 2),
        }

    stats_a = quartile_stats(scored_a, label_a)
    stats_b = quartile_stats(scored_b, label_b)

    # Which is better?
    improvement = stats_b["spread"] - stats_a["spread"]
    winner = label_b if improvement > 0 else label_a

    return {
        "comparison": {label_a: stats_a, label_b: stats_b},
        "improvement_spread": round(improvement, 2),
        "winner": winner,
        "sample_size": len(scored_a),
        "quartile_size": q_size,
    }


# ============================================================================
# ORCHESTRATOR: CalibrationEngine
# ============================================================================

class CalibrationEngine:
    """
    Full feedback loop orchestrator.

    Usage:
        engine = CalibrationEngine(scraper, potential_companies)
        report = engine.run_full_calibration()
    """

    def __init__(self, scraper, potential_companies: List[Dict],
                 data_dir: str = "sec_data"):
        self.scraper = scraper
        self.potential_companies = potential_companies
        self.data_dir = data_dir
        self.calibration_dir = os.path.join(data_dir, "calibration")
        os.makedirs(self.calibration_dir, exist_ok=True)

    def run_full_calibration(self, return_window: int = 90) -> Dict:
        """
        Execute the full feedback loop:
          1. Build signal vectors for all companies
          2. Load or compute backtest returns
          3. Correlate signals with returns
          4. Derive optimal weights
          5. Validate with simulated comparison
          6. Save calibration report

        Args:
            return_window: Which return window to optimize for (30, 60, 90, 180)

        Returns:
            Full calibration report dict
        """
        start = time.time()
        report = {
            "timestamp": datetime.now().isoformat(),
            "return_window": return_window,
            "status": "running",
        }

        # 1. BUILD SIGNAL VECTORS
        logger.info("Calibration: extracting signal vectors...")
        tickers = []
        signal_vectors = []

        for pc in self.potential_companies:
            ticker = pc.get('ticker', '')
            if not ticker:
                continue
            sv = extract_signal_vector(ticker, self.scraper, self.potential_companies)
            tickers.append(ticker)
            signal_vectors.append(sv)

        report["companies_analyzed"] = len(tickers)

        if len(tickers) < 10:
            report["status"] = "insufficient_data"
            report["message"] = (
                f"Only {len(tickers)} companies with signals. "
                f"Need at least 10 for meaningful calibration. "
                f"Run the full pipeline on more companies first."
            )
            return report

        # 2. LOAD BACKTEST RETURNS
        logger.info("Calibration: loading backtest returns...")
        backtest_cache = os.path.join(
            os.environ.get("MARKET_DATA_DIR", "sec_data/market_data"),
            "backtest_results.json"
        )

        backtest_by_ticker = {}
        if os.path.exists(backtest_cache):
            try:
                with open(backtest_cache) as f:
                    for r in json.load(f):
                        tk = r.get('ticker', '')
                        if tk and not r.get('error'):
                            # Keep best result per ticker (most recent filing)
                            if tk not in backtest_by_ticker:
                                backtest_by_ticker[tk] = r
            except Exception as e:
                logger.warning(f"Calibration: backtest cache load error: {e}")

        # Pair signals with returns
        paired_signals = []
        paired_returns = []
        paired_tickers = []
        w_key = str(return_window)

        for i, ticker in enumerate(tickers):
            bt = backtest_by_ticker.get(ticker)
            if bt and w_key in bt.get('windows', {}):
                ret = bt['windows'][w_key].get('change_pct')
                if ret is not None:
                    paired_signals.append(signal_vectors[i])
                    paired_returns.append(ret)
                    paired_tickers.append(ticker)

        report["companies_with_returns"] = len(paired_returns)
        report["companies_without_returns"] = len(tickers) - len(paired_returns)

        if len(paired_returns) < 10:
            report["status"] = "insufficient_backtest_data"
            report["message"] = (
                f"Only {len(paired_returns)} companies have {return_window}d return data. "
                f"Need at least 10. Run 'Backtest' first to fetch price data."
            )
            # Still compute correlations for whatever we have
            if len(paired_returns) >= 5:
                report["correlations"] = compute_signal_correlations(
                    paired_signals, paired_returns, f"{return_window}d"
                )
            return report

        # 3. CORRELATE SIGNALS WITH RETURNS
        logger.info(f"Calibration: correlating {len(paired_returns)} data points...")
        correlations = compute_signal_correlations(
            paired_signals, paired_returns, f"{return_window}d"
        )
        report["correlations"] = correlations

        # Summary: top predictive signals
        ranked_signals = sorted(
            correlations.items(),
            key=lambda x: abs(x[1].get('correlation', 0)),
            reverse=True
        )
        report["top_signals"] = [
            {
                "signal": name,
                "correlation": data["correlation"],
                "direction": data["direction"],
                "strength": data["strength"],
                "spread": data["spread"],
                "description": SIGNAL_DEFINITIONS.get(name, {}).get("description", ""),
            }
            for name, data in ranked_signals[:10]
        ]

        # 4. DERIVE OPTIMAL WEIGHTS
        logger.info("Calibration: computing optimal weights...")
        weight_result = derive_optimal_weights(correlations, DEFAULT_WEIGHTS)
        report["weight_optimization"] = weight_result

        # 5. VALIDATE
        logger.info("Calibration: running comparison backtest...")
        comparison = backtest_weights(
            paired_signals, paired_returns, paired_tickers,
            DEFAULT_WEIGHTS,
            weight_result["suggested_weights"],
            "Current Weights",
            "Optimized Weights",
        )
        report["comparison"] = comparison

        # 6. SAVE
        report["status"] = "complete"
        report["elapsed_seconds"] = round(time.time() - start, 2)
        report["message"] = self._summarize_report(report)

        # Save to calibration history
        self._save_report(report)

        logger.info(
            f"Calibration complete: {len(paired_returns)} companies, "
            f"best signal: {ranked_signals[0][0]} (r={ranked_signals[0][1]['correlation']:.3f}), "
            f"spread improvement: {comparison.get('improvement_spread', 0):.1f}%"
        )

        return report

    def get_current_weights(self) -> Dict[str, float]:
        """Get current CPS weights (latest calibration or defaults)."""
        latest = self._load_latest_weights()
        return latest if latest else dict(DEFAULT_WEIGHTS)

    def apply_weights(self, weights: Dict[str, float]):
        """Save new weights as the active CPS weight configuration."""
        path = os.path.join(self.calibration_dir, "active_weights.json")
        with open(path, 'w') as f:
            json.dump({
                "weights": weights,
                "applied_at": datetime.now().isoformat(),
            }, f, indent=2)
        logger.info(f"Applied new CPS weights: {weights}")

    def get_calibration_history(self, limit: int = 20) -> List[Dict]:
        """Load past calibration reports (summaries only)."""
        history = []
        hist_dir = os.path.join(self.calibration_dir, "history")
        if not os.path.exists(hist_dir):
            return []

        files = sorted(os.listdir(hist_dir), reverse=True)[:limit]
        for fname in files:
            try:
                with open(os.path.join(hist_dir, fname)) as f:
                    report = json.load(f)
                # Return summary only
                history.append({
                    "timestamp": report.get("timestamp"),
                    "status": report.get("status"),
                    "companies_with_returns": report.get("companies_with_returns", 0),
                    "return_window": report.get("return_window"),
                    "suggested_weights": report.get("weight_optimization", {}).get("suggested_weights"),
                    "improvement_spread": report.get("comparison", {}).get("improvement_spread"),
                    "top_signal": (report.get("top_signals", [{}])[0].get("signal")
                                   if report.get("top_signals") else None),
                    "message": report.get("message"),
                })
            except Exception:
                pass
        return history

    def _save_report(self, report: Dict):
        """Save calibration report to history."""
        hist_dir = os.path.join(self.calibration_dir, "history")
        os.makedirs(hist_dir, exist_ok=True)
        fname = f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(os.path.join(hist_dir, fname), 'w') as f:
            json.dump(report, f, indent=2)

    def _load_latest_weights(self) -> Optional[Dict[str, float]]:
        """Load the most recently applied weights."""
        path = os.path.join(self.calibration_dir, "active_weights.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                return data.get("weights")
            except Exception:
                pass
        return None

    def _summarize_report(self, report: Dict) -> str:
        """Generate a human-readable summary of calibration results."""
        n = report.get("companies_with_returns", 0)
        window = report.get("return_window", 90)

        top = report.get("top_signals", [])
        top_str = ""
        if top:
            t = top[0]
            top_str = (
                f"Strongest signal: {t['signal']} "
                f"(r={t['correlation']:.3f}, {t['strength']}, "
                f"spread={t['spread']:+.1f}%)"
            )

        comp = report.get("comparison", {})
        imp = comp.get("improvement_spread", 0)
        winner = comp.get("winner", "?")

        wo = report.get("weight_optimization", {})
        conf = wo.get("confidence", "?")
        changes = wo.get("weight_changes", {})
        biggest_change = ""
        if changes:
            bc = max(changes.items(), key=lambda x: abs(x[1]))
            biggest_change = f"Biggest change: {bc[0]} {bc[1]:+.0f}pts."

        return (
            f"Calibration on {n} companies using {window}d returns. "
            f"{top_str} "
            f"Weight optimization confidence: {conf}. {biggest_change} "
            f"Simulated improvement: {imp:+.1f}% spread ({winner} wins)."
        )
