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
from typing import Dict, List, Optional, Any
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

    # Tier 2 sub-scores (extracted from deep analysis)
    "mgmt_authenticity_score": {"source": "t2", "range": [1, 10], "description": "T2 management authenticity"},
    "competitive_position_score": {"source": "t2", "range": [1, 10], "description": "T2 competitive position"},
    "capital_allocation_score": {"source": "t2", "range": [1, 10], "description": "T2 capital allocation quality"},
    "catalyst_stacking_count": {"source": "t2", "range": [0, 8], "description": "Number of active catalyst categories"},

    # Opus deep signals (from local Opus T2 analysis)
    "has_contrarian_thesis": {"source": "opus", "range": [0, 1], "description": "Opus identified a contrarian thesis"},
    "asymmetry_level_numeric": {"source": "opus", "range": [0, 3], "description": "Information asymmetry (0=none, 1=low, 2=med, 3=high)"},
    "narrative_contradictions": {"source": "opus", "range": [0, 10], "description": "Number of narrative vs numbers contradictions"},
    "non_obvious_catalyst_count": {"source": "opus", "range": [0, 10], "description": "Number of non-obvious catalysts found by Opus"},

    # Amendment detection
    "has_amendment": {"source": "filing", "range": [0, 1], "description": "Has amended filing (10-K/A, 10-Q/A)"},

    # Customer/revenue concentration
    "revenue_concentration_score": {"source": "filing", "range": [0, 100], "description": "Revenue concentration risk (higher = more concentrated)"},
}

# The five CPS (Composite Prioritization Score) component names used throughout
# the module for weight optimization, backtest scoring, and walk-forward validation.
CPS_COMPONENTS = [
    'catalyst_strength', 'improvement_trajectory',
    'insider_conviction', 'turnaround_indicators', 'filing_recency',
]

# Maps categorical filing diff signals (from scraper filing comparisons) to numeric
# values for correlation analysis. Positive = improving fundamentals, negative = deteriorating.
DIFF_SIGNAL_MAP = {
    "strongly_improving": 2, "improving": 1.5, "slightly_improving": 1,
    "stable": 0,
    "slightly_deteriorating": -1, "deteriorating": -1.5, "strongly_deteriorating": -2,
}

GEM_POTENTIAL_MAP = {"high": 2, "medium": 1, "low": 0}
CONVICTION_MAP = {"high": 2, "medium": 1, "low": 0}
ASYMMETRY_LEVEL_MAP = {"high": 3, "medium": 2, "low": 1, "none": 0}


# ============================================================================
# CORE: SIGNAL EXTRACTION
# ============================================================================

def extract_signal_vector(ticker: str, scraper, potential_companies: List[Dict],
                          _lookups: Optional[Dict] = None) -> Dict[str, float]:
    """
    Extract all measurable signals for a company into a flat numeric vector.
    This is the key function — it captures everything we know about a company
    at the time of analysis so we can later correlate with actual returns.

    Args:
        _lookups: Optional pre-built lookup dicts to avoid O(n) scans per call.
                  Keys: 'cps_by_ticker', 'universe_by_ticker', 'filings_by_ticker'.
    """
    signals = {}
    tk_upper = ticker.upper()

    if _lookups:
        # O(1) lookups using pre-built dicts
        cps_data = _lookups.get('cps_by_ticker', {}).get(tk_upper)
        company = _lookups.get('universe_by_ticker', {}).get(tk_upper, {})
        filings = list(_lookups.get('filings_by_ticker', {}).get(tk_upper, []))
    else:
        # Find CPS data
        cps_data = next((c for c in potential_companies
                         if c.get('ticker', '').upper() == tk_upper), None)
        # Find company in universe
        company = next((c for c in scraper.universe
                        if c.get('ticker', '').upper() == tk_upper), {})
        # Find all filings for this ticker
        filings = [f for f in scraper.filings_index
                   if f.get('ticker', '').upper() == tk_upper]

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
    ins = company.get('insider_data') or {}
    if ins is not None and ins:
        signals['accumulation_score'] = ins.get('accumulation_score', 50)
        signals['cluster_score'] = ins.get('cluster_score', 0)
        signals['unique_buyers'] = ins.get('unique_buyers', 0)
    else:
        signals['accumulation_score'] = 50
        signals['cluster_score'] = 0
        signals['unique_buyers'] = 0

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

    # T2 sub-scores (from filing index, best across filings)
    best_mgmt = None
    best_comp_pos = None
    best_cap_alloc = None
    best_catalyst_stack = 0
    for f in filings:
        ms = f.get('mgmt_authenticity_score')
        if ms is not None:
            best_mgmt = max(best_mgmt or 0, ms)
        cs = f.get('competitive_position_score')
        if cs is not None:
            best_comp_pos = max(best_comp_pos or 0, cs)
        ca = f.get('capital_allocation_score')
        if ca is not None:
            best_cap_alloc = max(best_cap_alloc or 0, ca)
        csc = f.get('catalyst_stacking_count')
        if csc is not None:
            best_catalyst_stack = max(best_catalyst_stack, csc)
    if best_mgmt is not None:
        signals['mgmt_authenticity_score'] = best_mgmt
    if best_comp_pos is not None:
        signals['competitive_position_score'] = best_comp_pos
    if best_cap_alloc is not None:
        signals['capital_allocation_score'] = best_cap_alloc
    signals['catalyst_stacking_count'] = best_catalyst_stack

    # ---- Opus deep signals ----
    best_contrarian = False
    best_asymmetry = 0
    best_contradictions = 0
    best_catalysts_opus = 0
    has_amendment = False
    best_concentration = 0

    for f in filings:
        if f.get('has_contrarian_thesis'):
            best_contrarian = True
        al = ASYMMETRY_LEVEL_MAP.get(f.get('asymmetry_level', ''), 0)
        best_asymmetry = max(best_asymmetry, al)
        nc = f.get('narrative_contradictions', 0) or 0
        best_contradictions = max(best_contradictions, nc)
        noc = f.get('non_obvious_catalyst_count', 0) or 0
        best_catalysts_opus = max(best_catalysts_opus, noc)

        # Amendment detection
        ft = f.get('form_type', '')
        if ft.endswith('/A'):
            has_amendment = True

        # Revenue concentration
        rc = f.get('revenue_concentration_score', 0) or 0
        best_concentration = max(best_concentration, rc)

    signals['has_contrarian_thesis'] = 1.0 if best_contrarian else 0.0
    signals['asymmetry_level_numeric'] = best_asymmetry
    signals['narrative_contradictions'] = best_contradictions
    signals['non_obvious_catalyst_count'] = best_catalysts_opus
    signals['has_amendment'] = 1.0 if has_amendment else 0.0
    signals['revenue_concentration_score'] = best_concentration

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
# TEMPORAL SIGNAL WEIGHTING
# ============================================================================

def compute_temporal_weights(filing_dates: List[str], half_life_days: int = 180) -> List[float]:
    """
    Compute exponential decay weights based on filing age.
    Recent signals get more weight than old ones.

    Args:
        filing_dates: List of filing date strings (YYYY-MM-DD)
        half_life_days: Days after which a signal has half its original weight

    Returns:
        List of weights (0-1) corresponding to each filing date.
        Most recent filing gets weight 1.0, older filings decay exponentially.
    """
    if not filing_dates:
        return []

    now = datetime.now()
    decay_rate = math.log(2) / max(1, half_life_days)

    weights = []
    for fd in filing_dates:
        try:
            filed = datetime.strptime(fd[:10], '%Y-%m-%d')
            days_ago = max(0, (now - filed).days)
            w = math.exp(-decay_rate * days_ago)
            weights.append(w)
        except (ValueError, TypeError):
            weights.append(0.5)  # Default for unparseable dates

    return weights


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
    return_label: str = "90d",
    temporal_weights: Optional[List[float]] = None,
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
    if n <= 0:
        return {}
    # Significance threshold: |r| > 2/sqrt(n) is a rough rule of thumb
    sig_threshold = 2.0 / math.sqrt(n) if n >= 10 else 0.5

    # Get all signal names from the vectors
    all_signals = set()
    for sv in signal_vectors:
        all_signals.update(sv.keys())

    for signal_name in sorted(all_signals):
        # Extract this signal's values, paired with returns (and optional weights)
        pairs = []
        for i, sv in enumerate(signal_vectors):
            val = sv.get(signal_name)
            if val is not None and returns[i] is not None:
                tw = temporal_weights[i] if temporal_weights and i < len(temporal_weights) else 1.0
                pairs.append((float(val), float(returns[i]), tw))

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
        w_vals = [p[2] for p in pairs]

        corr = _spearman_rank_correlation(x_vals, y_vals)

        # Quartile analysis: what happened when the signal was high vs low?
        # Use temporal weights for weighted averages
        sorted_by_signal = sorted(pairs, key=lambda p: p[0])
        q_size = max(1, len(sorted_by_signal) // 4)
        bottom_q = sorted_by_signal[:q_size]
        top_q = sorted_by_signal[-q_size:]

        # Weighted average returns
        def _weighted_avg(quartile):
            if not quartile:
                return 0.0
            total_w = sum(p[2] for p in quartile)
            if total_w == 0:
                return sum(p[1] for p in quartile) / len(quartile)
            return sum(p[1] * p[2] for p in quartile) / total_w

        avg_when_low = _weighted_avg(bottom_q)
        avg_when_high = _weighted_avg(top_q)
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
    cps_signals = CPS_COMPONENTS

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
    def compute_score(sv, weights):
        """Compute weighted CPS from signal vector using given weights."""
        total_weight = sum(weights.get(k, 0) for k in CPS_COMPONENTS)
        if total_weight == 0:
            return 0
        score = 0
        for comp in CPS_COMPONENTS:
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
# CONFIDENCE-WEIGHTED SCORING (Phase 5)
# ============================================================================

def compute_score_confidence(gem_score: float, _cps: float, _signal_count: int,
                              backtest_results: List[Dict]) -> Dict:
    """
    Compute empirical confidence for a gem score based on historical backtest data.
    Groups backtests by score bucket and computes actual return distributions.
    Uses t-distribution correction for small samples.
    """
    if not backtest_results:
        return {
            "confidence_level": "no_data",
            "confidence_interval": None,
            "sample_size": 0,
            "reliability": "none",
            "historical_win_rate": None,
        }

    # Define score buckets
    buckets = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    bucket_label = None
    for low, high in buckets:
        if low <= gem_score < high or (high == 100 and gem_score == 100):
            bucket_label = f"{low}-{high}"
            break
    if bucket_label is None:
        bucket_label = "0-20"

    # Gather returns for this score bucket (use 90d window as primary)
    bucket_returns = []
    for r in backtest_results:
        score = r.get('gem_score') or r.get('final_gem_score') or 0
        try:
            score = float(score)
        except (ValueError, TypeError):
            continue
        low = int(bucket_label.split('-')[0])
        high = int(bucket_label.split('-')[1])
        if low <= score < high or (high == 100 and score == 100):
            w90 = r.get('windows', {}).get('90', {})
            if w90 and w90.get('change_pct') is not None:
                bucket_returns.append(w90['change_pct'])

    n = len(bucket_returns)
    if n == 0:
        return {
            "confidence_level": "no_data",
            "confidence_interval": None,
            "sample_size": 0,
            "bucket": bucket_label,
            "reliability": "none",
            "historical_win_rate": None,
        }

    mean_ret = sum(bucket_returns) / n
    win_rate = sum(1 for r in bucket_returns if r > 0) / n * 100

    if n >= 2:
        variance = sum((r - mean_ret) ** 2 for r in bucket_returns) / (n - 1)
        std_dev = math.sqrt(variance)
    else:
        std_dev = 0

    # t-distribution correction for small samples (approximate)
    # t critical values for 95% CI: use 2.0 for n>=30, higher for smaller n
    if n >= 30:
        t_crit = 1.96
    elif n >= 15:
        t_crit = 2.13
    elif n >= 10:
        t_crit = 2.26
    elif n >= 5:
        t_crit = 2.78
    else:
        t_crit = 4.30  # Very wide for tiny samples

    margin = t_crit * std_dev / math.sqrt(n)
    ci_low = round(mean_ret - margin, 2)
    ci_high = round(mean_ret + margin, 2)

    # Reliability assessment
    if n >= 20 and std_dev < abs(mean_ret) * 1.5:
        reliability = "high"
    elif n >= 10:
        reliability = "moderate"
    elif n >= 5:
        reliability = "low"
    else:
        reliability = "very_low"

    # Confidence level
    if reliability in ("high", "moderate") and win_rate >= 60:
        confidence_level = "high"
    elif reliability in ("high", "moderate", "low") and win_rate >= 50:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    return {
        "confidence_level": confidence_level,
        "confidence_interval": [ci_low, ci_high],
        "mean_return": round(mean_ret, 2),
        "std_dev": round(std_dev, 2),
        "sample_size": n,
        "bucket": bucket_label,
        "reliability": reliability,
        "historical_win_rate": round(win_rate, 1),
    }


# ============================================================================
# SIGNAL PERFORMANCE MATRIX (Phase 3B)
# ============================================================================

def compute_signal_performance_matrix(multi_window_correlations: Dict) -> Dict:
    """
    Reorganize multi-window correlations into a signal-centric view:
    {signal: {30d: corr, 60d: corr, 90d: corr, 180d: corr, best_horizon: "90"}}
    Also identifies horizon specialists.
    """
    # Collect all signal names
    all_signals = set()
    for window_corrs in multi_window_correlations.values():
        all_signals.update(window_corrs.keys())

    matrix = {}
    for signal in sorted(all_signals):
        row = {}
        best_corr = 0
        best_horizon = None
        for window, corrs in multi_window_correlations.items():
            sig_data = corrs.get(signal, {})
            corr = sig_data.get("correlation", 0)
            row[f"{window}d"] = {
                "correlation": corr,
                "strength": sig_data.get("strength", "none"),
                "spread": sig_data.get("spread", 0),
            }
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_horizon = window
        row["best_horizon"] = best_horizon
        row["best_correlation"] = round(best_corr, 4)
        matrix[signal] = row

    # Identify horizon specialists: which signal dominates at each window
    horizon_specialists = {}
    for window in multi_window_correlations:
        best_signal = None
        best_val = 0
        for signal, row in matrix.items():
            w_data = row.get(f"{window}d", {})
            corr = w_data.get("correlation", 0)
            if corr > best_val:
                best_val = corr
                best_signal = signal
        if best_signal:
            horizon_specialists[f"{window}d"] = {
                "signal": best_signal,
                "correlation": round(best_val, 4),
            }

    return {
        "matrix": matrix,
        "horizon_specialists": horizon_specialists,
    }


# ============================================================================
# MULTI-HORIZON WEIGHT BLENDING (Phase 4A)
# ============================================================================

def derive_multi_horizon_weights(multi_window_correlations: Dict,
                                  horizon_preferences: Dict = None) -> Dict:
    """
    Blend optimal weights across multiple return horizons.
    Default preferences favor medium-term (90d) horizons.
    """
    if horizon_preferences is None:
        horizon_preferences = {"30": 0.15, "60": 0.20, "90": 0.35, "180": 0.30}

    per_horizon_weights = {}
    for window, corrs in multi_window_correlations.items():
        w = derive_optimal_weights(corrs, DEFAULT_WEIGHTS)
        per_horizon_weights[window] = w.get("suggested_weights", {})

    if not per_horizon_weights:
        return {"blended_weights": dict(DEFAULT_WEIGHTS), "error": "no_horizon_data"}

    # Blend using preference weights
    cps_signals = CPS_COMPONENTS
    blended = {}
    total_pref = 0
    for window, weights in per_horizon_weights.items():
        pref = horizon_preferences.get(window, 0.25)
        total_pref += pref
        for s in cps_signals:
            blended[s] = blended.get(s, 0) + weights.get(s, 20) * pref

    if total_pref > 0:
        for s in blended:
            blended[s] = round(blended[s] / total_pref)

    # Normalize to 100
    total = sum(blended.values())
    if total > 0 and total != 100:
        for k in blended:
            blended[k] = round(blended[k] / total * 100)
        diff = 100 - sum(blended.values())
        if diff != 0:
            best = max(blended, key=blended.get)
            blended[best] += diff

    # Identify conflicts: signals with opposite directions at different horizons
    horizon_conflicts = []
    for s in cps_signals:
        vals = []
        for window, weights in per_horizon_weights.items():
            vals.append((window, weights.get(s, 20)))
        if vals:
            max_w = max(v[1] for v in vals)
            min_w = min(v[1] for v in vals)
            if max_w - min_w > 15:
                horizon_conflicts.append({
                    "signal": s,
                    "range": [min_w, max_w],
                    "per_horizon": {w: v for w, v in vals},
                })

    return {
        "blended_weights": blended,
        "per_horizon_weights": per_horizon_weights,
        "horizon_preferences": horizon_preferences,
        "horizon_conflicts": horizon_conflicts,
    }


# ============================================================================
# WALK-FORWARD VALIDATION (Phase 2)
# ============================================================================

def temporal_train_test_split(tickers, signal_vectors, returns, filing_dates,
                              test_fraction=0.25, min_train_size=8):
    """
    Split data temporally: oldest data for training, most recent for testing.
    Returns (train_set, test_set) where each is a dict with tickers/signals/returns/dates.
    """
    # Sort by filing date
    combined = list(zip(tickers, signal_vectors, returns, filing_dates))
    combined.sort(key=lambda x: x[3])  # Sort by date ascending

    n = len(combined)
    split_idx = max(min_train_size, int(n * (1 - test_fraction)))
    split_idx = min(split_idx, n - 1)  # Ensure at least 1 test sample

    train = combined[:split_idx]
    test = combined[split_idx:]

    def _to_dict(data):
        if not data:
            return {"tickers": [], "signals": [], "returns": [], "dates": []}
        t, s, r, d = zip(*data)
        return {"tickers": list(t), "signals": list(s), "returns": list(r), "dates": list(d)}

    return _to_dict(train), _to_dict(test)


def walk_forward_validate(tickers, signal_vectors, returns, filing_dates,
                           n_folds=3):
    """
    Walk-forward cross-validation: each fold trains on older data, tests on newer.
    For N < 15, degrades to leave-one-out cross-validation.
    Returns dict with fold_results, stability_score, recommended_weights, overfitting_risk.
    """
    n = len(tickers)
    min_train_size = 5

    if n < min_train_size:
        logger.warning("Dataset too small for walk-forward: n=%d", n)
        return {
            "fold_results": [],
            "stability_score": 0.0,
            "recommended_weights": dict(DEFAULT_WEIGHTS),
            "overfitting_risk": "insufficient_data",
            "message": f"Only {n} data points. Need at least 5 for walk-forward.",
        }

    # Sort by date
    combined = sorted(zip(tickers, signal_vectors, returns, filing_dates),
                      key=lambda x: x[3])
    tickers_s, signals_s, returns_s, dates_s = zip(*combined) if combined else ([], [], [], [])

    fold_results = []

    if n < 15:
        # Leave-one-out for small samples
        for i in range(max(3, n // 2), max(n, 4)):
            train_signals = signals_s[:i]
            train_returns = returns_s[:i]
            test_signals = [signals_s[i]]
            test_returns = [returns_s[i]]

            if len(train_returns) < 5:
                continue

            corrs = compute_signal_correlations(train_signals, train_returns)
            weights = derive_optimal_weights(corrs, DEFAULT_WEIGHTS)

            fold_results.append({
                "train_size": i,
                "test_size": 1,
                "train_end_date": dates_s[i - 1],
                "suggested_weights": weights.get("suggested_weights", {}),
                "test_ticker": tickers_s[i],
                "test_return": returns_s[i],
            })
    else:
        # Walk-forward with n_folds
        min_train = max(8, n // 3)
        step = max(1, (n - min_train) // n_folds)

        for fold_start in range(min_train, n, step):
            fold_end = min(fold_start + step, n)
            if fold_end <= fold_start:
                continue

            train_signals = signals_s[:fold_start]
            train_returns = returns_s[:fold_start]
            test_signals = signals_s[fold_start:fold_end]
            test_returns = returns_s[fold_start:fold_end]

            if len(train_returns) < 5 or len(test_returns) < 1:
                continue

            corrs = compute_signal_correlations(train_signals, train_returns)
            weights = derive_optimal_weights(corrs, DEFAULT_WEIGHTS)

            fold_results.append({
                "fold": len(fold_results) + 1,
                "train_size": fold_start,
                "test_size": fold_end - fold_start,
                "train_end_date": dates_s[fold_start - 1],
                "test_start_date": dates_s[fold_start],
                "suggested_weights": weights.get("suggested_weights", {}),
                "test_avg_return": round(sum(test_returns) / len(test_returns), 2),
            })

    if not fold_results:
        return {
            "fold_results": [],
            "stability_score": 0.0,
            "recommended_weights": dict(DEFAULT_WEIGHTS),
            "overfitting_risk": "insufficient_data",
        }

    # Compute stability: how consistent are weights across folds?
    cps_signals = CPS_COMPONENTS

    weight_vectors = []
    for fr in fold_results:
        sw = fr.get("suggested_weights", {})
        weight_vectors.append([sw.get(s, 20) for s in cps_signals])

    # Stability = 1 - normalized std dev across folds
    stability = 0.0
    if len(weight_vectors) >= 2:
        per_signal_std = []
        for j in range(len(cps_signals)):
            vals = [wv[j] for wv in weight_vectors]
            if len(vals) <= 1:
                per_signal_std.append(0.0)
                continue
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
            per_signal_std.append(std / max(mean, 1))
        avg_cv = sum(per_signal_std) / len(per_signal_std)
        stability = round(max(0, min(1, 1 - avg_cv)), 3)

    # Average weights across folds
    avg_weights = {}
    for s in cps_signals:
        vals = [fr["suggested_weights"].get(s, 20) for fr in fold_results]
        avg_weights[s] = round(sum(vals) / len(vals))
    # Normalize to 100
    total = sum(avg_weights.values())
    if total > 0 and total != 100:
        for k in avg_weights:
            avg_weights[k] = round(avg_weights[k] / total * 100)
        diff = 100 - sum(avg_weights.values())
        if diff != 0:
            best = max(avg_weights, key=avg_weights.get)
            avg_weights[best] += diff

    # Overfitting risk
    if stability >= 0.7:
        risk = "low"
    elif stability >= 0.4:
        risk = "medium"
    else:
        risk = "high"

    return {
        "fold_results": fold_results,
        "stability_score": stability,
        "recommended_weights": avg_weights,
        "overfitting_risk": risk,
        "n_folds": len(fold_results),
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

    def _build_ticker_filing_index(self) -> Dict[str, List[Dict]]:
        """Build a ticker -> filings lookup dict from scraper filings_index (O(n) once)."""
        ticker_filings = {}
        for fdata in self.scraper.filings_index:
            tk = fdata.get('ticker', '').upper()
            if tk:
                ticker_filings.setdefault(tk, []).append(fdata)
        return ticker_filings

    def _build_filing_date_list(self, tickers: List[str],
                                ticker_filings: Dict[str, List[Dict]]) -> List[str]:
        """Build a list of newest filing dates for given tickers using pre-built index."""
        filing_dates = []
        for tk in tickers:
            tk_fils = [fdata for fdata in ticker_filings.get(tk.upper(), [])
                       if fdata.get('filing_date')]
            if tk_fils:
                tk_fils.sort(key=lambda fdata: fdata.get('filing_date', ''), reverse=True)
                filing_dates.append(tk_fils[0]['filing_date'])
            else:
                filing_dates.append('')
        return filing_dates

    def run_full_calibration(self, return_window: int = 90,
                             multi_window: bool = False) -> Dict:
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
        # Build ticker filing lookup once (O(n)) to avoid repeated linear scans
        ticker_filing_index = self._build_ticker_filing_index()
        report = {
            "timestamp": datetime.now().isoformat(),
            "return_window": return_window,
            "status": "running",
        }

        # 1. BUILD SIGNAL VECTORS
        logger.info("Calibration: extracting signal vectors...")
        tickers = []
        signal_vectors = []

        # Build lookup dicts once (O(n)) to avoid O(T^2) nested scans
        cps_by_ticker = {}
        for c in self.potential_companies:
            tk = c.get('ticker', '').upper()
            if tk:
                cps_by_ticker[tk] = c
        universe_by_ticker = {}
        for c in self.scraper.universe:
            tk = c.get('ticker', '').upper()
            if tk:
                universe_by_ticker[tk] = c
        filings_by_ticker = {}
        for fdata in self.scraper.filings_index:
            tk = fdata.get('ticker', '').upper()
            if tk:
                filings_by_ticker.setdefault(tk, []).append(fdata)
        _lookups = {
            'cps_by_ticker': cps_by_ticker,
            'universe_by_ticker': universe_by_ticker,
            'filings_by_ticker': filings_by_ticker,
        }

        for pc in self.potential_companies:
            ticker = pc.get('ticker', '')
            if not ticker:
                continue
            sv = extract_signal_vector(ticker, self.scraper, self.potential_companies,
                                       _lookups=_lookups)
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
                with open(backtest_cache, encoding='utf-8') as f:
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

        # 3. CORRELATE SIGNALS WITH RETURNS (with temporal weighting)
        logger.info(f"Calibration: correlating {len(paired_returns)} data points...")

        # Build temporal weights from filing dates
        paired_filing_dates = self._build_filing_date_list(paired_tickers, ticker_filing_index)
        t_weights = compute_temporal_weights(paired_filing_dates, half_life_days=180)

        correlations = compute_signal_correlations(
            paired_signals, paired_returns, f"{return_window}d",
            temporal_weights=t_weights,
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

        # 3b. WALK-FORWARD VALIDATION (Phase 2)
        filing_dates_list = self._build_filing_date_list(paired_tickers, ticker_filing_index)

        n_paired = len(paired_returns)
        if n_paired >= 15:
            logger.info("Calibration: running walk-forward validation...")
            wf_result = walk_forward_validate(
                paired_tickers, paired_signals, paired_returns, filing_dates_list)
            report["walk_forward"] = wf_result
        elif n_paired >= 10:
            wf_result = walk_forward_validate(
                paired_tickers, paired_signals, paired_returns, filing_dates_list)
            wf_result.update({
                "diagnostics": {
                    "message": f"N={n_paired}, running basic walk-forward (high overfitting risk)",
                    "overfitting_risk": "high",
                },
            })
            report["walk_forward"] = wf_result
        else:
            report["walk_forward"] = {
                "message": f"N={n_paired} < 10, skipping walk-forward. Keeping defaults.",
                "overfitting_risk": "insufficient_data",
            }
            wf_result = None

        # 3c. MULTI-WINDOW CORRELATIONS (Phase 3)
        if multi_window:
            logger.info("Calibration: computing multi-window correlations...")
            multi_window_correlations = {}
            for mw in [30, 60, 90, 180]:
                mw_key = str(mw)
                mw_returns = []
                mw_signals = []
                for i, ticker in enumerate(paired_tickers):
                    bt = backtest_by_ticker.get(ticker)
                    if bt and mw_key in bt.get('windows', {}):
                        ret = bt['windows'][mw_key].get('change_pct')
                        if ret is not None:
                            mw_returns.append(ret)
                            mw_signals.append(paired_signals[i])
                if len(mw_returns) >= 5:
                    multi_window_correlations[mw_key] = compute_signal_correlations(
                        mw_signals, mw_returns, f"{mw}d")
            report["multi_window_correlations"] = multi_window_correlations

            # Signal performance matrix (Phase 3B)
            if multi_window_correlations:
                report["signal_performance_matrix"] = compute_signal_performance_matrix(
                    multi_window_correlations)

            # Multi-horizon weight blending (Phase 4A)
            if len(multi_window_correlations) >= 2:
                report["multi_horizon_weights"] = derive_multi_horizon_weights(
                    multi_window_correlations)

        # 4. DERIVE OPTIMAL WEIGHTS
        logger.info("Calibration: computing optimal weights...")
        weight_result = derive_optimal_weights(correlations, DEFAULT_WEIGHTS)
        report["weight_optimization"] = weight_result

        # Use walk-forward weights if stable enough
        if wf_result and wf_result.get('stability_score', 0) > 0.5:
            wf_weights = wf_result.get('recommended_weights', {})
            if wf_weights:
                report["weight_optimization"]["walk_forward_weights"] = wf_weights
                report["weight_optimization"]["walk_forward_stability"] = wf_result['stability_score']

        # If multi-window mode and we have blended weights, use those
        if multi_window and report.get("multi_horizon_weights", {}).get("blended_weights"):
            report["weight_optimization"]["multi_horizon_blended"] = report["multi_horizon_weights"]["blended_weights"]

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

        if not ranked_signals:
            logger.info(
                f"Calibration complete: {len(paired_returns)} companies, "
                f"no ranked signals, "
                f"spread improvement: {comparison.get('improvement_spread', 0):.1f}%"
            )
        else:
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
        try:
            path = os.path.join(self.calibration_dir, "active_weights.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({
                    "weights": weights,
                    "applied_at": datetime.now().isoformat(),
                }, f, indent=2)
            logger.info(f"Applied new CPS weights: {weights}")
        except Exception as e:
            logger.error("Failed to apply weights: %s", e)

    def get_calibration_history(self, limit: int = 20) -> List[Dict]:
        """Load past calibration reports (summaries only)."""
        history = []
        hist_dir = os.path.join(self.calibration_dir, "history")
        if not os.path.exists(hist_dir):
            return []

        files = sorted(os.listdir(hist_dir), reverse=True)[:limit]
        for fname in files:
            try:
                with open(os.path.join(hist_dir, fname), encoding='utf-8') as f:
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
        try:
            hist_dir = os.path.join(self.calibration_dir, "history")
            os.makedirs(hist_dir, exist_ok=True)
            fname = f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(os.path.join(hist_dir, fname), 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.error("Failed to save report: %s", e)

    def _load_latest_weights(self) -> Optional[Dict[str, float]]:
        """Load the most recently applied weights."""
        path = os.path.join(self.calibration_dir, "active_weights.json")
        if os.path.exists(path):
            try:
                with open(path, encoding='utf-8') as f:
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
