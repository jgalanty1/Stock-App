"""
inflection_detector.py — Financial Inflection Point Detector
==============================================================
Fetches structured financial data from EDGAR XBRL CompanyFacts API,
then detects inflection points that precede price moves in microcaps.

Key inflections detected:
- Revenue acceleration (growth rate itself is increasing)
- Approaching/crossing profitability (first-time positive EBITDA/net income)
- Gross margin expansion trend
- Operating leverage (revenue growing faster than OpEx)
- Cash flow turning positive
- Revenue approaching institutional threshold ($20-50M)

Also computes:
- Float and dilution analysis (shares outstanding trends)
- Filing timing analysis (early vs late filers)

Uses EDGAR XBRL endpoint:
  https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# XBRL taxonomy concepts we care about
# These are US-GAAP concepts that appear in the XBRL company facts
REVENUE_CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "SalesRevenueServicesNet",
]

NET_INCOME_CONCEPTS = [
    "NetIncomeLoss",
    "ProfitLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
]

GROSS_PROFIT_CONCEPTS = [
    "GrossProfit",
]

OPERATING_INCOME_CONCEPTS = [
    "OperatingIncomeLoss",
]

OPERATING_EXPENSE_CONCEPTS = [
    "OperatingExpenses",
    "CostsAndExpenses",
]

EBITDA_CONCEPTS = [
    "EarningsBeforeInterestTaxesDepreciationAndAmortization",
]

CASH_FLOW_CONCEPTS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]

SHARES_OUTSTANDING_CONCEPTS = [
    "CommonStockSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
    "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
]

TOTAL_ASSETS_CONCEPTS = [
    "Assets",
]

CASH_CONCEPTS = [
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsAndShortTermInvestments",
]

EPS_CONCEPTS = [
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
]

# Additional concepts for ratio inflections
CAPEX_CONCEPTS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "CapitalExpenditureDiscontinuedOperations",
    "PaymentsToAcquireProductiveAssets",
]

ACCOUNTS_RECEIVABLE_CONCEPTS = [
    "AccountsReceivableNetCurrent",
    "AccountsReceivableNet",
    "ReceivablesNetCurrent",
]

INVENTORY_CONCEPTS = [
    "InventoryNet",
    "InventoryFinishedGoods",
    "InventoryRawMaterials",
]

DEFERRED_REVENUE_CONCEPTS = [
    "DeferredRevenueCurrent",
    "DeferredRevenueCurrentAndNoncurrent",
    "ContractWithCustomerLiabilityCurrent",
    "DeferredRevenue",
]

SBC_CONCEPTS = [
    "ShareBasedCompensation",
    "AllocatedShareBasedCompensationExpense",
    "StockBasedCompensation",
]

INTEREST_EXPENSE_CONCEPTS = [
    "InterestExpense",
    "InterestExpenseDebt",
    "InterestIncomeExpenseNet",
]

TOTAL_LIABILITIES_CONCEPTS = [
    "Liabilities",
    "LiabilitiesCurrent",
]

LONG_TERM_DEBT_CONCEPTS = [
    "LongTermDebt",
    "LongTermDebtNoncurrent",
    "DebtInstrumentCarryingAmount",
    "LongTermDebtAndCapitalLeaseObligations",
]


# ============================================================================
# XBRL DATA FETCHING
# ============================================================================

def fetch_xbrl_financials(cik: str, session, limiter) -> Optional[Dict]:
    """
    Fetch company facts from EDGAR XBRL API.
    Returns the raw company facts JSON.
    """
    from scraper import _safe_request

    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    data = _safe_request(session, url, limiter=limiter)

    if not data or not isinstance(data, dict):
        logger.warning(f"CIK {cik}: no XBRL data returned")
        return None

    logger.info(f"CIK {cik}: fetched XBRL company facts")
    return data


def _extract_concept_series(facts: Dict, concept_names: List[str],
                             unit_key: str = "USD",
                             min_periods: int = 2,
                             quarterly_only: bool = False) -> List[Dict]:
    """
    Extract a time series for a concept from XBRL facts.
    Returns list of {period_end, value, form_type, filed} sorted by date.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    for concept in concept_names:
        concept_data = us_gaap.get(concept, {})
        units = concept_data.get("units", {})

        # Try the specified unit, then fall back
        values = units.get(unit_key, [])
        if not values and unit_key == "USD":
            # Try shares for share counts
            values = units.get("shares", [])
        if not values:
            # Try pure (dimensionless)
            values = units.get("pure", [])
        if not values:
            # Try USD/shares for EPS
            values = units.get("USD/shares", [])

        if not values:
            continue

        # Filter to 10-K and 10-Q filings, extract period data
        series = []
        seen = set()
        for entry in values:
            form = entry.get("form", "")
            if form not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                continue

            end_date = entry.get("end", "")
            start_date = entry.get("start", "")
            val = entry.get("val")
            filed = entry.get("filed", "")

            if val is None or not end_date:
                continue

            # For income statement items, we want quarterly data
            # Annual data has start→end spanning ~365 days
            # Quarterly data has start→end spanning ~90 days
            is_quarterly = False
            is_annual = False
            if start_date and end_date:
                try:
                    sd = datetime.strptime(start_date, "%Y-%m-%d")
                    ed = datetime.strptime(end_date, "%Y-%m-%d")
                    days = (ed - sd).days
                    is_quarterly = 60 <= days <= 120
                    is_annual = days >= 300
                except ValueError:
                    pass

            # Dedup by period end + form
            dedup_key = f"{end_date}_{form}_{is_quarterly}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            series.append({
                "period_end": end_date,
                "period_start": start_date,
                "value": val,
                "form_type": form.replace("/A", ""),
                "filed": filed,
                "is_quarterly": is_quarterly,
                "is_annual": is_annual,
            })

        if len(series) >= min_periods:
            # Sort by period end date
            series.sort(key=lambda x: x["period_end"])
            if quarterly_only:
                series = [s for s in series if s["is_quarterly"]]
            return series

    return []


# ============================================================================
# METRIC EXTRACTION
# ============================================================================

def extract_financial_metrics(facts: Dict) -> Dict:
    """
    Extract all key financial metric series from XBRL facts.
    Returns dict of metric_name → sorted time series.
    """
    metrics = {}

    # Revenue (quarterly preferred for trend detection)
    rev = _extract_concept_series(facts, REVENUE_CONCEPTS, quarterly_only=True)
    if not rev:
        rev = _extract_concept_series(facts, REVENUE_CONCEPTS)
    if rev:
        metrics["revenue"] = rev

    # Net income
    ni = _extract_concept_series(facts, NET_INCOME_CONCEPTS, quarterly_only=True)
    if not ni:
        ni = _extract_concept_series(facts, NET_INCOME_CONCEPTS)
    if ni:
        metrics["net_income"] = ni

    # Gross profit
    gp = _extract_concept_series(facts, GROSS_PROFIT_CONCEPTS, quarterly_only=True)
    if not gp:
        gp = _extract_concept_series(facts, GROSS_PROFIT_CONCEPTS)
    if gp:
        metrics["gross_profit"] = gp

    # Operating income
    oi = _extract_concept_series(facts, OPERATING_INCOME_CONCEPTS, quarterly_only=True)
    if not oi:
        oi = _extract_concept_series(facts, OPERATING_INCOME_CONCEPTS)
    if oi:
        metrics["operating_income"] = oi

    # Operating expenses
    oe = _extract_concept_series(facts, OPERATING_EXPENSE_CONCEPTS, quarterly_only=True)
    if not oe:
        oe = _extract_concept_series(facts, OPERATING_EXPENSE_CONCEPTS)
    if oe:
        metrics["operating_expenses"] = oe

    # Cash flow from operations
    cf = _extract_concept_series(facts, CASH_FLOW_CONCEPTS)
    if cf:
        metrics["operating_cash_flow"] = cf

    # Shares outstanding (balance sheet item, not quarterly-filtered)
    shares = _extract_concept_series(facts, SHARES_OUTSTANDING_CONCEPTS, unit_key="shares")
    if shares:
        metrics["shares_outstanding"] = shares

    # Cash
    cash = _extract_concept_series(facts, CASH_CONCEPTS)
    if cash:
        metrics["cash"] = cash

    # Total assets
    assets = _extract_concept_series(facts, TOTAL_ASSETS_CONCEPTS)
    if assets:
        metrics["total_assets"] = assets

    # EPS
    eps = _extract_concept_series(facts, EPS_CONCEPTS, unit_key="USD/shares")
    if eps:
        metrics["eps"] = eps

    # CapEx (for FCF computation)
    capex = _extract_concept_series(facts, CAPEX_CONCEPTS)
    if capex:
        metrics["capex"] = capex

    # Accounts receivable (for DSO computation)
    ar = _extract_concept_series(facts, ACCOUNTS_RECEIVABLE_CONCEPTS)
    if ar:
        metrics["accounts_receivable"] = ar

    # Inventory
    inv = _extract_concept_series(facts, INVENTORY_CONCEPTS)
    if inv:
        metrics["inventory"] = inv

    # Deferred revenue
    defrev = _extract_concept_series(facts, DEFERRED_REVENUE_CONCEPTS)
    if defrev:
        metrics["deferred_revenue"] = defrev

    # Stock-based compensation
    sbc = _extract_concept_series(facts, SBC_CONCEPTS, quarterly_only=True)
    if not sbc:
        sbc = _extract_concept_series(facts, SBC_CONCEPTS)
    if sbc:
        metrics["sbc"] = sbc

    # Interest expense
    intexp = _extract_concept_series(facts, INTEREST_EXPENSE_CONCEPTS, quarterly_only=True)
    if not intexp:
        intexp = _extract_concept_series(facts, INTEREST_EXPENSE_CONCEPTS)
    if intexp:
        metrics["interest_expense"] = intexp

    # Total liabilities (for debt/equity ratio)
    liab = _extract_concept_series(facts, TOTAL_LIABILITIES_CONCEPTS)
    if liab:
        metrics["total_liabilities"] = liab

    # Long-term debt (for debt/EBITDA)
    ltd = _extract_concept_series(facts, LONG_TERM_DEBT_CONCEPTS)
    if ltd:
        metrics["long_term_debt"] = ltd

    return metrics


# ============================================================================
# INFLECTION DETECTION
# ============================================================================

def _growth_rate(current: float, prior: float) -> Optional[float]:
    """Calculate growth rate, handling zero/negative priors."""
    if prior is not None and prior != 0:
        return (current - prior) / abs(prior)
    return None


def _get_recent_values(series: List[Dict], n: int = 8) -> List[Tuple[str, float]]:
    """Get the n most recent (date, value) pairs."""
    return [(s["period_end"], s["value"]) for s in series[-n:]]


def detect_inflections(metrics: Dict) -> Dict:
    """
    Analyze financial metric series to detect inflection points.

    Returns:
    {
        "inflections": [{"type": ..., "description": ..., "significance": ...}],
        "financial_summary": {...},
        "inflection_score": 1-100,
        "signal": "strong_positive" | "positive" | "neutral" | "negative" | ...
    }
    """
    inflections = []
    summary = {}
    score_adjustments = 0  # will add to base 50

    # ---- Revenue analysis ----
    rev = metrics.get("revenue", [])
    if len(rev) >= 3:
        recent_rev = _get_recent_values(rev, 8)
        latest_rev = recent_rev[-1][1]
        summary["latest_revenue"] = latest_rev
        summary["latest_revenue_date"] = recent_rev[-1][0]

        # Revenue growth rates
        growth_rates = []
        for i in range(1, len(recent_rev)):
            gr = _growth_rate(recent_rev[i][1], recent_rev[i - 1][1])
            if gr is not None:
                growth_rates.append((recent_rev[i][0], gr))

        if growth_rates:
            summary["latest_revenue_growth"] = round(growth_rates[-1][1] * 100, 1)

        # INFLECTION: Revenue acceleration
        # Growth rate itself is increasing over the last 3+ periods
        if len(growth_rates) >= 3:
            recent_growth = [gr for _, gr in growth_rates[-3:]]
            if all(recent_growth[i] > recent_growth[i - 1] for i in range(1, len(recent_growth))):
                inflections.append({
                    "type": "revenue_acceleration",
                    "description": (
                        f"Revenue growth is ACCELERATING: "
                        f"{' → '.join(f'{g*100:.0f}%' for g in recent_growth)}"
                    ),
                    "significance": "high",
                    "direction": "positive",
                })
                score_adjustments += 15
            elif len(recent_growth) >= 2 and recent_growth[-1] > recent_growth[-2]:
                inflections.append({
                    "type": "revenue_growth_uptick",
                    "description": (
                        f"Revenue growth ticked up: "
                        f"{recent_growth[-2]*100:.0f}% → {recent_growth[-1]*100:.0f}%"
                    ),
                    "significance": "medium",
                    "direction": "positive",
                })
                score_adjustments += 8

            # INFLECTION: Revenue deceleration
            if all(recent_growth[i] < recent_growth[i - 1] for i in range(1, len(recent_growth))):
                inflections.append({
                    "type": "revenue_deceleration",
                    "description": (
                        f"Revenue growth is DECELERATING: "
                        f"{' → '.join(f'{g*100:.0f}%' for g in recent_growth)}"
                    ),
                    "significance": "high",
                    "direction": "negative",
                })
                score_adjustments -= 12

        # INFLECTION: Revenue approaching institutional threshold
        if latest_rev is not None:
            # Assumes quarterly data when is_quarterly flag is set; annual otherwise
            annual_run_rate = latest_rev * 4 if rev[-1].get("is_quarterly") else latest_rev
            summary["annual_revenue_run_rate"] = annual_run_rate
            if 15_000_000 <= annual_run_rate <= 50_000_000:
                inflections.append({
                    "type": "approaching_institutional_threshold",
                    "description": (
                        f"Revenue run rate ${annual_run_rate/1e6:.1f}M — "
                        f"approaching $20-50M range where small funds start paying attention"
                    ),
                    "significance": "medium",
                    "direction": "positive",
                })
                score_adjustments += 5

    # ---- Profitability analysis ----
    ni = metrics.get("net_income", [])
    if len(ni) >= 2:
        recent_ni = _get_recent_values(ni, 6)
        summary["latest_net_income"] = recent_ni[-1][1]

        # INFLECTION: Crossing profitability threshold
        if len(recent_ni) >= 2:
            prev_profitable = recent_ni[-2][1] > 0
            curr_profitable = recent_ni[-1][1] > 0

            if not prev_profitable and curr_profitable:
                inflections.append({
                    "type": "first_profit",
                    "description": (
                        f"Turned PROFITABLE: net income went from "
                        f"${recent_ni[-2][1]/1e6:.2f}M to ${recent_ni[-1][1]/1e6:.2f}M"
                    ),
                    "significance": "high",
                    "direction": "positive",
                })
                score_adjustments += 20

            elif prev_profitable and not curr_profitable:
                inflections.append({
                    "type": "lost_profitability",
                    "description": (
                        f"Lost profitability: net income went from "
                        f"${recent_ni[-2][1]/1e6:.2f}M to ${recent_ni[-1][1]/1e6:.2f}M"
                    ),
                    "significance": "high",
                    "direction": "negative",
                })
                score_adjustments -= 15

        # INFLECTION: Losses narrowing (approaching breakeven)
        if len(recent_ni) >= 3:
            losses = [(d, v) for d, v in recent_ni if v < 0]
            if len(losses) >= 2:
                # Check if losses are getting smaller (less negative)
                recent_losses = losses[-3:] if len(losses) >= 3 else losses
                if all(recent_losses[i][1] > recent_losses[i - 1][1]
                       for i in range(1, len(recent_losses))):
                    inflections.append({
                        "type": "losses_narrowing",
                        "description": (
                            f"Losses narrowing toward breakeven: "
                            f"{' → '.join(f'${v/1e6:.2f}M' for _, v in recent_losses)}"
                        ),
                        "significance": "high",
                        "direction": "positive",
                    })
                    score_adjustments += 12

    # ---- Gross margin analysis ----
    gp = metrics.get("gross_profit", [])
    if rev and gp and len(gp) >= 3 and len(rev) >= 3:
        # Compute gross margins
        margins = []
        rev_lookup = {s["period_end"]: s["value"] for s in rev}
        for g in gp:
            r = rev_lookup.get(g["period_end"])
            if r and r > 0:
                margins.append((g["period_end"], g["value"] / r))

        if len(margins) >= 3:
            recent_margins = margins[-4:]
            summary["latest_gross_margin"] = round(recent_margins[-1][1] * 100, 1)

            # INFLECTION: Margin expansion
            if len(recent_margins) >= 3:
                margin_vals = [m for _, m in recent_margins[-3:]]
                if all(margin_vals[i] > margin_vals[i - 1] for i in range(1, len(margin_vals))):
                    expansion = (margin_vals[-1] - margin_vals[0]) * 100
                    inflections.append({
                        "type": "margin_expansion",
                        "description": (
                            f"Gross margin expanding: "
                            f"{margin_vals[0]*100:.1f}% → {margin_vals[-1]*100:.1f}% "
                            f"(+{expansion:.1f}pp over {len(margin_vals)} periods)"
                        ),
                        "significance": "high" if expansion > 3 else "medium",
                        "direction": "positive",
                    })
                    score_adjustments += min(10, int(expansion * 2))

                elif all(margin_vals[i] < margin_vals[i - 1] for i in range(1, len(margin_vals))):
                    compression = (margin_vals[0] - margin_vals[-1]) * 100
                    inflections.append({
                        "type": "margin_compression",
                        "description": (
                            f"Gross margin compressing: "
                            f"{margin_vals[0]*100:.1f}% → {margin_vals[-1]*100:.1f}% "
                            f"(-{compression:.1f}pp)"
                        ),
                        "significance": "high" if compression > 3 else "medium",
                        "direction": "negative",
                    })
                    score_adjustments -= min(10, int(compression * 2))

    # ---- Operating leverage ----
    oe = metrics.get("operating_expenses", [])
    if rev and oe and len(rev) >= 3 and len(oe) >= 3:
        rev_recent = _get_recent_values(rev, 4)
        oe_recent = _get_recent_values(oe, 4)

        if len(rev_recent) >= 2 and len(oe_recent) >= 2:
            rev_growth = _growth_rate(rev_recent[-1][1], rev_recent[-2][1])
            oe_growth = _growth_rate(oe_recent[-1][1], oe_recent[-2][1])

            if rev_growth is not None and oe_growth is not None:
                summary["latest_revenue_growth_pct"] = round((rev_growth or 0) * 100, 1)
                summary["latest_opex_growth_pct"] = round((oe_growth or 0) * 100, 1)

                if rev_growth > 0 and oe_growth is not None and rev_growth > oe_growth + 0.05:
                    inflections.append({
                        "type": "operating_leverage",
                        "description": (
                            f"Operating leverage kicking in: "
                            f"revenue growing {rev_growth*100:.0f}% vs "
                            f"OpEx growing {oe_growth*100:.0f}%"
                        ),
                        "significance": "high",
                        "direction": "positive",
                    })
                    score_adjustments += 10

    # ---- Cash flow inflection ----
    cf = metrics.get("operating_cash_flow", [])
    if len(cf) >= 2:
        recent_cf = _get_recent_values(cf, 4)
        summary["latest_operating_cash_flow"] = recent_cf[-1][1]

        if len(recent_cf) >= 2:
            if recent_cf[-2][1] < 0 and recent_cf[-1][1] > 0:
                inflections.append({
                    "type": "cash_flow_positive",
                    "description": (
                        f"Operating cash flow turned POSITIVE: "
                        f"${recent_cf[-2][1]/1e6:.2f}M → ${recent_cf[-1][1]/1e6:.2f}M"
                    ),
                    "significance": "high",
                    "direction": "positive",
                })
                score_adjustments += 15

    # ---- Shares outstanding / dilution ----
    shares = metrics.get("shares_outstanding", [])
    if len(shares) >= 2:
        recent_shares = _get_recent_values(shares, 6)
        summary["latest_shares_outstanding"] = recent_shares[-1][1]

        if len(recent_shares) >= 2:
            share_growth = _growth_rate(recent_shares[-1][1], recent_shares[0][1])
            if share_growth is not None:
                summary["share_dilution_pct"] = round(share_growth * 100, 1)

                if share_growth > 0.20:
                    inflections.append({
                        "type": "heavy_dilution",
                        "description": (
                            f"SIGNIFICANT share dilution: "
                            f"{recent_shares[0][1]/1e6:.1f}M → {recent_shares[-1][1]/1e6:.1f}M shares "
                            f"(+{share_growth*100:.0f}%)"
                        ),
                        "significance": "high",
                        "direction": "negative",
                    })
                    score_adjustments -= 15

                elif share_growth > 0.10:
                    inflections.append({
                        "type": "moderate_dilution",
                        "description": (
                            f"Moderate share dilution: +{share_growth*100:.0f}% over period"
                        ),
                        "significance": "medium",
                        "direction": "negative",
                    })
                    score_adjustments -= 8

                elif share_growth < -0.05:
                    inflections.append({
                        "type": "share_buyback",
                        "description": (
                            f"Share count DECREASING: {share_growth*100:.0f}% "
                            f"(buybacks or retirements)"
                        ),
                        "significance": "medium",
                        "direction": "positive",
                    })
                    score_adjustments += 5

    # ---- Cash position ----
    cash = metrics.get("cash", [])
    if cash:
        recent_cash = _get_recent_values(cash, 4)
        summary["latest_cash"] = recent_cash[-1][1]

        # Cash burn rate for unprofitable companies
        if ni and len(ni) >= 2:
            latest_ni_val = ni[-1]["value"]
            if latest_ni_val < 0 and recent_cash[-1][1] > 0:
                # Quarterly burn rate
                burn = abs(latest_ni_val)
                quarters_runway = recent_cash[-1][1] / burn if burn > 0 else 999
                summary["cash_runway_quarters"] = round(quarters_runway, 1)

                if quarters_runway < 4:
                    inflections.append({
                        "type": "low_cash_runway",
                        "description": (
                            f"LOW CASH RUNWAY: ~{quarters_runway:.1f} quarters at current burn rate "
                            f"(${recent_cash[-1][1]/1e6:.1f}M cash, ${burn/1e6:.1f}M/qtr burn)"
                        ),
                        "significance": "high",
                        "direction": "negative",
                    })
                    score_adjustments -= 12

    # ---- Free Cash Flow (OCF - CapEx) ----
    ocf = metrics.get("operating_cash_flow", [])
    capex = metrics.get("capex", [])
    if len(ocf) >= 2 and len(capex) >= 2:
        ocf_lookup = {s["period_end"]: s["value"] for s in ocf}
        fcf_series = []
        for c in capex:
            o = ocf_lookup.get(c["period_end"])
            if o is not None:
                fcf_series.append((c["period_end"], o - abs(c["value"])))
        if len(fcf_series) >= 2:
            summary["latest_fcf"] = fcf_series[-1][1]
            # FCF turning positive
            if len(fcf_series) >= 2 and fcf_series[-2][1] < 0 and fcf_series[-1][1] > 0:
                inflections.append({
                    "type": "fcf_positive",
                    "description": (
                        f"Free cash flow turned POSITIVE: "
                        f"${fcf_series[-2][1]/1e6:.2f}M → ${fcf_series[-1][1]/1e6:.2f}M"
                    ),
                    "significance": "high",
                    "direction": "positive",
                })
                score_adjustments += 12
            # FCF improving trend
            elif len(fcf_series) >= 3:
                recent_fcf = fcf_series[-3:]
                if all(recent_fcf[i][1] > recent_fcf[i-1][1] for i in range(1, len(recent_fcf))):
                    inflections.append({
                        "type": "fcf_improving",
                        "description": (
                            f"FCF trending up: "
                            f"{' → '.join(f'${v/1e6:.1f}M' for _, v in recent_fcf)}"
                        ),
                        "significance": "medium",
                        "direction": "positive",
                    })
                    score_adjustments += 6

    # ---- Days Sales Outstanding (DSO) ----
    ar = metrics.get("accounts_receivable", [])
    revenue_val = metrics.get("revenue", [])
    if len(ar) >= 2 and len(revenue_val) >= 2:
        rev_lookup = {s["period_end"]: s["value"] for s in revenue_val}
        dso_series = []
        for a in ar:
            r = rev_lookup.get(a["period_end"])
            if r and r > 0:
                dso = (a["value"] / r) * 90  # Approximate quarterly days
                dso_series.append((a["period_end"], dso))
        if len(dso_series) >= 2:
            summary["latest_dso"] = round(dso_series[-1][1], 1)
            # DSO increasing = negative (stretching collections)
            if len(dso_series) >= 3:
                recent_dso = dso_series[-3:]
                dso_vals = [d for _, d in recent_dso]
                if all(dso_vals[i] > dso_vals[i-1] + 2 for i in range(1, len(dso_vals))):
                    inflections.append({
                        "type": "dso_increasing",
                        "description": (
                            f"DSO rising (collection slowing): "
                            f"{dso_vals[0]:.0f} → {dso_vals[-1]:.0f} days"
                        ),
                        "significance": "medium",
                        "direction": "negative",
                    })
                    score_adjustments -= 6
                elif all(dso_vals[i] < dso_vals[i-1] - 2 for i in range(1, len(dso_vals))):
                    inflections.append({
                        "type": "dso_decreasing",
                        "description": (
                            f"DSO improving (faster collections): "
                            f"{dso_vals[0]:.0f} → {dso_vals[-1]:.0f} days"
                        ),
                        "significance": "medium",
                        "direction": "positive",
                    })
                    score_adjustments += 4

    # ---- Deferred Revenue trend ----
    defrev = metrics.get("deferred_revenue", [])
    if len(defrev) >= 3:
        recent_defrev = _get_recent_values(defrev, 4)
        summary["latest_deferred_revenue"] = recent_defrev[-1][1]
        if len(recent_defrev) >= 3:
            vals = [v for _, v in recent_defrev[-3:]]
            # Growing deferred revenue = future revenue in the pipeline
            if all(vals[i] > vals[i-1] * 1.05 for i in range(1, len(vals))):
                growth = _growth_rate(vals[-1], vals[0])
                inflections.append({
                    "type": "deferred_revenue_growth",
                    "description": (
                        (f"Deferred revenue growing (future revenue pipeline): "
                        f"${vals[0]/1e6:.1f}M → ${vals[-1]/1e6:.1f}M "
                        f"({growth*100:.0f}%)") if growth else ""
                    ),
                    "significance": "medium",
                    "direction": "positive",
                })
                score_adjustments += 6

    # ---- SBC as % of Revenue ----
    sbc = metrics.get("sbc", [])
    if len(sbc) >= 2 and len(rev) >= 2:
        rev_lookup_sbc = {s["period_end"]: s["value"] for s in rev}
        sbc_pct_series = []
        for s in sbc:
            r = rev_lookup_sbc.get(s["period_end"])
            if r and r > 0:
                sbc_pct_series.append((s["period_end"], s["value"] / r * 100))
        if len(sbc_pct_series) >= 2:
            summary["latest_sbc_pct_revenue"] = round(sbc_pct_series[-1][1], 1)
            # SBC > 20% of revenue = dilution concern
            if sbc_pct_series[-1][1] > 20:
                inflections.append({
                    "type": "high_sbc",
                    "description": (
                        f"High stock-based compensation: {sbc_pct_series[-1][1]:.0f}% of revenue"
                    ),
                    "significance": "medium",
                    "direction": "negative",
                })
                score_adjustments -= 5
            # SBC declining trend = positive
            if len(sbc_pct_series) >= 3:
                recent_sbc = [p for _, p in sbc_pct_series[-3:]]
                if all(recent_sbc[i] < recent_sbc[i-1] for i in range(1, len(recent_sbc))):
                    inflections.append({
                        "type": "sbc_declining",
                        "description": (
                            f"SBC as % of revenue declining: "
                            f"{recent_sbc[0]:.0f}% → {recent_sbc[-1]:.0f}%"
                        ),
                        "significance": "low",
                        "direction": "positive",
                    })
                    score_adjustments += 3

    # ---- Interest Coverage (Operating Income / Interest Expense) ----
    oi = metrics.get("operating_income", [])
    intexp = metrics.get("interest_expense", [])
    if len(oi) >= 2 and len(intexp) >= 2:
        intexp_lookup = {s["period_end"]: s["value"] for s in intexp}
        coverage_series = []
        for o in oi:
            ie = intexp_lookup.get(o["period_end"])
            if ie and abs(ie) > 0:
                coverage_series.append((o["period_end"], o["value"] / abs(ie)))
        if len(coverage_series) >= 2:
            summary["latest_interest_coverage"] = round(coverage_series[-1][1], 1)
            # Interest coverage < 1.5x = distress risk
            if coverage_series[-1][1] < 1.5 and coverage_series[-1][1] > 0:
                inflections.append({
                    "type": "low_interest_coverage",
                    "description": (
                        f"Low interest coverage ratio: {coverage_series[-1][1]:.1f}x "
                        f"(distress risk if <1.5x)"
                    ),
                    "significance": "high",
                    "direction": "negative",
                })
                score_adjustments -= 8
            # Coverage improving from low levels
            if len(coverage_series) >= 2:
                prev_cov = coverage_series[-2][1]
                curr_cov = coverage_series[-1][1]
                if prev_cov < 2 and curr_cov > prev_cov * 1.3:
                    inflections.append({
                        "type": "interest_coverage_improving",
                        "description": (
                            f"Interest coverage improving: {prev_cov:.1f}x → {curr_cov:.1f}x"
                        ),
                        "significance": "medium",
                        "direction": "positive",
                    })
                    score_adjustments += 5

    # ---- Debt / EBITDA ----
    ltd = metrics.get("long_term_debt", [])
    if ltd and (metrics.get("operating_income") or metrics.get("net_income")):
        # Approximate EBITDA from operating income (EBITDA concepts often unavailable)
        oi_data = metrics.get("operating_income", metrics.get("net_income", []))
        if len(ltd) >= 1 and len(oi_data) >= 1:
            latest_debt = ltd[-1]["value"]
            latest_oi_val = oi_data[-1]["value"]
            # Annualize if quarterly
            if oi_data[-1].get("is_quarterly"):
                latest_oi_val *= 4
            if latest_oi_val > 0:
                debt_ebitda = latest_debt / latest_oi_val
                summary["debt_to_ebitda_approx"] = round(debt_ebitda, 1)
                if debt_ebitda > 5:
                    inflections.append({
                        "type": "high_leverage",
                        "description": (
                            f"High leverage: Debt/EBITDA(approx) = {debt_ebitda:.1f}x"
                        ),
                        "significance": "high",
                        "direction": "negative",
                    })
                    score_adjustments -= 8
                elif debt_ebitda < 1.5:
                    inflections.append({
                        "type": "low_leverage",
                        "description": (
                            f"Low leverage: Debt/EBITDA(approx) = {debt_ebitda:.1f}x"
                        ),
                        "significance": "low",
                        "direction": "positive",
                    })
                    score_adjustments += 3

    # ---- Compute overall score ----
    inflection_score = max(1, min(100, 50 + score_adjustments))

    # ---- Signal classification ----
    positive = [i for i in inflections if i["direction"] == "positive"]
    negative = [i for i in inflections if i["direction"] == "negative"]

    if inflection_score >= 75:
        signal = "strong_positive"
    elif inflection_score >= 60:
        signal = "positive"
    elif inflection_score >= 45:
        signal = "neutral"
    elif inflection_score >= 30:
        signal = "negative"
    else:
        signal = "strong_negative"

    return {
        "inflections": inflections,
        "financial_summary": summary,
        "inflection_score": inflection_score,
        "signal": signal,
        "positive_inflections": len(positive),
        "negative_inflections": len(negative),
        "metrics_available": list(metrics.keys()),
    }


# ============================================================================
# FILING TIMING ANALYSIS
# ============================================================================

def analyze_filing_timing(filing_date: str, form_type: str,
                           period_end: str = None) -> Dict:
    """
    Analyze whether a filing was early or late relative to SEC deadlines.
    Early filers tend to have good news; late filers may be hiding problems.

    Deadlines for smaller reporting companies:
    - 10-K: 60 days after fiscal year end (accelerated) or 90 days (non-accelerated)
    - 10-Q: 40 days after quarter end (accelerated) or 45 days (non-accelerated)
    """
    result = {"filing_date": filing_date, "form_type": form_type}

    if not period_end or not filing_date:
        result["timing"] = "unknown"
        result["days_after_period"] = None
        return result

    try:
        filed = datetime.strptime(filing_date[:10], "%Y-%m-%d")
        period = datetime.strptime(period_end[:10], "%Y-%m-%d")
        days = (filed - period).days
    except ValueError:
        result["timing"] = "unknown"
        result["days_after_period"] = None
        return result

    result["days_after_period"] = days

    if form_type in ("10-K", "10-K/A"):
        # 10-K deadlines: 60 days (accelerated), 90 days (non-accelerated)
        if days <= 45:
            result["timing"] = "very_early"
        elif days <= 60:
            result["timing"] = "early"
        elif days <= 75:
            result["timing"] = "normal"
        elif days <= 90:
            result["timing"] = "late"
        else:
            result["timing"] = "very_late"
    elif form_type in ("10-Q", "10-Q/A"):
        # 10-Q deadlines: 40-45 days
        if days <= 30:
            result["timing"] = "very_early"
        elif days <= 40:
            result["timing"] = "early"
        elif days <= 45:
            result["timing"] = "normal"
        else:
            result["timing"] = "late"
    else:
        result["timing"] = "unknown"

    return result


# ============================================================================
# FORMATTING FOR LLM PROMPT
# ============================================================================

def format_inflection_for_prompt(inflection_data: Dict,
                                  filing_timing: Dict = None) -> str:
    """Format inflection detection data for inclusion in LLM analysis prompt."""
    if not inflection_data:
        return "(No financial inflection data available)"

    lines = [
        "=" * 60,
        "FINANCIAL INFLECTION ANALYSIS (from XBRL structured data)",
        "=" * 60,
        "",
        f"Inflection Score: {inflection_data.get('inflection_score', 50)}/100",
        f"Signal: {inflection_data.get('signal', 'unknown').replace('_', ' ').upper()}",
        f"Positive inflections: {inflection_data.get('positive_inflections', 0)}",
        f"Negative inflections: {inflection_data.get('negative_inflections', 0)}",
        f"Metrics available: {', '.join(inflection_data.get('metrics_available', []))}",
        "",
    ]

    # Financial summary
    fs = inflection_data.get("financial_summary", {})
    if fs:
        lines.append("--- Key Financial Metrics ---")
        if "latest_revenue" in fs:
            lines.append(f"  Latest revenue: ${fs['latest_revenue']/1e6:.2f}M ({fs.get('latest_revenue_date', '')})")
        if "annual_revenue_run_rate" in fs:
            lines.append(f"  Annual revenue run rate: ${fs['annual_revenue_run_rate']/1e6:.1f}M")
        if "latest_revenue_growth" in fs:
            lines.append(f"  Revenue growth (latest period): {fs['latest_revenue_growth']:.1f}%")
        if "latest_net_income" in fs:
            ni = fs['latest_net_income']
            lines.append(f"  Net income: ${ni/1e6:.2f}M {'(profitable)' if ni > 0 else '(unprofitable)'}")
        if "latest_gross_margin" in fs:
            lines.append(f"  Gross margin: {fs['latest_gross_margin']:.1f}%")
        if "latest_operating_cash_flow" in fs:
            lines.append(f"  Operating cash flow: ${fs['latest_operating_cash_flow']/1e6:.2f}M")
        if "latest_cash" in fs:
            lines.append(f"  Cash position: ${fs['latest_cash']/1e6:.1f}M")
        if "cash_runway_quarters" in fs:
            lines.append(f"  Cash runway: ~{fs['cash_runway_quarters']:.1f} quarters")
        if "share_dilution_pct" in fs:
            lines.append(f"  Share dilution over period: {fs['share_dilution_pct']:+.1f}%")
        if "latest_shares_outstanding" in fs:
            lines.append(f"  Shares outstanding: {fs['latest_shares_outstanding']/1e6:.1f}M")
        if "latest_revenue_growth_pct" in fs:
            lines.append(f"  Revenue growth: {fs['latest_revenue_growth_pct']:.1f}% vs OpEx growth: {fs.get('latest_opex_growth_pct', 0):.1f}%")
        if "latest_fcf" in fs:
            lines.append(f"  Free cash flow: ${fs['latest_fcf']/1e6:.2f}M")
        if "latest_dso" in fs:
            lines.append(f"  Days sales outstanding: {fs['latest_dso']:.0f} days")
        if "latest_deferred_revenue" in fs:
            lines.append(f"  Deferred revenue: ${fs['latest_deferred_revenue']/1e6:.1f}M")
        if "latest_sbc_pct_revenue" in fs:
            lines.append(f"  SBC as % of revenue: {fs['latest_sbc_pct_revenue']:.1f}%")
        if "latest_interest_coverage" in fs:
            lines.append(f"  Interest coverage: {fs['latest_interest_coverage']:.1f}x")
        if "debt_to_ebitda_approx" in fs:
            lines.append(f"  Debt/EBITDA (approx): {fs['debt_to_ebitda_approx']:.1f}x")
        lines.append("")

    # Inflection points
    inflections = inflection_data.get("inflections", [])
    if inflections:
        lines.append("--- Detected Inflection Points ---")
        for inf in inflections:
            icon = "🟢" if inf["direction"] == "positive" else "🔴"
            sig = inf["significance"].upper()
            lines.append(f"  {icon} [{sig}] {inf['type'].replace('_', ' ').upper()}")
            lines.append(f"     {inf['description']}")
        lines.append("")

    # Filing timing
    if filing_timing and filing_timing.get("timing") != "unknown":
        lines.append("--- Filing Timing ---")
        timing = filing_timing.get("timing", "unknown")
        days = filing_timing.get("days_after_period")
        timing_signal = {
            "very_early": "BULLISH (management eager to report)",
            "early": "Slightly positive (ahead of deadline)",
            "normal": "Neutral (within normal range)",
            "late": "Slightly negative (pushing deadline)",
            "very_late": "BEARISH (may be hiding problems or auditor issues)",
        }.get(timing, "Unknown")
        lines.append(f"  Filed {days} days after period end — {timing}: {timing_signal}")
        lines.append("")

    return "\n".join(lines)
