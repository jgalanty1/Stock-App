"""
Turnaround Catalyst Extractor
===============================
Algorithmic extraction of turnaround catalysts from SEC filings.
Purely pattern-based (no LLM). Outputs structured catalyst data that feeds into
the Tier 2 LLM prompt for interpretation.

Turnaround catalyst categories:
  1. Cost restructuring (cycle stage detection)
  2. Management shakeup
  3. Debt restructuring / going concern changes
  4. Margin inflection narrative
  5. Asset monetization
  6. Working capital normalization
  7. Activist / ownership changes
  8. Going concern language changes

Designed for microcap stocks ($20-100M market cap) where these events
have outsized price impact vs. large caps.
"""

import re
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


# ============================================================================
# SECTION EXTRACTION (shared with filing_signals.py patterns)
# ============================================================================

MDA_HEADER = re.compile(
    r'item\s+7[\.\s].*?(?:management|discussion|analysis)',
    re.IGNORECASE
)
RISK_HEADER = re.compile(
    r'item\s+1a[\.\s].*?risk\s+factor',
    re.IGNORECASE
)
LIQUIDITY_HEADER = re.compile(
    r'(?:liquidity|capital\s+resources|going\s+concern)',
    re.IGNORECASE
)
LEGAL_HEADER = re.compile(
    r'item\s+3[\.\s].*?legal\s+proceedings',
    re.IGNORECASE
)


def _extract_section(text: str, header_pattern: re.Pattern,
                     max_chars: int = 50000) -> Optional[str]:
    """Extract section from filing text. Returns None if <500 chars."""
    match = header_pattern.search(text)
    if not match:
        return None
    start = match.start()
    next_item = re.search(r'\bitem\s+\d+[a-z]?\.', text[start + 100:], re.IGNORECASE)
    end = start + 100 + next_item.start() if next_item else start + max_chars
    section = text[start:min(end, start + max_chars)].strip()
    return section if len(section) >= 500 else None


def _find_sentences(text: str, pattern: re.Pattern, context_chars: int = 0) -> List[str]:
    """Find sentences containing pattern matches. Optionally include surrounding context."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    results = []
    for i, sent in enumerate(sentences):
        if pattern.search(sent):
            if context_chars > 0:
                # Include neighboring sentences for context
                start = max(0, i - 1)
                end = min(len(sentences), i + 2)
                ctx = ' '.join(sentences[start:end])
                results.append(ctx[:context_chars])
            else:
                results.append(sent.strip())
    return results


def _extract_dollar_amounts(text: str) -> List[Dict[str, Any]]:
    """Extract dollar amounts with context."""
    pattern = re.compile(
        r'\$\s*([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|M|B|K)?',
        re.IGNORECASE
    )
    amounts = []
    for m in pattern.finditer(text):
        try:
            val = float(m.group(1).replace(',', ''))
            unit = (m.group(2) or '').lower()
            if unit in ('million', 'm'):
                val *= 1_000_000
            elif unit in ('billion', 'b'):
                val *= 1_000_000_000
            elif unit in ('thousand', 'k'):
                val *= 1_000
            # Get surrounding context
            start = max(0, m.start() - 80)
            end = min(len(text), m.end() + 80)
            ctx = text[start:end].strip()
            amounts.append({'value': val, 'raw': m.group(0), 'context': ctx})
        except (ValueError, TypeError):
            continue
    return amounts


# ============================================================================
# CATALYST 1: COST RESTRUCTURING
# ============================================================================

# Restructuring lifecycle phases
_RESTRUCTURING_ANNOUNCE = re.compile(
    r'(?:announced|initiated|commenced|approved|adopted|implemented)\s+'
    r'(?:a\s+)?(?:restructuring|reorganization|cost\s+reduction|transformation|'
    r'realignment|streamlining|rationalization)',
    re.IGNORECASE
)
_RESTRUCTURING_PROGRESS = re.compile(
    r'(?:restructuring\s+(?:charge|cost|expense|program|plan|initiative)|'
    r'severance\s+(?:charge|cost|expense|payment)|'
    r'(?:facility|plant|office)\s+(?:closure|consolidation|exit)|'
    r'workforce\s+(?:reduction|right-?sizing)|'
    r'headcount\s+(?:reduction|decrease)|'
    r'(?:reduced|eliminated)\s+approximately\s+\d+\s+(?:position|employee|role))',
    re.IGNORECASE
)
_RESTRUCTURING_COMPLETE = re.compile(
    r'(?:substantially\s+complet|fully\s+implement|restructuring\s+(?:is|was)\s+complete|'
    r'restructuring\s+program\s+(?:has\s+been|was)\s+complet|'
    r'completed\s+(?:the|our)\s+restructuring|'
    r'expect\s+to\s+(?:substantially\s+)?complete\s+(?:the|our)\s+restructuring|'
    r'annualized\s+(?:cost\s+)?savings)',
    re.IGNORECASE
)
_SAVINGS_REALIZED = re.compile(
    r'(?:cost\s+savings\s+of|annual(?:ized)?\s+savings|'
    r'reduced\s+(?:annual\s+)?(?:operating\s+)?(?:expense|cost)s?\s+by|'
    r'expect(?:ed)?\s+(?:to\s+)?(?:save|reduce|achieve)\s+(?:approximately\s+)?\$|'
    r'savings\s+(?:of\s+)?(?:approximately\s+)?\$)',
    re.IGNORECASE
)


def extract_restructuring_catalysts(text: str) -> Dict[str, Any]:
    """
    Detect restructuring activity and determine lifecycle phase.
    
    Early announcement = pain ahead (bearish near-term, potentially bullish long-term)
    In-progress with charges = mid-cycle
    Substantially complete + savings realized = INFLECTION POINT (strong turnaround catalyst)
    """
    result = {
        'found': False,
        'phase': 'none',  # none, announced, in_progress, nearing_completion, complete
        'catalysts': [],
        'restructuring_charges': [],
        'savings_mentioned': [],
        'headcount_changes': [],
        'facility_actions': [],
    }

    mda = _extract_section(text, MDA_HEADER, 60000) or text[:60000]
    
    # Check completion signals first (most bullish)
    complete_matches = _find_sentences(mda, _RESTRUCTURING_COMPLETE, 300)
    progress_matches = _find_sentences(mda, _RESTRUCTURING_PROGRESS, 300)
    announce_matches = _find_sentences(mda, _RESTRUCTURING_ANNOUNCE, 300)
    savings_matches = _find_sentences(mda, _SAVINGS_REALIZED, 300)

    if not (complete_matches or progress_matches or announce_matches):
        return result

    result['found'] = True

    # Determine phase
    if complete_matches:
        result['phase'] = 'complete' if not progress_matches else 'nearing_completion'
    elif progress_matches and announce_matches:
        result['phase'] = 'in_progress'
    elif announce_matches:
        result['phase'] = 'announced'
    else:
        result['phase'] = 'in_progress'

    # Extract headcount reductions
    hc_pattern = re.compile(
        r'(?:reduc|eliminat|cut)\w*\s+(?:approximately\s+)?(\d[\d,]*)\s+'
        r'(?:position|employee|role|job|staff|headcount|worker)',
        re.IGNORECASE
    )
    for m in hc_pattern.finditer(mda):
        try:
            count = int(m.group(1).replace(',', ''))
            result['headcount_changes'].append({
                'count': count,
                'context': mda[max(0, m.start()-60):m.end()+60].strip()
            })
        except ValueError:
            pass

    # Extract facility closures
    fac_pattern = re.compile(
        r'(?:clos|consolidat|exit|shut)\w*\s+(?:our\s+|the\s+)?'
        r'(?:\w+\s+){0,3}(?:facility|plant|office|warehouse|location|site)',
        re.IGNORECASE
    )
    result['facility_actions'] = _find_sentences(mda, fac_pattern)[:5]

    # Extract savings amounts
    if savings_matches:
        result['savings_mentioned'] = savings_matches[:5]
        # Try to extract dollar amounts from savings context
        for s in savings_matches:
            for amt in _extract_dollar_amounts(s):
                result['restructuring_charges'].append(amt)

    # Build catalyst entries
    phase_labels = {
        'announced': 'Restructuring announced — early stage, execution risk remains',
        'in_progress': 'Restructuring in progress — charges ongoing, savings building',
        'nearing_completion': 'Restructuring nearing completion — inflection approaching',
        'complete': 'Restructuring complete — savings should flow to bottom line',
    }

    cat = {
        'category': 'cost_restructuring',
        'subcategory': result['phase'],
        'phase': result['phase'],
        'description': phase_labels.get(result['phase'], ''),
        'bullish_if': result['phase'] in ('nearing_completion', 'complete'),
        'magnitude': 'high' if result['phase'] in ('nearing_completion', 'complete') else 'medium',
        'evidence': (complete_matches or progress_matches or announce_matches)[:3],
        'timeline': 'now' if result['phase'] == 'complete' else 'in_progress',
    }
    if result['headcount_changes']:
        cat['headcount_reduction'] = sum(h['count'] for h in result['headcount_changes'])
    if result['savings_mentioned']:
        cat['savings_evidence'] = result['savings_mentioned'][:2]

    result['catalysts'].append(cat)
    return result


# ============================================================================
# CATALYST 2: MANAGEMENT SHAKEUP
# ============================================================================

_MGMT_CHANGE = re.compile(
    r'(?:appointed|named|hired|elected|promoted)\s+(?:\w+\s+){0,6}'
    r'(?:chief\s+(?:executive|financial|operating|technology|commercial|revenue)\s+officer|'
    r'CEO|CFO|COO|CTO|CRO|president|chairman|vice\s+chairman)',
    re.IGNORECASE
)
_MGMT_DEPART = re.compile(
    r'(?:resign|depart|terminat|separat|retire)\w*\s+(?:\w+\s+){0,6}'
    r'(?:chief\s+(?:executive|financial|operating)\s+officer|CEO|CFO|COO|president)',
    re.IGNORECASE
)
_BOARD_REFRESH = re.compile(
    r'(?:appointed|elected|named)\s+(?:\w+\s+){0,6}(?:board\s+of\s+directors|'
    r'independent\s+director|new\s+director)',
    re.IGNORECASE
)


def extract_management_catalysts(text: str) -> Dict[str, Any]:
    """
    Detect management changes — new CEO/CFO is a strong turnaround signal.
    
    New management + insider buying (from Form 4) = highest conviction turnaround setup.
    """
    result = {
        'found': False,
        'catalysts': [],
        'new_appointments': [],
        'departures': [],
        'board_changes': [],
    }

    # Check full filing (management changes can appear anywhere)
    appoint_matches = _find_sentences(text, _MGMT_CHANGE, 250)
    depart_matches = _find_sentences(text, _MGMT_DEPART, 250)
    board_matches = _find_sentences(text, _BOARD_REFRESH, 250)

    if not (appoint_matches or depart_matches or board_matches):
        return result

    result['found'] = True
    result['new_appointments'] = appoint_matches[:5]
    result['departures'] = depart_matches[:3]
    result['board_changes'] = board_matches[:3]

    # Classify significance
    is_c_suite = False
    for m in appoint_matches:
        if re.search(r'CEO|chief\s+executive', m, re.IGNORECASE):
            is_c_suite = True
            result['catalysts'].append({
                'category': 'management_change',
                'subcategory': 'new_ceo',
                'description': 'New CEO appointed — strongest management turnaround signal',
                'bullish_if': True,
                'magnitude': 'high',
                'evidence': [m[:250]],
            })
        elif re.search(r'CFO|chief\s+financial', m, re.IGNORECASE):
            is_c_suite = True
            result['catalysts'].append({
                'category': 'management_change',
                'subcategory': 'new_cfo',
                'description': 'New CFO appointed — signals financial discipline focus',
                'bullish_if': True,
                'magnitude': 'medium',
                'evidence': [m[:250]],
            })

    if not is_c_suite and (appoint_matches or board_matches):
        result['catalysts'].append({
            'category': 'management_change',
            'subcategory': 'board_refresh',
            'description': f'Board/leadership changes detected ({len(appoint_matches)} appointments, {len(board_matches)} board changes)',
            'bullish_if': True,
            'magnitude': 'low',
            'evidence': (appoint_matches + board_matches)[:3],
        })

    return result


# ============================================================================
# CATALYST 3: DEBT RESTRUCTURING
# ============================================================================

_DEBT_PAYDOWN = re.compile(
    r'(?:repaid|paid\s+down|retired|redeemed|extinguished|reduced)\s+'
    r'(?:\w+\s+){0,4}(?:debt|indebtedness|borrowing|loan|note|credit\s+facility|'
    r'term\s+loan|revolving|senior\s+(?:secured|unsecured))',
    re.IGNORECASE
)
_DEBT_REFINANCE = re.compile(
    r'(?:refinanc|amend|extend|modif|renegotiat)\w*\s+'
    r'(?:\w+\s+){0,4}(?:credit\s+(?:facility|agreement)|debt|loan|note|'
    r'term\s+loan|revolving|indenture)',
    re.IGNORECASE
)
_COVENANT = re.compile(
    r'(?:covenant\s+(?:compliance|waiver|amendment|modification|relief|violation|breach)|'
    r'(?:waiv|amend)\w*\s+(?:\w+\s+){0,3}covenant|'
    r'(?:in\s+compliance|not\s+in\s+compliance)\s+with\s+(?:\w+\s+){0,3}covenant)',
    re.IGNORECASE
)
_MATURITY_EXTENSION = re.compile(
    r'(?:extend|push\w*\s+out|lengthen)\w*\s+(?:\w+\s+){0,4}'
    r'(?:maturity|due\s+date|expiration|term)|'
    r'maturity\s+(?:date\s+)?(?:was|has\s+been)\s+extended',
    re.IGNORECASE
)
_DEBT_MATURITY = re.compile(
    r'(?:matur|com(?:es?|ing)\s+due|payable)\s+(?:in\s+)?(?:20\d\d|fiscal\s+year)',
    re.IGNORECASE
)


def extract_debt_catalysts(text: str) -> Dict[str, Any]:
    """
    Detect debt restructuring, paydown, refinancing, and maturity events.
    
    For distressed microcaps, debt cleanup is often THE turnaround catalyst.
    Covenant relief + refinancing = survival signal. Paydown = balance sheet healing.
    """
    result = {
        'found': False,
        'catalysts': [],
        'paydown_signals': [],
        'refinance_signals': [],
        'covenant_signals': [],
        'maturity_dates': [],
    }

    mda = _extract_section(text, MDA_HEADER, 60000) or ''
    liquidity = _extract_section(text, LIQUIDITY_HEADER, 30000) or ''
    focus = (mda + ' ' + liquidity) if (mda or liquidity) else text[:60000]

    paydown = _find_sentences(focus, _DEBT_PAYDOWN, 300)
    refinance = _find_sentences(focus, _DEBT_REFINANCE, 300)
    covenant = _find_sentences(focus, _COVENANT, 300)
    extension = _find_sentences(focus, _MATURITY_EXTENSION, 300)
    maturity = _find_sentences(focus, _DEBT_MATURITY, 200)

    if not (paydown or refinance or covenant or extension):
        return result

    result['found'] = True
    result['paydown_signals'] = paydown[:5]
    result['refinance_signals'] = refinance[:5]
    result['covenant_signals'] = covenant[:3]
    result['maturity_dates'] = maturity[:3]

    if paydown:
        result['catalysts'].append({
            'category': 'debt_restructuring',
            'subcategory': 'debt_paydown',
            'description': 'Active debt reduction — balance sheet healing',
            'bullish_if': True,
            'magnitude': 'high',
            'evidence': paydown[:2],
        })

    if refinance or extension:
        # Check if terms improved or worsened
        lower_rate = any(
            re.search(r'(?:lower|reduced|favorable|improved)\s+(?:\w+\s+){0,3}'
                       r'(?:rate|interest|spread|pricing)', s, re.IGNORECASE)
            for s in (refinance + extension)
        )
        result['catalysts'].append({
            'category': 'debt_restructuring',
            'subcategory': 'refinancing',
            'description': 'Debt refinanced/extended' +
                          (' at improved terms' if lower_rate else '') +
                          ' — extends runway',
            'bullish_if': True,
            'magnitude': 'high' if lower_rate else 'medium',
            'evidence': (refinance + extension)[:2],
        })

    if covenant:
        # Distinguish compliance vs waiver vs violation
        has_waiver = any('waiv' in s.lower() for s in covenant)
        has_violation = any(
            re.search(r'(?:violat|breach|not\s+in\s+compliance)', s, re.IGNORECASE)
            for s in covenant
        )
        if has_violation:
            result['catalysts'].append({
                'category': 'debt_restructuring',
                'subcategory': 'covenant_breach',
                'description': 'Covenant violation detected — financial distress signal',
                'bullish_if': False,
                'magnitude': 'high',
                'evidence': covenant[:2],
            })
        elif has_waiver:
            result['catalysts'].append({
                'category': 'debt_restructuring',
                'subcategory': 'covenant_waiver',
                'description': 'Covenant waiver obtained — lender cooperation signal',
                'bullish_if': True,
                'magnitude': 'medium',
                'evidence': covenant[:2],
            })

    return result


# ============================================================================
# CATALYST 4: MARGIN INFLECTION NARRATIVE
# ============================================================================

_MARGIN_IMPROVE = re.compile(
    r'(?:gross\s+(?:profit\s+)?margin|operating\s+margin|(?:EBITDA|EBIT)\s+margin|'
    r'net\s+margin|contribution\s+margin)\s+'
    r'(?:improv|increas|expand|grew|higher|widen)',
    re.IGNORECASE
)
_MARGIN_SEQUENTIAL = re.compile(
    r'(?:sequential(?:ly)?\s+improv|quarter[\-\s]over[\-\s]quarter\s+(?:improv|increas)|'
    r'margin\s+(?:improv|expand)\w*\s+(?:sequentially|from\s+the\s+prior\s+quarter))',
    re.IGNORECASE
)
_OPERATING_LEVERAGE = re.compile(
    r'(?:operating\s+leverage|fixed\s+cost\s+(?:absorption|leverage)|'
    r'revenue\s+growth\s+(?:outpac|exceed)\w*\s+(?:\w+\s+){0,3}(?:cost|expense)\s+growth|'
    r'(?:cost|expense)s?\s+(?:grew|increased)\s+(?:at\s+)?(?:a\s+)?(?:slower|lower)\s+rate\s+than\s+revenue)',
    re.IGNORECASE
)
_PROFITABILITY_CROSSING = re.compile(
    r'(?:first\s+(?:time\s+)?(?:quarter|year|period)\s+(?:of\s+)?(?:profitab|positive\s+(?:net\s+)?income|'
    r'positive\s+(?:operating|adjusted)\s+(?:income|earnings))|'
    r'achiev\w+\s+(?:operating\s+)?profitab|'
    r'turn(?:ed|ing)\s+(?:a\s+)?profit|'
    r'(?:became|become)\s+(?:cash\s+flow\s+)?(?:profitable|positive)|'
    r'adjusted\s+EBITDA\s+(?:was|turned)\s+positive)',
    re.IGNORECASE
)


def extract_margin_catalysts(text: str) -> Dict[str, Any]:
    """
    Detect margin inflection and profitability crossing narratives.
    
    For turnarounds, margin improvement is the operational proof that restructuring works.
    First profitable quarter is a major re-rating catalyst for microcaps.
    """
    result = {
        'found': False,
        'catalysts': [],
        'margin_improvements': [],
        'sequential_improvements': [],
        'leverage_signals': [],
        'profitability_crossings': [],
    }

    mda = _extract_section(text, MDA_HEADER, 60000) or text[:60000]

    margin_imp = _find_sentences(mda, _MARGIN_IMPROVE, 250)
    sequential = _find_sentences(mda, _MARGIN_SEQUENTIAL, 250)
    leverage = _find_sentences(mda, _OPERATING_LEVERAGE, 250)
    profit_cross = _find_sentences(mda, _PROFITABILITY_CROSSING, 300)

    if not (margin_imp or sequential or leverage or profit_cross):
        return result

    result['found'] = True
    result['margin_improvements'] = margin_imp[:5]
    result['sequential_improvements'] = sequential[:3]
    result['leverage_signals'] = leverage[:3]
    result['profitability_crossings'] = profit_cross[:3]

    if profit_cross:
        result['catalysts'].append({
            'category': 'margin_inflection',
            'subcategory': 'profitability_crossing',
            'description': 'Profitability milestone reached — major re-rating catalyst',
            'bullish_if': True,
            'magnitude': 'very_high',
            'evidence': profit_cross[:2],
        })

    if sequential:
        result['catalysts'].append({
            'category': 'margin_inflection',
            'subcategory': 'sequential_improvement',
            'description': 'Sequential margin improvement — operational trajectory positive',
            'bullish_if': True,
            'magnitude': 'high',
            'evidence': sequential[:2],
        })
    elif margin_imp:
        result['catalysts'].append({
            'category': 'margin_inflection',
            'subcategory': 'margin_expansion',
            'description': 'Margin expansion mentioned in narrative',
            'bullish_if': True,
            'magnitude': 'medium',
            'evidence': margin_imp[:2],
        })

    if leverage:
        result['catalysts'].append({
            'category': 'margin_inflection',
            'subcategory': 'operating_leverage',
            'description': 'Operating leverage kicking in — costs growing slower than revenue',
            'bullish_if': True,
            'magnitude': 'high',
            'evidence': leverage[:2],
        })

    return result


# ============================================================================
# CATALYST 5: ASSET MONETIZATION
# ============================================================================

_ASSET_SALE = re.compile(
    r'(?:sold|divest|dispos|monetiz)\w*\s+(?:\w+\s+){0,4}'
    r'(?:asset|division|segment|subsidiary|business\s+(?:unit|line)|'
    r'property|real\s+estate|intellectual\s+property|patent|'
    r'non[\-\s]?core|underperform)|'
    r'(?:complet\w+\s+)?(?:the\s+)?sale\s+of\s+(?:\w+\s+){0,4}'
    r'(?:asset|division|segment|subsidiary|business\s+(?:unit|line)|'
    r'property|real\s+estate|non[\-\s]?core|underperform)',
    re.IGNORECASE
)
_IP_LICENSING = re.compile(
    r'(?:licens|royalt)\w*\s+(?:\w+\s+){0,4}'
    r'(?:agreement|revenue|income|arrangement|patent|technology|IP)',
    re.IGNORECASE
)
_STRATEGIC_ALTERNATIVES = re.compile(
    r'(?:strategic\s+alternative|exploring\s+(?:strategic\s+)?option|'
    r'retain(?:ed)?\s+(?:\w+\s+){0,3}(?:advisor|banker|investment\s+bank)|'
    r'sale\s+of\s+the\s+company|merger\s+(?:discussion|negotiation))',
    re.IGNORECASE
)


def extract_asset_catalysts(text: str) -> Dict[str, Any]:
    """
    Detect asset monetization and strategic review signals.
    
    Microcaps often trade below asset value. Divestiture or "strategic alternatives"
    language can signal 20-50%+ upside to liquidation/acquisition value.
    """
    result = {
        'found': False,
        'catalysts': [],
        'asset_sales': [],
        'licensing_revenue': [],
        'strategic_review': [],
    }

    asset_sales = _find_sentences(text, _ASSET_SALE, 300)
    licensing = _find_sentences(text, _IP_LICENSING, 250)
    strategic = _find_sentences(text, _STRATEGIC_ALTERNATIVES, 300)

    if not (asset_sales or strategic):
        return result

    result['found'] = True
    result['asset_sales'] = asset_sales[:5]
    result['licensing_revenue'] = licensing[:3]
    result['strategic_review'] = strategic[:3]

    if strategic:
        result['catalysts'].append({
            'category': 'asset_monetization',
            'subcategory': 'strategic_alternatives',
            'description': '"Strategic alternatives" language detected — potential sale/merger',
            'bullish_if': True,
            'magnitude': 'very_high',
            'evidence': strategic[:2],
        })

    if asset_sales:
        result['catalysts'].append({
            'category': 'asset_monetization',
            'subcategory': 'divestiture',
            'description': 'Non-core asset sales — focus sharpening, cash unlocked',
            'bullish_if': True,
            'magnitude': 'medium',
            'evidence': asset_sales[:2],
        })

    return result


# ============================================================================
# CATALYST 6: WORKING CAPITAL NORMALIZATION
# ============================================================================

_INVENTORY_IMPROVE = re.compile(
    r'(?:inventory\s+(?:decreas|reduc|decline|drew\s+down|drawdown|normaliz|improv)|'
    r'(?:reduc|decreas)\w*\s+(?:\w+\s+){0,3}inventor|'
    r'inventory\s+turn(?:over|s)?\s+(?:improv|increas))',
    re.IGNORECASE
)
_RECEIVABLES_IMPROVE = re.compile(
    r'(?:(?:accounts?\s+)?receivable\s+(?:decreas|reduc|improv|collect)|'
    r'DSO\s+(?:decreas|improv|reduc)|'
    r'days\s+sales\s+outstanding\s+(?:decreas|improv))',
    re.IGNORECASE
)
_CASH_CONVERSION = re.compile(
    r'(?:free\s+cash\s+flow\s+(?:improv|turn\w+\s+positive|increas|generat)|'
    r'(?:operat|positive)\s+cash\s+flow\s+(?:for\s+the\s+first|improv|increas)|'
    r'cash\s+(?:conversion|generation)\s+(?:improv|increas))',
    re.IGNORECASE
)


def extract_working_capital_catalysts(text: str) -> Dict[str, Any]:
    """
    Detect working capital normalization — signals operational healing.
    
    Inventory drawdowns + receivables improving + positive FCF = 
    the business is actually getting healthier, not just cutting costs.
    """
    result = {
        'found': False,
        'catalysts': [],
        'inventory_signals': [],
        'receivables_signals': [],
        'cash_flow_signals': [],
    }

    mda = _extract_section(text, MDA_HEADER, 60000) or text[:60000]

    inv = _find_sentences(mda, _INVENTORY_IMPROVE, 250)
    rec = _find_sentences(mda, _RECEIVABLES_IMPROVE, 250)
    cash = _find_sentences(mda, _CASH_CONVERSION, 250)

    if not (inv or rec or cash):
        return result

    result['found'] = True
    result['inventory_signals'] = inv[:3]
    result['receivables_signals'] = rec[:3]
    result['cash_flow_signals'] = cash[:3]

    signals_count = sum(1 for s in [inv, rec, cash] if s)

    if cash:
        result['catalysts'].append({
            'category': 'working_capital',
            'subcategory': 'cash_flow_improvement',
            'description': 'Cash flow improving — operational health signal',
            'bullish_if': True,
            'magnitude': 'high' if signals_count >= 2 else 'medium',
            'evidence': cash[:2],
        })

    if signals_count >= 2:
        result['catalysts'].append({
            'category': 'working_capital',
            'subcategory': 'broad_normalization',
            'description': f'Working capital normalizing across {signals_count} dimensions — business healing',
            'bullish_if': True,
            'magnitude': 'high',
            'evidence': (inv + rec + cash)[:3],
        })
    elif inv:
        result['catalysts'].append({
            'category': 'working_capital',
            'subcategory': 'inventory_reduction',
            'description': 'Inventory drawdown — demand absorbing excess or management tightening controls',
            'bullish_if': True,
            'magnitude': 'low',
            'evidence': inv[:2],
        })

    return result


# ============================================================================
# CATALYST 7: ACTIVIST / OWNERSHIP CHANGES
# ============================================================================

_ACTIVIST = re.compile(
    r'(?:schedule\s+13D|SC\s+13D|beneficial\s+ownership\s+(?:report|filing)|'
    r'activist\s+(?:investor|shareholder|stake)|'
    r'demand\w*\s+(?:\w+\s+){0,3}(?:board\s+(?:seat|representation)|special\s+meeting)|'
    r'consent\s+solicitation|proxy\s+contest|'
    r'nomin\w+\s+(?:\w+\s+){0,3}director)',
    re.IGNORECASE
)
_LARGE_HOLDER = re.compile(
    r'(?:acquir|purchas|accumulat)\w*\s+(?:approximately\s+)?(?:\d+[\.\d]*%|'
    r'a\s+(?:significant|substantial|large|major)\s+(?:stake|position|interest|block))',
    re.IGNORECASE
)
_BUYBACK = re.compile(
    r'(?:share\s+(?:repurchase|buyback)|stock\s+(?:repurchase|buyback)|'
    r'repurchas\w+\s+(?:\w+\s+){0,3}(?:share|stock)|'
    r'(?:authorized|approved)\s+(?:\w+\s+){0,3}(?:repurchase|buyback)\s+(?:program|plan))',
    re.IGNORECASE
)


def extract_ownership_catalysts(text: str) -> Dict[str, Any]:
    """
    Detect activist involvement and ownership structure changes.
    
    Activist taking a position = external pressure for change (can be very bullish for turnarounds).
    Buyback = management believes stock is undervalued.
    """
    result = {
        'found': False,
        'catalysts': [],
        'activist_signals': [],
        'large_holder_signals': [],
        'buyback_signals': [],
    }

    activist = _find_sentences(text, _ACTIVIST, 300)
    large_holder = _find_sentences(text, _LARGE_HOLDER, 250)
    buyback = _find_sentences(text, _BUYBACK, 250)

    if not (activist or buyback):
        return result

    result['found'] = True
    result['activist_signals'] = activist[:5]
    result['large_holder_signals'] = large_holder[:3]
    result['buyback_signals'] = buyback[:3]

    if activist:
        result['catalysts'].append({
            'category': 'ownership_change',
            'subcategory': 'activist_involvement',
            'description': 'Activist investor activity detected — external pressure for change',
            'bullish_if': True,
            'magnitude': 'very_high',
            'evidence': activist[:2],
        })

    if buyback:
        # Check if new authorization or just existing program
        new_auth = any(
            re.search(r'(?:authoriz|approv|new|additional)\w*', s, re.IGNORECASE)
            for s in buyback
        )
        result['catalysts'].append({
            'category': 'ownership_change',
            'subcategory': 'buyback',
            'description': ('New buyback authorization' if new_auth else 'Share repurchase program') +
                          ' — management signals undervaluation',
            'bullish_if': True,
            'magnitude': 'medium' if new_auth else 'low',
            'evidence': buyback[:2],
        })

    return result


# ============================================================================
# CATALYST 8: GOING CONCERN CHANGES
# ============================================================================

_GOING_CONCERN = re.compile(
    r'(?:going\s+concern|substantial\s+doubt\s+(?:about|regarding|as\s+to)\s+(?:\w+\s+){0,3}'
    r'(?:ability\s+to\s+continue|continued\s+existence)|'
    r'ability\s+to\s+continue\s+as\s+a\s+going\s+concern)',
    re.IGNORECASE
)
_GOING_CONCERN_RESOLVED = re.compile(
    r'(?:(?:no\s+longer|eliminated|resolved|removed|alleviated)\s+'
    r'(?:\w+\s+){0,6}(?:going\s+concern|substantial\s+doubt)|'
    r'(?:going\s+concern|substantial\s+doubt)\s+'
    r'(?:[\w,]+\s+){0,15}(?:has\s+been|was|is)\s+'
    r'(?:resolved|eliminated|removed|alleviated|addressed)|'
    r'(?:substantial\s+doubt)\s+(?:[\w,]+\s+){0,15}'
    r'(?:no\s+longer\s+exist|has\s+been\s+(?:resolved|eliminated|removed|alleviated)))',
    re.IGNORECASE
)
_GOING_CONCERN_NEW = re.compile(
    r'(?:raise|exist|identified|expressed)\w*\s+(?:\w+\s+){0,3}'
    r'(?:substantial\s+doubt|going\s+concern)',
    re.IGNORECASE
)


def extract_going_concern_catalysts(text: str) -> Dict[str, Any]:
    """
    Detect going concern qualification changes.
    
    Going concern REMOVED = existential risk resolved (massive turnaround catalyst).
    Going concern ADDED = existential risk (massive red flag).
    Binary signal with huge price impact for microcaps.
    """
    result = {
        'found': False,
        'catalysts': [],
        'going_concern_mentions': [],
        'resolved_signals': [],
        'new_concern_signals': [],
    }

    gc_mentions = _find_sentences(text, _GOING_CONCERN, 300)

    if not gc_mentions:
        return result

    result['found'] = True
    result['going_concern_mentions'] = gc_mentions[:5]

    resolved = _find_sentences(text, _GOING_CONCERN_RESOLVED, 300)
    new_concern = _find_sentences(text, _GOING_CONCERN_NEW, 300)

    result['resolved_signals'] = resolved[:3]
    result['new_concern_signals'] = new_concern[:3]

    if resolved:
        result['catalysts'].append({
            'category': 'going_concern',
            'subcategory': 'resolved',
            'description': 'Going concern doubt resolved — existential risk removed (major catalyst)',
            'bullish_if': True,
            'magnitude': 'very_high',
            'evidence': resolved[:2],
        })
    elif new_concern:
        result['catalysts'].append({
            'category': 'going_concern',
            'subcategory': 'new_doubt',
            'description': 'Going concern doubt raised — existential risk (major red flag)',
            'bullish_if': False,
            'magnitude': 'very_high',
            'evidence': new_concern[:2],
        })
    else:
        # Mentioned but can't determine direction from this filing alone
        result['catalysts'].append({
            'category': 'going_concern',
            'subcategory': 'mentioned',
            'description': 'Going concern language present — needs prior filing comparison for direction',
            'bullish_if': False,
            'magnitude': 'high',
            'evidence': gc_mentions[:2],
        })

    return result


# ============================================================================
# CATALYST 9: TIMELINE EXTRACTION (forward-looking dates)
# ============================================================================

_FORWARD_TIMELINE = re.compile(
    r'(?:expect|anticipat|plan|schedul|target|project)\w*\s+'
    r'(?:to\s+)?(?:\w+\s+){0,6}'
    r'(?:by\s+(?:the\s+)?(?:end\s+of\s+)?|in\s+|during\s+|(?:first|second|third|fourth)\s+quarter\s+(?:of\s+)?)?'
    r'(?:Q[1-4]\s+)?(?:20\d\d|fiscal\s+(?:year\s+)?20\d\d|'
    r'(?:first|second|third|fourth)\s+(?:quarter|half)|'
    r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d\d)',
    re.IGNORECASE
)


def extract_timeline_catalysts(text: str) -> List[Dict[str, str]]:
    """
    Extract forward-looking dates from filings to build a catalyst calendar.
    
    Returns list of {event, timeline, context} for events management expects to happen.
    """
    mda = _extract_section(text, MDA_HEADER, 60000) or text[:60000]
    matches = _find_sentences(mda, _FORWARD_TIMELINE, 250)

    timelines = []
    seen = set()
    for m in matches[:15]:  # cap at 15 to avoid noise
        # Deduplicate by first 60 chars
        key = m[:60].lower().strip()
        if key not in seen:
            seen.add(key)
            # Try to extract the actual date/quarter
            date_match = re.search(
                r'(Q[1-4]\s+20\d\d|20\d\d|'
                r'(?:first|second|third|fourth)\s+(?:quarter|half)\s+(?:of\s+)?20\d\d|'
                r'(?:January|February|March|April|May|June|July|August|September|'
                r'October|November|December)\s+20\d\d)',
                m, re.IGNORECASE
            )
            timelines.append({
                'event': m[:200],
                'timeline': date_match.group(0) if date_match else 'unspecified',
            })

    return timelines


# ============================================================================
# MASTER EXTRACTION FUNCTION
# ============================================================================

# Category weights for scoring (turnaround significance)
CATALYST_WEIGHTS = {
    'going_concern': {'resolved': 25, 'new_doubt': -30, 'mentioned': -10,
                      'removed_vs_prior': 25, 'added_vs_prior': -30},
    'cost_restructuring': {'complete': 20, 'nearing_completion': 15, 'in_progress': 5, 'announced': 3},
    'management_change': {'new_ceo': 18, 'new_cfo': 12, 'board_refresh': 5},
    'debt_restructuring': {'debt_paydown': 15, 'refinancing': 12, 'covenant_waiver': 8, 'covenant_breach': -15},
    'margin_inflection': {'profitability_crossing': 22, 'sequential_improvement': 15, 'margin_expansion': 8, 'operating_leverage': 12},
    'asset_monetization': {'strategic_alternatives': 20, 'divestiture': 10},
    'working_capital': {'cash_flow_improvement': 12, 'broad_normalization': 10, 'inventory_reduction': 4},
    'ownership_change': {'activist_involvement': 18, 'buyback': 6},
}


def extract_all_catalysts(text: str, prior_text: str = None) -> Dict[str, Any]:
    """
    Run all catalyst extractors on a filing and produce a turnaround catalyst profile.
    
    Args:
        text: Current filing text
        prior_text: Prior filing text (for going concern change detection)
    
    Returns:
        Dict with all catalyst categories, aggregate scoring, and catalyst calendar.
    """
    # Run all extractors
    restructuring = extract_restructuring_catalysts(text)
    management = extract_management_catalysts(text)
    debt = extract_debt_catalysts(text)
    margin = extract_margin_catalysts(text)
    assets = extract_asset_catalysts(text)
    working_cap = extract_working_capital_catalysts(text)
    ownership = extract_ownership_catalysts(text)
    going_concern = extract_going_concern_catalysts(text)
    timelines = extract_timeline_catalysts(text)

    # Collect all catalysts
    all_catalysts = []
    categories_found = []
    for extractor_result in [restructuring, management, debt, margin, assets,
                              working_cap, ownership, going_concern]:
        if extractor_result.get('found'):
            categories_found.append(extractor_result)
            all_catalysts.extend(extractor_result.get('catalysts', []))

    # Prior filing comparison for going concern
    gc_direction = None
    if going_concern.get('found'):
        if going_concern.get('resolved_signals'):
            # Current filing explicitly says GC resolved
            gc_direction = 'removed'
            if not any(c.get('subcategory') == 'removed_vs_prior' for c in all_catalysts):
                all_catalysts.append({
                    'category': 'going_concern',
                    'subcategory': 'removed_vs_prior',
                    'description': 'Going concern language RESOLVED in current filing — existential risk removed',
                    'bullish_if': True,
                    'magnitude': 'very_high',
                    'evidence': going_concern['resolved_signals'][:2],
                })
        elif going_concern.get('new_concern_signals'):
            gc_direction = 'added'
            if not any(c.get('subcategory') == 'added_vs_prior' for c in all_catalysts):
                all_catalysts.append({
                    'category': 'going_concern',
                    'subcategory': 'added_vs_prior',
                    'description': 'Going concern doubt raised — new existential risk',
                    'bullish_if': False,
                    'magnitude': 'very_high',
                    'evidence': going_concern['new_concern_signals'][:2],
                })
        if prior_text and gc_direction is None:
            # Fallback: compare presence in prior vs current when direction
            # was not determined from resolved/new signals above
            prior_gc = _find_sentences(prior_text, _GOING_CONCERN, 200)
            cur_gc = going_concern.get('going_concern_mentions', [])
            if prior_gc and not cur_gc:
                gc_direction = 'removed'
                if not any(c.get('subcategory') == 'removed_vs_prior' for c in all_catalysts):
                    all_catalysts.append({
                        'category': 'going_concern',
                        'subcategory': 'removed_vs_prior',
                        'description': 'Going concern language REMOVED since prior filing — existential risk resolved',
                        'bullish_if': True,
                        'magnitude': 'very_high',
                        'evidence': ['Prior filing had going concern language. Current filing does not.'],
                    })
            elif not prior_gc and cur_gc:
                gc_direction = 'added'
                if not any(c.get('subcategory') == 'added_vs_prior' for c in all_catalysts):
                    all_catalysts.append({
                        'category': 'going_concern',
                        'subcategory': 'added_vs_prior',
                        'description': 'Going concern language ADDED since prior filing — new existential risk',
                        'bullish_if': False,
                        'magnitude': 'very_high',
                        'evidence': ['Prior filing had no going concern language. Current filing does.'],
                    })

    # Score catalysts
    turnaround_score = 50  # neutral baseline
    bullish_catalysts = []
    bearish_catalysts = []

    for cat in all_catalysts:
        category = cat.get('category', '')
        subcat = cat.get('subcategory', '')
        weight = CATALYST_WEIGHTS.get(category, {}).get(subcat, 0)
        turnaround_score += weight

        if cat.get('bullish_if'):
            bullish_catalysts.append(cat)
        else:
            bearish_catalysts.append(cat)

    turnaround_score = max(0, min(100, turnaround_score))

    # Determine turnaround phase
    if restructuring.get('phase') == 'complete' and margin.get('found'):
        turnaround_phase = 'inflection'
    elif restructuring.get('phase') in ('nearing_completion', 'complete'):
        turnaround_phase = 'late_stage'
    elif restructuring.get('phase') in ('in_progress', 'announced'):
        turnaround_phase = 'early_stage'
    elif management.get('found') or debt.get('found'):
        turnaround_phase = 'catalyst_forming'
    elif any(e.get('found') for e in [margin, working_cap]):
        turnaround_phase = 'organic_improvement'
    else:
        turnaround_phase = 'no_turnaround_signals'

    # Overall signal
    if turnaround_score >= 75:
        signal = 'strong_turnaround'
        summary = 'Multiple strong turnaround catalysts detected — high conviction'
    elif turnaround_score >= 60:
        signal = 'turnaround_forming'
        summary = 'Turnaround catalysts forming — operational improvement visible'
    elif turnaround_score >= 48:
        signal = 'mixed'
        summary = 'Mixed signals — some positive catalysts offset by risks'
    elif turnaround_score >= 35:
        signal = 'distressed'
        summary = 'Distress signals outweigh recovery catalysts'
    else:
        signal = 'severe_distress'
        summary = 'Severe distress — going concern or major red flags detected'

    # Rank catalysts by magnitude
    magnitude_rank = {'very_high': 4, 'high': 3, 'medium': 2, 'low': 1}
    all_catalysts.sort(key=lambda c: magnitude_rank.get(c.get('magnitude', 'low'), 0), reverse=True)

    return {
        'turnaround_score': turnaround_score,
        'turnaround_signal': signal,
        'turnaround_phase': turnaround_phase,
        'turnaround_summary': summary,
        'catalyst_count': len(all_catalysts),
        'bullish_count': len(bullish_catalysts),
        'bearish_count': len(bearish_catalysts),
        'categories_found': len(categories_found),
        'gc_direction': gc_direction,
        'catalysts': all_catalysts,
        'bullish_catalysts': bullish_catalysts[:5],
        'bearish_catalysts': bearish_catalysts[:5],
        'timelines': timelines[:10],
        'detail': {
            'restructuring': restructuring,
            'management': management,
            'debt': debt,
            'margin': margin,
            'assets': assets,
            'working_capital': working_cap,
            'ownership': ownership,
            'going_concern': going_concern,
        },
    }


# ============================================================================
# PROMPT FORMATTING
# ============================================================================

def format_catalysts_for_prompt(catalyst_data: Dict[str, Any]) -> str:
    """Format catalyst extraction results for inclusion in LLM prompt."""
    if not catalyst_data or catalyst_data.get('catalyst_count', 0) == 0:
        return "(No turnaround catalysts detected)"

    lines = [
        "=== TURNAROUND CATALYST ANALYSIS (algorithmic extraction) ===",
        f"Turnaround Score: {catalyst_data['turnaround_score']}/100 "
        f"({catalyst_data['turnaround_signal'].replace('_', ' ')})",
        f"Phase: {catalyst_data['turnaround_phase'].replace('_', ' ')}",
        f"Catalysts: {catalyst_data['bullish_count']} bullish, "
        f"{catalyst_data['bearish_count']} bearish "
        f"({catalyst_data['categories_found']} categories active)",
        "",
    ]

    if catalyst_data.get('gc_direction'):
        gc = catalyst_data['gc_direction']
        lines.append(f"⚠️  GOING CONCERN: {gc.upper()} vs prior filing")
        lines.append("")

    # List catalysts by priority
    lines.append("DETECTED CATALYSTS (ranked by significance):")
    for i, cat in enumerate(catalyst_data.get('catalysts', [])[:8], 1):
        icon = '🟢' if cat.get('bullish_if') else '🔴'
        lines.append(
            f"  {i}. {icon} [{cat.get('category','?')}/{cat.get('subcategory','?')}] "
            f"({cat.get('magnitude','?')} impact)"
        )
        lines.append(f"     {cat.get('description', '')}")
        for ev in cat.get('evidence', [])[:1]:
            lines.append(f"     Evidence: \"{ev[:180]}\"")

    # Timeline
    timelines = catalyst_data.get('timelines', [])
    if timelines:
        lines.append("")
        lines.append("FORWARD-LOOKING TIMELINE (management-stated dates):")
        for t in timelines[:6]:
            lines.append(f"  • [{t['timeline']}] {t['event'][:150]}")

    # Restructuring phase detail
    restruct = catalyst_data.get('detail', {}).get('restructuring', {})
    if restruct.get('found'):
        lines.append("")
        lines.append(f"RESTRUCTURING DETAIL: Phase = {restruct['phase']}")
        if restruct.get('headcount_changes'):
            total_hc = sum(h['count'] for h in restruct['headcount_changes'])
            lines.append(f"  Headcount reduction: ~{total_hc:,} positions")
        if restruct.get('savings_mentioned'):
            lines.append(f"  Savings mentioned: {restruct['savings_mentioned'][0][:150]}")

    # Debt detail
    debt_d = catalyst_data.get('detail', {}).get('debt', {})
    if debt_d.get('found'):
        lines.append("")
        lines.append("DEBT SITUATION:")
        if debt_d.get('paydown_signals'):
            lines.append(f"  Paydown: {debt_d['paydown_signals'][0][:150]}")
        if debt_d.get('covenant_signals'):
            lines.append(f"  Covenants: {debt_d['covenant_signals'][0][:150]}")

    lines.append("")
    lines.append(
        "YOUR JOB: Interpret these catalyst signals. Are they genuine turnaround indicators "
        "or cosmetic? Do they align with the financial data and forensic analysis? "
        "What is the probability-weighted upside vs downside?"
    )

    return "\n".join(lines)
