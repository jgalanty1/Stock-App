"""
Filing Signals Module — Research-Backed Alpha Signals (10 Dimensions)
=====================================================================
Implements ten quantitative signals derived from academic research on SEC filings,
with weights calibrated to evidence strength and documented alpha:

PRIMARY SIGNALS (strongest evidence):
 1. MD&A Similarity       (20%) — Lazy Prices (Cohen+ JF 2020): 188bps/month alpha
 2. LM Sentiment          (15%) — Loughran-McDonald (JF 2011): R² 9.75% in-sample
 3. Risk Factor Changes   (13%) — Lazy Prices + Kirtac & Germano (2024)
 4. Positive Similarity   (12%) — Positive Similarity paper: uncorrelated alpha

SECONDARY SIGNALS (moderate evidence):
 5. Sentiment Delta       (10%) — Period-over-period sentiment change
 6. Legal Proceedings      (8%) — Offutt & Xie (2025): alpha unexplained by FF5+UMD
 7. Uncertainty Language   (7%) — LM uncertainty words: predicts IPO returns, volatility

TERTIARY SIGNALS (supporting evidence):
 8. Filing Timeliness      (5%) — Duarte-Silva et al. (2013): late = deterioration
 9. Document Complexity    (5%) — Fog index/readability: opacity = negative signal
10. Insider Pattern        (5%) — Form 4 cluster analysis: diminished post-disclosure

Each signal returns a 0-100 score:
  >60 = positive signal (stable filings, good sentiment, early filing)
  40-60 = neutral
  <40 = negative signal (big changes, negative tone, late filing)

Composite "research_signal_score" uses evidence-strength-weighted average.
"""

import re
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter

logger = logging.getLogger(__name__)

# ============================================================================
# TEXT SIMILARITY UTILITIES
# ============================================================================

_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'not',
    'no', 'nor', 'this', 'that', 'these', 'those', 'it', 'its', 'we',
    'our', 'us', 'they', 'their', 'them', 'he', 'she', 'his', 'her',
    'i', 'me', 'my', 'you', 'your', 'which', 'who', 'whom', 'what',
    'when', 'where', 'how', 'than', 'then', 'so', 'if', 'such', 'each',
    'all', 'any', 'both', 'more', 'other', 'some', 'only', 'also',
    'into', 'over', 'after', 'before', 'between', 'under', 'above',
    'very', 'just', 'about', 'up', 'out', 'off', 'down', 'through',
}

_WORD_RE = re.compile(r'\b[a-z]{2,}\b')


def _tokenize(text: str) -> List[str]:
    """Simple word tokenization, lowercase, remove stopwords."""
    words = _WORD_RE.findall(text.lower())
    return [w for w in words if w not in _STOPWORDS]


def _cosine_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute cosine similarity between two token lists using TF vectors."""
    if not tokens_a or not tokens_b:
        return 0.0

    counter_a = Counter(tokens_a)
    counter_b = Counter(tokens_b)

    all_words = set(counter_a.keys()) | set(counter_b.keys())
    dot_product = sum(counter_a.get(w, 0) * counter_b.get(w, 0) for w in all_words)
    norm_a = math.sqrt(sum(v ** 2 for v in counter_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in counter_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def _jaccard_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute Jaccard similarity (set overlap) between two token lists."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _word_count_ratio(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Ratio of document lengths (smaller/larger). Big size change = signal."""
    if not tokens_a or not tokens_b:
        return 1.0
    la, lb = len(tokens_a), len(tokens_b)
    return min(la, lb) / max(la, lb)


# ============================================================================
# SECTION EXTRACTION
# ============================================================================

LEGAL_PROCEEDINGS_HEADER_RE = re.compile(
    r'(?:item\s*(?:1|3)\.?\s*legal\s*proceedings'
    r'|legal\s*proceedings)',
    re.IGNORECASE,
)

MDA_HEADER_RE = re.compile(
    r"(?:item\s*(?:2|7)\.?\s*management'?s?\s*discussion|"
    r"management'?s?\s*discussion\s*and\s*analysis)",
    re.IGNORECASE,
)

RISK_FACTOR_HEADER_RE = re.compile(
    r'(?:item\s*1a\.?\s*risk\s*factors|risk\s*factors)',
    re.IGNORECASE,
)

# Keep backward-compatible aliases
LEGAL_PROCEEDINGS_HEADER = LEGAL_PROCEEDINGS_HEADER_RE
MDA_HEADER = MDA_HEADER_RE
RISK_FACTOR_HEADER = RISK_FACTOR_HEADER_RE


def _extract_section(text: str, header_pattern: re.Pattern,
                     max_chars: int = 50000,
                     min_chars: int = 500) -> Optional[str]:
    """Extract a section from filing text starting at header pattern.
    
    Returns None if section is shorter than min_chars (likely false positive match).
    """
    match = header_pattern.search(text)
    if not match:
        return None

    start = match.start()
    # Find the next major section header
    skip = len(match.group())
    next_item = re.search(r'\bitem\s+\d+[a-z]?\.', text[start + skip:], re.IGNORECASE)
    if next_item:
        end = start + skip + next_item.start()
    else:
        end = start + max_chars

    section = text[start:min(end, start + max_chars)]
    section = section.strip()
    
    # Return None if too short (likely false-positive header match)
    if len(section) < min_chars:
        return None
    
    return section


# ============================================================================
# SIGNAL WEIGHTS — calibrated to academic evidence strength
# ============================================================================
# Weights sum to 1.0 and reflect documented alpha / statistical significance.
# Primary signals (strongest evidence): 60% combined
# Secondary signals (moderate evidence): 25% combined
# Tertiary signals (supporting evidence): 15% combined
# ============================================================================

SIGNAL_WEIGHTS = {
    # PRIMARY — strongest documented alpha
    'mda_similarity':       0.20,  # Lazy Prices: 188bps/month (22% annual)
    'lm_sentiment':         0.15,  # Loughran-McDonald: R² 9.75%, widely validated
    'risk_factor_changes':  0.13,  # Lazy Prices: "especially informative"
    'positive_similarity':  0.12,  # Positive Similarity paper: uncorrelated to FF5

    # SECONDARY — moderate evidence
    'sentiment_delta':      0.10,  # Period-over-period sentiment change
    'legal_proceedings':    0.08,  # Offutt & Xie: alpha unexplained by FF5+UMD
    'uncertainty_language': 0.07,  # LM uncertainty: predicts IPO returns, volatility

    # TERTIARY — supporting evidence
    'filing_timeliness':    0.05,  # Duarte-Silva et al.: delays → deterioration
    'document_complexity':  0.05,  # Fog/readability: opacity = negative signal
    'insider_pattern':      0.05,  # Form 4 clusters: diminished but useful
}

# Evidence citations for each weight (displayed in prompt & UI)
SIGNAL_CITATIONS = {
    'mda_similarity':       'Lazy Prices (Cohen, Malloy, Nguyen — JF 2020)',
    'lm_sentiment':         'Loughran & McDonald (JF 2011); Management Sentiment Index (R² 9.75%)',
    'risk_factor_changes':  'Lazy Prices + Kirtac & Germano (2024)',
    'positive_similarity':  'Positive Similarity of Company Filings (2022)',
    'sentiment_delta':      'Brown & Tucker (2011); Bochkay (2014)',
    'legal_proceedings':    'Offutt & Xie (2025) — LLM legal risk scoring',
    'uncertainty_language': 'Loughran & McDonald uncertainty list; IPO underpricing lit.',
    'filing_timeliness':    'Duarte-Silva et al. (2013); Khalil et al. (2017)',
    'document_complexity':  'Li (2008) annual report readability; Lo et al. (2017)',
    'insider_pattern':      'Ozlen & Batumoglu (2025) — 70-80% alpha pre-disclosure',
}


# ============================================================================
# LOUGHRAN-MCDONALD FINANCE-SPECIFIC WORD LISTS
# ============================================================================
# Source: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
# Widely adopted: 36 of 78 textual analysis papers use LM dictionary
# Top terms from each category (representative subset for speed)
# ============================================================================

# NOTE: LM word sets intentionally overlap across categories (e.g., 'volatile' appears
# in both LM_NEGATIVE and LM_UNCERTAINTY, 'litigation' in LM_NEGATIVE and LM_LITIGIOUS).
# This mirrors the original Loughran-McDonald dictionary where words can carry multiple
# connotations. Each set is used independently for its respective signal dimension.
LM_NEGATIVE = {
    'abandon', 'abandoned', 'abandoning', 'abandonment', 'abrupt', 'absence',
    'adverse', 'adversely', 'against', 'aggravate', 'allegation', 'allegations',
    'alleged', 'annulled', 'breach', 'breached', 'burden', 'burdensome',
    'catastrophe', 'catastrophic', 'cease', 'ceased', 'claim', 'claims',
    'closing', 'complaint', 'complaints', 'concern', 'concerned', 'concerns',
    'condemn', 'convicted', 'conviction', 'crisis', 'critical', 'critically',
    'damage', 'damages', 'damaging', 'danger', 'dangerous', 'decline',
    'declined', 'declining', 'default', 'defaults', 'defect', 'defective',
    'deficiency', 'deficiencies', 'deficit', 'delinquency', 'delinquent',
    'denial', 'denied', 'deplete', 'depleted', 'depreciation', 'deteriorate',
    'deteriorated', 'deteriorating', 'deterioration', 'detrimental', 'diminish',
    'disadvantage', 'discontinue', 'discontinued', 'dispute', 'disputes',
    'disrupt', 'disruption', 'disruptions', 'dissolution', 'distress',
    'doubt', 'doubtful', 'downgrade', 'downturn', 'erode', 'eroded',
    'erosion', 'error', 'errors', 'exacerbate', 'excessive', 'fail',
    'failed', 'failing', 'failure', 'failures', 'fault', 'felony',
    'fine', 'fined', 'fines', 'forbid', 'force', 'forced', 'foreclose',
    'foreclosure', 'forfeit', 'forfeiture', 'fraud', 'fraudulent',
    'grievance', 'halt', 'halted', 'hamper', 'hampered', 'harm', 'harmed',
    'harmful', 'hardship', 'hinder', 'hindered', 'idle', 'idled',
    'impair', 'impaired', 'impairment', 'impede', 'impossible', 'inability',
    'inadequate', 'inadvertent', 'inconsistency', 'indictment', 'infringe',
    'infringement', 'injunction', 'insolvent', 'insufficient', 'investigate',
    'investigation', 'irregularity', 'jeopardize', 'lapse', 'liability',
    'liquidate', 'liquidation', 'litigation', 'loss', 'losses', 'lost',
    'malfeasance', 'malpractice', 'misappropriation', 'misconduct',
    'misrepresentation', 'misstate', 'misstated', 'misstatement', 'negative',
    'negatively', 'neglect', 'negligence', 'negligent', 'nonpayment',
    'obsolete', 'obsolescence', 'omission', 'onerous', 'penalty', 'penalties',
    'peril', 'plaintiffs', 'plea', 'plead', 'prohibit', 'prohibited',
    'prosecution', 'punitive', 'reassess', 'recall', 'recalled', 'recession',
    'reckless', 'redress', 'reject', 'rejected', 'relinquish', 'remediate',
    'remediation', 'repossess', 'restructure', 'restructuring', 'restate',
    'restated', 'restatement', 'restrict', 'restricted', 'revocation',
    'revoke', 'revoked', 'risk', 'risky', 'sanction', 'sanctions',
    'scandal', 'scrutiny', 'seizure', 'setback', 'shortfall', 'shutdown',
    'slowdown', 'subpoena', 'sue', 'sued', 'suffer', 'suffered', 'suspend',
    'suspended', 'suspension', 'terminate', 'terminated', 'termination',
    'threat', 'threaten', 'threatened', 'turmoil', 'unable', 'unauthorized',
    'uncertain', 'uncertainty', 'unfavorable', 'unforeseen', 'unlawful',
    'unpaid', 'unprecedented', 'unreliable', 'unstable', 'violate',
    'violated', 'violation', 'violations', 'volatile', 'volatility',
    'vulnerability', 'vulnerable', 'warn', 'warned', 'warning', 'weakness',
    'worsen', 'worsened', 'worsening', 'worthless', 'writedown', 'writeoff',
}

LM_POSITIVE = {
    'able', 'abundance', 'accomplish', 'accomplished', 'accomplishment',
    'achieve', 'achieved', 'achievement', 'achievements', 'advance',
    'advanced', 'advancement', 'advantage', 'advantages', 'assure',
    'attain', 'attained', 'attractive', 'beneficial', 'benefit',
    'benefited', 'benefits', 'best', 'better', 'boost', 'boosted',
    'breakthrough', 'collaborate', 'collaboration', 'commitment',
    'competence', 'competitive', 'compliment', 'confident', 'constructive',
    'creative', 'creativity', 'deliver', 'delivered', 'dependable',
    'desirable', 'diligent', 'distinction', 'distinctive', 'efficient',
    'efficiency', 'empower', 'enable', 'enabled', 'enabling', 'enhance',
    'enhanced', 'enhancement', 'enjoy', 'enjoyed', 'enthusiasm', 'excellence',
    'excellent', 'exceptional', 'exclusive', 'exemplary', 'expand',
    'expanded', 'expansion', 'favorable', 'favorably', 'gain', 'gained',
    'gains', 'great', 'greater', 'greatest', 'grew', 'grow', 'growing',
    'growth', 'highest', 'honor', 'improve', 'improved', 'improvement',
    'improvements', 'improving', 'increase', 'increased', 'increases',
    'increasing', 'innovative', 'innovation', 'integrity', 'leader',
    'leadership', 'leading', 'milestone', 'momentum', 'opportunity',
    'opportunities', 'optimal', 'optimistic', 'outpace', 'outperform',
    'outperformed', 'outstanding', 'overcome', 'pioneer', 'positive',
    'positively', 'proactive', 'productive', 'productivity', 'proficiency',
    'profit', 'profitable', 'profitability', 'progress', 'progressed',
    'promising', 'prosper', 'prosperity', 'record', 'recovery', 'reliable',
    'reward', 'rewarded', 'rewarding', 'robust', 'satisfaction', 'satisfied',
    'solid', 'solved', 'stability', 'stable', 'strength', 'strengthen',
    'strengthened', 'strong', 'stronger', 'strongest', 'succeed',
    'succeeded', 'succeeding', 'success', 'successes', 'successful',
    'successfully', 'superior', 'surpass', 'surpassed', 'sustain',
    'sustainable', 'thrive', 'thriving', 'top', 'transform', 'tremendous',
    'trustworthy', 'unmatched', 'upturn', 'valuable',
    'value', 'versatile', 'win', 'winning', 'won',
}

LM_UNCERTAINTY = {
    'almost', 'ambiguity', 'ambiguous', 'anticipate', 'anticipated',
    'apparent', 'apparently', 'appear', 'appeared', 'appears', 'approximate',
    'approximately', 'assumption', 'assumptions', 'believe', 'believed',
    'believes', 'conceivable', 'conditional', 'confuse', 'contingency',
    'contingent', 'could', 'crossroad', 'depend', 'depended', 'depending',
    'depends', 'destabilize', 'deviate', 'doubt', 'doubtful', 'doubts',
    'dubious', 'equivocal', 'estimate', 'estimated', 'estimates',
    'eventually', 'expect', 'expectation', 'expectations', 'expose',
    'exposed', 'exposure', 'fluctuate', 'fluctuated', 'fluctuation',
    'fluctuations', 'forecast', 'foreseeable', 'hypothetical',
    'imprecise', 'imprecision', 'improbable', 'inadvertently', 'incompletely',
    'indefinite', 'indefinitely', 'indeterminate', 'indicate', 'indication',
    'inherent', 'inherently', 'intend', 'likelihood', 'likely',
    'may', 'maybe', 'might', 'nearly', 'nonassessable', 'occasionally',
    'pending', 'perceive', 'perhaps', 'possibility', 'possible', 'possibly',
    'predict', 'predicted', 'prediction', 'predicting', 'predictability',
    'preliminary', 'presumably', 'presume', 'probable', 'probably',
    'project', 'projected', 'projection', 'projections', 'prospect',
    'random', 'randomly', 'reassess', 'reconsider', 'reexamine',
    'reinterpret', 'revise', 'revised', 'revision', 'roughly',
    'seem', 'seemed', 'seemingly', 'seems', 'somewhat', 'somewhere',
    'speculate', 'speculation', 'suggest', 'suggests', 'suppose',
    'susceptible', 'tend', 'tentative', 'tentatively', 'turbulence',
    'uncertain', 'uncertainly', 'uncertainties', 'uncertainty', 'unclear',
    'unconfirmed', 'undecided', 'undefined', 'undetermined', 'unforeseeable',
    'unforeseen', 'unknowingly', 'unknown', 'unlikely', 'unobservable',
    'unplanned', 'unpredictable', 'unpredictability', 'unproven',
    'unquantifiable', 'unsettled', 'unspecified', 'untested', 'unusual',
    'vagaries', 'vague', 'vaguely', 'variability', 'variable', 'variation',
    'vary', 'varying', 'volatile', 'volatility',
}

LM_LITIGIOUS = {
    'abovementioned', 'acquit', 'acquitted', 'adjudicate', 'adjudication',
    'aforesaid', 'allegation', 'allegations', 'allege', 'alleged',
    'appeal', 'appealed', 'arbitrate', 'arbitration', 'attorney',
    'attorneys', 'claimant', 'claimants', 'complain', 'complainant',
    'complaint', 'complaints', 'consent', 'counterclaim', 'court',
    'courts', 'damages', 'decree', 'defendant', 'defendants', 'depose',
    'deposition', 'discovery', 'dismiss', 'dismissed', 'docket', 'enforce',
    'enforceable', 'enforcement', 'enjoin', 'enjoined', 'estoppel',
    'evidentiary', 'filed', 'filing', 'guilty', 'hearing', 'hearings',
    'herein', 'hereto', 'indemnify', 'indemnification', 'infringe',
    'infringement', 'injunction', 'judge', 'judgment', 'judicial',
    'jurisdiction', 'jury', 'lawsuit', 'lawsuits', 'lawyer', 'lawyers',
    'legal', 'legally', 'litigate', 'litigated', 'litigation', 'motion',
    'motions', 'negligence', 'notary', 'oath', 'objection', 'objections',
    'order', 'ordinance', 'pending', 'petition', 'plaintiff', 'plaintiffs',
    'plea', 'plead', 'pleading', 'precedent', 'prejudice', 'proceedings',
    'prosecute', 'prosecution', 'punitive', 'ruling', 'sentence',
    'settlement', 'settlements', 'statute', 'statutes', 'statutory',
    'stipulate', 'stipulation', 'subpoena', 'sue', 'sued', 'suit',
    'suits', 'summon', 'summons', 'testify', 'testimony', 'tort',
    'tribunal', 'verdict', 'violate', 'violated', 'violation', 'violations',
    'witness', 'witnesses',
}


# ============================================================================
# SIGNAL FUNCTIONS
# Note: Signals are grouped by dependency, not by number.
# Order: 1 (MD&A) → 3 (Risk) → 6 (Legal) → 8 (Timeliness) →
#        2 (Sentiment) → 4 (Positive Sim) → 5 (Sentiment Delta) →
#        7 (Uncertainty) → 9 (Complexity) → 10 (Insider)
# ============================================================================

# --- Signal 1: MD&A Similarity ---
# ============================================================================
# SIGNAL 1: MD&A SIMILARITY (Weight: 20%)
# ============================================================================
# Source: "Lazy Prices" — Cohen, Malloy, Nguyen (JF 2020)
# Finding: Shorting changers / buying non-changers → 188bps/month alpha
# Changes concentrated in MD&A section — strongest documented textual signal
# ============================================================================

def compute_mda_similarity(current_text: str, prior_text: str) -> Dict[str, Any]:
    """
    Compute MD&A similarity between consecutive filings.

    Per Lazy Prices: high similarity = "non-changer" = positive signal.
    Low similarity = "changer" = negative signal (upcoming bad news/earnings).

    Returns score 0-100 where higher = more stable (positive).
    """
    result = {
        'signal': 'mda_similarity',
        'has_data': False,
        'score': 50,  # neutral default
        'cosine': None,
        'jaccard': None,
        'size_ratio': None,
        'has_prior': False,
        'mda_found_current': False,
        'mda_found_prior': False,
        'interpretation': '',
    }

    if not prior_text:
        result['interpretation'] = 'No prior filing for comparison'
        return result

    # Extract MD&A sections
    current_mda = _extract_section(current_text, MDA_HEADER, 60000)
    prior_mda = _extract_section(prior_text, MDA_HEADER, 60000)

    result['mda_found_current'] = current_mda is not None
    result['mda_found_prior'] = prior_mda is not None
    result['has_prior'] = True

    if not current_mda or not prior_mda:
        result['interpretation'] = 'MD&A section not found in one or both filings'
        return result

    # Tokenize
    tokens_cur = _tokenize(current_mda)
    tokens_pri = _tokenize(prior_mda)

    if len(tokens_cur) < 50 or len(tokens_pri) < 50:
        result['interpretation'] = 'MD&A sections too short for reliable comparison'
        return result

    # Compute three similarity metrics (as per Lazy Prices methodology)
    cosine = _cosine_similarity(tokens_cur, tokens_pri)
    jaccard = _jaccard_similarity(tokens_cur, tokens_pri)
    size_ratio = _word_count_ratio(tokens_cur, tokens_pri)

    result['cosine'] = round(cosine, 4)
    result['jaccard'] = round(jaccard, 4)
    result['size_ratio'] = round(size_ratio, 4)
    result['current_word_count'] = len(tokens_cur)
    result['prior_word_count'] = len(tokens_pri)

    # Combined similarity: weighted average of cosine (60%) + jaccard (30%) + size (10%)
    # Cosine is most predictive per literature, Jaccard captures vocabulary shifts
    combined = cosine * 0.60 + jaccard * 0.30 + size_ratio * 0.10

    # Map combined similarity (typically 0.7-0.99) to score 0-100
    # Based on Lazy Prices quintile breakpoints:
    #   Top quintile (non-changers): similarity > 0.95 → score 80-100
    #   Q2: 0.90-0.95 → 65-80
    #   Q3: 0.85-0.90 → 50-65
    #   Q4: 0.80-0.85 → 35-50
    #   Bottom quintile (changers): < 0.80 → 0-35
    if combined >= 0.95:
        score = 80 + (combined - 0.95) / 0.05 * 20  # 80-100
    elif combined >= 0.90:
        score = 65 + (combined - 0.90) / 0.05 * 15  # 65-80
    elif combined >= 0.85:
        score = 50 + (combined - 0.85) / 0.05 * 15  # 50-65
    elif combined >= 0.80:
        score = 35 + (combined - 0.80) / 0.05 * 15  # 35-50
    elif combined >= 0.70:
        score = 15 + (combined - 0.70) / 0.10 * 20  # 15-35
    else:
        score = max(0, combined / 0.70 * 15)          # 0-15

    score = max(0, min(100, round(score)))
    result['has_data'] = True
    result['score'] = score
    result['combined_similarity'] = round(combined, 4)

    # Interpretation
    if score >= 75:
        result['interpretation'] = 'Highly stable MD&A — "non-changer" (Lazy Prices: positive signal)'
    elif score >= 60:
        result['interpretation'] = 'Moderately stable MD&A — minor language evolution'
    elif score >= 45:
        result['interpretation'] = 'Notable MD&A changes — warrant attention'
    elif score >= 30:
        result['interpretation'] = 'Significant MD&A rewrite — "changer" (Lazy Prices: negative signal)'
    else:
        result['interpretation'] = 'Major MD&A overhaul — strong "changer" signal (historically bearish)'

    return result


# ============================================================================
# SIGNAL 3: RISK FACTOR CHANGES (Weight: 13%)
# ============================================================================
# Source: "Lazy Prices" — risk factor + litigation language "especially informative"
# Also: Kirtac & Germano (2024) — risk factor tone predicts index returns
# ============================================================================

def compute_risk_factor_signal(current_text: str, prior_text: str) -> Dict[str, Any]:
    """
    Analyze risk factor section changes between consecutive filings.

    Per research: new risk factors predict negative outcomes.
    Risk factor section changes are "especially informative for future returns."

    Returns score 0-100 where higher = fewer/stable risk factors (positive).
    """
    result = {
        'signal': 'risk_factor_changes',
        'has_data': False,
        'score': 50,
        'cosine': None,
        'jaccard': None,
        'risks_added': 0,
        'risks_removed': 0,
        'current_risk_count': 0,
        'prior_risk_count': 0,
        'has_prior': False,
        'interpretation': '',
    }

    if not prior_text:
        result['interpretation'] = 'No prior filing for comparison'
        return result

    result['has_prior'] = True

    # Extract risk factor sections
    current_rf = _extract_section(current_text, RISK_FACTOR_HEADER, 60000)
    prior_rf = _extract_section(prior_text, RISK_FACTOR_HEADER, 60000)

    if not current_rf or not prior_rf:
        result['interpretation'] = 'Risk factors section not found in one or both filings'
        return result

    # Tokenize and compute similarity
    tokens_cur = _tokenize(current_rf)
    tokens_pri = _tokenize(prior_rf)

    if len(tokens_cur) < 30 or len(tokens_pri) < 30:
        result['interpretation'] = 'Risk factor sections too short for comparison'
        return result

    cosine = _cosine_similarity(tokens_cur, tokens_pri)
    jaccard = _jaccard_similarity(tokens_cur, tokens_pri)

    result['cosine'] = round(cosine, 4)
    result['jaccard'] = round(jaccard, 4)

    # Count distinct risk factor headings
    def _count_risks(section):
        titles = re.findall(
            r'(?:^|\n)\s*([A-Z][A-Za-z ,\'\-]{10,80})(?:\.|—|\n)',
            section
        )
        return [t.strip() for t in titles if len(t.strip()) > 15]

    cur_risks = _count_risks(current_rf)
    pri_risks = _count_risks(prior_rf)

    result['current_risk_count'] = len(cur_risks)
    result['prior_risk_count'] = len(pri_risks)

    # Compare risk titles (fuzzy match)
    cur_set = set(r.lower()[:50] for r in cur_risks)
    pri_set = set(r.lower()[:50] for r in pri_risks)
    risks_added = len(cur_set - pri_set)
    risks_removed = len(pri_set - cur_set)
    result['risks_added'] = risks_added
    result['risks_removed'] = risks_removed

    # Score components:
    # 1. Section similarity (60% weight) — per Lazy Prices
    sim_component = cosine * 0.6 + jaccard * 0.4

    # 2. Net risk factor change (40% weight)
    net_change = risks_added - risks_removed
    # Normalize: 0 change = neutral, each added risk = -5pts, each removed = +3pts
    risk_change_score = 50 - (net_change * 5)
    risk_change_score = max(0, min(100, risk_change_score))

    # Map similarity to 0-100 (same as MD&A but risk sections change more)
    if sim_component >= 0.90:
        sim_score = 75 + (sim_component - 0.90) / 0.10 * 25
    elif sim_component >= 0.80:
        sim_score = 55 + (sim_component - 0.80) / 0.10 * 20
    elif sim_component >= 0.70:
        sim_score = 35 + (sim_component - 0.70) / 0.10 * 20
    else:
        sim_score = max(0, sim_component / 0.70 * 35)

    # Combined: similarity (60%) + risk count change (40%)
    score = round(sim_score * 0.60 + risk_change_score * 0.40)
    score = max(0, min(100, score))
    result['has_data'] = True
    result['score'] = score

    # Interpretation
    if score >= 70:
        result['interpretation'] = f'Stable risk profile — {risks_added} new, {risks_removed} removed'
    elif score >= 50:
        result['interpretation'] = f'Moderate risk changes — {risks_added} new risks added'
    elif score >= 30:
        result['interpretation'] = f'Significant risk disclosure changes — {risks_added} new risks (bearish signal)'
    else:
        result['interpretation'] = f'Major risk factor overhaul — {risks_added} new risks (strong bearish signal)'

    return result


# ============================================================================
# SIGNAL 6: LEGAL PROCEEDINGS CHANGES (Weight: 8%)
# ============================================================================
# Source: Offutt & Xie (2025) — LLM-scored legal risk shifts → alpha
#         unexplained by Fama-French 5-factor + momentum
# Also: Lazy Prices — litigation language changes "especially informative"
# ============================================================================

def compute_legal_proceedings_signal(current_text: str, prior_text: str) -> Dict[str, Any]:
    """
    Track changes in Legal Proceedings section (Item 1 for 10-Q, Item 3 for 10-K).

    Expanding legal proceedings = negative signal.
    Stable/shrinking = positive signal.

    Returns score 0-100 where higher = stable/shrinking legal exposure.
    """
    result = {
        'signal': 'legal_proceedings',
        'has_data': False,
        'score': 50,
        'has_prior': False,
        'legal_found_current': False,
        'legal_found_prior': False,
        'current_word_count': 0,
        'prior_word_count': 0,
        'size_change_pct': 0,
        'cosine': None,
        'new_legal_entities': [],
        'new_dollar_amounts': [],
        'interpretation': '',
    }

    # Extract legal proceedings sections
    current_legal = _extract_section(current_text, LEGAL_PROCEEDINGS_HEADER, 30000)

    if not current_legal:
        result['interpretation'] = 'No legal proceedings section found'
        result['has_data'] = True
        result['score'] = 60  # absence of legal section is mildly positive
        return result

    result['legal_found_current'] = True
    cur_tokens = _tokenize(current_legal)
    result['current_word_count'] = len(cur_tokens)

    if not prior_text:
        result['interpretation'] = 'No prior filing for legal comparison'
        return result

    result['has_prior'] = True
    prior_legal = _extract_section(prior_text, LEGAL_PROCEEDINGS_HEADER, 30000)

    if not prior_legal:
        # Legal section exists now but didn't before — new litigation
        result['legal_found_prior'] = False
        result['has_data'] = True
        result['score'] = 25
        result['interpretation'] = 'NEW legal proceedings section (was absent in prior filing — bearish)'
        return result

    result['legal_found_prior'] = True
    pri_tokens = _tokenize(prior_legal)
    result['prior_word_count'] = len(pri_tokens)

    # Size change
    if len(pri_tokens) > 0:
        size_change_pct = round((len(cur_tokens) - len(pri_tokens)) / len(pri_tokens) * 100, 1)
    else:
        size_change_pct = 0
    result['size_change_pct'] = size_change_pct

    # Similarity
    if len(cur_tokens) >= 20 and len(pri_tokens) >= 20:
        cosine = _cosine_similarity(cur_tokens, pri_tokens)
        result['cosine'] = round(cosine, 4)
    else:
        cosine = 0.8  # assume stable for very short sections

    # Detect new legal entity names (capitalized multi-word phrases, min 8 chars, in current not in prior)
    _entity_re = re.compile(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}')
    cur_entities = set(e for e in _entity_re.findall(current_legal) if len(e) >= 8)
    pri_entities = set(e for e in _entity_re.findall(prior_legal) if len(e) >= 8)
    new_entities = list(cur_entities - pri_entities)[:5]
    result['new_legal_entities'] = new_entities

    # Detect new dollar amounts (matches $10.5M, $X million, USD X million)
    _dollar_re = re.compile(
        r'(?:\$[\d,.]+\s*[MBKmb]?(?:illion|housand)?'
        r'|(?:USD|U\.S\.\s*dollars?)\s+[\d,.]+\s*(?:million|billion|thousand)?)',
        re.IGNORECASE
    )
    cur_amounts = set(_dollar_re.findall(current_legal))
    pri_amounts = set(_dollar_re.findall(prior_legal))
    new_amounts = list(cur_amounts - pri_amounts)[:5]
    result['new_dollar_amounts'] = new_amounts

    # Score:
    # 1. Similarity component (40%)
    if cosine >= 0.90:
        sim_score = 75 + (cosine - 0.90) / 0.10 * 25
    elif cosine >= 0.75:
        sim_score = 45 + (cosine - 0.75) / 0.15 * 30
    else:
        sim_score = max(0, cosine / 0.75 * 45)

    # 2. Size change component (30%) — growing legal section is bad
    if size_change_pct <= -10:
        size_score = 80  # shrinking legal section = very positive
    elif size_change_pct <= 5:
        size_score = 60  # stable
    elif size_change_pct <= 25:
        size_score = 40  # moderate growth
    elif size_change_pct <= 50:
        size_score = 25  # significant growth
    else:
        size_score = 10  # major expansion

    # 3. New entities/amounts component (30%)
    novelty_penalty = len(new_entities) * 8 + len(new_amounts) * 10
    novelty_score = max(0, 70 - novelty_penalty)

    score = round(sim_score * 0.40 + size_score * 0.30 + novelty_score * 0.30)
    score = max(0, min(100, score))
    result['has_data'] = True
    result['score'] = score

    # Interpretation
    if score >= 70:
        result['interpretation'] = 'Stable legal proceedings — no significant new litigation'
    elif score >= 50:
        result['interpretation'] = f'Minor legal changes — {len(new_entities)} new entities mentioned'
    elif score >= 30:
        result['interpretation'] = f'Growing legal exposure — section expanded {size_change_pct:.0f}%, {len(new_amounts)} new dollar figures'
    else:
        result['interpretation'] = f'Major legal proceedings expansion — significant new litigation risk'

    return result


# ============================================================================
# SIGNAL 8: FILING TIMELINESS (Weight: 5%)
# ============================================================================
# Source: Duarte-Silva et al. (2013) — delays signal performance deterioration
#         Khalil et al. (2017) — bond markets react negatively to late filings
# ============================================================================

def compute_filing_timeliness_signal(filing_date: str, form_type: str,
                                      period_end: str = None) -> Dict[str, Any]:
    """
    Score filing timeliness.

    Early filers signal confidence. Late filers predict deterioration.
    Also: first-time filers (IPO-adjacent) get neutral treatment.

    Returns score 0-100 where higher = timelier filing.
    """
    result = {
        'signal': 'filing_timeliness',
        'has_data': False,
        'score': 50,  # neutral default
        'timing': 'unknown',
        'days_after_period': None,
        'deadline_days': None,
        'interpretation': '',
    }

    if not filing_date:
        result['interpretation'] = 'No filing date available'
        return result

    if not period_end:
        # Try to infer period end from filing date and form type
        try:
            filed = datetime.strptime(filing_date[:10], "%Y-%m-%d")
            if form_type in ("10-K", "10-K/A"):
                # Most 10-Ks filed ~60-90 days after fiscal year end
                # Estimate: period_end ≈ filing_date - 75 days
                period = filed - timedelta(days=75)
                period_end = period.strftime("%Y-%m-%d")
            elif form_type in ("10-Q", "10-Q/A"):
                period = filed - timedelta(days=40)
                period_end = period.strftime("%Y-%m-%d")
            else:
                result['interpretation'] = f'Cannot assess timing for {form_type}'
                return result
        except ValueError:
            result['interpretation'] = 'Invalid filing date'
            return result

    try:
        filed = datetime.strptime(filing_date[:10], "%Y-%m-%d")
        period = datetime.strptime(period_end[:10], "%Y-%m-%d")
        days = (filed - period).days
    except ValueError:
        result['interpretation'] = 'Invalid date format'
        return result

    result['days_after_period'] = days

    # Determine deadline and score based on form type
    if form_type in ("10-K", "10-K/A"):
        # SEC deadlines: Large accelerated=60d, Accelerated=60d, Non-accel=90d
        # We use 75 as midpoint for small-cap universe
        deadline = 75
        result['deadline_days'] = deadline
        result['has_data'] = True

        if days <= 45:
            result['timing'] = 'very_early'
            result['score'] = 85
        elif days <= 60:
            result['timing'] = 'early'
            result['score'] = 75
        elif days <= 75:
            result['timing'] = 'on_time'
            result['score'] = 60
        elif days <= 90:
            result['timing'] = 'late'
            result['score'] = 35
        elif days <= 120:
            result['timing'] = 'very_late'
            result['score'] = 15
        else:
            result['timing'] = 'severely_late'
            result['score'] = 5

    elif form_type in ("10-Q", "10-Q/A"):
        deadline = 42
        result['deadline_days'] = deadline
        result['has_data'] = True

        if days <= 30:
            result['timing'] = 'very_early'
            result['score'] = 85
        elif days <= 40:
            result['timing'] = 'early'
            result['score'] = 70
        elif days <= 45:
            result['timing'] = 'on_time'
            result['score'] = 55
        elif days <= 60:
            result['timing'] = 'late'
            result['score'] = 30
        else:
            result['timing'] = 'very_late'
            result['score'] = 10

    else:
        result['interpretation'] = f'Timing assessment not applicable for {form_type}'
        return result

    # Interpretation
    timing_labels = {
        'very_early': f'Filed {days}d after period end — very early (confidence signal)',
        'early': f'Filed {days}d after period end — early filer',
        'on_time': f'Filed {days}d after period end — on schedule',
        'late': f'Filed {days}d after period end — LATE (research: predicts deterioration)',
        'very_late': f'Filed {days}d after period end — VERY LATE (strong negative signal)',
        'severely_late': f'Filed {days}d after period end — SEVERELY LATE (possible SEC extension)',
    }
    result['interpretation'] = timing_labels.get(result['timing'], f'Filed {days}d after period end')

    return result


# ============================================================================
# SIGNAL 2: LOUGHRAN-MCDONALD SENTIMENT (Weight: 15%)
# ============================================================================
# Source: Loughran & McDonald (JF 2011) — finance-specific word lists
# Finding: Management Sentiment Index has R² 9.75% in-sample, 8.38% OOS
#          Outperforms general dictionaries (Harvard GI) for financial texts
#          36 of 78 reviewed papers use LM dictionary
# ============================================================================

def compute_lm_sentiment(text: str) -> Dict[str, Any]:
    """
    Score sentiment using Loughran-McDonald finance-specific word lists.

    Higher positive-to-negative ratio → more optimistic → higher score.
    Research: optimistic tone correlates with higher returns.

    Returns score 0-100 where higher = more positive sentiment.
    """
    result = {
        'signal': 'lm_sentiment',
        'has_data': False,
        'score': 50,
        'positive_count': 0,
        'negative_count': 0,
        'uncertainty_count': 0,
        'litigious_count': 0,
        'total_words': 0,
        'net_sentiment': 0.0,
        'positive_pct': 0.0,
        'negative_pct': 0.0,
        'uncertainty_pct': 0.0,
        'interpretation': '',
    }

    if not text or len(text) < 500:
        result['interpretation'] = 'Insufficient text for sentiment analysis'
        return result

    # Tokenize entire filing
    words = _WORD_RE.findall(text.lower())
    total = len(words)
    if total < 100:
        result['interpretation'] = 'Insufficient tokens for sentiment analysis'
        return result

    result['total_words'] = total

    # Count category words — single pass over all words
    pos_count = 0
    neg_count = 0
    unc_count = 0
    lit_count = 0
    for w in words:
        if w in LM_POSITIVE:
            pos_count += 1
        if w in LM_NEGATIVE:
            neg_count += 1
        if w in LM_UNCERTAINTY:
            unc_count += 1
        if w in LM_LITIGIOUS:
            lit_count += 1

    result['positive_count'] = pos_count
    result['negative_count'] = neg_count
    result['uncertainty_count'] = unc_count
    result['litigious_count'] = lit_count

    pos_pct = pos_count / total * 100
    neg_pct = neg_count / total * 100
    unc_pct = unc_count / total * 100
    result['positive_pct'] = round(pos_pct, 3)
    result['negative_pct'] = round(neg_pct, 3)
    result['uncertainty_pct'] = round(unc_pct, 3)

    # Net sentiment: positive% - negative% (typical range: -2% to +1%)
    net = pos_pct - neg_pct
    result['net_sentiment'] = round(net, 3)

    # Map net sentiment to score 0-100
    # Based on LM paper distributions across 10-K filings:
    #   Mean negative%: ~1.4%, Mean positive%: ~0.75%
    #   Most filings have net sentiment between -1.5% and +0.5%
    #   Scores: net > 0 = bullish, net < -1.0 = bearish
    if net >= 0.5:
        score = 85 + min(15, (net - 0.5) * 15)      # 85-100
    elif net >= 0.0:
        score = 70 + (net / 0.5) * 15                 # 70-85
    elif net >= -0.5:
        score = 55 + (net + 0.5) / 0.5 * 15           # 55-70
    elif net >= -1.0:
        score = 40 + (net + 1.0) / 0.5 * 15           # 40-55
    elif net >= -1.5:
        score = 25 + (net + 1.5) / 0.5 * 15           # 25-40
    else:
        score = max(5, 25 + (net + 1.5) * 10)         # 5-25

    score = max(0, min(100, round(score)))
    result['has_data'] = True
    result['score'] = score

    if score >= 70:
        result['interpretation'] = f'Positive filing sentiment — net {net:+.2f}% (bullish per LM research)'
    elif score >= 55:
        result['interpretation'] = f'Moderately positive sentiment — net {net:+.2f}%'
    elif score >= 40:
        result['interpretation'] = f'Neutral-to-negative sentiment — net {net:+.2f}%'
    elif score >= 25:
        result['interpretation'] = f'Negative sentiment — net {net:+.2f}% (bearish per LM research)'
    else:
        result['interpretation'] = f'Strongly negative sentiment — net {net:+.2f}% (historically bearish)'

    return result


# ============================================================================
# SIGNAL 4: POSITIVE SIMILARITY (Weight: 12%)
# ============================================================================
# Source: "Positive Similarity of Company Filings & Cross-Section of Returns"
# Finding: Low positive similarity stocks outperform high positive similarity
#          Uncorrelated to common factors (value, size, momentum, profitability)
#          1-month holding period, economically and statistically significant
# Key insight: changing positive language = new genuine positive developments
# ============================================================================

def compute_positive_similarity(current_text: str, prior_text: str) -> Dict[str, Any]:
    """
    Compare similarity of POSITIVE (optimistic) language between consecutive filings.

    Per research: LOW positive similarity → stocks OUTPERFORM.
    This is INVERSE to MD&A similarity (where high similarity = good).
    Interpretation: companies changing their positive language have new developments.

    Returns score 0-100 where higher = lower positive similarity (bullish).
    """
    result = {
        'signal': 'positive_similarity',
        'has_data': False,
        'score': 50,
        'cosine': None,
        'jaccard': None,
        'current_positive_count': 0,
        'prior_positive_count': 0,
        'has_prior': False,
        'interpretation': '',
    }

    if not prior_text:
        result['interpretation'] = 'No prior filing for positive similarity comparison'
        return result

    result['has_prior'] = True

    # Extract MD&A or full filing and filter to positive word contexts
    current_mda = _extract_section(current_text, MDA_HEADER, 60000) or current_text[:60000]
    prior_mda = _extract_section(prior_text, MDA_HEADER, 60000) or prior_text[:60000]

    # Get all words, then filter to positive words and their 3-word contexts
    def _positive_context_tokens(text):
        """Extract positive words and their surrounding context. Returns (context_tokens, positive_count)."""
        words = _WORD_RE.findall(text.lower())
        positive_tokens = []
        positive_count = 0
        for i, w in enumerate(words):
            if w in LM_POSITIVE:
                positive_count += 1
                # Include the positive word and 2 words on each side (context)
                start = max(0, i - 2)
                end = min(len(words), i + 3)
                positive_tokens.extend(words[start:end])
        return positive_tokens, positive_count

    cur_pos, cur_pos_count = _positive_context_tokens(current_mda)
    pri_pos, pri_pos_count = _positive_context_tokens(prior_mda)

    result['current_positive_count'] = cur_pos_count
    result['prior_positive_count'] = pri_pos_count

    if len(cur_pos) < 20 or len(pri_pos) < 20:
        result['interpretation'] = 'Insufficient positive language for comparison'
        return result

    # Compute similarity of positive language contexts
    cosine = _cosine_similarity(cur_pos, pri_pos)
    jaccard = _jaccard_similarity(cur_pos, pri_pos)

    result['cosine'] = round(cosine, 4)
    result['jaccard'] = round(jaccard, 4)

    combined = cosine * 0.6 + jaccard * 0.4

    # INVERTED scoring: LOW similarity = HIGH score (bullish per research)
    # Typical positive similarity range: 0.60-0.95
    if combined <= 0.60:
        score = 85 + (0.60 - combined) * 25         # 85-100 (very different = very bullish)
    elif combined <= 0.70:
        score = 70 + (0.70 - combined) / 0.10 * 15  # 70-85
    elif combined <= 0.80:
        score = 55 + (0.80 - combined) / 0.10 * 15  # 55-70
    elif combined <= 0.90:
        score = 40 + (0.90 - combined) / 0.10 * 15  # 40-55
    else:
        score = max(15, 40 - (combined - 0.90) / 0.10 * 25)  # 15-40 (very similar = muted signal)

    # Score is clamped to [0, 100] below
    score = max(0, min(100, round(score)))
    result['has_data'] = True
    result['score'] = score
    result['combined_similarity'] = round(combined, 4)

    if score >= 70:
        result['interpretation'] = f'Low positive similarity ({combined:.2f}) — new positive developments (bullish)'
    elif score >= 55:
        result['interpretation'] = f'Moderate positive language change ({combined:.2f})'
    elif score >= 40:
        result['interpretation'] = f'Stable positive language ({combined:.2f}) — neutral signal'
    else:
        result['interpretation'] = f'High positive similarity ({combined:.2f}) — boilerplate repetition (muted signal)'

    return result


# ============================================================================
# SIGNAL 5: SENTIMENT DELTA (Weight: 10%)
# ============================================================================
# Source: Brown & Tucker (2011) — firms modify MD&A more with economic changes
#         Bochkay (2014) — text-enhanced models more accurate than quantitative alone
# Finding: Improving sentiment period-over-period → positive future returns
# ============================================================================

def compute_sentiment_delta(current_text: str, prior_text: str) -> Dict[str, Any]:
    """
    Compute change in LM sentiment between consecutive filings.

    Improving sentiment = bullish, deteriorating = bearish.
    More informative than absolute sentiment levels.

    Returns score 0-100 where higher = improving sentiment.
    """
    result = {
        'signal': 'sentiment_delta',
        'has_data': False,
        'score': 50,
        'current_net_sentiment': None,
        'prior_net_sentiment': None,
        'delta': None,
        'has_prior': False,
        'interpretation': '',
    }

    if not prior_text:
        result['interpretation'] = 'No prior filing for sentiment comparison'
        return result

    result['has_prior'] = True

    # Compute LM sentiment for both periods
    def _quick_net_sentiment(text):
        words = re.findall(r'\b[a-z]{2,}\b', text.lower())
        total = len(words)
        if total < 100:
            return None
        pos = sum(1 for w in words if w in LM_POSITIVE)
        neg = sum(1 for w in words if w in LM_NEGATIVE)
        return (pos - neg) / total * 100

    # Use MD&A sections if available, else full text
    cur_section = _extract_section(current_text, MDA_HEADER, 60000) or current_text[:60000]
    pri_section = _extract_section(prior_text, MDA_HEADER, 60000) or prior_text[:60000]

    cur_sent = _quick_net_sentiment(cur_section)
    pri_sent = _quick_net_sentiment(pri_section)

    if cur_sent is None or pri_sent is None:
        result['interpretation'] = 'Insufficient text for sentiment delta'
        return result

    result['current_net_sentiment'] = round(cur_sent, 3)
    result['prior_net_sentiment'] = round(pri_sent, 3)

    delta = cur_sent - pri_sent
    result['delta'] = round(delta, 3)

    # Map delta to score 0-100
    # Typical delta range: -0.5% to +0.5%
    if delta >= 0.4:
        score = 85 + min(15, (delta - 0.4) * 20)
    elif delta >= 0.2:
        score = 70 + (delta - 0.2) / 0.2 * 15
    elif delta >= 0.0:
        score = 55 + (delta / 0.2) * 15
    elif delta >= -0.2:
        score = 40 + (delta + 0.2) / 0.2 * 15
    elif delta >= -0.4:
        score = 25 + (delta + 0.4) / 0.2 * 15
    else:
        score = max(5, 25 + (delta + 0.4) * 25)

    score = max(0, min(100, round(score)))
    result['has_data'] = True
    result['score'] = score

    if score >= 70:
        result['interpretation'] = f'Improving sentiment (Δ{delta:+.2f}%) — management tone brightening'
    elif score >= 55:
        result['interpretation'] = f'Slightly improving sentiment (Δ{delta:+.2f}%)'
    elif score >= 40:
        result['interpretation'] = f'Stable sentiment (Δ{delta:+.2f}%) — minimal change'
    elif score >= 25:
        result['interpretation'] = f'Deteriorating sentiment (Δ{delta:+.2f}%) — cautious language shift'
    else:
        result['interpretation'] = f'Sharply deteriorating sentiment (Δ{delta:+.2f}%) — significant tone change (bearish)'

    return result


# ============================================================================
# SIGNAL 7: UNCERTAINTY LANGUAGE (Weight: 7%)
# ============================================================================
# Source: Loughran & McDonald uncertainty word list
# Finding: High uncertainty language predicts IPO underpricing, stock volatility
#          Forward-looking statements with uncertainty terms less credible
# ============================================================================

def compute_uncertainty_signal(text: str) -> Dict[str, Any]:
    """
    Score filing based on density of uncertainty language.

    High uncertainty = less confident management = bearish.
    Low uncertainty = confident, clear messaging = bullish.

    Returns score 0-100 where higher = lower uncertainty (positive).
    """
    result = {
        'signal': 'uncertainty_language',
        'has_data': False,
        'score': 50,
        'uncertainty_count': 0,
        'uncertainty_pct': 0.0,
        'total_words': 0,
        'top_uncertainty_terms': [],
        'interpretation': '',
    }

    if not text or len(text) < 500:
        result['interpretation'] = 'Insufficient text for uncertainty analysis'
        return result

    # Focus on forward-looking sections: MD&A + risk factors
    mda = _extract_section(text, MDA_HEADER, 60000) or ''
    risk = _extract_section(text, RISK_FACTOR_HEADER, 40000) or ''
    focus_text = (mda + ' ' + risk) if (mda or risk) else text[:80000]

    words = re.findall(r'\b[a-z]{2,}\b', focus_text.lower())
    total = len(words)
    if total < 100:
        result['interpretation'] = 'Insufficient tokens for uncertainty analysis'
        return result

    result['total_words'] = total

    # Count uncertainty terms
    unc_words = [w for w in words if w in LM_UNCERTAINTY]
    unc_count = len(unc_words)
    unc_pct = unc_count / total * 100

    result['uncertainty_count'] = unc_count
    result['uncertainty_pct'] = round(unc_pct, 3)

    # Top uncertainty terms by frequency
    top_terms = Counter(unc_words).most_common(5)
    result['top_uncertainty_terms'] = [f'{term}({cnt})' for term, cnt in top_terms]

    # Map uncertainty percentage to INVERTED score (less uncertainty = higher score)
    # Typical uncertainty% in 10-K filings: 2-6%
    if unc_pct <= 1.5:
        score = 90                                       # Very confident
    elif unc_pct <= 2.5:
        score = 75 + (2.5 - unc_pct) / 1.0 * 15         # 75-90
    elif unc_pct <= 3.5:
        score = 55 + (3.5 - unc_pct) / 1.0 * 20         # 55-75
    elif unc_pct <= 4.5:
        score = 35 + (4.5 - unc_pct) / 1.0 * 20         # 35-55
    elif unc_pct <= 6.0:
        score = 15 + (6.0 - unc_pct) / 1.5 * 20         # 15-35
    else:
        score = max(5, 15 - (unc_pct - 6.0) * 5)        # 5-15

    score = max(0, min(100, round(score)))
    result['has_data'] = True
    result['score'] = score

    if score >= 70:
        result['interpretation'] = f'Low uncertainty ({unc_pct:.1f}%) — confident management language'
    elif score >= 50:
        result['interpretation'] = f'Moderate uncertainty ({unc_pct:.1f}%) — typical hedging'
    elif score >= 30:
        result['interpretation'] = f'High uncertainty ({unc_pct:.1f}%) — elevated hedging language (cautious)'
    else:
        result['interpretation'] = f'Very high uncertainty ({unc_pct:.1f}%) — excessive hedging (bearish signal)'

    return result


# ============================================================================
# SIGNAL 9: DOCUMENT COMPLEXITY (Weight: 5%)
# ============================================================================
# Source: Li (2008) — annual report readability predicts earnings persistence
#         Lo et al. (2017) — complex filings associated with information asymmetry
# Finding: More complex filings = more opaque = bearish signal
#          Deliberate obfuscation hypothesis: firms hide bad news in complexity
# ============================================================================

def compute_document_complexity(text: str) -> Dict[str, Any]:
    """
    Assess filing complexity using fog-index-inspired metrics.

    Simpler filings = more transparent = bullish.
    Complex filings = potential obfuscation = bearish.

    Returns score 0-100 where higher = simpler/more readable (positive).
    """
    result = {
        'signal': 'document_complexity',
        'has_data': False,
        'score': 50,
        'avg_sentence_length': 0.0,
        'complex_word_pct': 0.0,
        'fog_index': 0.0,
        'total_sentences': 0,
        'interpretation': '',
    }

    if not text or len(text) < 1000:
        result['interpretation'] = 'Insufficient text for complexity analysis'
        return result

    # Use MD&A section (most important per research) or first 50K chars
    focus = _extract_section(text, MDA_HEADER, 50000) or text[:50000]

    # Split into sentences (lookahead for capital letter avoids splitting on abbreviations like U.S.)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', focus)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if len(sentences) < 10:
        result['interpretation'] = 'Insufficient sentences for complexity analysis'
        return result

    result['total_sentences'] = len(sentences)

    # Average sentence length (in words)
    word_counts = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    avg_sent_len = sum(word_counts) / len(word_counts) if word_counts else 0
    result['avg_sentence_length'] = round(avg_sent_len, 1)

    # Complex word percentage (3+ syllables, approximated)
    all_words = re.findall(r'\b[a-z]{2,}\b', focus.lower())
    total_words = len(all_words)

    def _approx_syllables(word):
        """Approximate syllable count using vowel groups."""
        vowels = len(re.findall(r'[aeiouy]+', word.lower()))
        if word.endswith('e') and vowels > 1:
            vowels -= 1
        return max(1, vowels)

    complex_words = sum(1 for w in all_words if _approx_syllables(w) >= 3)
    complex_pct = (complex_words / total_words * 100) if total_words > 0 else 0
    result['complex_word_pct'] = round(complex_pct, 1)

    # Simplified Gunning Fog Index: 0.4 * (avg_sent_len + complex_word_pct)
    fog = 0.4 * (avg_sent_len + complex_pct)
    result['fog_index'] = round(fog, 1)

    # Map fog index to score (INVERTED: lower fog = higher score)
    # SEC filings typically range 18-25 fog index
    # < 16 = very readable, 16-20 = good, 20-24 = average, > 24 = complex
    if fog <= 14:
        score = 90
    elif fog <= 17:
        score = 75 + (17 - fog) / 3 * 15
    elif fog <= 20:
        score = 55 + (20 - fog) / 3 * 20
    elif fog <= 23:
        score = 35 + (23 - fog) / 3 * 20
    elif fog <= 27:
        score = 15 + (27 - fog) / 4 * 20
    else:
        score = max(5, 15 - (fog - 27) * 2)

    score = max(0, min(100, round(score)))
    result['has_data'] = True
    result['score'] = score

    if score >= 70:
        result['interpretation'] = f'Clear, readable filing (fog={fog:.0f}) — transparent communication'
    elif score >= 50:
        result['interpretation'] = f'Average complexity (fog={fog:.0f}) — typical SEC filing'
    elif score >= 30:
        result['interpretation'] = f'Complex filing (fog={fog:.0f}) — potential obfuscation'
    else:
        result['interpretation'] = f'Very complex filing (fog={fog:.0f}) — high information asymmetry risk'

    return result


# ============================================================================
# SIGNAL 10: INSIDER PATTERN (Weight: 5%)
# ============================================================================
# Source: Ozlen & Batumoglu (2025) — 70-80% of alpha pre-disclosure
# Also: Pre-SOX research: insider purchases → 50bps/month (6% annual)
# Note: Post-disclosure value diminished but cluster signals remain useful
# ============================================================================

def compute_insider_pattern_signal(insider_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Score insider trading pattern quality from Form 4 data.

    Evaluates cluster buying, executive vs director trades, and conviction signals.
    Most alpha is pre-disclosure, so weight is low. Cluster signals still useful.

    Returns score 0-100 where higher = more bullish insider signal.
    """
    result = {
        'signal': 'insider_pattern',
        'has_data': False,
        'score': 50,  # neutral when no data
        'net_purchases': 0,
        'buyer_count': 0,
        'seller_count': 0,
        'executive_buyers': 0,
        'cluster_buy': False,
        'total_value': 0,
        'interpretation': '',
    }

    if not insider_data:
        result['interpretation'] = 'No insider trading data available'
        return result

    # Extract key metrics from insider_data
    # Expected format from insider_tracker module:
    transactions = insider_data.get('transactions', [])
    summary = insider_data.get('summary', {})

    if not transactions and not summary:
        result['interpretation'] = 'No insider transactions found'
        return result

    # Use summary if available
    net_purch = summary.get('net_purchases', 0)
    buyers = summary.get('buyer_count', 0)
    sellers = summary.get('seller_count', 0)
    exec_buyers = summary.get('executive_buyers', 0)
    total_val = summary.get('total_purchase_value', 0)
    cluster = summary.get('cluster_buy', False)

    # If no summary, compute from transactions
    if not summary and transactions:
        buy_txns = [t for t in transactions if t.get('type', '').lower() in ('purchase', 'buy', 'p')]
        sell_txns = [t for t in transactions if t.get('type', '').lower() in ('sale', 'sell', 's')]
        buyers = len(set(t.get('insider_name', '') for t in buy_txns))
        sellers = len(set(t.get('insider_name', '') for t in sell_txns))
        net_purch = len(buy_txns) - len(sell_txns)  # Count transactions, not unique persons
        exec_buyers = sum(1 for t in buy_txns if t.get('is_executive', False) or
                         t.get('title', '').upper() in ('CEO', 'CFO', 'COO', 'CTO', 'PRESIDENT'))
        total_val = sum(t.get('value', 0) for t in buy_txns)
        cluster = buyers >= 3

    result['net_purchases'] = net_purch
    result['buyer_count'] = buyers
    result['seller_count'] = sellers
    result['executive_buyers'] = exec_buyers
    result['cluster_buy'] = cluster
    result['total_value'] = total_val

    # Score components:
    # 1. Net direction (40%): more buyers than sellers = positive
    if net_purch >= 3:
        dir_score = 85
    elif net_purch >= 1:
        dir_score = 65
    elif net_purch == 0:
        dir_score = 50
    elif net_purch >= -1:
        dir_score = 35
    else:
        dir_score = 20

    # 2. Cluster buying (30%): multiple insiders buying = strong signal
    if cluster and exec_buyers >= 2:
        cluster_score = 90  # Multiple executives buying = very strong
    elif cluster:
        cluster_score = 75  # 3+ buyers = strong
    elif buyers >= 2:
        cluster_score = 60
    elif buyers == 1:
        cluster_score = 50
    else:
        cluster_score = 40

    # 3. Executive involvement (30%): CEO/CFO trades more informative
    if exec_buyers >= 2:
        exec_score = 85
    elif exec_buyers == 1:
        exec_score = 70
    else:
        exec_score = 45  # Directors only = less informative

    score = round(dir_score * 0.40 + cluster_score * 0.30 + exec_score * 0.30)
    score = max(0, min(100, score))
    result['has_data'] = True
    result['score'] = score

    if score >= 70:
        result['interpretation'] = f'Bullish insider pattern — {buyers} buyers ({exec_buyers} executives), cluster={cluster}'
    elif score >= 55:
        result['interpretation'] = f'Mildly positive insider activity — {buyers} buyers, {sellers} sellers'
    elif score >= 40:
        result['interpretation'] = f'Mixed insider activity — net direction unclear'
    else:
        result['interpretation'] = f'Net insider selling — {sellers} sellers outweigh {buyers} buyers (bearish)'

    return result


# --- Utility / Aggregation Functions ---
# ============================================================================
# COMPOSITE SIGNAL & UTILITY FUNCTIONS
# ============================================================================

def detect_amendment_restatement(text: str, form_type: str = '') -> Dict[str, Any]:
    """
    Detect amendments and restatements in filing text.

    Amendments (10-K/A, 10-Q/A) often indicate accounting problems.
    Restatements are even more serious — they mean prior financials were wrong.

    Returns:
        {
            'is_amendment': bool,
            'has_restatement': bool,
            'restatement_keywords': list,
            'amendment_score': 0-100 (lower = more concern),
            'signal': str
        }
    """
    is_amendment = form_type.endswith('/A')
    text_lower = text[:50000].lower() if text else ''

    restatement_phrases = [
        'restatement', 'restated', 'restate our',
        'material weakness', 'material misstatement',
        'error correction', 'correcting error',
        'revised financial statements', 'revised consolidated',
        'reclassification', 'prior period adjustment',
        'as previously reported', 'as originally reported',
    ]

    found_phrases = [p for p in restatement_phrases if p in text_lower]
    has_restatement = len(found_phrases) >= 2  # Multiple keywords = likely actual restatement

    # Score: 50 = neutral, lower = worse
    score = 50
    if is_amendment:
        score -= 15
    if has_restatement:
        score -= 25
    if 'going concern' in text_lower:
        score -= 10

    score = max(0, min(100, score))

    if score < 20:
        signal = 'strong_negative'
    elif score < 35:
        signal = 'negative'
    elif score < 45:
        signal = 'slightly_negative'
    else:
        signal = 'neutral'

    return {
        'is_amendment': is_amendment,
        'has_restatement': has_restatement,
        'restatement_keywords': found_phrases,
        'amendment_score': score,
        'signal': signal,
    }


def detect_revenue_concentration(text: str) -> Dict[str, Any]:
    """
    Parse "major customer" / revenue concentration disclosures from filing text.

    SEC requires disclosure when a single customer represents >10% of revenue.
    High concentration = risk (customer loss = revenue cliff).
    Growing concentration from prior filings = additional concern.

    Returns:
        {
            'concentration_score': 0-100 (higher = more concentrated),
            'major_customers_mentioned': int,
            'concentration_phrases': list,
            'signal': str
        }
    """
    text_lower = text[:50000].lower() if text else ''
    if not text_lower:
        return {
            'concentration_score': 0,
            'major_customers_mentioned': 0,
            'concentration_phrases': [],
            'signal': 'no_data',
        }

    # Phrases indicating customer concentration
    concentration_phrases = []
    patterns = [
        r'(?:one|a single|1)\s+customer\s+(?:accounted|represented|comprised)',
        r'(?:two|three|2|3)\s+customers?\s+(?:accounted|represented|comprised)',
        r'(?:\d+)%\s+of\s+(?:our\s+)?(?:total\s+)?(?:revenue|sales|net revenue)',
        r'major\s+customer',
        r'significant\s+customer',
        r'largest\s+customer',
        r'customer\s+concentration',
        r'revenue\s+concentration',
        r'(?:loss|losing)\s+(?:of\s+)?(?:a\s+)?(?:key|major|significant)\s+customer',
        r'single\s+source\s+(?:of\s+)?revenue',
    ]

    for pat in patterns:
        matches = re.findall(pat, text_lower)
        concentration_phrases.extend(matches[:3])

    # Extract percentage mentions near concentration language
    pct_near_customer = []
    for match in re.finditer(r'(\d{1,3})%\s+of\s+(?:our\s+)?(?:total\s+)?(?:revenue|sales|net\s+revenue)', text_lower):
        try:
            pct = int(match.group(1))
            if 10 <= pct <= 100:
                pct_near_customer.append(pct)
        except ValueError:
            pass

    # Score: 0 = no concentration, 100 = extreme concentration
    score = 0
    n_phrases = len(concentration_phrases)

    if pct_near_customer:
        max_pct = max(pct_near_customer)
        score = max_pct  # Direct mapping: 50% customer = score 50
    elif n_phrases >= 3:
        score = 60
    elif n_phrases >= 2:
        score = 40
    elif n_phrases >= 1:
        score = 20

    if score >= 60:
        signal = 'high_concentration'
    elif score >= 30:
        signal = 'moderate_concentration'
    elif score > 0:
        signal = 'low_concentration'
    else:
        signal = 'no_concentration_risk'

    return {
        'concentration_score': score,
        'major_customers_mentioned': len(pct_near_customer),
        'max_customer_pct': max(pct_near_customer) if pct_near_customer else None,
        'concentration_phrases': concentration_phrases[:10],
        'signal': signal,
    }


def compute_all_signals(current_text: str, prior_text: str = None,
                        filing_date: str = '', form_type: str = '10-K',
                        period_end: str = None,
                        insider_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compute all ten research-backed signals and the weighted composite.

    Weights derived from academic evidence strength:
      PRIMARY (60%):
        MD&A Similarity:      20% (188bps/month — Lazy Prices JF 2020)
        LM Sentiment:         15% (R² 9.75% — Loughran & McDonald JF 2011)
        Risk Factor Changes:  13% ("especially informative" — Lazy Prices)
        Positive Similarity:  12% (uncorrelated alpha — Positive Similarity paper)
      SECONDARY (25%):
        Sentiment Delta:      10% (period-over-period change — Brown & Tucker 2011)
        Legal Proceedings:     8% (alpha unexplained by FF5+UMD — Offutt & Xie 2025)
        Uncertainty Language:  7% (predicts volatility — LM uncertainty list)
      TERTIARY (15%):
        Filing Timeliness:     5% (late = deterioration — Duarte-Silva et al. 2013)
        Document Complexity:   5% (opacity = bearish — Li 2008)
        Insider Pattern:       5% (diminished post-disclosure — Ozlen & Batumoglu 2025)
    """
    # Compute individual signals
    # Pass None (not '') for missing prior text — functions check `if not prior_text:`

    # SC 13D/A filings lack MD&A, risk factors, legal proceedings sections.
    # Only compute text-level signals that work on any prose.
    is_ownership = form_type.startswith('SC 13')

    _NO_DATA = {
        'score': 50, 'has_data': False, 'signal': 'not_applicable',
        'interpretation': 'Not applicable for this filing type',
        'cosine': None, 'jaccard': None, 'size_ratio': None,
        'has_prior': False, 'delta': None,
        'current_net_sentiment': None, 'prior_net_sentiment': None,
        'size_change_pct': 0, 'new_legal_entities': [], 'new_dollar_amounts': [],
        'risks_added': 0, 'risks_removed': 0,
        'current_positive_count': 0, 'prior_positive_count': 0,
        'timing': 'unknown', 'days_after_period': None, 'deadline_days': None,
    }

    if is_ownership:
        mda = _NO_DATA.copy()
        risk = _NO_DATA.copy()
        pos_sim = _NO_DATA.copy()
        sent_delta = _NO_DATA.copy()
        legal = _NO_DATA.copy()
        timing = _NO_DATA.copy()
    else:
        mda = compute_mda_similarity(current_text, prior_text)
        risk = compute_risk_factor_signal(current_text, prior_text)
        pos_sim = compute_positive_similarity(current_text, prior_text)
        sent_delta = compute_sentiment_delta(current_text, prior_text)
        legal = compute_legal_proceedings_signal(current_text, prior_text)
        timing = compute_filing_timeliness_signal(filing_date, form_type, period_end)

    # These work on any text (word-level analysis)
    sentiment = compute_lm_sentiment(current_text)
    uncertainty = compute_uncertainty_signal(current_text)
    complexity = compute_document_complexity(current_text)
    insider = compute_insider_pattern_signal(insider_data)

    signals = {
        'mda_similarity': mda,
        'lm_sentiment': sentiment,
        'risk_factor_changes': risk,
        'positive_similarity': pos_sim,
        'sentiment_delta': sent_delta,
        'legal_proceedings': legal,
        'uncertainty_language': uncertainty,
        'filing_timeliness': timing,
        'document_complexity': complexity,
        'insider_pattern': insider,
    }

    # Count how many signals have usable data (using explicit has_data flag)
    has_data = [key for key, sig in signals.items() if sig.get('has_data', False)]

    # Compute weighted composite
    if has_data:
        # Re-normalize weights for signals that have data
        total_weight = sum(SIGNAL_WEIGHTS[k] for k in has_data)
        weighted_sum = sum(
            signals[k]['score'] * SIGNAL_WEIGHTS[k] / total_weight
            for k in has_data
        )
        composite_score = round(weighted_sum)
    else:
        composite_score = 50  # no data = neutral

    # Overall interpretation with strength tiers
    if composite_score >= 72:
        overall = 'strong_positive'
        summary = 'Research signals strongly positive — stable filings, good sentiment, no red flags'
    elif composite_score >= 60:
        overall = 'positive'
        summary = 'Research signals lean positive — mostly stable with favorable indicators'
    elif composite_score >= 48:
        overall = 'neutral'
        summary = 'Research signals mixed — some positive/negative indicators cancel out'
    elif composite_score >= 35:
        overall = 'negative'
        summary = 'Research signals negative — significant changes, deteriorating sentiment'
    else:
        overall = 'strong_negative'
        summary = 'Research signals strongly negative — major changes, bearish sentiment across dimensions'

    # Categorize signals by strength tier
    primary_signals = {k: v for k, v in signals.items()
                       if k in ('mda_similarity', 'lm_sentiment', 'risk_factor_changes', 'positive_similarity')}
    secondary_signals = {k: v for k, v in signals.items()
                         if k in ('sentiment_delta', 'legal_proceedings', 'uncertainty_language')}
    tertiary_signals = {k: v for k, v in signals.items()
                        if k in ('filing_timeliness', 'document_complexity', 'insider_pattern')}

    return {
        'research_signal_score': composite_score,
        'research_signal': overall,
        'research_summary': summary,
        'signals': signals,
        'signals_with_data': len(has_data),
        'signal_count': 10,
        'weights': SIGNAL_WEIGHTS,
        'citations': SIGNAL_CITATIONS,
        'tiers': {
            'primary': {k: v['score'] for k, v in primary_signals.items()},
            'secondary': {k: v['score'] for k, v in secondary_signals.items()},
            'tertiary': {k: v['score'] for k, v in tertiary_signals.items()},
        },
    }


# ============================================================================
# FORMAT FOR LLM PROMPT
# ============================================================================

def format_signals_for_prompt(signals_data: Dict[str, Any]) -> str:
    """Format all 10 research signals into a concise block for the LLM analysis prompt."""
    if not signals_data:
        return "(No research signal data available)"

    lines = [
        "=" * 65,
        "RESEARCH-BACKED FILING SIGNALS (10 dimensions, academic literature)",
        "=" * 65,
        "",
        f"Composite Research Signal Score: {signals_data.get('research_signal_score', 50)}/100",
        f"Overall Signal: {signals_data.get('research_signal', 'neutral').upper()}",
        f"Summary: {signals_data.get('research_summary', '')}",
        f"Signals with data: {signals_data.get('signals_with_data', 0)}/10",
        "",
        "--- PRIMARY SIGNALS (strongest evidence, 60% combined weight) ---",
        "",
    ]

    sigs = signals_data.get('signals', {})
    weights = signals_data.get('weights', SIGNAL_WEIGHTS)
    citations = signals_data.get('citations', SIGNAL_CITATIONS)

    # 1. MD&A Similarity (20%)
    mda = sigs.get('mda_similarity', {})
    pct = round(weights.get('mda_similarity', 0.20) * 100)
    lines.append(f"1. MD&A SIMILARITY [{pct}% — {citations.get('mda_similarity', '')}]")
    lines.append(f"   Score: {mda.get('score', 50)}/100")
    if mda.get('cosine') is not None:
        lines.append(f"   Cosine: {mda['cosine']:.3f} | Jaccard: {mda.get('jaccard', 0):.3f} | Size ratio: {mda.get('size_ratio', 0):.3f}")
    lines.append(f"   {mda.get('interpretation', 'N/A')}")
    lines.append("")

    # 2. LM Sentiment (15%)
    sent = sigs.get('lm_sentiment', {})
    pct = round(weights.get('lm_sentiment', 0.15) * 100)
    lines.append(f"2. LM SENTIMENT [{pct}% — {citations.get('lm_sentiment', '')}]")
    lines.append(f"   Score: {sent.get('score', 50)}/100")
    if sent.get('positive_count'):
        lines.append(f"   Positive: {sent['positive_count']} ({sent.get('positive_pct', 0):.2f}%) | Negative: {sent.get('negative_count', 0)} ({sent.get('negative_pct', 0):.2f}%) | Net: {sent.get('net_sentiment', 0):+.2f}%")
    lines.append(f"   {sent.get('interpretation', 'N/A')}")
    lines.append("")

    # 3. Risk Factor Changes (13%)
    rf = sigs.get('risk_factor_changes', {})
    pct = round(weights.get('risk_factor_changes', 0.13) * 100)
    lines.append(f"3. RISK FACTOR CHANGES [{pct}% — {citations.get('risk_factor_changes', '')}]")
    lines.append(f"   Score: {rf.get('score', 50)}/100")
    if rf.get('risks_added') or rf.get('risks_removed'):
        lines.append(f"   Risks added: {rf.get('risks_added', 0)} | Removed: {rf.get('risks_removed', 0)} | Section similarity: {rf.get('cosine', 0):.3f}" if rf.get('cosine') is not None else f"   Risks added: {rf.get('risks_added', 0)} | Removed: {rf.get('risks_removed', 0)}")
    lines.append(f"   {rf.get('interpretation', 'N/A')}")
    lines.append("")

    # 4. Positive Similarity (12%)
    ps = sigs.get('positive_similarity', {})
    pct = round(weights.get('positive_similarity', 0.12) * 100)
    lines.append(f"4. POSITIVE SIMILARITY [{pct}% — {citations.get('positive_similarity', '')}]")
    lines.append(f"   Score: {ps.get('score', 50)}/100 (INVERTED: low similarity = bullish)")
    if ps.get('cosine') is not None:
        lines.append(f"   Positive language cosine: {ps['cosine']:.3f} | Jaccard: {ps.get('jaccard', 0):.3f}")
    lines.append(f"   {ps.get('interpretation', 'N/A')}")
    lines.append("")

    lines.append("--- SECONDARY SIGNALS (moderate evidence, 25% combined weight) ---")
    lines.append("")

    # 5. Sentiment Delta (10%)
    sd = sigs.get('sentiment_delta', {})
    pct = round(weights.get('sentiment_delta', 0.10) * 100)
    lines.append(f"5. SENTIMENT DELTA [{pct}% — {citations.get('sentiment_delta', '')}]")
    lines.append(f"   Score: {sd.get('score', 50)}/100")
    if sd.get('delta') is not None:
        lines.append(f"   Current sentiment: {sd.get('current_net_sentiment', 0):+.2f}% | Prior: {sd.get('prior_net_sentiment', 0):+.2f}% | Delta: {sd['delta']:+.2f}%")
    lines.append(f"   {sd.get('interpretation', 'N/A')}")
    lines.append("")

    # 6. Legal Proceedings (8%)
    legal = sigs.get('legal_proceedings', {})
    pct = round(weights.get('legal_proceedings', 0.08) * 100)
    lines.append(f"6. LEGAL PROCEEDINGS [{pct}% — {citations.get('legal_proceedings', '')}]")
    lines.append(f"   Score: {legal.get('score', 50)}/100")
    if legal.get('size_change_pct') is not None:
        lines.append(f"   Section size change: {legal['size_change_pct']:+.0f}%")
    if legal.get('new_legal_entities'):
        lines.append(f"   New entities: {', '.join(legal['new_legal_entities'][:3])}")
    if legal.get('new_dollar_amounts'):
        lines.append(f"   New dollar amounts: {', '.join(legal['new_dollar_amounts'][:3])}")
    lines.append(f"   {legal.get('interpretation', 'N/A')}")
    lines.append("")

    # 7. Uncertainty Language (7%)
    unc = sigs.get('uncertainty_language', {})
    pct = round(weights.get('uncertainty_language', 0.07) * 100)
    lines.append(f"7. UNCERTAINTY LANGUAGE [{pct}% — {citations.get('uncertainty_language', '')}]")
    lines.append(f"   Score: {unc.get('score', 50)}/100")
    if unc.get('uncertainty_pct'):
        lines.append(f"   Uncertainty word density: {unc['uncertainty_pct']:.2f}% ({unc.get('uncertainty_count', 0)} terms)")
    if unc.get('top_uncertainty_terms'):
        lines.append(f"   Top terms: {', '.join(unc['top_uncertainty_terms'][:4])}")
    lines.append(f"   {unc.get('interpretation', 'N/A')}")
    lines.append("")

    lines.append("--- TERTIARY SIGNALS (supporting evidence, 15% combined weight) ---")
    lines.append("")

    # 8. Filing Timeliness (5%)
    timing = sigs.get('filing_timeliness', {})
    pct = round(weights.get('filing_timeliness', 0.05) * 100)
    lines.append(f"8. FILING TIMELINESS [{pct}% — {citations.get('filing_timeliness', '')}]")
    lines.append(f"   Score: {timing.get('score', 50)}/100")
    lines.append(f"   {timing.get('interpretation', 'N/A')}")
    lines.append("")

    # 9. Document Complexity (5%)
    cpx = sigs.get('document_complexity', {})
    pct = round(weights.get('document_complexity', 0.05) * 100)
    lines.append(f"9. DOCUMENT COMPLEXITY [{pct}% — {citations.get('document_complexity', '')}]")
    lines.append(f"   Score: {cpx.get('score', 50)}/100")
    if cpx.get('fog_index') is not None:
        lines.append(f"   Fog index: {cpx['fog_index']:.0f} | Avg sentence length: {cpx.get('avg_sentence_length', 0):.0f} words | Complex words: {cpx.get('complex_word_pct', 0):.0f}%")
    lines.append(f"   {cpx.get('interpretation', 'N/A')}")
    lines.append("")

    # 10. Insider Pattern (5%)
    ins = sigs.get('insider_pattern', {})
    pct = round(weights.get('insider_pattern', 0.05) * 100)
    lines.append(f"10. INSIDER PATTERN [{pct}% — {citations.get('insider_pattern', '')}]")
    lines.append(f"    Score: {ins.get('score', 50)}/100")
    if ins.get('buyer_count') or ins.get('seller_count'):
        lines.append(f"    Buyers: {ins.get('buyer_count', 0)} | Sellers: {ins.get('seller_count', 0)} | Exec buyers: {ins.get('executive_buyers', 0)} | Cluster: {ins.get('cluster_buy', False)}")
    lines.append(f"    {ins.get('interpretation', 'N/A')}")
    lines.append("")

    return "\n".join(lines)
