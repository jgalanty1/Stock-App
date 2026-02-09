"""
Forensic Language Analysis Module
==================================
Pure Python (no LLM) extraction of quantitative linguistic signals from SEC filings.
Produces concrete metrics that get passed INTO the LLM prompts to ground the analysis.

Two main capabilities:
  1. Single-filing forensics: hedging, confidence, specificity, readability
  2. Filing-over-filing diff: what sentences/sections changed, metric shifts
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# ============================================================================
# WORD LISTS
# ============================================================================

HEDGING_WORDS = {
    "may", "might", "could", "would", "possibly", "potentially", "uncertain",
    "uncertain", "approximately", "substantially", "generally", "typically",
    "believe", "anticipate", "expect", "estimate", "intend", "plan",
    "preliminary", "subject to", "contingent", "if applicable",
}

STRONG_HEDGES = {
    "no assurance", "cannot predict", "may not", "might not", "could adversely",
    "uncertain whether", "there can be no", "we cannot guarantee",
    "subject to significant", "material adverse", "significant uncertainty",
}

CONFIDENCE_WORDS = {
    "confident", "well-positioned", "strong", "robust", "proven", "delivered",
    "achieved", "exceeded", "outperformed", "momentum", "track record",
    "competitive advantage", "disciplined", "committed", "demonstrated",
    "accelerated", "strengthened", "improved", "expanded", "record",
}

BLAME_LANGUAGE = {
    "macroeconomic", "headwinds", "industry-wide", "unprecedented",
    "unforeseen", "beyond our control", "challenging environment",
    "regulatory burden", "market conditions", "supply chain disruptions",
    "inflationary pressures", "geopolitical", "force majeure",
    "pandemic", "covid", "one-time", "non-recurring", "extraordinary",
}

SPECIFICITY_MARKERS = re.compile(
    r'\$[\d,.]+[MBK]?'           # dollar amounts
    r'|\d+\.?\d*\s*%'            # percentages
    r'|\d+\.?\d*x'              # multiples
    r'|\d{1,3}(?:,\d{3})+'      # large numbers with commas
    r'|increased \d+'            # specific increases
    r'|decreased \d+'            # specific decreases
    r'|grew \d+'                 # specific growth
)

PASSIVE_MARKERS = re.compile(
    r'\b(?:was|were|been|being|is|are)\s+(?:\w+ed|made|done|taken|given)\b',
    re.IGNORECASE,
)

FORWARD_LOOKING = re.compile(
    r'\b(?:expect|anticipate|believe|intend|plan|project|forecast|'
    r'outlook|guidance|target|goal|objective|will\s+(?:be|continue|'
    r'increase|decrease|remain|grow|expand))\b',
    re.IGNORECASE,
)

# Section header patterns for extracting key sections
RISK_FACTOR_HEADER = re.compile(
    r'(?:item\s*1a\.?\s*risk\s*factors|risk\s*factors)',
    re.IGNORECASE,
)

MDA_HEADER = re.compile(
    r"(?:item\s*(?:2|7)\.?\s*management'?s?\s*discussion|"
    r"management'?s?\s*discussion\s*and\s*analysis)",
    re.IGNORECASE,
)

FORWARD_LOOKING_HEADER = re.compile(
    r'(?:forward[- ]looking\s*statements|cautionary\s*(?:note|statement))',
    re.IGNORECASE,
)


# ============================================================================
# TEXT EXTRACTION HELPERS
# ============================================================================

def _normalize_text(text: str) -> str:
    """Normalize whitespace for analysis."""
    return re.sub(r'\s+', ' ', text).strip()


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences (approximate)."""
    # Split on period/question/exclamation followed by space+capital or end
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in parts if len(s.strip()) > 20]


def _extract_section(text: str, header_pattern: re.Pattern,
                     max_chars: int = 50000) -> Optional[str]:
    """Extract a section from the filing starting at a header pattern."""
    match = header_pattern.search(text)
    if not match:
        return None

    start = match.start()
    # Find the next major section header (Item N.)
    next_item = re.search(r'\bitem\s+\d+[a-z]?\.', text[start + 100:], re.IGNORECASE)
    if next_item:
        end = start + 100 + next_item.start()
    else:
        end = start + max_chars

    section = text[start:min(end, start + max_chars)]
    return _normalize_text(section)


def _count_pattern_hits(text: str, word_set: set) -> Tuple[int, List[str]]:
    """Count occurrences of words/phrases from a set. Return count and examples."""
    text_lower = text.lower()
    hits = []
    total = 0
    for phrase in word_set:
        count = text_lower.count(phrase.lower())
        if count > 0:
            total += count
            hits.append(f"{phrase} ({count})")
    # Sort by frequency descending
    hits.sort(key=lambda x: int(re.search(r'\((\d+)\)', x).group(1)), reverse=True)
    return total, hits[:10]


# ============================================================================
# SINGLE-FILING FORENSICS
# ============================================================================

def analyze_language(text: str) -> Dict[str, Any]:
    """
    Analyze a single filing's language patterns.
    Returns quantitative metrics that can be passed to LLM prompts.
    """
    if not text or len(text) < 500:
        return {"error": "Text too short for analysis"}

    normalized = _normalize_text(text)
    sentences = _split_sentences(normalized)
    word_count = len(normalized.split())

    if word_count < 100:
        return {"error": "Too few words for analysis"}

    per_1000 = 1000.0 / word_count  # multiplier to normalize to per-1000-words

    # ---- Hedging analysis ----
    hedge_count, hedge_examples = _count_pattern_hits(normalized, HEDGING_WORDS)
    strong_hedge_count, strong_hedge_examples = _count_pattern_hits(normalized, STRONG_HEDGES)
    hedge_ratio = round(hedge_count * per_1000, 2)

    # ---- Confidence analysis ----
    conf_count, conf_examples = _count_pattern_hits(normalized, CONFIDENCE_WORDS)
    conf_ratio = round(conf_count * per_1000, 2)

    # ---- Confidence-to-hedging ratio (higher = more confident) ----
    ch_ratio = round(conf_count / max(hedge_count, 1), 2)

    # ---- Blame language ----
    blame_count, blame_examples = _count_pattern_hits(normalized, BLAME_LANGUAGE)
    blame_ratio = round(blame_count * per_1000, 2)

    # ---- Specificity (numbers, $, %) ----
    specificity_hits = SPECIFICITY_MARKERS.findall(normalized)
    specificity_ratio = round(len(specificity_hits) * per_1000, 2)

    # ---- Passive voice ----
    passive_hits = PASSIVE_MARKERS.findall(normalized)
    passive_ratio = round(len(passive_hits) * per_1000, 2)

    # ---- Forward-looking statements ----
    fwd_hits = FORWARD_LOOKING.findall(normalized)
    fwd_ratio = round(len(fwd_hits) * per_1000, 2)

    # ---- Sentence length (readability proxy) ----
    sent_lengths = [len(s.split()) for s in sentences]
    avg_sent_len = round(sum(sent_lengths) / max(len(sent_lengths), 1), 1)

    # ---- Extract risk factors section ----
    risk_section = _extract_section(text, RISK_FACTOR_HEADER)
    risk_factor_count = 0
    risk_factor_titles = []
    if risk_section:
        # Count distinct risk factor headings (often bold or all-caps phrases)
        rf_titles = re.findall(
            r'(?:^|\n)\s*([A-Z][A-Za-z\s,\'-]{10,80}?)(?:\.|—|\n)',
            risk_section
        )
        risk_factor_titles = [t.strip() for t in rf_titles if len(t.strip()) > 15][:30]
        risk_factor_count = len(risk_factor_titles)

    # ---- Extract MD&A section length ----
    mda_section = _extract_section(text, MDA_HEADER)
    mda_word_count = len(mda_section.split()) if mda_section else 0

    # ---- Compute overall forensic scores (1-10 scale) ----
    # Confidence score: higher confidence words + lower hedging = better
    # Typical CH ratio: 0.1 (very hedged) to 3.0+ (very confident)
    confidence_score = min(10, max(1, round(2 + ch_ratio * 2.5)))

    # Transparency score: more specific numbers = better
    # Typical specificity: 5-50 per 1000 words
    transparency_score = min(10, max(1, round(specificity_ratio / 5 + 1)))

    # Directness score: less passive voice + less blame = better
    # Typical blame: 0-30 per 1000w, passive: 0-10 per 1000w
    directness_score = min(10, max(1, round(8 - blame_ratio * 0.15 - passive_ratio * 0.3)))

    return {
        "word_count": word_count,
        "sentence_count": len(sentences),
        "avg_sentence_length": avg_sent_len,
        "hedging": {
            "count": hedge_count,
            "ratio_per_1k": hedge_ratio,
            "strong_hedges": strong_hedge_count,
            "top_phrases": hedge_examples[:5],
        },
        "confidence": {
            "count": conf_count,
            "ratio_per_1k": conf_ratio,
            "top_phrases": conf_examples[:5],
        },
        "confidence_hedge_ratio": ch_ratio,
        "blame_language": {
            "count": blame_count,
            "ratio_per_1k": blame_ratio,
            "top_phrases": blame_examples[:5],
        },
        "specificity": {
            "count": len(specificity_hits),
            "ratio_per_1k": specificity_ratio,
            "examples": specificity_hits[:10],
        },
        "passive_voice": {
            "count": len(passive_hits),
            "ratio_per_1k": passive_ratio,
        },
        "forward_looking": {
            "count": len(fwd_hits),
            "ratio_per_1k": fwd_ratio,
        },
        "risk_factors": {
            "count": risk_factor_count,
            "titles": risk_factor_titles[:15],
        },
        "mda_word_count": mda_word_count,
        "scores": {
            "confidence": confidence_score,
            "transparency": transparency_score,
            "directness": directness_score,
        },
    }


# ============================================================================
# FILING-OVER-FILING DIFF
# ============================================================================

def diff_filings(current_text: str, prior_text: str) -> Dict[str, Any]:
    """
    Compare two filings and identify what changed.
    Returns structured diff data for LLM consumption.
    """
    if not current_text or not prior_text:
        return {"error": "Need both current and prior text"}

    current_forensics = analyze_language(current_text)
    prior_forensics = analyze_language(prior_text)

    if "error" in current_forensics or "error" in prior_forensics:
        return {"error": "One or both texts too short for analysis"}

    # ---- Metric deltas ----
    metric_changes = {}
    for key in ["confidence_hedge_ratio"]:
        cur = current_forensics.get(key, 0)
        pri = prior_forensics.get(key, 0)
        metric_changes[key] = {
            "current": cur, "prior": pri,
            "change": round(cur - pri, 2),
            "direction": "up" if cur > pri else "down" if cur < pri else "flat",
        }

    for category in ["hedging", "confidence", "blame_language", "specificity",
                      "passive_voice", "forward_looking"]:
        cur = current_forensics.get(category, {}).get("ratio_per_1k", 0)
        pri = prior_forensics.get(category, {}).get("ratio_per_1k", 0)
        pct = round((cur - pri) / max(pri, 0.5) * 100, 1)  # floor at 0.5 to avoid huge %
        pct = max(-999, min(999, pct))  # cap at ±999%
        metric_changes[category] = {
            "current": cur, "prior": pri,
            "change": round(cur - pri, 2),
            "pct_change": pct,
            "direction": "up" if cur > pri else "down" if cur < pri else "flat",
        }

    # ---- Score deltas ----
    score_changes = {}
    for key in ["confidence", "transparency", "directness"]:
        cur = current_forensics.get("scores", {}).get(key, 5)
        pri = prior_forensics.get("scores", {}).get(key, 5)
        score_changes[key] = {
            "current": cur, "prior": pri, "change": cur - pri,
        }

    # ---- Risk factor diff ----
    current_risks = set(current_forensics.get("risk_factors", {}).get("titles", []))
    prior_risks = set(prior_forensics.get("risk_factors", {}).get("titles", []))
    added_risks = list(current_risks - prior_risks)[:10]
    removed_risks = list(prior_risks - current_risks)[:10]

    # ---- Key sentence diff (MD&A section) ----
    current_mda = _extract_section(current_text, MDA_HEADER, 30000) or ""
    prior_mda = _extract_section(prior_text, MDA_HEADER, 30000) or ""

    added_sentences, removed_sentences = _diff_sentences(current_mda, prior_mda)

    # ---- Risk factors section sentence diff ----
    current_rf = _extract_section(current_text, RISK_FACTOR_HEADER, 30000) or ""
    prior_rf = _extract_section(prior_text, RISK_FACTOR_HEADER, 30000) or ""

    rf_added, rf_removed = _diff_sentences(current_rf, prior_rf)

    # ---- Overall diff signal ----
    # Positive signals: confidence up, hedging down, blame down, specificity up
    positive_shifts = 0
    negative_shifts = 0

    if metric_changes["confidence"]["direction"] == "up":
        positive_shifts += 1
    elif metric_changes["confidence"]["direction"] == "down":
        negative_shifts += 1

    if metric_changes["hedging"]["direction"] == "down":
        positive_shifts += 1
    elif metric_changes["hedging"]["direction"] == "up":
        negative_shifts += 1

    if metric_changes["blame_language"]["direction"] == "down":
        positive_shifts += 1
    elif metric_changes["blame_language"]["direction"] == "up":
        negative_shifts += 1

    if metric_changes["specificity"]["direction"] == "up":
        positive_shifts += 1
    elif metric_changes["specificity"]["direction"] == "down":
        negative_shifts += 1

    if len(removed_risks) > len(added_risks):
        positive_shifts += 1  # fewer risk factors = improving
    elif len(added_risks) > len(removed_risks):
        negative_shifts += 1

    if positive_shifts >= 3:
        overall_signal = "improving"
    elif negative_shifts >= 3:
        overall_signal = "deteriorating"
    elif positive_shifts > negative_shifts:
        overall_signal = "slightly_improving"
    elif negative_shifts > positive_shifts:
        overall_signal = "slightly_deteriorating"
    else:
        overall_signal = "stable"

    return {
        "metric_changes": metric_changes,
        "score_changes": score_changes,
        "risk_factor_diff": {
            "current_count": len(current_risks),
            "prior_count": len(prior_risks),
            "added": added_risks,
            "removed": removed_risks,
        },
        "mda_changes": {
            "added_sentences": added_sentences[:8],
            "removed_sentences": removed_sentences[:8],
        },
        "risk_factor_language_changes": {
            "added_sentences": rf_added[:8],
            "removed_sentences": rf_removed[:8],
        },
        "overall_signal": overall_signal,
        "positive_shifts": positive_shifts,
        "negative_shifts": negative_shifts,
        "current_forensics": current_forensics,
        "prior_forensics": prior_forensics,
    }


def _diff_sentences(current_text: str, prior_text: str,
                    similarity_threshold: float = 0.6) -> Tuple[List[str], List[str]]:
    """
    Find sentences added to current and removed from prior.
    Uses fuzzy matching to ignore minor rewording.
    """
    if not current_text or not prior_text:
        return [], []

    current_sents = _split_sentences(current_text)
    prior_sents = _split_sentences(prior_text)

    # For efficiency, limit to reasonable counts
    current_sents = current_sents[:200]
    prior_sents = prior_sents[:200]

    # Find truly new sentences (no close match in prior)
    added = []
    for cs in current_sents:
        if len(cs) < 30:
            continue
        best_match = max(
            (SequenceMatcher(None, cs.lower(), ps.lower()).ratio()
             for ps in prior_sents),
            default=0,
        )
        if best_match < similarity_threshold:
            added.append(cs[:200])

    # Find removed sentences (no close match in current)
    removed = []
    for ps in prior_sents:
        if len(ps) < 30:
            continue
        best_match = max(
            (SequenceMatcher(None, ps.lower(), cs.lower()).ratio()
             for cs in current_sents),
            default=0,
        )
        if best_match < similarity_threshold:
            removed.append(ps[:200])

    return added, removed


# ============================================================================
# FORMAT FOR LLM PROMPT
# ============================================================================

def format_forensics_for_prompt(forensics: Dict[str, Any],
                                diff: Optional[Dict[str, Any]] = None) -> str:
    """
    Format forensic data into a concise text block for inclusion in LLM prompts.
    """
    lines = ["=== FORENSIC LANGUAGE METRICS ==="]

    s = forensics.get("scores", {})
    lines.append(f"Confidence Score: {s.get('confidence', '?')}/10 | "
                 f"Transparency Score: {s.get('transparency', '?')}/10 | "
                 f"Directness Score: {s.get('directness', '?')}/10")

    lines.append(f"Confidence-to-Hedge Ratio: {forensics.get('confidence_hedge_ratio', '?')} "
                 f"(>1.5=confident, <0.5=heavily hedged)")

    h = forensics.get("hedging", {})
    lines.append(f"Hedging: {h.get('ratio_per_1k', '?')}/1000w "
                 f"({h.get('strong_hedges', 0)} strong hedges)")
    if h.get("top_phrases"):
        lines.append(f"  Top hedges: {', '.join(h['top_phrases'][:3])}")

    c = forensics.get("confidence", {})
    lines.append(f"Confidence language: {c.get('ratio_per_1k', '?')}/1000w")
    if c.get("top_phrases"):
        lines.append(f"  Top confidence: {', '.join(c['top_phrases'][:3])}")

    b = forensics.get("blame_language", {})
    if b.get("count", 0) > 0:
        lines.append(f"Blame/excuse language: {b.get('ratio_per_1k', '?')}/1000w")
        if b.get("top_phrases"):
            lines.append(f"  Top blame phrases: {', '.join(b['top_phrases'][:3])}")

    sp = forensics.get("specificity", {})
    lines.append(f"Specificity (numbers/$/%): {sp.get('ratio_per_1k', '?')}/1000w")

    pv = forensics.get("passive_voice", {})
    lines.append(f"Passive voice: {pv.get('ratio_per_1k', '?')}/1000w")

    rf = forensics.get("risk_factors", {})
    lines.append(f"Risk factors identified: {rf.get('count', '?')}")

    # ---- Diff section ----
    if diff and "error" not in diff:
        lines.append("")
        lines.append("=== FILING-OVER-FILING CHANGES ===")
        lines.append(f"Overall signal: {diff.get('overall_signal', '?').upper()} "
                     f"(+{diff.get('positive_shifts', 0)} positive, "
                     f"-{diff.get('negative_shifts', 0)} negative shifts)")

        mc = diff.get("metric_changes", {})
        for key in ["hedging", "confidence", "blame_language", "specificity"]:
            if key in mc:
                m = mc[key]
                arrow = "↑" if m["direction"] == "up" else "↓" if m["direction"] == "down" else "→"
                lines.append(f"  {key}: {m['prior']} → {m['current']} ({arrow} {m['pct_change']:+.1f}%)")

        rfd = diff.get("risk_factor_diff", {})
        if rfd.get("added"):
            lines.append(f"\nNEW risk factors ({len(rfd['added'])}):")
            for r in rfd["added"][:5]:
                lines.append(f"  + {r}")
        if rfd.get("removed"):
            lines.append(f"\nREMOVED risk factors ({len(rfd['removed'])}):")
            for r in rfd["removed"][:5]:
                lines.append(f"  - {r}")

        mda = diff.get("mda_changes", {})
        if mda.get("added_sentences"):
            lines.append(f"\nKEY NEW STATEMENTS in MD&A ({len(mda['added_sentences'])}):")
            for s in mda["added_sentences"][:5]:
                lines.append(f"  + \"{s[:150]}\"")
        if mda.get("removed_sentences"):
            lines.append(f"\nREMOVED STATEMENTS from MD&A ({len(mda['removed_sentences'])}):")
            for s in mda["removed_sentences"][:5]:
                lines.append(f"  - \"{s[:150]}\"")

        rfl = diff.get("risk_factor_language_changes", {})
        if rfl.get("added_sentences"):
            lines.append(f"\nNEW risk factor language ({len(rfl['added_sentences'])}):")
            for s in rfl["added_sentences"][:3]:
                lines.append(f"  + \"{s[:150]}\"")
        if rfl.get("removed_sentences"):
            lines.append(f"\nREMOVED risk factor language ({len(rfl['removed_sentences'])}):")
            for s in rfl["removed_sentences"][:3]:
                lines.append(f"  - \"{s[:150]}\"")

    return "\n".join(lines)
