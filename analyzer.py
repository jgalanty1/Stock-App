"""
SEC Filing Analyzer - Core Analysis Module
==========================================
Analyzes SEC filings for flags, sentiment, and changes.
"""

import re
import logging
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class RiskLevel(Enum):
    CRITICAL = 4
    HIGH = 3
    MODERATE = 2
    LOW = 1
    INFO = 0


class SignalType(Enum):
    RED_FLAG = "red_flag"
    YELLOW_FLAG = "yellow_flag"
    GREEN_FLAG = "green_flag"
    NEUTRAL = "neutral"


class ChangeType(Enum):
    NEW = "new"
    WORSENING = "worsening"
    IMPROVING = "improving"
    RESOLVED = "resolved"
    UNCHANGED = "unchanged"
    FIRST_FILING = "first_filing"


class SentimentLevel(Enum):
    VERY_CONFIDENT = 5
    CONFIDENT = 4
    NEUTRAL = 3
    CAUTIOUS = 2
    DEFENSIVE = 1
    ALARMING = 0


class Category(Enum):
    LIQUIDITY = "liquidity"
    DEBT = "debt"
    REVENUE = "revenue"
    PROFITABILITY = "profitability"
    GOVERNANCE = "governance"
    OPERATIONS = "operations"
    ACCOUNTING = "accounting"
    LEGAL = "legal"
    INSIDER = "insider"
    RELATED_PARTY = "related_party"
    AUDITOR = "auditor"
    MANAGEMENT = "management"
    DILUTION = "dilution"


# =============================================================================
# WEIGHTS AND MULTIPLIERS
# =============================================================================

CATEGORY_WEIGHTS = {
    Category.LIQUIDITY: 1.5,
    Category.DEBT: 1.4,
    Category.AUDITOR: 1.5,
    Category.GOVERNANCE: 1.2,
    Category.ACCOUNTING: 1.3,
    Category.RELATED_PARTY: 1.3,
    Category.REVENUE: 1.0,
    Category.PROFITABILITY: 1.0,
    Category.OPERATIONS: 0.9,
    Category.LEGAL: 1.1,
    Category.INSIDER: 1.0,
    Category.MANAGEMENT: 1.0,
    Category.DILUTION: 1.1,
}

CHANGE_MULTIPLIERS = {
    ChangeType.NEW: 2.0,
    ChangeType.WORSENING: 1.75,
    ChangeType.UNCHANGED: 1.0,
    ChangeType.IMPROVING: 0.5,
    ChangeType.RESOLVED: 0.0,
    ChangeType.FIRST_FILING: 1.0,
}


# =============================================================================
# FLAG DETECTION RULES
# =============================================================================

FLAG_RULES = {
    # CRITICAL RED FLAGS
    "going_concern": {
        "category": Category.AUDITOR,
        "signal_type": SignalType.RED_FLAG,
        "risk_level": RiskLevel.CRITICAL,
        "title": "Going Concern Warning",
        "patterns": [
            r"substantial doubt.*ability to continue as a going concern",
            r"going concern",
            r"raise substantial doubt",
            r"ability to continue as a going concern"
        ],
        "description": "Auditor or management has expressed doubt about the company's ability to continue operating"
    },
    "auditor_change": {
        "category": Category.AUDITOR,
        "signal_type": SignalType.RED_FLAG,
        "risk_level": RiskLevel.CRITICAL,
        "title": "Auditor Resignation/Dismissal",
        "patterns": [
            r"auditor resigned",
            r"dismissal of.*accounting firm",
            r"change in certifying accountant",
            r"item 4\.01"
        ],
        "description": "Change in auditor, especially if mid-engagement"
    },
    "restatement": {
        "category": Category.ACCOUNTING,
        "signal_type": SignalType.RED_FLAG,
        "risk_level": RiskLevel.CRITICAL,
        "title": "Financial Restatement",
        "patterns": [
            r"restatement of previously issued",
            r"restated financial statements",
            r"material misstatement",
            r"item 4\.02"
        ],
        "description": "Company has restated or will restate financial statements"
    },
    "sec_investigation": {
        "category": Category.LEGAL,
        "signal_type": SignalType.RED_FLAG,
        "risk_level": RiskLevel.CRITICAL,
        "title": "SEC Investigation",
        "patterns": [
            r"sec investigation",
            r"securities and exchange commission.*investigation",
            r"enforcement action",
            r"subpoena.*sec",
            r"wells notice"
        ],
        "description": "Company is under SEC investigation or enforcement action"
    },
    "covenant_violation": {
        "category": Category.DEBT,
        "signal_type": SignalType.RED_FLAG,
        "risk_level": RiskLevel.CRITICAL,
        "title": "Debt Covenant Violation",
        "patterns": [
            r"covenant violation",
            r"breach of covenant",
            r"event of default",
            r"waiver of covenant",
            r"not in compliance with.*covenant"
        ],
        "description": "Company has violated or is at risk of violating debt covenants"
    },
    
    # HIGH RED FLAGS
    "material_weakness": {
        "category": Category.ACCOUNTING,
        "signal_type": SignalType.RED_FLAG,
        "risk_level": RiskLevel.HIGH,
        "title": "Material Weakness in Internal Controls",
        "patterns": [
            r"material weakness",
            r"internal control.*was not effective",
            r"significant deficiency"
        ],
        "description": "Deficiency in internal controls that could result in material misstatement"
    },
    "refinancing_risk": {
        "category": Category.DEBT,
        "signal_type": SignalType.RED_FLAG,
        "risk_level": RiskLevel.HIGH,
        "title": "Debt Refinancing Risk",
        "patterns": [
            r"refinancing",
            r"debt matur",
            r"credit agreement.*expir",
            r"evaluate refinancing",
            r"speaking with.*partners.*refinanc"
        ],
        "description": "Significant debt maturing with refinancing uncertainty"
    },
    "negative_equity": {
        "category": Category.LIQUIDITY,
        "signal_type": SignalType.RED_FLAG,
        "risk_level": RiskLevel.HIGH,
        "title": "Negative Shareholders' Equity",
        "patterns": [
            r"shareholders.*deficit",
            r"stockholders.*deficit",
            r"negative.*equity"
        ],
        "description": "Total liabilities exceed total assets"
    },
    "related_party": {
        "category": Category.RELATED_PARTY,
        "signal_type": SignalType.RED_FLAG,
        "risk_level": RiskLevel.HIGH,
        "title": "Material Related Party Transactions",
        "patterns": [
            r"related party transaction",
            r"transactions with affiliates",
            r"controlled by.*officer",
            r"related parties"
        ],
        "description": "Significant transactions with insiders or affiliates"
    },
    
    # MODERATE RED FLAGS
    "ceo_departure": {
        "category": Category.MANAGEMENT,
        "signal_type": SignalType.RED_FLAG,
        "risk_level": RiskLevel.MODERATE,
        "title": "CEO/CFO Departure",
        "patterns": [
            r"interim.*chief executive",
            r"interim.*president",
            r"interim.*ceo",
            r"interim.*chief financial",
            r"resignation of.*officer",
            r"departure of.*officer"
        ],
        "description": "Key executive has departed or is interim"
    },
    "goodwill_impairment": {
        "category": Category.ACCOUNTING,
        "signal_type": SignalType.RED_FLAG,
        "risk_level": RiskLevel.MODERATE,
        "title": "Goodwill/Asset Impairment",
        "patterns": [
            r"goodwill impairment",
            r"impairment charge",
            r"write-down",
            r"carrying value exceeds"
        ],
        "description": "Assets written down indicating overpayment for acquisitions"
    },
    
    # YELLOW FLAGS
    "operating_losses": {
        "category": Category.PROFITABILITY,
        "signal_type": SignalType.YELLOW_FLAG,
        "risk_level": RiskLevel.MODERATE,
        "title": "Operating Losses",
        "patterns": [
            r"accumulated deficit",
            r"history of.*losses",
            r"operating loss",
            r"net loss"
        ],
        "description": "Pattern of operating losses"
    },
    "dilution": {
        "category": Category.DILUTION,
        "signal_type": SignalType.YELLOW_FLAG,
        "risk_level": RiskLevel.MODERATE,
        "title": "Share Dilution Risk",
        "patterns": [
            r"shelf registration",
            r"at-the-market offering",
            r"warrant.*exercise",
            r"convertible",
            r"equity offering"
        ],
        "description": "Potential or actual significant increase in shares outstanding"
    },
    "litigation": {
        "category": Category.LEGAL,
        "signal_type": SignalType.YELLOW_FLAG,
        "risk_level": RiskLevel.MODERATE,
        "title": "Material Litigation",
        "patterns": [
            r"class action",
            r"securities litigation",
            r"material litigation",
            r"contingent liability",
            r"probable loss"
        ],
        "description": "Significant pending or threatened litigation"
    },
    
    # GREEN FLAGS
    "positive_cash_flow": {
        "category": Category.LIQUIDITY,
        "signal_type": SignalType.GREEN_FLAG,
        "risk_level": RiskLevel.MODERATE,
        "title": "Positive Operating Cash Flow",
        "patterns": [
            r"cash provided by operating",
            r"positive.*cash.*from operations",
            r"consecutive quarter.*positive.*cash"
        ],
        "description": "Company generating cash from operations"
    },
    "debt_reduction": {
        "category": Category.DEBT,
        "signal_type": SignalType.GREEN_FLAG,
        "risk_level": RiskLevel.MODERATE,
        "title": "Debt Reduction",
        "patterns": [
            r"repayment of.*debt",
            r"debt reduction",
            r"paid down.*debt",
            r"reduced.*borrowings",
            r"prepayment"
        ],
        "description": "Company actively reducing debt"
    },
    "same_store_growth": {
        "category": Category.OPERATIONS,
        "signal_type": SignalType.GREEN_FLAG,
        "risk_level": RiskLevel.MODERATE,
        "title": "Same-Store Sales Growth",
        "patterns": [
            r"same.store sales.*increas",
            r"comparable.*sales.*\d+\.?\d*%",
            r"comp.*sales.*growth"
        ],
        "description": "Organic growth at existing locations"
    },
    "covenant_compliance": {
        "category": Category.DEBT,
        "signal_type": SignalType.GREEN_FLAG,
        "risk_level": RiskLevel.LOW,
        "title": "Covenant Compliance",
        "patterns": [
            r"in compliance with.*covenant",
            r"compliance with.*requirements.*financing"
        ],
        "description": "Company in compliance with debt covenants"
    },
    "effective_controls": {
        "category": Category.GOVERNANCE,
        "signal_type": SignalType.GREEN_FLAG,
        "risk_level": RiskLevel.LOW,
        "title": "Effective Internal Controls",
        "patterns": [
            r"disclosure controls.*were effective",
            r"internal control.*was effective",
            r"no material weakness"
        ],
        "description": "No material weaknesses in internal controls"
    },
}


# =============================================================================
# SENTIMENT INDICATORS
# =============================================================================

SENTIMENT_PATTERNS = {
    SentimentLevel.VERY_CONFIDENT: [
        r"excited to announce",
        r"strong momentum",
        r"exceeded expectations",
        r"record results",
        r"well positioned for growth",
        r"confident in our ability",
        r"significant opportunity"
    ],
    SentimentLevel.CONFIDENT: [
        r"pleased with",
        r"on track",
        r"continue to execute",
        r"positive trends",
        r"good progress",
        r"demonstrated improvement",
        r"gaining traction"
    ],
    SentimentLevel.NEUTRAL: [
        r"as expected",
        r"in line with",
        r"consistent with",
        r"maintained",
        r"stable"
    ],
    SentimentLevel.CAUTIOUS: [
        r"challenging environment",
        r"headwinds",
        r"uncertainty",
        r"pressure on",
        r"monitoring closely",
        r"taking steps to address"
    ],
    SentimentLevel.DEFENSIVE: [
        r"despite challenges",
        r"factors beyond our control",
        r"market conditions",
        r"macroeconomic pressures",
        r"unprecedented",
        r"adjusted basis"
    ],
    SentimentLevel.ALARMING: [
        r"substantial doubt",
        r"going concern",
        r"exploring strategic alternatives",
        r"evaluating all options",
        r"liquidity constraints",
        r"material uncertainty"
    ],
}


# =============================================================================
# ANALYZER CLASS
# =============================================================================

class SECFilingAnalyzer:
    """Main analyzer class for SEC filings"""
    
    def __init__(self, use_llm: bool = False):
        self.flag_rules = FLAG_RULES
        self.sentiment_patterns = SENTIMENT_PATTERNS
        self.use_llm = use_llm
        self._llm_module = None
        
        if use_llm:
            try:
                import llm_analysis
                if llm_analysis.is_llm_available():
                    self._llm_module = llm_analysis
                    logger.info("[Analyzer] LLM mode ACTIVE — using Claude API")
                else:
                    logger.info("[Analyzer] LLM requested but API key not set — falling back to regex")
                    self.use_llm = False
            except ImportError:
                logger.info("[Analyzer] LLM module not found — falling back to regex")
                self.use_llm = False
    
    def analyze_filing(
        self,
        ticker: str,
        company_name: str,
        current_text: str,
        prior_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze SEC filing and return comprehensive results.
        Uses LLM analysis when available, otherwise falls back to regex.
        """
        
        # Normalize text for regex matching
        current_text_lower = current_text.lower()
        prior_text_lower = prior_text.lower() if prior_text else None
        
        # ---- FLAG DETECTION ----
        regex_flags = self._detect_flags(current_text_lower, prior_text_lower, current_text)
        
        llm_flags = []
        llm_flag_meta = {}
        if self.use_llm and self._llm_module:
            logger.info("[Analyzer] Running LLM flag analysis...")
            llm_result = self._llm_module.llm_analyze_flags(current_text, prior_text)
            if llm_result:
                llm_flags = llm_result.get("flags", [])
                llm_flag_meta = {
                    "resolved_risks": llm_result.get("resolved_risks", []),
                    "llm_change_summary": llm_result.get("change_summary", {}),
                    "overall_assessment": llm_result.get("overall_assessment", ""),
                    "risks_checked_not_found": llm_result.get("risks_checked_not_found", []),
                }
                logger.info("[Analyzer] LLM found %d flags", len(llm_flags))
        
        # Merge flags: LLM flags take priority, regex fills gaps
        flags = self._merge_flags(regex_flags, llm_flags)
        
        # ---- SENTIMENT ANALYSIS ----
        regex_sentiment = self._analyze_sentiment(current_text_lower, prior_text_lower)
        
        llm_sentiment = []
        llm_sentiment_meta = {}
        if self.use_llm and self._llm_module:
            logger.info("[Analyzer] Running LLM sentiment analysis...")
            llm_sent_result = self._llm_module.llm_analyze_sentiment(current_text, prior_text)
            if llm_sent_result:
                llm_sentiment = llm_sent_result.get("sentiment_results", [])
                llm_sentiment_meta = llm_sent_result.get("sentiment_meta", {})
                logger.info("[Analyzer] LLM returned %d sentiment categories", len(llm_sentiment))
        
        # Use LLM sentiment if available (5 categories), else regex (1 category)
        if llm_sentiment:
            sentiment_results = llm_sentiment
        else:
            sentiment_results = regex_sentiment
        
        # ---- METRICS ----
        metrics = self._extract_metrics(current_text)
        
        # ---- SCORING ----
        flag_impact = sum(f['score_impact'] for f in flags)
        sentiment_impact = sum(s['score_impact'] for s in sentiment_results)
        
        base_score = 50
        final_score = max(0, min(100, base_score + flag_impact + sentiment_impact))
        
        # Determine risk rating
        if final_score >= 70:
            risk_rating = "LOW RISK"
        elif final_score >= 55:
            risk_rating = "MODERATE RISK"
        elif final_score >= 40:
            risk_rating = "ELEVATED RISK"
        elif final_score >= 25:
            risk_rating = "HIGH RISK"
        else:
            risk_rating = "CRITICAL RISK"
        
        # Determine sentiment trajectory
        if llm_sentiment_meta.get("overall_trajectory"):
            sentiment_trajectory = llm_sentiment_meta["overall_trajectory"]
        else:
            improving = sum(1 for s in sentiment_results if s.get('change_direction') == 'improving')
            worsening = sum(1 for s in sentiment_results if s.get('change_direction') == 'worsening')
            if improving > worsening:
                sentiment_trajectory = "improving"
            elif worsening > improving:
                sentiment_trajectory = "worsening"
            else:
                sentiment_trajectory = "stable"
        
        # Generate summaries
        red_flags = [f for f in flags if f['signal_type'] == 'red_flag']
        yellow_flags = [f for f in flags if f['signal_type'] == 'yellow_flag']
        green_flags = [f for f in flags if f['signal_type'] == 'green_flag']
        
        key_concerns = [f['title'] for f in red_flags + yellow_flags 
                        if f['risk_level'] in ['CRITICAL', 'HIGH', 'MODERATE']][:5]
        key_positives = [f['title'] for f in green_flags][:5]
        
        # Build results
        results = {
            "ticker": ticker,
            "company_name": company_name,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_mode": "llm" if (self.use_llm and self._llm_module) else "regex",
            "final_score": final_score,
            "risk_rating": risk_rating,
            "sentiment_trajectory": sentiment_trajectory,
            "score_breakdown": {
                "base_score": base_score,
                "flag_impact": flag_impact,
                "sentiment_impact": sentiment_impact,
                "red_flag_count": len(red_flags),
                "yellow_flag_count": len(yellow_flags),
                "green_flag_count": len(green_flags)
            },
            "key_concerns": key_concerns,
            "key_positives": key_positives,
            "flags": flags,
            "sentiment_analysis": sentiment_results,
            "sentiment_meta": llm_sentiment_meta,
            "metrics": metrics,
            "change_summary": self._summarize_changes(flags),
        }
        
        # Add LLM-only metadata if available
        if llm_flag_meta:
            results["llm_assessment"] = llm_flag_meta.get("overall_assessment", "")
            results["resolved_risks"] = llm_flag_meta.get("resolved_risks", [])
            results["risks_checked_not_found"] = llm_flag_meta.get("risks_checked_not_found", [])
        
        return results
    
    def _merge_flags(
        self,
        regex_flags: List[Dict[str, Any]],
        llm_flags: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge regex and LLM flags. LLM flags take priority when they
        cover the same category+signal_type. Regex fills in anything
        the LLM missed.
        """
        if not llm_flags:
            return regex_flags
        
        if not regex_flags:
            return llm_flags
        
        # Build a set of (category, signal_type) pairs covered by LLM
        llm_covered = set()
        for f in llm_flags:
            llm_covered.add((f.get("category", ""), f.get("title", "").lower()))
        
        # Keep all LLM flags, add regex flags only if not duplicated
        merged = list(llm_flags)
        for rf in regex_flags:
            # Check if this regex flag overlaps with any LLM flag
            title_lower = rf.get("title", "").lower()
            cat = rf.get("category", "")
            
            # Simple overlap check: same category or similar title words
            is_duplicate = False
            for lf in llm_flags:
                lf_title = lf.get("title", "").lower()
                lf_cat = lf.get("category", "")
                # Same category and significant word overlap
                if cat == lf_cat:
                    title_words = set(title_lower.split())
                    lf_words = set(lf_title.split())
                    if len(title_words & lf_words) >= 2:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                rf["source"] = "regex"
                merged.append(rf)
        
        return sorted(merged, key=lambda x: abs(x.get("score_impact", 0)), reverse=True)
    
    def _detect_flags(
        self,
        current_text: str,
        prior_text: Optional[str],
        original_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Detect flags in filing text. original_text preserves casing for evidence."""
        # Use original_text for evidence extraction to preserve casing
        evidence_source = original_text if original_text else current_text
        flags = []

        # Safe-harbor / forward-looking disclaimer pattern to skip matches within
        _safe_harbor_re = re.compile(
            r'(?:forward[- ]looking\s+statements?|safe[- ]harbor|cautionary\s+(?:note|statement))',
            re.IGNORECASE
        )

        for rule_id, rule in self.flag_rules.items():
            # Check if any pattern matches
            matched = False
            evidence = ""

            for pattern in rule['patterns']:
                match = re.search(pattern, current_text, re.IGNORECASE)
                if match:
                    # Check if match is inside a safe-harbor disclaimer section
                    context_start = max(0, match.start() - 500)
                    context_before = current_text[context_start:match.start()]
                    if _safe_harbor_re.search(context_before):
                        continue  # Skip matches within safe-harbor boilerplate
                    matched = True
                    # Extract surrounding context as evidence from original (cased) text
                    start = max(0, match.start() - 100)
                    end = min(len(evidence_source), match.end() + 100)
                    evidence = evidence_source[start:end].strip()
                    break
            
            if matched:
                # Determine change type
                change_type = ChangeType.FIRST_FILING
                if prior_text:
                    prior_matched = any(
                        re.search(p, prior_text, re.IGNORECASE) 
                        for p in rule['patterns']
                    )
                    if prior_matched:
                        change_type = ChangeType.UNCHANGED
                    else:
                        change_type = ChangeType.NEW
                
                # Calculate impact
                base_impact = rule['risk_level'].value * 10
                category_weight = CATEGORY_WEIGHTS.get(rule['category'], 1.0)
                change_mult = CHANGE_MULTIPLIERS.get(change_type, 1.0)
                
                if rule['signal_type'] == SignalType.RED_FLAG:
                    signal_mult = -1
                elif rule['signal_type'] == SignalType.YELLOW_FLAG:
                    signal_mult = -0.5
                elif rule['signal_type'] == SignalType.GREEN_FLAG:
                    signal_mult = 0.5
                    # For green flags, NEW is good
                    if change_type == ChangeType.NEW:
                        change_mult = 1.5
                else:
                    signal_mult = 0
                
                score_impact = int(base_impact * category_weight * signal_mult * change_mult)
                
                flags.append({
                    "rule_id": rule_id,
                    "title": rule['title'],
                    "category": rule['category'].value,
                    "signal_type": rule['signal_type'].value,
                    "risk_level": rule['risk_level'].name,
                    "change_type": change_type.value,
                    "score_impact": score_impact,
                    "description": rule['description'],
                    "evidence": evidence[:300] + "..." if len(evidence) > 300 else evidence
                })
        
        return sorted(flags, key=lambda x: abs(x['score_impact']), reverse=True)
    
    def _analyze_sentiment(
        self, 
        current_text: str, 
        prior_text: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Analyze sentiment in filing text"""
        results = []
        
        # Score current sentiment
        current_scores = {level: 0 for level in SentimentLevel}
        for level, patterns in self.sentiment_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, current_text, re.IGNORECASE)
                current_scores[level] += len(matches)
        
        # Determine current sentiment level; break ties toward neutral (CAUTIOUS)
        max_score = max(current_scores.values())
        if max_score > 0:
            tied_levels = [k for k, v in current_scores.items() if v == max_score]
            if len(tied_levels) > 1:
                # Break tie toward neutral (CAUTIOUS=2, NEUTRAL=3)
                current_level = min(tied_levels, key=lambda k: abs(k.value - SentimentLevel.CAUTIOUS.value))
            else:
                current_level = tied_levels[0]
        else:
            current_level = SentimentLevel.NEUTRAL
        
        # Score prior sentiment if available
        prior_level = None
        if prior_text:
            prior_scores = {level: 0 for level in SentimentLevel}
            for level, patterns in self.sentiment_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, prior_text, re.IGNORECASE)
                    prior_scores[level] += len(matches)
            
            max_prior = max(prior_scores.values())
            if max_prior > 0:
                prior_level = max(prior_scores.keys(), key=lambda k: prior_scores[k])
            else:
                prior_level = SentimentLevel.NEUTRAL
        
        # Determine change direction
        if prior_level is None:
            change_direction = "first_filing"
        elif current_level.value > prior_level.value:
            change_direction = "improving"
        elif current_level.value < prior_level.value:
            change_direction = "worsening"
        else:
            change_direction = "stable"
        
        # Calculate sentiment impact
        base_impacts = {
            SentimentLevel.VERY_CONFIDENT: 10,
            SentimentLevel.CONFIDENT: 5,
            SentimentLevel.NEUTRAL: 0,
            SentimentLevel.CAUTIOUS: -5,
            SentimentLevel.DEFENSIVE: -10,
            SentimentLevel.ALARMING: -20
        }
        base = base_impacts.get(current_level, 0)
        
        if change_direction == "worsening":
            multiplier = 2.0
        elif change_direction == "improving":
            multiplier = 1.5
        else:
            multiplier = 1.0
        
        if base < 0:
            score_impact = int(base * multiplier)
        else:
            score_impact = int(base * (multiplier if change_direction == "improving" else 1.0))
        
        # Find key phrases
        key_phrases = []
        for pattern in self.sentiment_patterns.get(current_level, []):
            matches = re.findall(pattern, current_text, re.IGNORECASE)
            key_phrases.extend(matches[:2])
        
        results.append({
            "category": "overall_tone",
            "current_level": current_level.name,
            "prior_level": prior_level.name if prior_level else None,
            "change_direction": change_direction,
            "score_impact": score_impact,
            "key_phrases": key_phrases[:5]
        })
        
        return results
    
    def _extract_metrics(self, text: str) -> Dict[str, Any]:
        """Extract financial metrics from filing text"""
        metrics = {}
        
        # Try to extract common metrics using patterns
        patterns = {
            "revenue": r"(?:total\s+)?revenue[s]?\s*(?:of\s*)?\$?([\d,]+\.?\d*)\s*(?:million|M)",
            "net_income": r"net\s+(?:income|loss)\s*(?:of\s*)?\$?([\d,]+\.?\d*)\s*(?:million|M)",
            "cash": r"cash\s+and\s+(?:cash\s+)?equivalents?\s*(?:of\s*)?\$?([\d,]+\.?\d*)\s*(?:million|M)",
            "total_debt": r"(?:total\s+)?(?:long-term\s+)?debt\s*(?:of\s*)?\$?([\d,]+\.?\d*)\s*(?:million|M)",
            "operating_income": r"operating\s+(?:income|loss)\s*(?:of\s*)?\$?([\d,]+\.?\d*)\s*(?:million|M)"
        }
        
        for metric_name, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1).replace(',', ''))
                    metrics[metric_name] = value
                except (ValueError, TypeError):
                    pass
        
        return metrics
    
    def _summarize_changes(self, flags: List[Dict]) -> Dict[str, int]:
        """Summarize change types across all flags"""
        summary = {
            "new": 0,
            "worsening": 0,
            "improving": 0,
            "unchanged": 0,
            "first_filing": 0
        }
        
        for flag in flags:
            change_type = flag.get('change_type', 'first_filing')
            if change_type in summary:
                summary[change_type] += 1
        
        return summary
