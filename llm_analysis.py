"""
LLM Analysis Module - Dual Model Support (OpenAI + Anthropic)
==============================================================
Two-tier analysis system for finding hidden gem microcap stocks:
  Tier 1: GPT-4o-mini (cheap bulk screening)
  Tier 2: Claude Haiku (deep analysis on top candidates)

Focus: Finding IMPROVING companies with authentic management tone,
not just avoiding disasters.
"""

import os
import json
import re
import logging
import copy
import time as _time
import threading
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODELS = {
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
        "max_context": 128000,
        "tier": 1,
    },
    "claude-haiku": {
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
        "input_cost_per_1m": 1.00,
        "output_cost_per_1m": 5.00,
        "max_context": 200000,
        "tier": 2,
    },
    "claude-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-5-20250929",
        "input_cost_per_1m": 3.00,
        "output_cost_per_1m": 15.00,
        "max_context": 200000,
        "tier": 2,
    },
    "claude-opus": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-6",
        "input_cost_per_1m": 5.00,
        "output_cost_per_1m": 25.00,
        "max_context": 200000,
        "tier": 3,
        "supports_thinking": True,
        "supports_effort": True,
    },
}

DEFAULT_TIER1_MODEL = "gpt-4o-mini"
DEFAULT_TIER2_MODEL = "claude-haiku"

# Effort level presets: controls intelligence vs speed/cost tradeoff for Opus
EFFORT_LEVELS = {
    "low": {"description": "Fast, cheap — simple filings", "thinking_budget": 1024},
    "medium": {"description": "Balanced — standard analysis", "thinking_budget": 2048},
    "high": {"description": "Thorough — conflicting signals", "thinking_budget": 8192},
    "max": {"description": "Maximum depth — hardest cases", "thinking_budget": 16384},
}


# ============================================================================
# API CLIENTS
# ============================================================================

# Cached API clients (reuse connections)
_openai_client = None
_anthropic_client = None

MAX_LLM_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds

# Token usage tracking for actual cost computation (#23)
# Thread safety: _cumulative_usage protected by _usage_lock
_last_token_usage = {}  # Updated after each LLM call
_cumulative_usage = {"total_input_tokens": 0, "total_output_tokens": 0,
                     "total_cost_usd": 0.0, "calls": 0, "by_model": {}}
_usage_lock = threading.Lock()


def get_last_token_usage() -> Dict:
    """Get token usage from the most recent LLM call."""
    return dict(_last_token_usage)


def get_cumulative_usage() -> Dict:
    """Get cumulative token usage and cost across all calls."""
    return copy.deepcopy(_cumulative_usage)


def _track_usage(model_name: str, input_tokens: int, output_tokens: int):
    """Track cumulative usage and compute actual cost."""
    config = None
    for name, cfg in MODELS.items():
        if cfg.get("model_id") == model_name or name == model_name:
            config = cfg
            break

    cost = 0.0
    if config:
        cost = (input_tokens * config.get("input_cost_per_1m", 0) / 1_000_000 +
                output_tokens * config.get("output_cost_per_1m", 0) / 1_000_000)

    with _usage_lock:
        _cumulative_usage["total_input_tokens"] += input_tokens
        _cumulative_usage["total_output_tokens"] += output_tokens
        _cumulative_usage["total_cost_usd"] += cost
        _cumulative_usage["calls"] += 1

        if model_name not in _cumulative_usage["by_model"]:
            _cumulative_usage["by_model"][model_name] = {
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "calls": 0
            }
        m = _cumulative_usage["by_model"][model_name]
        m["input_tokens"] += input_tokens
        m["output_tokens"] += output_tokens
        m["cost_usd"] += cost
        m["calls"] += 1

    return cost


def reset_cumulative_usage():
    """Reset cumulative usage counters (e.g., at start of analysis run)."""
    _cumulative_usage.update({
        "total_input_tokens": 0, "total_output_tokens": 0,
        "total_cost_usd": 0.0, "calls": 0, "by_model": {},
    })


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _call_openai(prompt: str, system: str, model: str = "gpt-4o-mini",
                 max_tokens: int = 2000) -> Optional[str]:
    """Call OpenAI API with retry on transient errors."""
    global _last_token_usage
    client = _get_openai_client()

    last_error = None
    for attempt in range(MAX_LLM_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            # Track actual token usage (#23)
            if response.usage:
                _last_token_usage = {
                    "model": model,
                    "provider": "openai",
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }
                _track_usage(model, response.usage.prompt_tokens,
                             response.usage.completion_tokens)
            return response.choices[0].message.content
        except (ConnectionError, TimeoutError, OSError) as e:
            last_error = e
            wait = RETRY_BACKOFF_BASE * (2 ** attempt)
            logger.warning(f"OpenAI retry {attempt+1}/{MAX_LLM_RETRIES} after {wait}s: {e}")
            _time.sleep(wait)
            continue
        except Exception as e:
            last_error = e
            # Check for retryable HTTP status codes via exception type
            try:
                from openai import RateLimitError, APIStatusError, APITimeoutError
                if isinstance(e, (RateLimitError, APITimeoutError)):
                    wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(f"OpenAI retry {attempt+1}/{MAX_LLM_RETRIES} after {wait}s: {e}")
                    _time.sleep(wait)
                    continue
                if isinstance(e, APIStatusError) and getattr(e, 'status_code', 0) in (500, 502, 503, 529):
                    wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(f"OpenAI retry {attempt+1}/{MAX_LLM_RETRIES} after {wait}s: {e}")
                    _time.sleep(wait)
                    continue
            except ImportError:
                pass
            # Non-retryable error
            break

    logger.error(f"OpenAI API error after {MAX_LLM_RETRIES} attempts: {last_error}")
    raise RuntimeError(f"OpenAI API call failed: {last_error}")


def _call_anthropic(prompt: str, system: str, model: str = "claude-haiku-4-5-20251001",
                    max_tokens: int = 2000, thinking_budget: int = 0,
                    effort: str = "") -> Optional[str]:
    """Call Anthropic Claude API with retry, optional extended thinking and effort levels."""
    global _last_token_usage
    client = _get_anthropic_client()

    # Build request kwargs
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }

    # Add extended thinking if budget > 0
    if thinking_budget > 0:
        kwargs["temperature"] = 1  # Required for extended thinking
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
        # max_tokens must account for thinking + response tokens
        kwargs["max_tokens"] = max_tokens + thinking_budget
    else:
        # Use low temperature for deterministic JSON output (Fix #7)
        kwargs["temperature"] = 0.3

    # Add effort level if specified (Opus 4.6+ feature)
    if effort and effort in ("low", "medium", "high", "max"):
        kwargs["effort"] = effort

    last_error = None
    for attempt in range(MAX_LLM_RETRIES):
        try:
            message = client.messages.create(**kwargs)

            # Extract text from response, skipping thinking blocks
            text_parts = []
            for block in message.content:
                if hasattr(block, 'type'):
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "thinking":
                        thinking_text = getattr(block, 'thinking', '')
                        if thinking_text:
                            logger.debug(f"Extended thinking ({len(thinking_text)} chars): "
                                         f"{thinking_text[:200]}...")
                elif hasattr(block, 'text'):
                    text_parts.append(block.text)

            result = "\n".join(text_parts)

            # Log token usage
            if hasattr(message, 'usage'):
                usage = message.usage
                input_t = getattr(usage, 'input_tokens', 0)
                output_t = getattr(usage, 'output_tokens', 0)
                _last_token_usage = {
                    "model": model,
                    "provider": "anthropic",
                    "input_tokens": input_t,
                    "output_tokens": output_t,
                }
                _track_usage(model, input_t, output_t)
                logger.info(f"Anthropic {model}: {input_t} input + {output_t} output tokens"
                            f"{f' (thinking budget: {thinking_budget})' if thinking_budget else ''}")

            return result

        except (ConnectionError, TimeoutError, OSError) as e:
            last_error = e
            wait = RETRY_BACKOFF_BASE * (2 ** attempt)
            logger.warning(f"Anthropic retry {attempt+1}/{MAX_LLM_RETRIES} after {wait}s: {e}")
            _time.sleep(wait)
            continue
        except Exception as e:
            last_error = e
            # Check for retryable errors via exception type
            try:
                from anthropic import RateLimitError, APIStatusError, APITimeoutError
                if isinstance(e, (RateLimitError, APITimeoutError)):
                    wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(f"Anthropic retry {attempt+1}/{MAX_LLM_RETRIES} after {wait}s: {e}")
                    _time.sleep(wait)
                    continue
                if isinstance(e, APIStatusError) and getattr(e, 'status_code', 0) in (500, 502, 503, 529):
                    wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(f"Anthropic retry {attempt+1}/{MAX_LLM_RETRIES} after {wait}s: {e}")
                    _time.sleep(wait)
                    continue
            except ImportError:
                pass
            break

    logger.error(f"Anthropic API error after {MAX_LLM_RETRIES} attempts (model={model}): {last_error}")
    raise RuntimeError(f"Anthropic API call failed: {last_error}")


def _call_llm(prompt: str, system: str, model_name: str = "gpt-4o-mini",
              max_tokens: int = 2000, thinking_budget: int = 0,
              effort: str = "", fallback: bool = True) -> Optional[str]:
    """Route to appropriate API based on model. Falls back to alternative on failure (#15)."""
    config = MODELS.get(model_name)
    if not config:
        logger.error(f"Unknown model: {model_name}")
        return None

    # Define fallback chains per tier
    FALLBACK_CHAINS = {
        "gpt-4o-mini": ["claude-haiku"],
        "claude-haiku": ["gpt-4o-mini", "claude-sonnet"],
        "claude-sonnet": ["claude-haiku", "gpt-4o-mini"],
        "claude-opus": ["claude-sonnet", "claude-haiku"],
    }

    try:
        if config["provider"] == "openai":
            return _call_openai(prompt, system, config["model_id"], max_tokens)
        elif config["provider"] == "anthropic":
            return _call_anthropic(prompt, system, config["model_id"], max_tokens,
                                   thinking_budget=thinking_budget, effort=effort)
    except Exception as primary_error:
        if not fallback:
            raise
        # Try fallback models
        fallbacks = FALLBACK_CHAINS.get(model_name, [])
        for fb_model in fallbacks:
            fb_config = MODELS.get(fb_model)
            if not fb_config:
                continue
            logger.warning(
                f"Primary model {model_name} failed ({type(primary_error).__name__}), "
                f"falling back to {fb_model}"
            )
            try:
                if fb_config["provider"] == "openai":
                    return _call_openai(prompt, system, fb_config["model_id"], max_tokens)
                elif fb_config["provider"] == "anthropic":
                    return _call_anthropic(prompt, system, fb_config["model_id"], max_tokens)
            except Exception as fb_error:
                logger.warning(f"Fallback {fb_model} also failed: {fb_error}")
                continue
        # All fallbacks failed
        logger.error(f"All models failed for request (primary: {model_name})")
        raise primary_error


def _parse_json_response(text: str) -> Optional[Dict]:
    """Parse JSON from LLM response, handling markdown fences and truncation."""
    if not text:
        return None

    cleaned = text.strip()

    # Strip ALL markdown code fences (```json, ```, possibly multiple)
    cleaned = re.sub(r"```(?:json|JSON)?\s*\n?", "", cleaned)
    cleaned = cleaned.strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Extract the outermost { ... } block
    # Find the first { and try to parse from there
    first_brace = cleaned.find('{')
    if first_brace == -1:
        logger.warning(f"No JSON object found in response: {text[:200]}...")
        return None

    json_str = cleaned[first_brace:]

    # Try parsing as-is
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Try first { to last } (handles trailing text after valid JSON)
    last_brace = json_str.rfind('}')
    if last_brace > 0:
        try:
            return json.loads(json_str[:last_brace + 1])
        except json.JSONDecodeError:
            pass

    # Response likely truncated at max_tokens — try to repair
    # Strategy: find the deepest valid JSON by progressively closing brackets
    # First, try trimming trailing incomplete content and closing brackets
    repaired = json_str.rstrip()

    # Remove trailing partial strings (cut off mid-value)
    # Look for last complete key-value or array element
    last_good = max(
        repaired.rfind('",'),
        repaired.rfind('"},'),
        repaired.rfind('}],'),
        repaired.rfind('"],'),
        repaired.rfind('" },'),
        repaired.rfind('" }'),
        repaired.rfind('"}'),
        repaired.rfind('"]'),
        repaired.rfind('true,'),
        repaired.rfind('false,'),
        repaired.rfind('null,'),
    )

    if last_good > 0:
        # Find the end of the last complete value
        # Move past the comma if there is one
        repaired = repaired[:last_good]
        # Find what was the last complete token — only add closing quote if not already quoted
        if repaired.endswith('"') and not repaired.endswith('""'):
            pass  # Already ends with a quote, no need to add another
        # Count open braces/brackets and close them
        open_braces = repaired.count('{') - repaired.count('}')
        open_brackets = repaired.count('[') - repaired.count(']')
        repaired += ']' * max(0, open_brackets)
        repaired += '}' * max(0, open_braces)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # More aggressive repair: find last valid } or ] and close from there
    max_attempts = 1000
    attempts = 0
    for trim_pos in range(len(json_str) - 1, max(len(json_str) // 2, 100), -1):
        attempts += 1
        if attempts > max_attempts:
            break
        ch = json_str[trim_pos]
        if ch in ('}', ']', '"'):
            candidate = json_str[:trim_pos + 1]
            open_braces = candidate.count('{') - candidate.count('}')
            open_brackets = candidate.count('[') - candidate.count(']')
            if open_braces >= 0 and open_brackets >= 0:
                candidate += ']' * open_brackets + '}' * open_braces
                try:
                    result = json.loads(candidate)
                    logger.info(f"Repaired truncated JSON (trimmed {len(json_str) - trim_pos - 1} chars)")
                    return result
                except json.JSONDecodeError:
                    continue

    logger.warning(f"JSON parse failed (response len={len(text)}). Preview: {text[:200]}...")
    return None


def _truncate_text(text: str, max_chars: int = 100000) -> str:
    """Truncate text while preserving key sections."""
    if len(text) <= max_chars:
        return text
    # Keep beginning and end (often most important)
    half = max_chars // 2
    return text[:half] + "\n\n[...CONTENT TRUNCATED FOR LENGTH...]\n\n" + text[-half:]


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

# NOTE: Filing text is user-controlled content from SEC EDGAR. Prompt injection risk documented.
SYSTEM_PROMPT_GEM_FINDER = """You are an expert microcap stock analyst searching for HIDDEN GEMS -
undervalued companies with improving fundamentals and authentic, confident management.

Your job is NOT just to avoid disasters. You're looking for:
- Companies quietly executing well
- Management that's confident but not promotional  
- Improving trends that the market may be missing
- Understated positives and conservative guidance

You analyze SEC filings with a focus on TONE and TRAJECTORY.
CRITICAL: Respond with RAW JSON only. No markdown, no ```json fences, no backticks."""


SYSTEM_PROMPT_DEEP_ANALYSIS = """You are a forensic financial analyst specializing in microcap stocks.
You perform deep qualitative analysis of SEC filings, grounded in QUANTITATIVE linguistic forensics.

You don't just summarize filings. You:
- INTERPRET concrete metrics (hedging ratios, confidence scores, specificity)
- EXPLAIN what filing-over-filing changes mean for investors
- INTERPRET insider buying/selling patterns and what they signal
- ANALYZE financial inflection points (revenue acceleration, profitability crossings, margin trends)
- IDENTIFY contradictions between management narrative and forensic/financial signals
- FIND what management is trying to hide or downplay

When forensic diff data shows risk factors were removed, that could mean the risk was
resolved (bullish) or that they're hiding it (bearish) - your job is to determine which.

When hedging increases but management claims confidence, that's a contradiction worth flagging.
When specificity decreases, ask why they stopped giving concrete numbers.

CRITICAL FORMATTING RULES:
- Respond with RAW JSON only. No markdown, no ```json fences, no backticks.
- Start your response with { and end with }
- Keep evidence arrays to 2 items max. Keep assessments to 2 sentences.
- Keep hidden_strengths and subtle_red_flags to 3 items max each.
- Be concise — quality over quantity in every field."""


# ============================================================================
# TIER 1: SCREENING PROMPT (GPT-4o-mini)
# ============================================================================

TIER1_SCREENING_PROMPT = """Analyze this SEC filing to identify if this company could be a hidden gem.

Score each dimension from 1-10:

TONE DIMENSIONS:
1. confidence (1=desperate/promotional, 5=neutral, 10=quietly assured)
2. transparency (1=vague/evasive, 5=standard, 10=unusually specific/honest)
3. consistency (1=changing story, 5=neutral, 10=same strategy, steady execution)
4. operational_focus (1=excuses/blame, 5=neutral, 10=execution-focused, metrics-driven)
5. capital_discipline (1=dilutive/wasteful, 5=neutral, 10=ROIC-focused, smart buybacks)

FINANCIAL SIGNALS:
6. liquidity_health (1=going concern, 5=adequate, 10=strong/improving)
7. growth_quality (1=declining, 5=flat, 10=organic growth with good unit economics)
8. profitability_trend (1=worsening losses, 5=stable, 10=margin expansion)

For each dimension provide:
- score (1-10)
- evidence (one short quote, max 12 words)

Also identify:
- top_3_positives: Best things about this company (be specific)
- top_3_concerns: Biggest risks (be specific)
- composite_score: Overall score from 1 to 100 (NOT 1-10. Calculate as average of all dimension scores multiplied by 10)
- gem_potential: "high", "medium", "low", or "avoid"
- gem_reasoning: One sentence explaining gem potential

Respond with JSON:
{{
  "ticker": "{ticker}",
  "dimensions": {{
    "confidence": {{"score": N, "evidence": "..."}},
    "transparency": {{"score": N, "evidence": "..."}},
    "consistency": {{"score": N, "evidence": "..."}},
    "operational_focus": {{"score": N, "evidence": "..."}},
    "capital_discipline": {{"score": N, "evidence": "..."}},
    "liquidity_health": {{"score": N, "evidence": "..."}},
    "growth_quality": {{"score": N, "evidence": "..."}},
    "profitability_trend": {{"score": N, "evidence": "..."}}
  }},
  "composite_score": 1-100,
  "top_3_positives": ["...", "...", "..."],
  "top_3_concerns": ["...", "...", "..."],
  "gem_potential": "high|medium|low|avoid",
  "gem_reasoning": "..."
}}

FILING ({form_type}) for {ticker} - {company_name}:
{filing_text}"""


# ============================================================================
# TIER 1: OWNERSHIP FILING PROMPT (SC 13D, SC 13D/A, SC 13G, SC 13G/A)
# ============================================================================

TIER1_OWNERSHIP_PROMPT = """Analyze this {form_type} ownership filing for turnaround catalyst potential.

FILING TYPE CONTEXT:
- SC 13D / SCHEDULE 13D: ACTIVIST filing — investor acquired >5% with intent to influence the company.
  These are HIGH-VALUE signals. Look for board seats, management changes, strategic proposals.
- SC 13G / SCHEDULE 13G: PASSIVE filing — investor acquired >5% as a passive holder (no intent to influence).
  These are LOWER-VALUE signals but still indicate institutional interest and potential future activism.
- /A suffix means AMENDMENT — look for changes in stake size, purpose, or plans since original filing.

THIS FILING IS: {form_type}

Score each dimension from 1-10:

OWNERSHIP SIGNAL DIMENSIONS:
1. confidence (1=vague/passive/routine, 5=neutral, 10=clear activist plan with specific demands)
2. transparency (1=boilerplate/no detail, 5=standard, 10=specific proposals, detailed analysis)
3. consistency (1=contradictory/unclear intent, 5=neutral, 10=coherent strategy with timeline)
4. operational_focus (1=purely financial/passive, 5=mixed, 10=focused on operational improvements)
5. capital_discipline (1=asset stripping, 5=neutral, 10=value-creating proposals — board seats, cost cuts, spinoffs)

CATALYST STRENGTH:
6. liquidity_health (1=filer is distressed, 5=neutral, 10=well-funded with ability to increase stake)
7. growth_quality (1=no plan for value creation, 5=generic, 10=specific operational turnaround plan)
8. profitability_trend (1=filer intends to liquidate, 5=neutral, 10=filer has track record of value creation)

KEY QUESTIONS:
- What is the stated "Purpose of Transaction"?
- Is this an activist with specific demands or a passive >5% holder?
- If SC 13G: is there any language suggesting future activism or conversion to 13D?
- Does the filer's track record suggest they create value?
- For amendments (/A): what changed vs. the prior filing? (stake size, purpose, plans)

For each dimension provide:
- score (1-10)
- evidence (one short quote, max 12 words)

Also identify:
- top_3_positives: Best catalyst signals from this filing (be specific)
- top_3_concerns: Biggest risks (entrenchment, litigation, distraction, or just passive with no plan)
- composite_score: Overall score from 1 to 100 (NOT 1-10. Calculate as average of all dimension scores multiplied by 10)
- gem_potential: "high" if activist with clear value plan, "medium" if active but vague, "low" if passive/routine, "avoid" if harmful
- gem_reasoning: One sentence explaining turnaround catalyst potential

Respond with JSON:
{{
  "ticker": "{ticker}",
  "dimensions": {{
    "confidence": {{"score": N, "evidence": "..."}},
    "transparency": {{"score": N, "evidence": "..."}},
    "consistency": {{"score": N, "evidence": "..."}},
    "operational_focus": {{"score": N, "evidence": "..."}},
    "capital_discipline": {{"score": N, "evidence": "..."}},
    "liquidity_health": {{"score": N, "evidence": "..."}},
    "growth_quality": {{"score": N, "evidence": "..."}},
    "profitability_trend": {{"score": N, "evidence": "..."}}
  }},
  "composite_score": 1-100,
  "top_3_positives": ["...", "...", "..."],
  "top_3_concerns": ["...", "...", "..."],
  "gem_potential": "high|medium|low|avoid",
  "gem_reasoning": "..."
}}

FILING ({form_type}) for {ticker} - {company_name}:
{filing_text}"""


# ============================================================================
# TIER 1: 8-K EVENT FILING PROMPT (#2)
# ============================================================================

TIER1_8K_EVENT_PROMPT = """Analyze this 8-K event filing for IMMEDIATE CATALYST potential.

8-K filings are the most time-sensitive SEC filings — they report material events within days.
This filing may contain one or more of: management changes, M&A activity, material agreements,
guidance revisions, restructuring, asset sales, going private, bankruptcy, or other events.

FOCUS: What is the EVENT and does it represent a catalyst for stock price movement?

Score each dimension from 1-10:

EVENT CATALYST DIMENSIONS:
1. confidence (1=routine/administrative, 5=neutral event, 10=transformative catalyst event)
2. transparency (1=vague disclosure, 5=standard, 10=detailed explanation of impact and rationale)
3. consistency (1=contradicts prior strategy, 5=neutral, 10=reinforces turnaround narrative)
4. operational_focus (1=financial distress event, 5=neutral, 10=operational improvement catalyst)
5. capital_discipline (1=dilutive event, 5=neutral, 10=value-creating event like buyback/sale)

EVENT IMPACT:
6. liquidity_health (1=funding crisis, 5=neutral, 10=strengthening balance sheet)
7. growth_quality (1=contract/downsizing, 5=neutral, 10=new revenue/partnership catalyst)
8. profitability_trend (1=charges/writedowns, 5=neutral, 10=margin improvement catalyst)

KEY QUESTIONS:
- What specific event(s) are being reported?
- Is this a positive catalyst (M&A, new management, raised guidance)?
- Or negative event (departure, litigation, going concern, covenant breach)?
- What is the timeline? Does this trigger near-term price action?
- Any strategic alternatives language? Board changes?

For each dimension provide:
- score (1-10)
- evidence (one short quote, max 12 words)

Also identify:
- top_3_positives: Best catalyst signals from this event
- top_3_concerns: Biggest risks from this event
- composite_score: Overall score from 1 to 100
- gem_potential: "high" if major positive catalyst, "medium" if mixed, "low" if routine, "avoid" if negative
- gem_reasoning: One sentence on event's catalyst potential
- event_type: Brief label for the event (e.g., "CEO change", "acquisition", "raised guidance")

Respond with JSON:
{{
  "ticker": "{ticker}",
  "dimensions": {{
    "confidence": {{"score": N, "evidence": "..."}},
    "transparency": {{"score": N, "evidence": "..."}},
    "consistency": {{"score": N, "evidence": "..."}},
    "operational_focus": {{"score": N, "evidence": "..."}},
    "capital_discipline": {{"score": N, "evidence": "..."}},
    "liquidity_health": {{"score": N, "evidence": "..."}},
    "growth_quality": {{"score": N, "evidence": "..."}},
    "profitability_trend": {{"score": N, "evidence": "..."}}
  }},
  "composite_score": 1-100,
  "top_3_positives": ["...", "...", "..."],
  "top_3_concerns": ["...", "...", "..."],
  "gem_potential": "high|medium|low|avoid",
  "gem_reasoning": "...",
  "event_type": "..."
}}

FILING ({form_type}) for {ticker} - {company_name}:
{filing_text}"""


# ============================================================================
# TIER 1: DEF 14A PROXY STATEMENT PROMPT (#6)
# ============================================================================

TIER1_DEF14A_PROMPT = """Analyze this DEF 14A proxy statement for GOVERNANCE CATALYST potential.

Proxy statements reveal executive compensation, board composition, shareholder proposals,
and governance structures. These can signal upcoming turnarounds, activist wins, or entrenched management.

FOCUS: Does the governance picture suggest a positive catalyst or entrenched mediocrity?

Score each dimension from 1-10:

GOVERNANCE SIGNAL DIMENSIONS:
1. confidence (1=defensive/entrenched, 5=neutral governance, 10=shareholder-friendly changes)
2. transparency (1=minimal disclosure, 5=standard, 10=unusually detailed/honest proxy)
3. consistency (1=contradictory comp vs performance, 5=neutral, 10=aligned incentives)
4. operational_focus (1=board captured by management, 5=mixed, 10=independent/activist-aligned board)
5. capital_discipline (1=excessive comp despite poor results, 5=neutral, 10=comp tied to value creation)

GOVERNANCE CATALYSTS:
6. liquidity_health (1=poison pill/anti-takeover, 5=neutral, 10=shareholder-friendly capital structure)
7. growth_quality (1=no shareholder proposals, 5=routine, 10=activist proposals with board seats)
8. profitability_trend (1=comp increasing while profits falling, 5=neutral, 10=comp restructured for performance)

KEY QUESTIONS:
- Are there shareholder proposals for board seats, executive compensation, or strategic review?
- Is executive pay aligned with company performance?
- Are there poison pills, classified boards, or other anti-takeover provisions?
- Any new board members from activist investors?
- Related party transactions or conflicts of interest?
- Golden parachutes or change-of-control provisions (M&A catalyst)?

For each dimension provide:
- score (1-10)
- evidence (one short quote, max 12 words)

Also identify:
- top_3_positives: Best governance catalyst signals
- top_3_concerns: Biggest governance risks (entrenchment, misalignment, self-dealing)
- composite_score: Overall score from 1 to 100
- gem_potential: "high" if governance changes signal catalyst, "medium" if mixed, "low" if routine, "avoid" if problematic
- gem_reasoning: One sentence on governance catalyst potential

Respond with JSON:
{{
  "ticker": "{ticker}",
  "dimensions": {{
    "confidence": {{"score": N, "evidence": "..."}},
    "transparency": {{"score": N, "evidence": "..."}},
    "consistency": {{"score": N, "evidence": "..."}},
    "operational_focus": {{"score": N, "evidence": "..."}},
    "capital_discipline": {{"score": N, "evidence": "..."}},
    "liquidity_health": {{"score": N, "evidence": "..."}},
    "growth_quality": {{"score": N, "evidence": "..."}},
    "profitability_trend": {{"score": N, "evidence": "..."}}
  }},
  "composite_score": 1-100,
  "top_3_positives": ["...", "...", "..."],
  "top_3_concerns": ["...", "...", "..."],
  "gem_potential": "high|medium|low|avoid",
  "gem_reasoning": "..."
}}

FILING ({form_type}) for {ticker} - {company_name}:
{filing_text}"""

TIER1_TRAJECTORY_PROMPT = """Compare these two SEC filings for {ticker} to assess IMPROVEMENT trajectory.

You're looking for companies getting BETTER - not just avoiding getting worse.

For each dimension, score BOTH filings (1-10) and determine trajectory:

DIMENSIONS:
1. confidence - Management's tone (promotional=bad, quietly assured=good)
2. transparency - Specificity and honesty
3. operational_focus - Execution vs excuses
4. capital_discipline - Smart capital allocation
5. liquidity_health - Cash and debt position
6. growth_quality - Revenue trend and quality
7. profitability_trend - Margin direction

TRAJECTORY SCORING:
- "strongly_improving" = +3 or more points
- "improving" = +1 to +2 points  
- "stable" = no change
- "declining" = -1 to -2 points
- "deteriorating" = -3 or more points

Look specifically for:
- Word substitutions ("will" → "may", "expect" → "hope")
- Metrics no longer highlighted
- New blame language
- OR the opposite: increasing confidence, better metrics, less hedging

Respond with JSON:
{{
  "ticker": "{ticker}",
  "prior_filing_date": "{prior_date}",
  "current_filing_date": "{current_date}",
  "dimensions": {{
    "confidence": {{
      "prior_score": N,
      "current_score": N,
      "trajectory": "strongly_improving|improving|stable|declining|deteriorating",
      "key_shift": "what changed (max 15 words)"
    }},
    ... (all 7 dimensions)
  }},
  "overall_trajectory": "strongly_improving|improving|stable|declining|deteriorating",
  "trajectory_score": N,
  "improvement_signals": ["specific positive change 1", "...", "..."],
  "warning_signals": ["specific negative change 1", "...", "..."],
  "notable_quote_current": "most telling quote from current filing",
  "notable_quote_prior": "comparable quote from prior filing",
  "gem_potential": "high|medium|low|avoid",
  "summary": "2-3 sentence trajectory summary"
}}

PRIOR FILING ({prior_form_type}, {prior_date}):
{prior_text}

CURRENT FILING ({current_form_type}, {current_date}):
{current_text}"""


# ============================================================================
# TIER 2: DEEP ANALYSIS PROMPT (Claude Haiku) - Forensic-Enhanced
# ============================================================================

TIER2_DEEP_ANALYSIS_PROMPT = """Perform deep qualitative analysis on this potential hidden gem.

This company scored well in initial screening. You have been provided with:
1. The SEC filing text
2. QUANTITATIVE forensic language metrics (hedging ratios, confidence patterns, etc.)
3. Filing-over-filing DIFF DATA showing what changed vs the prior filing (if available)
4. INSIDER & INSTITUTIONAL ACTIVITY data (Form 4 insider buys/sells, institutional filings)
5. FINANCIAL INFLECTION ANALYSIS from XBRL data (revenue trends, margin shifts, profitability crossings)
6. RESEARCH-BACKED FILING SIGNALS — quantitative scores from academic research on SEC filing predictive power
7. TURNAROUND CATALYST EXTRACTION — algorithmically detected operational catalysts from the filing text

YOUR JOB IS NOT TO SUMMARIZE. Your job is to INTERPRET all signals holistically and explain
what they mean for this company's investment potential. Focus on:

FORENSIC INTERPRETATION:
- What does the confidence-to-hedge ratio tell us? Is management genuinely confident?
- Is the specificity score high (they give real numbers) or low (vague hand-waving)?
- Is blame language present? Who are they blaming and is it legitimate?
- Are the forensic scores consistent with what management claims?

FILING DIFF INTERPRETATION (if prior filing data provided):
- What risk factors were ADDED or REMOVED? What does each change signal?
- What new statements appeared in MD&A? Are they positive or concerning?
- What was REMOVED from MD&A? Did they stop talking about something important?
- Did hedging increase or decrease? Did specificity change?
- Do the metric shifts tell a different story than management's narrative?

INSIDER ACTIVITY INTERPRETATION (if insider data provided):
- Are insiders buying with their own money? This is the strongest bullish signal for microcaps.
- Is there CLUSTER buying (multiple insiders buying in a short window)? Even stronger signal.
- Are C-suite executives (CEO/CFO) buying? They have the most information.
- Is there an activist investor (SC 13D)? What does this mean for the stock?
- Is there net selling? Could be routine or could signal problems — context matters.
- Does insider behavior CONFIRM or CONTRADICT management's optimistic language in the filing?

FINANCIAL INFLECTION INTERPRETATION (if XBRL data provided):
- Is revenue ACCELERATING (growth rate increasing)? This is the most important forward indicator.
- Is the company approaching profitability for the first time? This is a major re-rating catalyst.
- Are margins expanding? This suggests pricing power or operating leverage kicking in.
- Is there heavy dilution? This can destroy shareholder value even if the business improves.
- Is cash runway adequate? Low runway means potential dilutive financing ahead.
- Does the financial trajectory support or contradict the filing's qualitative narrative?

RESEARCH SIGNALS INTERPRETATION (from academic literature — weights reflect evidence strength):
You are given four quantitative signals derived from peer-reviewed research on SEC filings:
- MD&A SIMILARITY (35% weight, from "Lazy Prices" Cohen et al. JF 2020):
  Measures how much the MD&A section changed vs the prior filing.
  HIGH similarity = "non-changer" = historically POSITIVE (188bps/month alpha).
  LOW similarity = "changer" = historically NEGATIVE (predicts bad earnings, news, bankruptcies).
  This is the SINGLE STRONGEST documented textual alpha signal from SEC filings.
- RISK FACTOR CHANGES (30% weight, from "Lazy Prices" + Kirtac & Germano 2024):
  Risk factor section changes are "especially informative for future returns."
  New risk factors predict negative outcomes. Stable risk section = positive.
- LEGAL PROCEEDINGS (20% weight, from Offutt & Xie 2025):
  Expanding legal proceedings = priced legal risk increasing. Alpha unexplained by FF5+momentum.
  New litigation parties or dollar amounts = bearish.
- FILING TIMELINESS (15% weight, from Duarte-Silva et al. 2013):
  Late filers experience performance deterioration. Early = confidence signal.

IMPORTANT: Integrate these research scores with all other signals. Do they CONFIRM or CONTRADICT
the qualitative narrative? A high gem score with poor research signals suggests the narrative
may be overly optimistic. A moderate gem score with strong research signals suggests underappreciated stability.

TURNAROUND CATALYST INTERPRETATION (if catalyst data provided):
You are given algorithmically extracted turnaround catalyst signals across 8 categories:
cost restructuring, management changes, debt restructuring, margin inflection,
asset monetization, working capital, activist/ownership, and going concern.

CRITICAL TURNAROUND ASSESSMENT FRAMEWORK:
- WHERE IN THE TURNAROUND CYCLE? Early (announced restructuring, new management just arrived),
  mid-cycle (charges flowing, cost cuts executing), or inflection (savings realized, margins turning)?
  Early-stage turnarounds have high execution risk. Late-stage/inflection = highest conviction setups.
- ARE CATALYSTS GENUINE OR COSMETIC? Management can restructure forever without results.
  Look for PROOF: margin improvement, debt paydown, working capital normalization, FCF turning positive.
  If the catalyst data shows restructuring "complete" but margins aren't improving, be skeptical.
- CATALYST STACKING: Multiple reinforcing catalysts (e.g., new CEO + debt refinancing + margin improvement)
  are far more bullish than a single catalyst. Count how many categories are active.
- GOING CONCERN DIRECTION: If going concern was REMOVED vs prior filing, this is one of the highest-conviction
  turnaround setups in microcap investing. If ADDED, it's an existential red flag.
- TIMELINE: Are there specific forward-looking dates? Catalysts with defined timelines are actionable.
- DOES INSIDER BUYING CONFIRM THE TURNAROUND? Management buying stock during a restructuring
  is the strongest confirmation signal. If insiders are selling during a "turnaround," be very skeptical.

DEEP QUALITATIVE ANALYSIS:
1. MANAGEMENT AUTHENTICITY
   - Does forensic data CONFIRM or CONTRADICT management's narrative?
   - Are they getting more or less specific over time?
   - Is guidance conservative (sandbagging) or aggressive?

2. COMPETITIVE POSITION
   - What's the actual moat (if any)?
   - Pricing power evidence in the language?

3. CAPITAL ALLOCATION QUALITY
   - Are they investing at good returns?
   - Buyback timing and discipline

4. HIDDEN STRENGTHS
   - Understated positives buried in footnotes
   - New positive language that wasn't in prior filing
   - Removed risk factors suggesting resolved concerns

5. SUBTLE RED FLAGS
   - New hedging or blame language
   - Decreased specificity (hiding something?)
   - Related party transactions
   - Revenue recognition nuances

Respond with JSON (CRITICAL FIELDS FIRST — these must appear early in your response):
{{
  "ticker": "{ticker}",
  "final_gem_score": 1-100,
  "conviction_level": "high|medium|low",
  "recommendation": "strong_buy|buy|hold|avoid",
  "one_liner": "One sentence pitch referencing specific evidence",
  "insider_signal": {{
    "assessment": "1-2 sentences interpreting insider activity in context of the filing",
    "confirms_thesis": true/false,
    "key_observation": "most important insider activity finding"
  }},
  "financial_trajectory": {{
    "assessment": "1-2 sentences interpreting financial inflection data",
    "revenue_momentum": "accelerating|stable|decelerating|declining",
    "profitability_trajectory": "improving|stable|worsening",
    "key_inflection": "most significant financial inflection detected (or 'none')",
    "dilution_risk": "none|low|moderate|high"
  }},
  "investment_thesis": {{
    "bull_case": "2 sentences grounded in specific forensic, insider, AND financial evidence",
    "bear_case": "2 sentences grounded in specific forensic, insider, AND financial evidence",
    "key_risk": "what could go wrong"
  }},
  "turnaround_assessment": {{
    "is_turnaround": true/false,
    "turnaround_phase": "no_turnaround|early_stage|mid_cycle|inflection|post_turnaround",
    "phase_rationale": "1-2 sentences: what evidence places it in this phase?",
    "catalysts_genuine": true/false,
    "genuineness_evidence": "1-2 sentences: are margin/cash/debt ACTUALLY improving, or just talk?",
    "catalyst_stacking_count": 0-8,
    "top_catalysts": [
      {{
        "event": "specific catalyst description",
        "category": "cost_restructuring|management_change|debt_restructuring|margin_inflection|asset_monetization|working_capital|ownership_change|going_concern",
        "timeline": "Q2 2026 or 'in_progress' or 'completed'",
        "probability": "high|medium|low",
        "magnitude": "very_high|high|medium|low",
        "price_asymmetry": "1 sentence: is upside or downside larger?"
      }}
    ],
    "insider_confirms_turnaround": true/false/null,
    "overall_turnaround_conviction": "high|medium|low|none"
  }},
  "deep_analysis": {{
    "management_authenticity": {{
      "score": 1-10,
      "genuine_confidence_level": "high|medium|low",
      "guidance_style": "conservative|balanced|aggressive",
      "forensic_confirmation": "confirmed|mixed|contradicted",
      "evidence": ["quote1"],
      "assessment": "1-2 sentences"
    }},
    "competitive_position": {{
      "score": 1-10,
      "moat_type": "none|weak|moderate|strong",
      "share_trend": "gaining|stable|losing",
      "evidence": ["quote1"],
      "assessment": "1-2 sentences"
    }},
    "capital_allocation": {{
      "score": 1-10,
      "quality": "poor|mediocre|good|excellent",
      "evidence": ["quote1"],
      "assessment": "1-2 sentences"
    }},
    "hidden_strengths": [
      {{"finding": "...", "significance": "high|medium|low", "evidence": "..."}}
    ],
    "subtle_red_flags": [
      {{"finding": "...", "severity": "high|medium|low", "evidence": "..."}}
    ]
  }},
  "filing_diff_insights": {{
    "tone_shift": "improving|stable|deteriorating",
    "key_changes": ["change 1", "change 2"],
    "what_they_stopped_saying": ["removed topic 1"],
    "what_they_started_saying": ["new topic 1"],
    "red_flag_changes": ["concerning change"],
    "positive_changes": ["encouraging change"]
  }},
  "research_signals_assessment": {{
    "overall_interpretation": "1-2 sentences: do the research signals above CONFIRM or CONTRADICT your qualitative analysis? Call out any key disagreements.",
    "research_confirms_thesis": true/false
  }}
}}

{forensic_data}

{insider_data}

{inflection_data}

{research_signals_data}

{catalyst_data}

FILING ({form_type}) for {ticker} - {company_name}:
{filing_text}"""


# ============================================================================
# TIER 2: OPUS-LEVEL DEEP ANALYSIS PROMPT (Claude Code local analysis)
# ============================================================================

# TODO: Refactor to build on TIER2_DEEP_ANALYSIS_PROMPT + opus additions to reduce duplication
TIER2_OPUS_DEEP_ANALYSIS_PROMPT = """Perform deep qualitative analysis on this potential hidden gem.

This company scored well in initial screening. You have been provided with:
1. The SEC filing text
2. QUANTITATIVE forensic language metrics (hedging ratios, confidence patterns, etc.)
3. Filing-over-filing DIFF DATA showing what changed vs the prior filing (if available)
4. INSIDER & INSTITUTIONAL ACTIVITY data (Form 4 insider buys/sells, institutional filings)
5. FINANCIAL INFLECTION ANALYSIS from XBRL data (revenue trends, margin shifts, profitability crossings)
6. RESEARCH-BACKED FILING SIGNALS — quantitative scores from academic research on SEC filing predictive power
7. TURNAROUND CATALYST EXTRACTION — algorithmically detected operational catalysts from the filing text

YOUR JOB IS NOT TO SUMMARIZE. Your job is to INTERPRET all signals holistically and explain
what they mean for this company's investment potential. Focus on:

FORENSIC INTERPRETATION:
- What does the confidence-to-hedge ratio tell us? Is management genuinely confident?
- Is the specificity score high (they give real numbers) or low (vague hand-waving)?
- Is blame language present? Who are they blaming and is it legitimate?
- Are the forensic scores consistent with what management claims?

FILING DIFF INTERPRETATION (if prior filing data provided):
- What risk factors were ADDED or REMOVED? What does each change signal?
- What new statements appeared in MD&A? Are they positive or concerning?
- What was REMOVED from MD&A? Did they stop talking about something important?
- Did hedging increase or decrease? Did specificity change?
- Do the metric shifts tell a different story than management's narrative?

INSIDER ACTIVITY INTERPRETATION (if insider data provided):
- Are insiders buying with their own money? This is the strongest bullish signal for microcaps.
- Is there CLUSTER buying (multiple insiders buying in a short window)? Even stronger signal.
- Are C-suite executives (CEO/CFO) buying? They have the most information.
- Is there an activist investor (SC 13D)? What does this mean for the stock?
- Is there net selling? Could be routine or could signal problems — context matters.
- Does insider behavior CONFIRM or CONTRADICT management's optimistic language in the filing?

FINANCIAL INFLECTION INTERPRETATION (if XBRL data provided):
- Is revenue ACCELERATING (growth rate increasing)? This is the most important forward indicator.
- Is the company approaching profitability for the first time? This is a major re-rating catalyst.
- Are margins expanding? This suggests pricing power or operating leverage kicking in.
- Is there heavy dilution? This can destroy shareholder value even if the business improves.
- Is cash runway adequate? Low runway means potential dilutive financing ahead.
- Does the financial trajectory support or contradict the filing's qualitative narrative?

RESEARCH SIGNALS INTERPRETATION (from academic literature — weights reflect evidence strength):
You are given four quantitative signals derived from peer-reviewed research on SEC filings:
- MD&A SIMILARITY (35% weight, from "Lazy Prices" Cohen et al. JF 2020):
  Measures how much the MD&A section changed vs the prior filing.
  HIGH similarity = "non-changer" = historically POSITIVE (188bps/month alpha).
  LOW similarity = "changer" = historically NEGATIVE (predicts bad earnings, news, bankruptcies).
  This is the SINGLE STRONGEST documented textual alpha signal from SEC filings.
- RISK FACTOR CHANGES (30% weight, from "Lazy Prices" + Kirtac & Germano 2024):
  Risk factor section changes are "especially informative for future returns."
  New risk factors predict negative outcomes. Stable risk section = positive.
- LEGAL PROCEEDINGS (20% weight, from Offutt & Xie 2025):
  Expanding legal proceedings = priced legal risk increasing. Alpha unexplained by FF5+momentum.
  New litigation parties or dollar amounts = bearish.
- FILING TIMELINESS (15% weight, from Duarte-Silva et al. 2013):
  Late filers experience performance deterioration. Early = confidence signal.

IMPORTANT: Integrate these research scores with all other signals. Do they CONFIRM or CONTRADICT
the qualitative narrative? A high gem score with poor research signals suggests the narrative
may be overly optimistic. A moderate gem score with strong research signals suggests underappreciated stability.

TURNAROUND CATALYST INTERPRETATION (if catalyst data provided):
You are given algorithmically extracted turnaround catalyst signals across 8 categories:
cost restructuring, management changes, debt restructuring, margin inflection,
asset monetization, working capital, activist/ownership, and going concern.

CRITICAL TURNAROUND ASSESSMENT FRAMEWORK:
- WHERE IN THE TURNAROUND CYCLE? Early (announced restructuring, new management just arrived),
  mid-cycle (charges flowing, cost cuts executing), or inflection (savings realized, margins turning)?
  Early-stage turnarounds have high execution risk. Late-stage/inflection = highest conviction setups.
- ARE CATALYSTS GENUINE OR COSMETIC? Management can restructure forever without results.
  Look for PROOF: margin improvement, debt paydown, working capital normalization, FCF turning positive.
  If the catalyst data shows restructuring "complete" but margins aren't improving, be skeptical.
- CATALYST STACKING: Multiple reinforcing catalysts (e.g., new CEO + debt refinancing + margin improvement)
  are far more bullish than a single catalyst. Count how many categories are active.
- GOING CONCERN DIRECTION: If going concern was REMOVED vs prior filing, this is one of the highest-conviction
  turnaround setups in microcap investing. If ADDED, it's an existential red flag.
- TIMELINE: Are there specific forward-looking dates? Catalysts with defined timelines are actionable.
- DOES INSIDER BUYING CONFIRM THE TURNAROUND? Management buying stock during a restructuring
  is the strongest confirmation signal. If insiders are selling during a "turnaround," be very skeptical.

OPUS-LEVEL DEEP SIGNAL ANALYSIS:
You are the most capable reasoning model available. Your edge is finding what
surface-level analysis misses. For EVERY filing, perform these additional analyses:

1. NARRATIVE vs. NUMBERS CROSS-CHECK
   - For each material claim in MD&A, verify against the financial data provided
   - "Strong cash generation" -> check operating cash flow. Positive or negative?
   - "Margin expansion" -> check if gross/operating margins actually improved
   - "Revenue growth" -> organic or acquisition-driven? One-time or recurring?
   - Flag EVERY claim the numbers don't support. This is where alpha lives.

2. OMISSION ANALYSIS — What's NOT Being Said
   - What SHOULD be discussed given the industry/business, but isn't mentioned?
   - If prior filing discussed a key initiative, is there a progress update? Silence = failure.
   - If a competitor is dominating, does management acknowledge it? Denial = risk.
   - If the macro environment changed, did the risk factors update? Stale risks = lazy management.
   - Omissions are often more informative than what IS said.

3. INFORMATION ASYMMETRY DETECTION
   - Where does management have information the market likely doesn't?
   - New contract/partnership language implying a signed but unannounced deal
   - Capacity expansion + hiring language suggesting demand visibility
   - Unusually conservative guidance when business metrics are improving (sandbagging = bullish)
   - Unusual caution around a segment that's outperforming (managing expectations for upside surprise)

4. SECOND-ORDER IMPLICATIONS — If X Then Y
   - Debt refinancing at lower rates -> improved cash flow -> potential for buybacks or M&A
   - New restructuring advisor hired -> board likely evaluating strategic alternatives
   - Auditor change -> could be accounting concerns OR new management cleaning house
   - CFO departure + insider buying by CEO -> CEO knows something CFO didn't agree with
   - Each implication should state what it means for the stock price

5. LANGUAGE EVOLUTION PATTERNS (requires prior filing diff)
   - Track hedging direction: "will" -> "expect" -> "may" = progressive deterioration
   - Track specificity: vague -> specific = growing confidence; specific -> vague = hiding
   - Track tone: defensive -> neutral -> confident = turnaround taking hold
   - One-filing changes are noise. Multi-filing trends are signal.

6. CONTRARIAN SIGNAL DETECTION
   - What does consensus likely think about this stock? (The obvious read)
   - Where might consensus be WRONG?
   - The highest-alpha microcap opportunities are non-consensus correct calls
   - A company everyone thinks is dying that shows subtle signs of recovery
   - A company everyone thinks is fine but showing early cracks
   - State your contrarian thesis and what specific evidence supports it

7. FOOTNOTE MINING
   - Critical disclosures often hide in footnotes: related party deals, off-balance-sheet
     obligations, contingent liabilities, revenue recognition changes, lease commitments
   - Changes in accounting policies that affect comparability
   - Subsequent events (Note: filed after period end — most recent information available)

DEEP QUALITATIVE ANALYSIS:
1. MANAGEMENT AUTHENTICITY
   - Does forensic data CONFIRM or CONTRADICT management's narrative?
   - Are they getting more or less specific over time?
   - Is guidance conservative (sandbagging) or aggressive?

2. COMPETITIVE POSITION
   - What's the actual moat (if any)?
   - Pricing power evidence in the language?

3. CAPITAL ALLOCATION QUALITY
   - Are they investing at good returns?
   - Buyback timing and discipline

4. HIDDEN STRENGTHS
   - Understated positives buried in footnotes
   - New positive language that wasn't in prior filing
   - Removed risk factors suggesting resolved concerns

5. SUBTLE RED FLAGS
   - New hedging or blame language
   - Decreased specificity (hiding something?)
   - Related party transactions
   - Revenue recognition nuances

Respond with JSON (CRITICAL FIELDS FIRST — these must appear early in your response):
{{
  "ticker": "{ticker}",
  "final_gem_score": 1-100,
  "conviction_level": "high|medium|low",
  "recommendation": "strong_buy|buy|hold|avoid",
  "one_liner": "One sentence pitch referencing specific evidence",
  "insider_signal": {{
    "assessment": "1-2 sentences interpreting insider activity in context of the filing",
    "confirms_thesis": true/false,
    "key_observation": "most important insider activity finding"
  }},
  "financial_trajectory": {{
    "assessment": "1-2 sentences interpreting financial inflection data",
    "revenue_momentum": "accelerating|stable|decelerating|declining",
    "profitability_trajectory": "improving|stable|worsening",
    "key_inflection": "most significant financial inflection detected (or 'none')",
    "dilution_risk": "none|low|moderate|high"
  }},
  "investment_thesis": {{
    "bull_case": "2 sentences grounded in specific forensic, insider, AND financial evidence",
    "bear_case": "2 sentences grounded in specific forensic, insider, AND financial evidence",
    "key_risk": "what could go wrong"
  }},
  "turnaround_assessment": {{
    "is_turnaround": true/false,
    "turnaround_phase": "no_turnaround|early_stage|mid_cycle|inflection|post_turnaround",
    "phase_rationale": "1-2 sentences: what evidence places it in this phase?",
    "catalysts_genuine": true/false,
    "genuineness_evidence": "1-2 sentences: are margin/cash/debt ACTUALLY improving, or just talk?",
    "catalyst_stacking_count": 0-8,
    "top_catalysts": [
      {{
        "event": "specific catalyst description",
        "category": "cost_restructuring|management_change|debt_restructuring|margin_inflection|asset_monetization|working_capital|ownership_change|going_concern",
        "timeline": "Q2 2026 or 'in_progress' or 'completed'",
        "probability": "high|medium|low",
        "magnitude": "very_high|high|medium|low",
        "price_asymmetry": "1 sentence: is upside or downside larger?"
      }}
    ],
    "insider_confirms_turnaround": true/false/null,
    "overall_turnaround_conviction": "high|medium|low|none"
  }},
  "opus_deep_signals": {{
    "narrative_vs_numbers": {{
      "confirmed_claims": [{{"claim": "...", "evidence": "..."}}],
      "contradicted_claims": [{{"claim": "...", "contradiction": "..."}}],
      "unverifiable_claims": ["..."]
    }},
    "critical_omissions": [
      {{"expected_topic": "...", "why_missing_matters": "...", "signal": "bullish|bearish|neutral"}}
    ],
    "information_asymmetry": {{
      "signals": [{{"observation": "...", "what_mgmt_likely_knows": "...", "direction": "bullish|bearish"}}],
      "asymmetry_level": "high|medium|low|none"
    }},
    "second_order_implications": [
      {{"observation": "...", "implication": "...", "price_impact": "...", "conviction": "high|medium|low"}}
    ],
    "language_evolution": {{
      "hedging_trend": "increasing|stable|decreasing",
      "specificity_trend": "increasing|stable|decreasing",
      "confidence_trend": "increasing|stable|decreasing",
      "key_language_shifts": ["specific word/phrase changes"]
    }},
    "contrarian_thesis": {{
      "consensus_view": "what the market likely thinks",
      "contrarian_view": "why consensus might be wrong",
      "edge_source": "what specific evidence supports the contrarian view",
      "conviction": "high|medium|low"
    }},
    "non_obvious_catalysts": [
      {{"catalyst": "...", "timeline": "...", "probability": "high|medium|low", "why_others_miss_it": "..."}}
    ],
    "overall_edge_assessment": "1-2 sentences: what is the single most important non-obvious insight from this filing?"
  }},
  "deep_analysis": {{
    "management_authenticity": {{
      "score": 1-10,
      "genuine_confidence_level": "high|medium|low",
      "guidance_style": "conservative|balanced|aggressive",
      "forensic_confirmation": "confirmed|mixed|contradicted",
      "evidence": ["quote1"],
      "assessment": "1-2 sentences"
    }},
    "competitive_position": {{
      "score": 1-10,
      "moat_type": "none|weak|moderate|strong",
      "share_trend": "gaining|stable|losing",
      "evidence": ["quote1"],
      "assessment": "1-2 sentences"
    }},
    "capital_allocation": {{
      "score": 1-10,
      "quality": "poor|mediocre|good|excellent",
      "evidence": ["quote1"],
      "assessment": "1-2 sentences"
    }},
    "hidden_strengths": [
      {{"finding": "...", "significance": "high|medium|low", "evidence": "..."}}
    ],
    "subtle_red_flags": [
      {{"finding": "...", "severity": "high|medium|low", "evidence": "..."}}
    ]
  }},
  "filing_diff_insights": {{
    "tone_shift": "improving|stable|deteriorating",
    "key_changes": ["change 1", "change 2"],
    "what_they_stopped_saying": ["removed topic 1"],
    "what_they_started_saying": ["new topic 1"],
    "red_flag_changes": ["concerning change"],
    "positive_changes": ["encouraging change"]
  }},
  "research_signals_assessment": {{
    "overall_interpretation": "1-2 sentences: do the research signals above CONFIRM or CONTRADICT your qualitative analysis? Call out any key disagreements.",
    "research_confirms_thesis": true/false
  }}
}}

{forensic_data}

{insider_data}

{inflection_data}

{research_signals_data}

{catalyst_data}

FILING ({form_type}) for {ticker} - {company_name}:
{filing_text}"""


# Opus-specific deep analysis instructions (included in T2 bundles for Claude Code analysis)
# TODO: This is a third copy of the opus-specific instructions (also in TIER2_OPUS_DEEP_ANALYSIS_PROMPT).
# Refactor to use a single source: extract shared opus instructions into a constant and compose both from it.
OPUS_DEEP_ANALYSIS_INSTRUCTIONS = """OPUS-LEVEL DEEP SIGNAL ANALYSIS:
You are the most capable reasoning model available. Your edge is finding what
surface-level analysis misses. For EVERY filing, perform these additional analyses:

1. NARRATIVE vs. NUMBERS CROSS-CHECK
   - For each material claim in MD&A, verify against the financial data provided
   - "Strong cash generation" -> check operating cash flow. Positive or negative?
   - "Margin expansion" -> check if gross/operating margins actually improved
   - "Revenue growth" -> organic or acquisition-driven? One-time or recurring?
   - Flag EVERY claim the numbers don't support. This is where alpha lives.

2. OMISSION ANALYSIS — What's NOT Being Said
   - What SHOULD be discussed given the industry/business, but isn't mentioned?
   - If prior filing discussed a key initiative, is there a progress update? Silence = failure.
   - If a competitor is dominating, does management acknowledge it? Denial = risk.
   - If the macro environment changed, did the risk factors update? Stale risks = lazy management.
   - Omissions are often more informative than what IS said.

3. INFORMATION ASYMMETRY DETECTION
   - Where does management have information the market likely doesn't?
   - New contract/partnership language implying a signed but unannounced deal
   - Capacity expansion + hiring language suggesting demand visibility
   - Unusually conservative guidance when business metrics are improving (sandbagging = bullish)
   - Unusual caution around a segment that's outperforming (managing expectations for upside surprise)

4. SECOND-ORDER IMPLICATIONS — If X Then Y
   - Debt refinancing at lower rates -> improved cash flow -> potential for buybacks or M&A
   - New restructuring advisor hired -> board likely evaluating strategic alternatives
   - Auditor change -> could be accounting concerns OR new management cleaning house
   - CFO departure + insider buying by CEO -> CEO knows something CFO didn't agree with
   - Each implication should state what it means for the stock price

5. LANGUAGE EVOLUTION PATTERNS (requires prior filing diff)
   - Track hedging direction: "will" -> "expect" -> "may" = progressive deterioration
   - Track specificity: vague -> specific = growing confidence; specific -> vague = hiding
   - Track tone: defensive -> neutral -> confident = turnaround taking hold
   - One-filing changes are noise. Multi-filing trends are signal.

6. CONTRARIAN SIGNAL DETECTION
   - What does consensus likely think about this stock? (The obvious read)
   - Where might consensus be WRONG?
   - The highest-alpha microcap opportunities are non-consensus correct calls
   - A company everyone thinks is dying that shows subtle signs of recovery
   - A company everyone thinks is fine but showing early cracks
   - State your contrarian thesis and what specific evidence supports it

7. FOOTNOTE MINING
   - Critical disclosures often hide in footnotes: related party deals, off-balance-sheet
     obligations, contingent liabilities, revenue recognition changes, lease commitments
   - Changes in accounting policies that affect comparability
   - Subsequent events (Note: filed after period end — most recent information available)

Include an "opus_deep_signals" section in your JSON output with:
- narrative_vs_numbers: confirmed_claims, contradicted_claims, unverifiable_claims
- critical_omissions: expected_topic, why_missing_matters, signal (bullish/bearish/neutral)
- information_asymmetry: signals with observation/what_mgmt_likely_knows/direction, asymmetry_level
- second_order_implications: observation, implication, price_impact, conviction
- language_evolution: hedging_trend, specificity_trend, confidence_trend, key_language_shifts
- contrarian_thesis: consensus_view, contrarian_view, edge_source, conviction
- non_obvious_catalysts: catalyst, timeline, probability, why_others_miss_it
- overall_edge_assessment: 1-2 sentences on the single most important non-obvious insight"""


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_filing_tier1(
    ticker: str,
    company_name: str,
    filing_text: str,
    form_type: str = "10-K",
    model: str = DEFAULT_TIER1_MODEL,
) -> Optional[Dict[str, Any]]:
    """
    Tier 1 screening analysis using GPT-4o-mini.
    Fast and cheap - run on all filings.
    Routes to ownership-specific prompt for SC 13D/G filings.
    """
    # Select prompt based on filing type
    if form_type.startswith('SC 13'):
        template = TIER1_OWNERSHIP_PROMPT
    elif form_type in ('8-K', '8-K/A'):
        template = TIER1_8K_EVENT_PROMPT
    elif form_type in ('DEF 14A', 'DEFA14A'):
        template = TIER1_DEF14A_PROMPT
    else:
        template = TIER1_SCREENING_PROMPT

    prompt = template.format(
        ticker=ticker,
        company_name=company_name,
        form_type=form_type,
        filing_text=_truncate_text(filing_text, 80000),
    )

    raw = _call_llm(prompt, SYSTEM_PROMPT_GEM_FINDER, model)
    result = _parse_json_response(raw)

    if not result:
        preview = (raw or "")[:200]
        raise RuntimeError(f"JSON parse failed. Raw response: {preview}...")

    result["analysis_tier"] = 1
    result["model_used"] = model
    result["analyzed_at"] = datetime.now().isoformat()

    # Calculate composite if not provided
    if "composite_score" not in result and "dimensions" in result:
        scores = [d.get("score", 5) for d in result["dimensions"].values()]
        result["composite_score"] = round(sum(scores) / len(scores) * 10, 1) if scores else 50

    # Warn if LLM returns score in 0-10 range instead of 0-100
    cs = result.get("composite_score")
    if cs is not None and cs <= 10:
        logger.warning("LLM returned score in 0-10 range: %s", cs)

    return result


def analyze_trajectory_tier1(
    ticker: str,
    current_text: str,
    prior_text: str,
    current_date: str,
    prior_date: str,
    current_form_type: str = "10-K",
    prior_form_type: str = "10-K",
    model: str = DEFAULT_TIER1_MODEL,
) -> Optional[Dict[str, Any]]:
    """
    Tier 1 trajectory analysis - compare two filings.
    Identifies improving vs declining companies.
    """
    prompt = TIER1_TRAJECTORY_PROMPT.format(
        ticker=ticker,
        current_text=_truncate_text(current_text, 50000),
        prior_text=_truncate_text(prior_text, 50000),
        current_date=current_date,
        prior_date=prior_date,
        current_form_type=current_form_type,
        prior_form_type=prior_form_type,
    )

    raw = _call_llm(prompt, SYSTEM_PROMPT_GEM_FINDER, model)
    result = _parse_json_response(raw)

    if not result:
        logger.warning("Failed to parse tier1 result for %s", ticker)

    if result:
        result["analysis_tier"] = 1
        result["analysis_type"] = "trajectory"
        result["model_used"] = model
        result["analyzed_at"] = datetime.now().isoformat()

        # Calculate trajectory score if not provided
        if "trajectory_score" not in result and "dimensions" in result:
            traj_map = {
                "strongly_improving": 2,
                "improving": 1,
                "stable": 0,
                "declining": -1,
                "deteriorating": -2,
            }
            traj_scores = [
                traj_map.get(d.get("trajectory", "stable"), 0)
                for d in result["dimensions"].values()
            ]
            result["trajectory_score"] = sum(traj_scores)

    return result


def analyze_filing_tier2(
    ticker: str,
    company_name: str,
    filing_text: str,
    form_type: str = "10-K",
    model: str = DEFAULT_TIER2_MODEL,
    forensic_data: str = "",
    insider_data: str = "",
    inflection_data: str = "",
    research_signals_data: str = "",
    catalyst_data: str = "",
    thinking_budget: int = 0,
    effort: str = "",
    cross_ref_context: str = "",      # Multi-filing cross-reference (#3)
    peer_comparison: str = "",         # Peer/sector comparison (#5)
    short_interest: str = "",          # Short interest data (#13)
    transcript_signals: str = "",      # Earnings call signals (#12)
    insider_timing: str = "",          # Insider timing correlation (#4)
    market_data_context: str = "",     # Quant screen + institutional data (#11, #14)
) -> Optional[Dict[str, Any]]:
    """
    Tier 2 deep analysis using Claude (Haiku, Sonnet, or Opus).
    Run on top candidates from Tier 1 screening.

    Args:
        forensic_data: Pre-formatted forensic metrics + diff data string
        insider_data: Pre-formatted insider transaction analysis string
        inflection_data: Pre-formatted financial inflection analysis string
        research_signals_data: Pre-formatted research-backed filing signals string
        catalyst_data: Pre-formatted turnaround catalyst extraction string
        thinking_budget: Extended thinking token budget (Opus only, 0=disabled)
        effort: Effort level for Opus: low|medium|high|max (empty=default)
        cross_ref_context: Summaries of other filings for the same company (#3)
        peer_comparison: Sector/peer comparison data (#5)
        short_interest: Short interest and squeeze potential data (#13)
        transcript_signals: Earnings call transcript signals (#12)
        insider_timing: Insider trading timing vs filing dates (#4)
        market_data_context: Quant screen + institutional ownership data (#11, #14)
    """
    if not forensic_data:
        forensic_data = "(No forensic data available)"
    if not insider_data:
        insider_data = "(No insider activity data available)"
    if not inflection_data:
        inflection_data = "(No financial inflection data available)"
    if not research_signals_data:
        research_signals_data = "(No research signal data available)"
    if not catalyst_data:
        catalyst_data = "(No turnaround catalyst data available)"

    # Build extended context sections for new data sources
    extended_context = ""

    if cross_ref_context:
        extended_context += f"\n\n=== OTHER FILINGS FOR THIS COMPANY (cross-reference) ===\n{cross_ref_context}\n"

    if peer_comparison:
        extended_context += f"\n\n=== SECTOR/PEER COMPARISON ===\n{peer_comparison}\n"

    if short_interest:
        extended_context += f"\n\n=== SHORT INTEREST DATA ===\n{short_interest}\n"

    if insider_timing:
        extended_context += f"\n\n=== INSIDER TIMING ANALYSIS ===\n{insider_timing}\n"

    if transcript_signals:
        extended_context += f"\n\n=== EARNINGS CALL SIGNALS ===\n{transcript_signals}\n"

    if market_data_context:
        extended_context += f"\n\n=== QUANTITATIVE SCREENING & INSTITUTIONAL DATA ===\n{market_data_context}\n"

    # Add form-type-specific context for special filing types
    form_context = ""
    if form_type.startswith('SC 13'):
        is_activist = 'SC 13D' in form_type.upper()
        filing_label = "ACTIVIST (SC 13D)" if is_activist else "PASSIVE LARGE HOLDER (SC 13G)"
        form_context = f"""
NOTE: This is a {form_type} OWNERSHIP FILING ({filing_label}), not a 10-K/10-Q periodic filing.
{"SC 13D is filed by investors who acquired >5% with intent to INFLUENCE the company." if is_activist else "SC 13G is filed by PASSIVE investors who acquired >5% with no current intent to influence."}
Focus analysis on:
- PURPOSE OF TRANSACTION: What does the filer intend to do? (activist plan, passive hold, acquisition)
- OWNERSHIP STAKE: How large is the position? Increasing or decreasing?
{"- ACTIVIST PROPOSALS: Board seats, management changes, strategic alternatives, cost cutting?" if is_activist else "- CONVERSION RISK: Any language suggesting possible future activism or switch to 13D?"}
- SOURCE OF FUNDS: How is the position being financed?
- PLANS OR PROPOSALS: Specific change proposals are high-value turnaround signals.
Some analysis sections (MD&A, risk factors, financial statements) do NOT apply to this filing type.
Focus on what this ownership change means as a CATALYST for the underlying company.
"""
    elif form_type in ('8-K', '8-K/A'):
        form_context = """
NOTE: This is an 8-K EVENT FILING — the most time-sensitive SEC filing type.
8-K filings report MATERIAL EVENTS that happened within the last few days.
Focus analysis on:
- WHAT EVENT(S): Management changes? M&A? Material agreements? Guidance? Restructuring?
- CATALYST IMPACT: Does this event create a near-term catalyst for stock price movement?
- STRATEGIC IMPLICATIONS: Does this change the company's trajectory?
- TIMING: How recent is this event? Is there urgency or deadline?
Some standard 10-K sections (detailed financial analysis, risk factors) may not apply.
Focus on the EVENT itself and its immediate implications as a CATALYST.
"""
    elif form_type in ('DEF 14A', 'DEFA14A'):
        form_context = """
NOTE: This is a DEF 14A PROXY STATEMENT — reveals governance and compensation structure.
Focus analysis on:
- EXECUTIVE COMPENSATION: Is pay aligned with performance? Changes in comp structure?
- BOARD COMPOSITION: Independent directors? New members from activist investors? Classified board?
- SHAREHOLDER PROPOSALS: Any proposals for strategic review, board seats, compensation reform?
- ANTI-TAKEOVER: Poison pills, staggered boards, super-majority requirements?
- RELATED PARTY TRANSACTIONS: Conflicts of interest, self-dealing?
- CHANGE-OF-CONTROL PROVISIONS: Golden parachutes that might signal M&A catalyst?
Focus on whether the governance picture supports or blocks a turnaround catalyst.
"""

    # NOTE: Filing text is user-controlled content from SEC EDGAR. Prompt injection risk documented.
    prompt = TIER2_DEEP_ANALYSIS_PROMPT.format(
        ticker=ticker,
        company_name=company_name,
        form_type=form_type,
        filing_text=form_context + _truncate_text(filing_text, 80000) + extended_context,
        forensic_data=forensic_data,
        insider_data=insider_data,
        inflection_data=inflection_data,
        research_signals_data=research_signals_data,
        catalyst_data=catalyst_data,
    )

    # Prompt size validation — warn if excessively large
    estimated_tokens = len(prompt) // 4
    if estimated_tokens > 200000:
        logger.warning("Prompt exceeds 200K estimated tokens: %d", estimated_tokens)

    raw = _call_llm(prompt, SYSTEM_PROMPT_DEEP_ANALYSIS, model, max_tokens=8192,
                    thinking_budget=thinking_budget, effort=effort)
    result = _parse_json_response(raw)

    if not result:
        preview = (raw or "<no response>")[:300]
        logger.error(f"Tier 2 {ticker}: JSON parse failed. len={len(raw or '')} preview={preview}")
        raise RuntimeError(f"JSON parse failed (response len={len(raw or '')}). Preview: {preview}...")

    result["analysis_tier"] = 2
    result["model_used"] = model
    result["analyzed_at"] = datetime.now().isoformat()
    if thinking_budget > 0:
        result["thinking_enabled"] = True
        result["thinking_budget"] = thinking_budget
    if effort:
        result["effort_level"] = effort

    # Warn if LLM returns score in 0-10 range instead of 0-100
    fgs = result.get("final_gem_score")
    if fgs is not None and fgs <= 10:
        logger.warning("LLM returned score in 0-10 range: %s", fgs)

    return result


# ============================================================================
# BATCH ANALYSIS ORCHESTRATOR
# ============================================================================

def run_tiered_analysis(
    filings: List[Dict],
    tier1_model: str = DEFAULT_TIER1_MODEL,
    tier2_model: str = DEFAULT_TIER2_MODEL,
    tier2_percentile: float = 0.25,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Run two-tier analysis on a list of filings.

    Args:
        filings: List of dicts with keys: ticker, company_name, text, form_type, filing_date
        tier1_model: Model for initial screening
        tier2_model: Model for deep analysis
        tier2_percentile: Top X% to analyze with Tier 2 (0.25 = top 25%)
        progress_callback: fn(current, total, ticker, tier, status)

    Returns:
        Dict with all results and summary stats
    """
    results = {
        "tier1_results": [],
        "tier2_results": [],
        "errors": [],
        "stats": {
            "total_filings": len(filings),
            "tier1_completed": 0,
            "tier2_completed": 0,
            "tier2_eligible": 0,
        },
    }

    total = len(filings)

    # ---- TIER 1: Screen all filings ----
    if progress_callback:
        progress_callback(0, total, "", 1, "Starting Tier 1 screening...")

    for i, filing in enumerate(filings):
        ticker = filing.get("ticker", "UNKNOWN")
        try:
            t1_result = analyze_filing_tier1(
                ticker=ticker,
                company_name=filing.get("company_name", ""),
                filing_text=filing.get("text", ""),
                form_type=filing.get("form_type", "10-K"),
                model=tier1_model,
            )
            if t1_result:
                t1_result["filing_date"] = filing.get("filing_date", "")
                t1_result["accession_number"] = filing.get("accession_number", "")
                results["tier1_results"].append(t1_result)
                results["stats"]["tier1_completed"] += 1
        except Exception as e:
            results["errors"].append(f"{ticker}: Tier 1 error - {str(e)}")
            logger.error(f"Tier 1 error for {ticker}: {e}")

        if progress_callback:
            progress_callback(i + 1, total, ticker, 1, "Tier 1 screening")

    # ---- Sort by composite score and select top percentile for Tier 2 ----
    tier1_sorted = sorted(
        results["tier1_results"],
        key=lambda x: x.get("composite_score", 0),
        reverse=True,
    )

    tier2_count = max(0, int(len(tier1_sorted) * tier2_percentile))
    tier2_candidates = tier1_sorted[:tier2_count]
    results["stats"]["tier2_eligible"] = tier2_count

    # Build lookup for filing text by accession_number to avoid collisions
    filing_lookup = {f.get("accession_number"): f for f in filings}

    # ---- TIER 2: Deep analysis on top candidates ----
    if tier2_count == 0:
        logger.info("No filings eligible for Tier 2 analysis (count=0)")
        return results

    if progress_callback:
        progress_callback(0, tier2_count, "", 2, "Starting Tier 2 deep analysis...")

    for i, t1_result in enumerate(tier2_candidates):
        ticker = t1_result.get("ticker", "UNKNOWN")
        filing = filing_lookup.get(t1_result.get("accession_number"), {})

        try:
            t2_result = analyze_filing_tier2(
                ticker=ticker,
                company_name=filing.get("company_name", ""),
                filing_text=filing.get("text", ""),
                form_type=filing.get("form_type", "10-K"),
                model=tier2_model,
            )
            if t2_result:
                # Merge Tier 1 data
                t2_result["tier1_composite_score"] = t1_result.get("composite_score")
                t2_result["tier1_gem_potential"] = t1_result.get("gem_potential")
                t2_result["filing_date"] = t1_result.get("filing_date", "")
                t2_result["accession_number"] = t1_result.get("accession_number", "")
                results["tier2_results"].append(t2_result)
                results["stats"]["tier2_completed"] += 1
        except Exception as e:
            results["errors"].append(f"{ticker}: Tier 2 error - {str(e)}")
            logger.error(f"Tier 2 error for {ticker}: {e}")

        if progress_callback:
            progress_callback(i + 1, tier2_count, ticker, 2, "Tier 2 deep analysis")

    # ---- Final ranking ----
    results["tier2_results"] = sorted(
        results["tier2_results"],
        key=lambda x: x.get("final_gem_score", 0),
        reverse=True,
    )

    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_api_status() -> Dict[str, Any]:
    """Check which APIs are available."""
    status = {
        "openai": {"available": False, "has_key": False, "has_sdk": False},
        "anthropic": {"available": False, "has_key": False, "has_sdk": False},
    }

    # OpenAI
    status["openai"]["has_key"] = bool(os.environ.get("OPENAI_API_KEY"))
    try:
        import openai
        status["openai"]["has_sdk"] = True
    except ImportError:
        pass
    status["openai"]["available"] = (
        status["openai"]["has_key"] and status["openai"]["has_sdk"]
    )

    # Anthropic
    status["anthropic"]["has_key"] = bool(os.environ.get("ANTHROPIC_API_KEY"))
    try:
        import anthropic
        status["anthropic"]["has_sdk"] = True
    except ImportError:
        pass
    status["anthropic"]["available"] = (
        status["anthropic"]["has_key"] and status["anthropic"]["has_sdk"]
    )

    return status


def estimate_cost(num_filings: int, tier2_percentile: float = 0.25,
                  tier2_model: str = "", effort: str = "") -> Dict[str, Any]:
    """Estimate cost for analyzing filings including insider + inflection data.

    Note: Token counts (t1_avg_input, t1_avg_output, t2_avg_input, t2_avg_output)
    are rough estimates based on typical filing sizes and response lengths.
    Actual costs may vary significantly depending on filing length and model verbosity.
    """
    tier2_count = max(1, int(num_filings * tier2_percentile))
    t2_model_name = tier2_model if tier2_model in MODELS else DEFAULT_TIER2_MODEL

    # ---- Tier 1: GPT-4o-mini screening ----
    t1_avg_input = 30000
    t1_avg_output = 1500

    tier1_input = num_filings * t1_avg_input / 1_000_000
    tier1_output = num_filings * t1_avg_output / 1_000_000

    # ---- Tier 2: model-dependent ----
    # Filing text + forensic + insider + inflection + prompt = ~35K input tokens
    t2_avg_input = 35000
    t2_avg_output = 3000

    # Extended thinking adds output tokens (billed at output rate)
    t2_config = MODELS[t2_model_name]
    thinking_tokens = 0
    if effort and t2_config.get("supports_thinking"):
        thinking_tokens = EFFORT_LEVELS.get(effort, {}).get("thinking_budget", 0)

    # Thinking tokens are billed as output tokens
    t2_avg_output_total = t2_avg_output + thinking_tokens

    tier2_input = tier2_count * t2_avg_input / 1_000_000
    tier2_output = tier2_count * t2_avg_output_total / 1_000_000

    t1_config = MODELS[DEFAULT_TIER1_MODEL]

    tier1_cost = (tier1_input * t1_config["input_cost_per_1m"] +
                  tier1_output * t1_config["output_cost_per_1m"])
    tier2_cost = (tier2_input * t2_config["input_cost_per_1m"] +
                  tier2_output * t2_config["output_cost_per_1m"])

    # ---- EDGAR API calls (free) ----
    unique_tickers = max(1, int(tier2_count * 0.8))
    edgar_calls_per_ticker = 1 + 15 + 1
    total_edgar_calls = unique_tickers * edgar_calls_per_ticker

    # ---- Time estimates ----
    tier1_time_min = round(num_filings * 4 / 60, 1)
    # Opus is slower per call (~15-30s vs Haiku's ~3-5s)
    if t2_model_name == "claude-opus":
        t2_sec_per = 25 if thinking_tokens > 0 else 18
    elif t2_model_name == "claude-sonnet":
        t2_sec_per = 15
    else:
        t2_sec_per = 12
    tier2_time_min = round(tier2_count * t2_sec_per / 60, 1)
    total_time_min = round(tier1_time_min + tier2_time_min, 1)

    # ---- Build description ----
    tier2_desc = "forensics + insider tracking + financial inflections"
    if thinking_tokens > 0:
        tier2_desc += f" + extended thinking ({effort})"

    return {
        "tier1_filings": num_filings,
        "tier2_filings": tier2_count,
        "tier1_cost": round(tier1_cost, 2),
        "tier2_cost": round(tier2_cost, 2),
        "total_cost": round(tier1_cost + tier2_cost, 2),
        "tier1_model": DEFAULT_TIER1_MODEL,
        "tier2_model": t2_model_name,
        "tier2_includes": tier2_desc,
        "edgar_api_calls": total_edgar_calls,
        "unique_tickers_tier2": unique_tickers,
        "est_time_min": total_time_min,
        "tier1_time_min": tier1_time_min,
        "tier2_time_min": tier2_time_min,
        "thinking_tokens_per_filing": thinking_tokens,
        "effort_level": effort or "default",
        "cost_per_filing_tier2": round(tier2_cost / max(tier2_count, 1), 3),
    }


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

def is_llm_available() -> bool:
    """Check if any LLM API is available."""
    status = check_api_status()
    return status["openai"]["available"] or status["anthropic"]["available"]


def llm_analyze_sentiment(current_text: str, prior_text: Optional[str] = None) -> Optional[Dict]:
    """Legacy function - now uses Tier 1 analysis.

    .. deprecated::
        Use :func:`analyze_filing_tier1` directly instead. This wrapper adds overhead
        by converting results to an outdated format and may be removed in a future version.
    """
    try:
        result = analyze_filing_tier1(
            ticker="UNKNOWN",
            company_name="Unknown",
            filing_text=current_text,
            form_type="10-K",
        )
    except Exception as e:
        logger.error(f"llm_analyze_sentiment failed: {e}")
        return None
    if result:
        # Convert to legacy format
        return {
            "sentiment_results": [
                {
                    "category": k,
                    "current_level": "CONFIDENT" if v.get("score", 5) >= 7 else
                                    "NEUTRAL" if v.get("score", 5) >= 4 else "CAUTIOUS",
                    "score_impact": (v.get("score", 5) - 5) * 2,
                    "key_phrases": [v.get("evidence", "")],
                }
                for k, v in result.get("dimensions", {}).items()
            ],
            "sentiment_meta": {
                "overall_trajectory": result.get("gem_potential", "medium"),
            },
        }
    return None


def llm_analyze_flags(current_text: str, prior_text: Optional[str] = None) -> Optional[Dict]:
    """Legacy function - now uses Tier 1 analysis.

    .. deprecated::
        Use :func:`analyze_filing_tier1` directly instead. This wrapper adds overhead
        by converting results to an outdated format and may be removed in a future version.
    """
    try:
        result = analyze_filing_tier1(
            ticker="UNKNOWN",
            company_name="Unknown",
            filing_text=current_text,
            form_type="10-K",
        )
    except Exception as e:
        logger.error(f"llm_analyze_flags failed: {e}")
        return None
    if result:
        flags = []
        for concern in result.get("top_3_concerns", []):
            flags.append({
                "rule_id": "llm_concern",
                "title": concern,
                "signal_type": "yellow_flag",
                "risk_level": "MODERATE",
                "score_impact": -5,
                "source": "llm",
            })
        for positive in result.get("top_3_positives", []):
            flags.append({
                "rule_id": "llm_positive",
                "title": positive,
                "signal_type": "green_flag",
                "risk_level": "LOW",
                "score_impact": 5,
                "source": "llm",
            })
        return {
            "flags": flags,
            "overall_assessment": result.get("gem_reasoning", ""),
        }
    return None
