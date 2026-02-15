"""
insider_tracker.py — Insider & Institutional Accumulation Tracker
=================================================================
Fetches Form 4 (insider transactions) and SC 13D/G (institutional holdings)
from EDGAR, then computes accumulation signals for microcap stock analysis.

Key signals:
- Insider cluster buying (multiple insiders buying in a short window)
- Large purchases relative to compensation/holdings
- First-time institutional holders appearing
- Buy/sell ratio trends
- Open market purchases vs option exercises

All functions are designed to be called during Tier 2 analysis,
feeding structured data into LLM prompts.
"""

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _normalize_form(form_type: str) -> str:
    """Normalize SEC form type: SCHEDULE 13D → SC 13D, etc."""
    ft = form_type.strip().upper()
    ft = ft.replace("SCHEDULE 13D/A", "SC 13D/A")
    ft = ft.replace("SCHEDULE 13D", "SC 13D")
    ft = ft.replace("SCHEDULE 13G/A", "SC 13G/A")
    ft = ft.replace("SCHEDULE 13G", "SC 13G")
    return ft

# ============================================================================
# FORM 4 FETCHING & PARSING
# ============================================================================

def fetch_insider_transactions(cik: str, session, limiter,
                                lookback_days: int = 365) -> List[Dict]:
    """
    Fetch Form 4 filings for a company from EDGAR submissions endpoint.
    Returns list of insider transaction records.
    """
    from scraper import _safe_request

    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    data = _safe_request(session, url, limiter=limiter)
    if not data:
        logger.warning(f"CIK {cik}: no submission data returned")
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    # Date strings must be ISO format (YYYY-MM-DD) for string comparison to work correctly
    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    cik_num = cik.lstrip('0')

    form4_entries = []
    for i in range(len(forms)):
        form_type = forms[i] if i < len(forms) else ""
        filing_date = dates[i] if i < len(dates) else ""
        accession = accessions[i] if i < len(accessions) else ""
        primary_doc = primary_docs[i] if i < len(primary_docs) else ""

        if form_type in ("4", "4/A") and filing_date >= cutoff:
            accession_clean = accession.replace("-", "")
            xml_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_num}/{accession_clean}/{primary_doc}"
            )
            form4_entries.append({
                "filing_date": filing_date,
                "accession": accession,
                "url": xml_url,
                "primary_doc": primary_doc,
            })

    logger.info(f"CIK {cik}: found {len(form4_entries)} Form 4 filings in last {lookback_days}d")

    # Parse each Form 4 (limit to most recent 30 to stay within rate limits)
    transactions = []
    for entry in form4_entries[:30]:
        try:
            parsed = _fetch_and_parse_form4(entry["url"], entry["filing_date"],
                                             session, limiter)
            transactions.extend(parsed)
        except Exception as e:
            logger.debug(f"Form 4 parse error ({entry['url']}): {e}")
            continue

    return transactions


def _fetch_and_parse_form4(url: str, filing_date: str,
                            session, limiter) -> List[Dict]:
    """Fetch a single Form 4 XML and extract transaction details."""
    from scraper import _safe_request

    # _safe_request returns parsed JSON for .json URLs, raw text for others.
    # Form 4 XMLs come back as raw text strings, so we fall back to text handling.
    content = _safe_request(session, url, limiter=limiter)
    if not content or not isinstance(content, str):
        return []

    # Input size limit to prevent excessive memory usage
    if len(content) > 10_000_000:
        logger.warning("XML too large (%d bytes), skipping", len(content))
        return []

    # Form 4 can be XML or HTML wrapper around XML
    transactions = []

    try:
        # Try direct XML parse
        root = ET.fromstring(content)
    except ET.ParseError as e:
        # Try to extract XML from HTML
        xml_match = re.search(r'<ownershipDocument>(.*?)</ownershipDocument>',
                              content, re.DOTALL | re.IGNORECASE)
        if xml_match:
            try:
                root = ET.fromstring(
                    f'<ownershipDocument>{xml_match.group(1)}</ownershipDocument>'
                )
            except ET.ParseError as e2:
                logger.warning("Failed to parse Form 4 XML: %s", e2)
                return []
        else:
            logger.warning("Failed to parse Form 4 XML: %s", e)
            return []

    # Extract reporting owner info — collect all owners from the document
    owners = []
    current_owner_name = ""
    current_owner_relationship = ""
    for owner_elem in root.iter():
        if 'rptOwnerName' in owner_elem.tag:
            # Save previous owner if we're starting a new one
            if current_owner_name:
                owners.append({"name": current_owner_name, "relationship": current_owner_relationship})
            current_owner_name = (owner_elem.text or "").strip()
            current_owner_relationship = ""
        elif 'isDirector' in owner_elem.tag and (owner_elem.text or "").strip() == "1":
            current_owner_relationship = "director" if not current_owner_relationship else current_owner_relationship
        elif 'isOfficer' in owner_elem.tag and (owner_elem.text or "").strip() == "1":
            current_owner_relationship = "officer"
        elif 'isTenPercentOwner' in owner_elem.tag and (owner_elem.text or "").strip() == "1":
            current_owner_relationship = "10pct_owner" if not current_owner_relationship else current_owner_relationship
        elif 'officerTitle' in owner_elem.tag and owner_elem.text:
            officer_title = owner_elem.text.strip()
            if officer_title:
                current_owner_relationship = f"officer:{officer_title}"
    # Don't forget the last owner
    if current_owner_name:
        owners.append({"name": current_owner_name, "relationship": current_owner_relationship})

    # Use first owner as default for transactions (most Form 4s have one owner)
    owner_name = owners[0]["name"] if owners else ""
    owner_relationship = owners[0]["relationship"] if owners else ""

    # Extract non-derivative transactions
    for txn in root.iter():
        if 'nonDerivativeTransaction' not in txn.tag:
            continue

        txn_data = _parse_transaction_element(txn)
        if txn_data:
            txn_data["owner_name"] = owner_name
            txn_data["relationship"] = owner_relationship
            txn_data["filing_date"] = filing_date
            txn_data["derivative"] = False
            transactions.append(txn_data)

    # Extract derivative transactions (options, warrants)
    for txn in root.iter():
        if 'derivativeTransaction' not in txn.tag:
            continue

        txn_data = _parse_transaction_element(txn)
        if txn_data:
            txn_data["owner_name"] = owner_name
            txn_data["relationship"] = owner_relationship
            txn_data["filing_date"] = filing_date
            txn_data["derivative"] = True
            transactions.append(txn_data)

    return transactions


def _parse_transaction_element(txn_elem) -> Optional[Dict]:
    """Parse a single transaction element from Form 4 XML."""
    data = {}

    for child in txn_elem.iter():
        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        text = (child.text or "").strip()

        if tag == "transactionDate":
            # transactionDate has a <value> child
            for v in child.iter():
                if 'value' in v.tag and v.text:
                    data["transaction_date"] = v.text.strip()
        elif tag == "transactionCode":
            data["transaction_code"] = text
        elif tag == "transactionShares":
            for v in child.iter():
                if 'value' in v.tag and v.text:
                    try:
                        data["shares"] = float(v.text.strip())
                    except ValueError:
                        pass
        elif tag == "transactionPricePerShare":
            for v in child.iter():
                if 'value' in v.tag and v.text:
                    try:
                        data["price_per_share"] = float(v.text.strip())
                    except ValueError:
                        pass
        elif tag == "transactionAcquiredDisposedCode":
            for v in child.iter():
                if 'value' in v.tag and v.text:
                    data["acquired_disposed"] = v.text.strip()  # A=acquired, D=disposed
        elif tag == "sharesOwnedFollowingTransaction":
            for v in child.iter():
                if 'value' in v.tag and v.text:
                    try:
                        data["shares_after"] = float(v.text.strip())
                    except ValueError:
                        pass
        elif tag == "directOrIndirectOwnership":
            for v in child.iter():
                if 'value' in v.tag and v.text:
                    data["ownership_type"] = v.text.strip()  # D=direct, I=indirect

    if not data.get("transaction_code"):
        return None

    return data


# ============================================================================
# TRANSACTION CODE MEANINGS
# ============================================================================

TRANSACTION_CODES = {
    "P": "open_market_purchase",    # STRONGEST buy signal
    "S": "open_market_sale",        # sell signal
    "A": "grant_award",             # compensation, not informative
    "M": "option_exercise",         # converting options, moderate signal
    "F": "tax_withholding",         # forced sale for taxes, ignore
    "G": "gift",                    # not informative
    "J": "other",
    "C": "conversion",
    "D": "disposition_to_issuer",
    "E": "expiration",
    "H": "expiration",
    "I": "discretionary",
    "L": "small_acquisition",
    "W": "acquisition_by_will",
    "Z": "trust_deposit",
}


def classify_transaction(txn: Dict) -> str:
    """Classify a transaction as buy, sell, neutral, or exercise."""
    code = txn.get("transaction_code", "")
    acq_disp = txn.get("acquired_disposed", "")

    if code == "P":
        return "open_market_buy"
    elif code == "S":
        return "open_market_sell"
    elif code == "M":
        return "option_exercise"
    elif code == "A":
        return "grant"
    elif code == "F":
        return "tax_sale"
    elif code in ("G", "W", "Z"):
        return "neutral"
    elif acq_disp == "A":
        return "other_acquisition"
    elif acq_disp == "D":
        return "other_disposition"
    else:
        return "unknown"


# ============================================================================
# INSTITUTIONAL HOLDER DETECTION (SC 13D/G)
# ============================================================================

def fetch_institutional_filings(cik: str, session, limiter,
                                 lookback_days: int = 365,
                                 ticker: str = "",
                                 company_name: str = "") -> List[Dict]:
    """
    Check for SC 13D, SC 13G, and 13F-HR filings indicating institutional
    accumulation. Returns basic filing metadata (not parsed content).

    SC 13D/G are filed by the INVESTOR about the COMPANY — they do NOT
    appear under the target company's CIK submissions. We use EDGAR EFTS
    full-text search to find them.
    """
    from scraper import _safe_request

    institutional_forms = {"SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"}
    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")

    filings = []

    # ---- EFTS search for filings mentioning this company ----
    from scraper import _efts_search

    search_terms = []
    # Company name first — SC 13D filings reference target by name, not ticker
    if company_name:
        words = [w for w in company_name.split() if w.upper() not in
                 ('INC', 'INC.', 'CORP', 'CORP.', 'LTD', 'LLC', 'CO', 'CO.',
                  'THE', 'GROUP', 'HOLDINGS', 'INTERNATIONAL')]
        if len(words) >= 2:
            search_terms.append(f'"{words[0]} {words[1]}"')
    if ticker:
        search_terms.append(f'"{ticker}"')
    if not search_terms:
        search_terms.append(cik.lstrip("0"))

    seen_accessions = set()
    forms_list = list(institutional_forms)

    for query in search_terms:
        try:
            data = _efts_search(session, query, forms_list, cutoff, today)
            if not data or not isinstance(data, dict):
                continue

            # Parse hits — handle multiple possible response structures
            hits = []
            if 'hits' in data:
                h = data['hits']
                hits = h.get('hits', []) if isinstance(h, dict) else h
            elif 'filings' in data:
                hits = data['filings']

            for hit in hits:
                if isinstance(hit, dict) and '_source' in hit:
                    src = hit['_source']
                    raw_id = hit.get('_id', '')
                else:
                    src = hit
                    raw_id = hit.get('accession_number', '') or hit.get('accessionNo', '')

                accession = raw_id.split(':')[0] if ':' in raw_id else raw_id
                if not accession or accession in seen_accessions:
                    continue
                seen_accessions.add(accession)

                form_type = src.get('form_type', '') or src.get('formType', '') or ''
                form_type = _normalize_form(form_type)
                if form_type not in institutional_forms:
                    continue

                filings.append({
                    "form_type": form_type,
                    "filing_date": (src.get('file_date', '') or src.get('filedAt', '') or '')[:10],
                    "accession": accession,
                    "filer_name": src.get('entity_name', '') or src.get('companyName', '') or '',
                })
        except Exception as e:
            logger.warning(f"EFTS institutional search for {query}: {e}")

    # ---- Fallback: also check CIK submissions (some may be cross-referenced) ----
    try:
        cik_padded = cik.zfill(10)
        sub_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        sub_data = _safe_request(session, sub_url, limiter=limiter)
        if sub_data and isinstance(sub_data, dict):
            recent = sub_data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            for i in range(len(forms)):
                ft = forms[i] if i < len(forms) else ""
                ft = _normalize_form(ft)
                fd = dates[i] if i < len(dates) else ""
                acc = accessions[i] if i < len(accessions) else ""
                if ft in institutional_forms and fd >= cutoff and acc not in seen_accessions:
                    seen_accessions.add(acc)
                    filings.append({
                        "form_type": ft,
                        "filing_date": fd,
                        "accession": acc,
                    })
    except Exception as e:
        logger.warning(f"CIK {cik}: fallback institutional check failed: {e}")

    logger.info(f"CIK {cik}: found {len(filings)} institutional filings (SC 13D/G)")
    return filings


# ============================================================================
# SIGNAL COMPUTATION
# ============================================================================

def analyze_insider_activity(transactions: List[Dict],
                              institutional_filings: List[Dict] = None
                              ) -> Dict:
    """
    Analyze insider transactions and institutional filings to produce
    accumulation signals.

    Returns a structured dict with:
    - Summary counts and ratios
    - Cluster detection
    - Notable transactions
    - Accumulation score (1-100)
    """
    if institutional_filings is None:
        institutional_filings = []

    result = {
        "transaction_count": len(transactions),
        "institutional_filing_count": len(institutional_filings),
    }

    if not transactions and not institutional_filings:
        result["accumulation_score"] = 50  # neutral when no data
        result["signal"] = "no_data"
        result["summary"] = "No insider transaction or institutional filing data found."
        return result

    # ---- Classify transactions ----
    open_buys = []
    open_sells = []
    exercises = []
    grants = []
    other = []

    for txn in transactions:
        cls = classify_transaction(txn)
        shares = txn.get("shares", 0) or 0
        price = txn.get("price_per_share", 0) or 0
        value = shares * price

        enriched = {**txn, "classification": cls, "value": value}

        if cls == "open_market_buy":
            open_buys.append(enriched)
        elif cls == "open_market_sell":
            open_sells.append(enriched)
        elif cls == "option_exercise":
            exercises.append(enriched)
        elif cls == "grant":
            grants.append(enriched)
        else:
            other.append(enriched)

    # ---- Buy/sell metrics ----
    total_buy_shares = sum(t.get("shares", 0) or 0 for t in open_buys)
    total_sell_shares = sum(t.get("shares", 0) or 0 for t in open_sells)
    total_buy_value = sum(t.get("value", 0) for t in open_buys)
    total_sell_value = sum(t.get("value", 0) for t in open_sells)

    buy_count = len(open_buys)
    sell_count = len(open_sells)

    # Buy/sell ratio (by transaction count)
    if sell_count > 0:
        buy_sell_ratio = round(buy_count / sell_count, 2)
    elif buy_count > 0:
        buy_sell_ratio = float('inf')
    else:
        buy_sell_ratio = 1.0

    # Net buying (value)
    net_value = total_buy_value - total_sell_value

    result["open_market_buys"] = buy_count
    result["open_market_sells"] = sell_count
    result["option_exercises"] = len(exercises)
    result["grants"] = len(grants)
    result["total_buy_shares"] = total_buy_shares
    result["total_sell_shares"] = total_sell_shares
    result["total_buy_value"] = round(total_buy_value, 2)
    result["total_sell_value"] = round(total_sell_value, 2)
    result["net_value"] = round(net_value, 2)
    result["buy_sell_ratio"] = buy_sell_ratio if buy_sell_ratio != float('inf') else 999

    # ---- Unique buyers/sellers ----
    unique_buyers = set(t.get("owner_name", "").lower() for t in open_buys if t.get("owner_name"))
    unique_sellers = set(t.get("owner_name", "").lower() for t in open_sells if t.get("owner_name"))
    result["unique_buyers"] = len(unique_buyers)
    result["unique_sellers"] = len(unique_sellers)

    # ---- Cluster detection ----
    # Multiple insiders buying within a 30-day window = cluster
    cluster_score = 0
    if open_buys:
        buy_dates = []
        for b in open_buys:
            d = b.get("transaction_date") or b.get("filing_date", "")
            if d:
                try:
                    buy_dates.append(datetime.strptime(d[:10], "%Y-%m-%d"))
                except ValueError:
                    pass

        if len(buy_dates) >= 2:
            buy_dates.sort()
            # Check for clusters (3+ buys within 30 days by different people)
            for i in range(len(buy_dates)):
                window_end = buy_dates[i] + timedelta(days=30)
                cluster_buys = [b for b in open_buys
                                if _txn_date(b) and buy_dates[i] <= _txn_date(b) <= window_end]
                cluster_people = set(b.get("owner_name", "").lower()
                                     for b in cluster_buys if b.get("owner_name"))
                if len(cluster_people) >= 3:
                    cluster_score = 3  # strong cluster
                elif len(cluster_people) >= 2:
                    cluster_score = max(cluster_score, 2)
                elif len(cluster_buys) >= 3:
                    cluster_score = max(cluster_score, 1)

    result["cluster_score"] = cluster_score  # 0=none, 1=mild, 2=moderate, 3=strong

    # ---- Notable transactions ----
    notable = []

    # Large open market purchases (>$25K for microcaps is significant)
    for b in sorted(open_buys, key=lambda x: x.get("value", 0), reverse=True)[:5]:
        if b.get("value", 0) >= 10000:
            notable.append({
                "type": "large_buy",
                "owner": b.get("owner_name", "Unknown"),
                "role": b.get("relationship", ""),
                "shares": b.get("shares", 0),
                "value": b.get("value", 0),
                "date": b.get("transaction_date") or b.get("filing_date", ""),
                "shares_after": b.get("shares_after"),
            })

    # Large sales
    for s in sorted(open_sells, key=lambda x: x.get("value", 0), reverse=True)[:3]:
        if s.get("value", 0) >= 25000:
            notable.append({
                "type": "large_sell",
                "owner": s.get("owner_name", "Unknown"),
                "role": s.get("relationship", ""),
                "shares": s.get("shares", 0),
                "value": s.get("value", 0),
                "date": s.get("transaction_date") or s.get("filing_date", ""),
            })

    # CEO/CFO buys are always notable regardless of size
    for b in open_buys:
        role = (b.get("relationship", "") or "").lower()
        if any(title in role for title in ["ceo", "cfo", "chief executive", "chief financial", "president"]):
            if not any(n.get("owner") == b.get("owner_name") and n["type"] == "large_buy"
                       for n in notable):
                notable.append({
                    "type": "executive_buy",
                    "owner": b.get("owner_name", "Unknown"),
                    "role": b.get("relationship", ""),
                    "shares": b.get("shares", 0),
                    "value": b.get("value", 0),
                    "date": b.get("transaction_date") or b.get("filing_date", ""),
                })

    result["notable_transactions"] = notable[:10]

    # ---- Recent institutional filings ----
    recent_institutional = []
    for inf in institutional_filings:
        recent_institutional.append({
            "form_type": inf.get("form_type", ""),
            "filing_date": inf.get("filing_date", ""),
        })
    result["institutional_filings"] = recent_institutional

    # SC 13D = activist/large holder (>5%), SC 13G = passive large holder
    has_13d = any(f.get("form_type", "").startswith("SC 13D") for f in institutional_filings)
    has_13g = any(f.get("form_type", "").startswith("SC 13G") for f in institutional_filings)
    result["has_activist_holder"] = has_13d
    result["has_passive_large_holder"] = has_13g

    # ---- Compute accumulation score (1-100) ----
    score = 50  # neutral baseline

    # Open market buys are the strongest signal
    if buy_count > 0 and sell_count == 0:
        score += min(20, buy_count * 4)  # up to +20 for pure buying
    elif buy_count > sell_count:
        score += min(15, (buy_count - sell_count) * 3)
    elif sell_count > buy_count:
        score -= min(15, (sell_count - buy_count) * 3)

    # Cluster buying is very bullish
    score += cluster_score * 8  # up to +24

    # Multiple unique buyers
    if len(unique_buyers) >= 3:
        score += 10
    elif len(unique_buyers) >= 2:
        score += 5

    # Net buying value
    if net_value > 100000:
        score += 10
    elif net_value > 25000:
        score += 5
    elif net_value < -100000:
        score -= 10
    elif net_value < -25000:
        score -= 5

    # Institutional interest
    if has_13d:
        score += 8  # activist taking position
    if has_13g:
        score += 5  # passive accumulation
    if len(institutional_filings) >= 3:
        score += 5  # multiple institutional filings

    # CEO/CFO buying is a strong signal
    exec_buys = [n for n in notable if n["type"] == "executive_buy"]
    if exec_buys:
        score += 8

    # Clamp
    score = max(1, min(100, score))
    result["accumulation_score"] = score

    # ---- Signal classification ----
    if score >= 75:
        result["signal"] = "strong_accumulation"
    elif score >= 60:
        result["signal"] = "moderate_accumulation"
    elif score >= 45:
        result["signal"] = "neutral"
    elif score >= 30:
        result["signal"] = "moderate_distribution"
    else:
        result["signal"] = "strong_distribution"

    # ---- Human-readable summary ----
    parts = []
    if buy_count > 0:
        parts.append(f"{buy_count} open-market buys (${total_buy_value:,.0f})")
    if sell_count > 0:
        parts.append(f"{sell_count} open-market sells (${total_sell_value:,.0f})")
    if len(unique_buyers) > 1:
        parts.append(f"{len(unique_buyers)} different insiders buying")
    if cluster_score >= 2:
        parts.append("CLUSTER buying detected")
    if exec_buys:
        parts.append(f"C-suite buying: {', '.join(e['owner'] for e in exec_buys)}")
    if has_13d:
        parts.append("Activist investor filing (SC 13D)")
    if has_13g:
        parts.append("New institutional holder (SC 13G)")
    if not parts:
        parts.append("No significant insider activity detected")
    result["summary"] = "; ".join(parts)

    return result


def _txn_date(txn: Dict) -> Optional[datetime]:
    """Parse transaction date, returning datetime or None."""
    d = txn.get("transaction_date") or txn.get("filing_date", "")
    if d:
        try:
            return datetime.strptime(d[:10], "%Y-%m-%d")
        except ValueError:
            pass
    return None


# ============================================================================
# FORMATTING FOR LLM PROMPT
# ============================================================================

def format_insider_for_prompt(insider_data: Dict) -> str:
    """Format insider activity data for inclusion in LLM analysis prompt."""
    if not insider_data or insider_data.get("signal") == "no_data":
        return "(No insider transaction data available)"

    lines = [
        "=" * 60,
        "INSIDER & INSTITUTIONAL ACTIVITY",
        "=" * 60,
        "",
        f"Accumulation Score: {insider_data.get('accumulation_score', 50)}/100",
        f"Signal: {insider_data.get('signal', 'unknown').replace('_', ' ').upper()}",
        "",
        "--- Transaction Summary ---",
        f"Open-market buys:  {insider_data.get('open_market_buys', 0)} "
        f"(${insider_data.get('total_buy_value', 0):,.0f})",
        f"Open-market sells: {insider_data.get('open_market_sells', 0)} "
        f"(${insider_data.get('total_sell_value', 0):,.0f})",
        f"Option exercises:  {insider_data.get('option_exercises', 0)}",
        f"Net insider value: ${insider_data.get('net_value', 0):,.0f}",
        f"Buy/sell ratio:    {insider_data.get('buy_sell_ratio', 1.0)}",
        f"Unique buyers:     {insider_data.get('unique_buyers', 0)}",
        f"Unique sellers:    {insider_data.get('unique_sellers', 0)}",
        f"Cluster buying:    {'YES (level ' + str(insider_data.get('cluster_score', 0)) + ')' if insider_data.get('cluster_score', 0) > 0 else 'None detected'}",
        "",
    ]

    # Notable transactions
    notable = insider_data.get("notable_transactions", [])
    if notable:
        lines.append("--- Notable Transactions ---")
        for n in notable:
            typ = n.get("type", "").replace("_", " ").upper()
            lines.append(
                f"  {typ}: {n.get('owner', '?')} ({n.get('role', '?')}) — "
                f"{n.get('shares', 0):,.0f} shares @ ${n.get('value', 0):,.0f} "
                f"on {n.get('date', '?')}"
            )
        lines.append("")

    # Institutional filings
    inst = insider_data.get("institutional_filings", [])
    if inst:
        lines.append("--- Institutional Filings ---")
        for f in inst:
            lines.append(f"  {f.get('form_type', '?')} filed {f.get('filing_date', '?')}")
        if insider_data.get("has_activist_holder"):
            lines.append("  ⚠ ACTIVIST INVESTOR (SC 13D) — indicates >5% stake with active intent")
        if insider_data.get("has_passive_large_holder"):
            lines.append("  📊 Passive institutional holder (SC 13G) — indicates >5% passive stake")
        lines.append("")

    lines.append(f"Summary: {insider_data.get('summary', '')}")

    return "\n".join(lines)
