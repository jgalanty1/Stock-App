# SEC Analyzer — Full Code Review

**Date**: 2026-02-15
**Reviewer**: Claude Opus 4.6 (automated)
**Scope**: All 14 Python modules (~17,861 lines)

---

## FILE: `engine.py` (2674 lines)

### Bugs

🔴 BUG: [105-110] `get_scraper()` singleton is not thread-safe. Two threads could race past the `if _scraper is None` check and create two instances. Called from thread-pool workers (e.g., `_prefetch_ticker_data` at line 822) and the main thread concurrently.

🔴 BUG: [113-124] `get_db()` singleton is not thread-safe. Same issue as `get_scraper()`. Additionally, `get_db()` calls `get_scraper()` internally, so a deadlock-free lock ordering must be ensured if locking is added to both.

🔴 BUG: [589-601] `analysis_status` is a mutable global dict modified from multiple threads without a lock. `_progress` function (line 635) reads and mutates `analysis_status["message"]` from both threads without synchronization. Lines 960-961 (`analysis_status["failed_tier2"] += 1`) are in the main thread with no lock, but workers also modify it at lines 922 and 935 under `_t2_lock`.

🔴 BUG: [739-740] `scraper._save_state()` called outside the lock. At line 739, state save is triggered after exiting the `with _t1_lock:` block. Since `_save_state()` serializes `scraper.filings_index`, and other T1 worker threads are concurrently modifying filings, the save could capture a partially-updated state.

🔴 BUG: [502-503] File write inside `_do_update()` is not atomic. `json.dump` to `result_path` writes directly (no temp-file + rename pattern). If the process crashes mid-write, the result JSON is corrupted. Contrast with `save_config()` at line 45-49 which correctly uses atomic write via `.tmp` + `os.replace`.

🔴 BUG: [326] `turnaround_score * 0.10` can exceed 20. `turnaround_pts` is set to `round(ts * 0.10)` where `ts` could be 0-200+. The subsequent `min(20, ...)` calls only apply to additive bonuses, not to the base value.

🔴 BUG: [1003] Variable name `pf` shadows loop variable. The `for pf in all_filings_by_ticker.get(...)` loop uses `pf` as loop variable, then at line 1003 `with open(pf_path, ...) as pf:` reuses the same name as a file handle.

🔴 BUG: [1885-1886] Incorrect score normalization. `if cs is not None and cs <= 10` assumes scores on a 0-10 scale should be normalized to 0-100. But a valid composite_score of 8 (on a 0-100 scale) would be incorrectly multiplied to 80.

🔴 BUG: [1947-1948] Same normalization issue for `final_gem_score`. `if fgs is not None and fgs <= 10` would incorrectly multiply a legitimate score of 7/100 to 70.

🔴 BUG: [397-398] `ins` variable may be unbound or empty dict (falsy). When `ins` is `{}`, the result dict gets `'insider_signal': ''` and `'accumulation_score': None`, which is inconsistent with the default of 50 in the `insider_pts` calculation.

🔴 BUG: [2557] Potential division by zero. `g['gem_score'] / 100.0 if g['gem_score'] else 0.5` -- if `gem_score` is exactly `0`, this evaluates to `0.5` (since `0` is falsy). A score of 0 should yield 0.0, not 0.5.

### Warnings

🟡 WARN: [39] Silent swallowing of corrupt config. `json.JSONDecodeError` on `config.json` is silently ignored, returning `{}`. If the config file becomes corrupted, the user gets no indication and all API keys are silently lost.

🟡 WARN: [50-51] Orphaned temp file on write failure. If `os.replace(tmp, CONFIG_FILE)` fails, the `.tmp` file remains. The IOError handler logs but does not clean up.

🟡 WARN: [54-60] API keys stored in plaintext. `persist_key()` writes API keys directly to `config.json` on disk with no encryption.

🟡 WARN: [132-133] Silent exception swallowing in `sync_db()`. `except Exception: pass` hides all errors. Database sync failures are completely invisible.

🟡 WARN: [836, 844, 857, ...] Excessive bare `except Exception: pass` blocks. 40+ instances throughout the file make debugging extremely difficult.

🟡 WARN: [674-701] `_analyze_one_t1` reads files inside thread pool without size limits. With `TIER1_WORKERS=6` threads each reading large files simultaneously, this could cause memory pressure.

🟡 WARN: [718-719] Hardcoded cost estimation. `80000 * 0.15 / 1_000_000 + 2000 * 0.60 / 1_000_000` assumes fixed token counts. Actual costs could differ significantly.

🟡 WARN: [915-916] Hardcoded T2 cost estimation uses Sonnet prices, not the selected model. Uses $3/$15 (Claude Sonnet) pricing regardless of the actual `t2_model_name` selected.

🟡 WARN: [813] `edgar_session` shared across threads without thread-safety guarantee. `requests.Session` is not inherently thread-safe and is used in `ThreadPoolExecutor`.

🟡 WARN: [500] Ticker used in filename without sanitization. If `tk` contains path separators or special characters, this could create files in unexpected locations.

🟡 WARN: [1813] `prepared_dir` parameter allows path traversal. A caller could pass `prepared_dir='../../etc'` to read from arbitrary directories.

🟡 WARN: [2037] Import at function level without ImportError handling. `from market_data import batch_backtest, BACKTEST_WINDOWS` will crash with `ImportError` if market_data.py is missing.

🟡 WARN: [2471] `os.makedirs(os.path.dirname(output_path))` can fail with empty dirname. If `output_path` is just a filename, `os.path.dirname` returns `''`, and `os.makedirs('')` raises.

🟡 WARN: [2664-2669] `backup_data` traverses entire data directory without size limit. Could produce enormous zip files.

🟡 WARN: [2674] Module-level side effect on import. `restore_keys()` is called at import time, which reads `config.json` and modifies `os.environ`. Simply importing `engine.py` has side effects.

🟡 WARN: [941] `t2_executor` is never properly shut down on exception. Should be used as a context manager (`with ThreadPoolExecutor(...)`).

### Style

🔵 STYLE: [774] `top_companies` set is computed but never used.

🔵 STYLE: [252-256] `_ORIG_MAX` duplicates `_DEFAULT_WEIGHTS`. These should be a single constant.

🔵 STYLE: [977-985, 1550-1558] `_same_family()` function defined twice. Identical helper defined inside both `run_analysis()` and `prepare_analysis()`.

🔵 STYLE: [1039-1051, 1612-1624] Insider signal data construction duplicated.

🔵 STYLE: [822-880, 1449-1507] `_prefetch_ticker_data()` defined twice. Nearly identical ~60-line functions in both `run_analysis()` and `prepare_analysis()`. DRY violation.

🔵 STYLE: [1080-1095, 1643-1658] Cross-reference context construction duplicated.

🔵 STYLE: [1098-1137, 1660-1700] Market data context construction duplicated. Same ~40 lines of formatting logic in both functions.

🔵 STYLE: [12] `traceback` imported but never used.

🔵 STYLE: [903] `_t2_filing_times` list is populated but never read. Dead code.

🔵 STYLE: [917] `analysis_status["tier2_last_completed"]` set but never consumed.

🔵 STYLE: [2068-2069] Magic number `CALIBRATION_THRESHOLD = 15` defined locally but should be module-level.

🔵 STYLE: [186-187] Integer division could produce unexpected formatting. `int((min_market_cap or 20_000_000)/1e6)` truncates rather than rounds.

---

## FILE: `cli.py` (1197 lines)

### Bugs

🔴 BUG: [159-160] `--json` flag on `status` command outputs both Rich table AND JSON. When `args.json` is True, the function first prints the full Rich table (lines 134-157), then also dumps JSON. Same pattern affects `cmd_gems`, `cmd_potential`, `cmd_company`, `cmd_backtest`, `cmd_calibrate`, `cmd_outcomes`, `cmd_prepare`, `cmd_import_analysis`, and `cmd_scan`.

🔴 BUG: [563] `_fmt_ret` function defined inside a `for` loop, creating a new closure on every iteration. Unnecessary object churn.

🔴 BUG: [940-941] Section comment says "EXPORT COMMAND" but the function defined is `cmd_portfolio`. Misleading labeling.

🔴 BUG: [1047] Global `--json` flag is dead code. Every subparser re-declares `--json`, overwriting the namespace attribute. The global flag can never be read.

🔴 BUG: [700-725] `cmd_watchlist` silently ignores its `--json` flag. The argparse definition promises it, but the handler never checks `args.json`.

### Warnings

🟡 WARN: [96, 105, 114] API keys entered via `input()` are echoed to the terminal in plaintext. Should use `getpass.getpass()`.

🟡 WARN: [84-119] `cmd_setup` has no exception handling for `engine.persist_key()` failures. Success message is unconditional.

🟡 WARN: [1020-1021] `cmd_export` will crash with `FileNotFoundError` if path doesn't exist. No try/except around `os.path.getsize(path)`.

🟡 WARN: [1034-1036] Same issue in `cmd_backup` — `os.path.getsize(path)` with no guard.

🟡 WARN: [742-745] `cmd_outcomes` reaches into internal `market_data._price_cache` to clear it. Fragile — relies on internal implementation detail.

🟡 WARN: [755] `cmd_outcomes` calls `engine.get_scraper()` to directly iterate `scraper.filings_index`. Breaks abstraction boundary.

🟡 WARN: [328] Potential `TypeError` if `diff_signal` is explicitly `None`. `'improving' in diff` will raise if `diff` is `None`.

🟡 WARN: [470] Rich markup inside conditional expression could produce malformed markup if `gem` is non-numeric.

🟡 WARN: [721] Rich markup in format string width specifier produces misaligned output. `_cyan(ticker)` returns markup string, `:8` operates on markup length.

🟡 WARN: [597] `getattr(args, 'multi_window', False)` masks potential argparse configuration bugs silently.

🟡 WARN: [1019] `cmd_export` and `cmd_backup` lack `--json` flags despite project convention.

### Style

🔵 STYLE: [53-64] `_FallbackConsole` is minimal and does not implement all methods used. `print(str(a))` on a Rich `Table` object produces unhelpful `__repr__`.

🔵 STYLE: [170] Ternary for `steps` could use `or` idiom.

🔵 STYLE: [299] `limit = args.limit or len(gems)` treats `0` as falsy. `--limit 0` evaluates to `len(gems)`.

🔵 STYLE: [227-233] `model_map` is recreated on every invocation of `cmd_analyze`.

🔵 STYLE: [356] Fallback table format omits columns that Rich table includes.

🔵 STYLE: [477-479] Non-Rich fallback in `cmd_company` shows far less detail than Rich path.

🔵 STYLE: [961] Redundant import of `Table` inside `cmd_portfolio` — already imported at module level.

🔵 STYLE: [33] `sys.path.insert(0, ...)` modifies Python path at module level.

🔵 STYLE: [1015-1036] `cmd_export` and `cmd_backup` have no `--json` support despite project convention.

---

## FILE: `llm_analysis.py` (1978 lines)

### Bugs

🔴 BUG: [277] `_last_token_usage` assigned without `global` declaration in `_call_anthropic`. The assignment on line 277 creates a local variable instead of updating the module-level global. `get_last_token_usage()` will never reflect Anthropic API calls.

🔴 BUG: [116-129] Race condition on `_cumulative_usage` and `_last_token_usage` — not thread-safe. Module docstring states analysis uses `ThreadPoolExecutor` with locks, but `_track_usage()` mutates global mutable dicts without any lock.

🔴 BUG: [1453-1454] Score normalization falsely triggers on valid score of 10. `if cs is not None and cs <= 10` normalizes a composite score of exactly 10 to 100. A legitimately terrible company scoring 8-10 would be incorrectly inflated to 80-100. Same issue at line 1655 for `final_gem_score`.

🔴 BUG: [1739] `filing_lookup` keyed by ticker causes collisions for multiple filings of the same ticker. Only the last filing is kept. Tier 2 analysis may use the wrong filing text.

🔴 BUG: [418-421] Truncated JSON repair double-quotes a string ending with `"`. After truncating to `last_good`, if the string already ends with `"`, appending another `"` creates invalid JSON.

🔴 BUG: [351] Unreachable `return None` in `_call_llm`. Dead code that could mask logic changes.

🔴 BUG: [1734] `tier2_count` is always at least 1 even with zero tier1 results. `max(1, int(len(tier1_sorted) * tier2_percentile))` reports `tier2_eligible: 1` when it should be 0.

### Warnings

🟡 WARN: [149, 163] API keys read from environment — could be logged in stack traces during client initialization.

🟡 WARN: [200-210] Overly broad exception catching in `_call_openai`. Catches `TypeError`, `AttributeError`, `KeyError` in response parsing and may retry on those, wasting API calls.

🟡 WARN: [289-297] Same overly broad exception catching in `_call_anthropic`.

🟡 WARN: [202-204, 291-293] Error classification by string matching is fragile. `any(kw in err_str for kw in ('rate', '429', ...))` could match non-transient errors like "Invalid rate parameter".

🟡 WARN: [244-249] `effort` parameter silently ignored with try/except pass. The `try` block only manipulates a dict (never raises), so the try/except does nothing.

🟡 WARN: [434] Aggressive JSON repair loop could be CPU-intensive. For a 100KB response, could attempt ~50,000 JSON parses.

🟡 WARN: [466-476, 1623-1633] Filing text directly interpolated into prompts — prompt injection vulnerability. SEC filings are public documents that anyone can create.

🟡 WARN: [1627] No total prompt size validation before sending to the API. Combined prompt content can exceed model context limits.

🟡 WARN: [1823] `estimate_cost` computes `tier2_count` differently than `run_tiered_analysis`. Estimate uses `int()` (truncation), actual run uses `max(1, int(...))`.

🟡 WARN: [1909-1938, 1941-1977] Legacy functions make full API calls — expensive for simple compatibility. If legacy code calls both on the same text, it doubles API cost.

🟡 WARN: [93-95, 98-100] `get_cumulative_usage()` returns shallow copy. The nested `by_model` dict is still mutable by the caller.

🟡 WARN: [78] `import time as _time` placed after the initial import block. Violates PEP 8 import ordering.

### Style

🔵 STYLE: [67] `EFFORT_LEVELS["low"]` has `thinking_budget: 0` — same as no thinking. Makes "low" functionally identical to non-thinking path.

🔵 STYLE: [1043-1334] `TIER2_OPUS_DEEP_ANALYSIS_PROMPT` is a near-complete duplicate of `TIER2_DEEP_ANALYSIS_PROMPT`. ~90% duplicated content. Changes must be made in two places.

🔵 STYLE: [1337-1398] `OPUS_DEEP_ANALYSIS_INSTRUCTIONS` is a third copy of the Opus analysis instructions.

🔵 STYLE: [1486-1507] `analyze_trajectory_tier1` silently returns `None` on JSON parse failure. Inconsistent with `analyze_filing_tier1` which raises `RuntimeError`.

🔵 STYLE: [1718-1719] Dead else branch. `analyze_filing_tier1` either returns a dict or raises, so `if t1_result:` / `else:` branch is dead code.

🔵 STYLE: [1765-1766] Same dead else branch for tier2 in `run_tiered_analysis`.

🔵 STYLE: [466-476] `SYSTEM_PROMPT_GEM_FINDER` instructs "Start with {" but OpenAI call uses `response_format={"type": "json_object"}`. Redundant for OpenAI.

🔵 STYLE: [88-90] Global mutable state for usage tracking makes testing difficult.

🔵 STYLE: [1820-1896] `estimate_cost` uses hardcoded token estimates that may drift from reality. The Opus prompt alone is ~1300 lines.

---

## FILE: `calibration.py` (1459 lines)

### Bugs

🔴 BUG: [1057] Population standard deviation used instead of sample standard deviation in `walk_forward_validate` stability computation. Uses `/ len(vals)` (population) while line 739 correctly uses `/ (n - 1)` (sample). With small fold counts (3-5), this systematically inflates stability scores.

🔴 BUG: [1274-1281] Walk-forward result for N=10-14 is computed but the initial dict is immediately overwritten. Lines 1275-1278 build a dict with "high overfitting risk" warning, then lines 1279-1281 overwrite it with the actual result. Warning is silently lost.

🔴 BUG: [1358] Potential `IndexError` if `ranked_signals` is empty. Logger accesses `ranked_signals[0][0]` and `ranked_signals[0][1]` without checking.

🔴 BUG: [979] Off-by-one in leave-one-out loop start index. For n=5, `range(max(5, n//2), n)` yields `range(5, 5)` which is empty — no folds produced.

🔴 BUG: [934-935] `temporal_train_test_split` can produce an empty test set when `n < min_train_size`, violating the function's own contract.

🔴 BUG: [396] Division by zero when `n` is 0. `math.sqrt(n)` in `sig_threshold` calculation. Guard only checks `n >= 10`, not `n > 0`.

### Warnings

🟡 WARN: [305] Division by zero if `half_life_days` is 0. `decay_rate = math.log(2) / half_life_days` — no validation or guard.

🟡 WARN: [1172-1181, 1389, 1412, 1420] File I/O without encoding specification. All `open()` calls lack explicit `encoding='utf-8'`.

🟡 WARN: [1407-1413] `_save_report` has no error handling. If disk is full or permissions wrong, the report result is lost after all computation.

🟡 WARN: [1369-1377] `apply_weights` has no error handling. Same issue: no try/except around file write.

🟡 WARN: [131-133, 1221-1222, 1259-1260] Repeated O(n) linear scans of `scraper.filings_index`. For every ticker, full scan. Total complexity O(T * F * 3). Should build a lookup dict.

🟡 WARN: [124-129] Repeated O(n) scans of `potential_companies` and `scraper.universe`. O(T^2) for potential_companies lookups.

🟡 WARN: [1217-1228, 1257-1266] Duplicate code for building filing date lists. Built with identical logic twice.

🟡 WARN: [371] `temporal_weights` parameter typed as `List[float]` but defaults to `None`. Should be `Optional[List[float]]`.

🟡 WARN: [438] Potential division by zero in `_weighted_avg` fallback if `quartile` is empty.

🟡 WARN: [150] `if ins else 50` check is misleading since empty dict `{}` is falsy. `ins.get('accumulation_score', 50)` on an empty dict would also return 50.

🟡 WARN: [757] Redundant `if n > 0` guard inside block that only executes when `n >= 1`.

### Style

🔵 STYLE: [29] `Tuple` imported from typing but never used.

🔵 STYLE: [683] Unused parameters `cps` and `signal_count` in `compute_score_confidence`.

🔵 STYLE: [1297] Variable `mw_tickers_filtered` assigned but never used. Dead write.

🔵 STYLE: [1021] `test_corrs` computed but never used in walk-forward validation. Wasted CPU.

🔵 STYLE: [606-607, 871-872, 1042-1043] CPS component list `['catalyst_strength', ...]` defined as local variable in 5 different places. Should be module-level constant.

🔵 STYLE: [100-108] `DIFF_SIGNAL_MAP` has asymmetric value spacing but no documentation.

🔵 STYLE: [149] Double-default pattern: `.get('insider_data', {}) or {}` — the `or {}` is redundant unless value is explicitly `None`.

🔵 STYLE: [970-973] Verbose unpacking. Could use `zip(*combined)`.

🔵 STYLE: [1225] Lambda parameter `f` shadows outer variable name.

---

## FILE: `scraper.py` (1908 lines)

### Bugs

🔴 BUG: [37-46] Fallback User-Agent defeats the purpose of the validation check. Sets fake `"SECFilingAnalyzer/1.0 (user@configure-me.com)"` that SEC EDGAR will eventually block.

🔴 BUG: [154-166] Singleton session `_get_session()` is not thread-safe. Global `_session` without any lock. If two threads call simultaneously before initialization, race condition.

🔴 BUG: [437-438] `min_market_cap=0` treated as "not provided". `int(min_market_cap) if min_market_cap else MIN_MARKET_CAP` — `0` is falsy so it becomes `MIN_MARKET_CAP`. Should use `is not None`.

🔴 BUG: [1287] Unhandled exception from `future.result()` in step2 thread pool. A single company failure crashes the entire step2 process, losing all progress since the last `_save_state()`.

🔴 BUG: [1369] Unhandled exception from `future.result()` in step3 thread pool. Same issue for download workers.

🔴 BUG: [1475] Unhandled exception from `future.result()` in step4 thread pool. Same pattern for insider data fetching.

🔴 BUG: [976] `max(best_html_size, best_xml_size)` reports incorrect size in log. Should report the size of the chosen document, not the max of both.

🔴 BUG: [1714-1715] Score normalization converts scores <= 10 to 10x. A filing legitimately scoring 8 out of 100 would become 80. No reliable way to distinguish scales.

### Warnings

🟡 WARN: [98-101] Relative paths for data directories. If working directory changes, paths resolve incorrectly.

🟡 WARN: [290, 337, 376] Diagnostic `test_fmp_api` bypasses rate limiter. Direct `session.get()` calls skip `_fmp_limiter`.

🟡 WARN: [292, 339] API key leaked in `response_preview` of diagnostic results. FMP may echo key in error responses.

🟡 WARN: [878-879] Potential path traversal via ticker/form_type in filename. `filename = f"{ticker}_{form_type}_{date}.html"` — no sanitization.

🟡 WARN: [1140-1146] JSON state loading has no error handling for corrupt files. `json.load(f)` without try/except crashes with unhelpful JSONDecodeError.

🟡 WARN: [1149-1153] Non-atomic state save risks data corruption. Two JSON files written sequentially. If process crashes between writes, files out of sync.

🟡 WARN: [1149-1153] `_save_state` called from within lock AND outside locks. Multiple threads could trigger state save, causing data corruption.

🟡 WARN: [1054-1055] Large filing read without size limit. `f.read()` on SEC filings that can be 20-50MB of HTML.

🟡 WARN: [200-204] Double JSON parse attempt on responses. A response with `Content-Type: text/html` that is valid JSON gets parsed as JSON.

🟡 WARN: [693-705] EFTS response parsing has no JSON decode error handling. `data = resp.json()` without try/except.

🟡 WARN: [894-896] Mutating `filing["filing_url"]` is a side effect on shared data without lock protection.

🟡 WARN: [1003] Regex `r'<\?xml.*'` with `re.DOTALL` could capture huge content into memory.

### Style

🔵 STYLE: [24, 982] `import re` at module level and again inside `_resolve_filing_url`. Redundant.

🔵 STYLE: [187, 259, 175] `import requests` imported inside multiple functions inconsistently.

🔵 STYLE: [95] `_MIN_INTERVAL` computed at module level but never used. Dead code.

🔵 STYLE: [66-67] `TARGET_FORM_TYPES` is just an alias for `TARGET_FORM_TYPES_CIK`. Ambiguous which to use.

🔵 STYLE: [636-640] First `if i < len(forms)` check is always true when iterating `range(len(forms))`.

🔵 STYLE: [1706-1716] `_normalize_score` defined as method but has no dependency on `self`. Should be static.

🔵 STYLE: [1904-1907] `clear_universe` doesn't log the action. Inconsistent with other state-modifying methods.

🔵 STYLE: [1513-1523] Market cap bucketing doesn't account for companies below $20M.

🔵 STYLE: [599-601] Logging dropped-CIK tickers truncates at 20 without indicating how many were omitted.

---

## FILE: `filing_signals.py` (1860 lines)

### Bugs

🔴 BUG: [288] Duplicate entry 'doubt' in `LM_UNCERTAINTY` set. `'doubt', 'doubtful', 'doubt'` — the second `'doubt'` was likely meant to be a different word (possibly `'doubts'`), meaning a word may be missing from the dictionary.

🔴 BUG: [1016] Score can exceed 100 before clamping in `compute_positive_similarity`. The formula at extreme values is unbounded, though `min(100, ...)` on line 1026 saves it.

🔴 BUG: [1385] `net_purch` calculated as `buyers - sellers` (unique person count) instead of net transaction count. Produces semantically different data than the summary path, leading to inconsistent scoring.

🔴 BUG: [1629] `_NO_DATA` dict is missing keys that downstream code expects. While `.get()` prevents crashes, the `tiers` dict at line 1715 does `v['score']` (not `.get('score')`) — fragile if any signal lacks 'score'.

🔴 BUG: [135] `_extract_section` skips first 100 characters for "next item" search using fixed offset. If the header is shorter than 100 chars, it skips actual section content.

### Warnings

🟡 WARN: [45-57] STOPWORDS set rebuilt on every call to `_tokenize()`. Should be module-level constant.

🟡 WARN: [58] Regex `r'\b[a-z]{2,}\b'` compiled on every call. Appears in 8+ locations. Should be compiled once at module level.

🟡 WARN: [510-514] Regex for counting risk factor headings has potential for pathological backtracking (ReDoS) with `[A-Za-z\s,'-]{10,80}?`.

🟡 WARN: [649-650] Entity extraction regex matches false positives like section headers. Matches any sequence of 2-5 capitalized words.

🟡 WARN: [655-656] Dollar amount regex misses common SEC formats: `$10.5M`, `USD 10 million`, `10 million dollars`.

🟡 WARN: [1191] Redundant import `from collections import Counter as C2`. `Counter` already imported at line 35.

🟡 WARN: [997-998] Redundant re-tokenization in `compute_positive_similarity`. Full text tokenized again just to count positive words, despite earlier pass doing same work.

🟡 WARN: [882-885] Four separate O(n) iterations over word list for LM sentiment. A single pass would be 4x more efficient.

🟡 WARN: [1263] Sentence splitting regex `r'[.!?]+'` is naive. Splits on abbreviations (U.S., Inc., Corp.), decimals (3.14), dates (Dec. 31). Biases fog index downward.

🟡 WARN: [1469, 1530] `text.lower()` called on potentially 500K+ character strings just to search for ~12 phrases. Should search only first portion.

🟡 WARN: [1528] Redundant `import re` inside `detect_revenue_concentration`. Already imported at line 31.

🟡 WARN: [744, 749] `from datetime import timedelta` imported conditionally inside two branches. Should be at module level.

🟡 WARN: [712] Parameter `filing_count_for_ticker` declared but never used in `compute_filing_timeliness_signal`.

🟡 WARN: [1644] `compute_filing_timeliness_signal` called without `filing_count_for_ticker` parameter — parameter is dead code.

### Style

🔵 STYLE: [240-242] Overlap between LM_NEGATIVE and LM_UNCERTAINTY sets is intentional per LM dictionary but undocumented.

🔵 STYLE: [315-338] Overlap between LM_NEGATIVE and LM_LITIGIOUS sets also undocumented.

🔵 STYLE: [450-452] Functions not in numeric order (Signal 1, 3, 6, 8, 2, 4, 5, 7, 9, 10). Inconsistent ordering makes navigation harder.

🔵 STYLE: [1452-1597] `detect_amendment_restatement` and `detect_revenue_concentration` are not part of the 10-signal framework. Placement is confusing.

🔵 STYLE: [1807] Falsy check on `size_change_pct` will skip display when change is exactly 0.0. Should use `is not None`.

🔵 STYLE: [1844] Same falsy-check issue with `fog_index`. A value of 0.0 would be suppressed.

🔵 STYLE: [104-118] Compiled regex patterns use inconsistent naming.

---

## FILE: `market_data.py` (1133 lines)

### Bugs

🔴 BUG: [29] `FMP_API_KEY` read once at module load, never refreshed. If `engine.persist_key()` sets the key later, the module-level constant remains empty. Every function that checks `if not FMP_API_KEY` will continue to skip FMP requests.

🔴 BUG: [191] `price_source` detection is wrong. `"fmp" if FMP_API_KEY else "yfinance"` reports source based on key existence, not which provider actually returned data.

🔴 BUG: [267] Benchmark cache file-load skipped after first failed lookup. `if not _benchmark_cache` becomes False once any entry is stored. Later calls for different dates miss disk cache.

🔴 BUG: [418] Median calculation incorrect for even-length lists. `sorted(gains)[len(gains) // 2]` returns element at index n//2, not the average of middle two elements.

### Warnings

🟡 WARN: [43-56] Global `_session` is not thread-safe. Used from `ThreadPoolExecutor` in engine.py.

🟡 WARN: [86, 144] `_price_cache` is not thread-safe. Check-then-set TOCTOU race across threads.

🟡 WARN: [247, 260, 317-320] `_benchmark_cache` has race conditions on read/write including file I/O.

🟡 WARN: [319-320] Benchmark cache file written without atomic write. Process crash mid-write corrupts file.

🟡 WARN: [337-343] Backtest cache loaded without validation. Missing `ticker` or `filing_date` keys throw swallowed `KeyError`.

🟡 WARN: [370-375] Database connection created per-filing in loop. New `SECDatabase` instance for every single filing.

🟡 WARN: [382] Fixed `time.sleep(0.3)` rate limiting is crude. Doesn't differentiate between FMP and yfinance.

🟡 WARN: [74] `resp.json()` called without catching `JSONDecodeError`. FMP could return non-JSON body.

🟡 WARN: [127] yfinance `row.get('Open', 0)` — recent yfinance versions changed column naming conventions.

🟡 WARN: [486] Misleading `shortTermCoverageRatios` used as `short_ratio`. This is a liquidity ratio, NOT short interest data. Feeds incorrect data into squeeze analysis.

🟡 WARN: [846] `import re` inside function body. Unconventional and adds overhead.

🟡 WARN: [644-646] Division by zero when `avg_pe` or `avg_ptb` is very small. Also inconsistent fallbacks (`None` vs `0`).

### Style

🔵 STYLE: [20] `Tuple` imported but only used once.

🔵 STYLE: [82] Comment references external numbering system (#13, #14, #5, etc.) without context.

🔵 STYLE: [488-489] Inconsistent field naming in fallback short interest data. Primary path has 4 fields, fallback has 1.

🔵 STYLE: [117-118] `datetime.now()` should be `datetime.utcnow()` or timezone-aware. Price data dates are in exchange time zones.

🔵 STYLE: [625] No coordination with global `RateLimiter`. Uses ad-hoc `time.sleep(0.2)`.

🔵 STYLE: [36] `BACKTEST_WINDOWS` could be a tuple since it's never mutated.

🔵 STYLE: [151] `if not history` is ambiguous — catches both `None` and `[]`.

---

## FILE: `inflection_detector.py` (1092 lines)

### Bugs

🔴 BUG: [791-794] Conditional expression has wrong precedence. `f"Deferred revenue growing..." if growth else ""` — the ternary evaluates on the entire f-string concatenation, not just the percentage part. When `_growth_rate` returns `None`, the entire `description` field becomes an empty string instead of a meaningful partial description.

### Warnings

🟡 WARN: [380-382] `_growth_rate` returns `None` for zero priors, but callers only check `if gr is not None` before appending. Some downstream calculations (line 466) treat `0` revenue as falsy, silently skipping analysis.

🟡 WARN: [741] Variable `rev` is reassigned, shadowing the `rev` already retrieved on line 407. Works because both point to same data, but confusing.

🟡 WARN: [466] `annual_run_rate = latest_rev * 4 if rev[-1].get("is_quarterly") else latest_rev` — the `is_quarterly` field could be `False` for a half-year period, leading to an incorrect run rate.

### Style

🔵 STYLE: [24] `import math` is unused — no `math.` references anywhere in the file.

🔵 STYLE: [25] `from datetime import datetime, timedelta` — `timedelta` is never used.

🔵 STYLE: [139-144] `LONG_TERM_DEBT_CONCEPTS` duplicates entries from `DEBT_CONCEPTS` on lines 86-89. `DEBT_CONCEPTS` at line 85 is dead code.

---

## FILE: `catalyst_extractor.py` (1197 lines)

### Warnings

🟡 WARN: [1018-1041] Dead code: `if prior_gc and not cur_gc` condition can never be True when `going_concern['found']` is True (which is the guard for entering this block). "Going concern removed" detection is unreachable.

🟡 WARN: [951-959] Going concern catalysts can be double-counted — once from `extract_going_concern_catalysts` (subcategories `'resolved'`, `'new_doubt'`, `'mentioned'`) and again from the `prior_text` comparison block (subcategories `'removed_vs_prior'`, `'added_vs_prior'`).

### Style

🔵 STYLE: [25] `from collections import defaultdict` — imported but never used.

🔵 STYLE: [24] `from typing import ... Tuple` — `Tuple` is imported but never used.

---

## FILE: `insider_tracker.py` (729 lines)

### Bugs

🔴 BUG: [37-39] `FORM4_NS` namespace dict is defined but never used. The namespace URL points to an EDGAR search URL, not an XML namespace URI. Dead and incorrect constant.

🔴 BUG: [117-132] `ET.fromstring(content)` on untrusted XML from EDGAR without any size/entity restrictions. Vulnerable to XML entity expansion attacks (billion laughs / XML bomb). Should use `defusedxml.ElementTree`.

🔴 BUG: [110] `_safe_request` returns parsed JSON by default, but Form 4 files are XML. The function tries `resp.json()` first, which raises an exception, then falls back to `resp.text`. Works but unnecessarily brittle.

### Warnings

🟡 WARN: [295-304] Two separate deferred imports from scraper with unused imports. `_edgar_limiter` on line 295 is unused. `edgar_lim` alias on line 304 is also unused.

🟡 WARN: [502-513] Cluster detection is O(n^2). Each buy date creates a 30-day window and re-scans all buys. Acceptable for ~30 Form 4 filings but inefficient.

🟡 WARN: [137-149] Owner relationship parsing uses `root.iter()` with tag substring matching. Could match nested elements unintentionally. A more precise XPath would be safer.

🟡 WARN: [66] Date comparison uses string comparison of ISO-format dates. Works but fragile if format changes.

### Style

🔵 STYLE: [37-39] Dead constant `FORM4_NS` — should be removed.

---

## FILE: `forensics.py` (543 lines)

### Warnings

🟡 WARN: [138] `hits.sort(key=lambda x: int(re.search(r'\((\d+)\)', x).group(1)), ...)` — if `re.search` returns `None`, `.group(1)` raises `AttributeError`. Fragile: if a phrase contains parenthesized numbers, the regex matches the wrong group.

🟡 WARN: [25-26] `HEDGING_WORDS` set contains `"uncertain"` twice. Duplicate suggests a different word was intended.

🟡 WARN: [423-449] `_diff_sentences` has O(n*m) complexity with SequenceMatcher, which is itself O(n^2) per comparison. For large filings (200 sentences cap each), could take seconds to tens of seconds.

🟡 WARN: [52-60] `SPECIFICITY_MARKERS` regex matches false positives: addresses, phone numbers, `$.`, `$,`.

### Style

🔵 STYLE: [15] `from collections import Counter` — imported but never used.

🔵 STYLE: [80-84] `MDA_HEADER` regex pattern duplicates the pattern from `catalyst_extractor.py`. Both modules define section-extraction logic independently, violating DRY.

---

## FILE: `database.py` (731 lines)

### Bugs

🔴 BUG: [208-209] `PRAGMA journal_mode=WAL` and `PRAGMA synchronous=NORMAL` executed on every `_conn()` call. WAL mode is persistent and only needs to be set once. `synchronous=NORMAL` with WAL means writes can be lost on OS crash — risky for financial data.

### Warnings

🟡 WARN: [627-679] `INSERT OR REPLACE` clobbers partial updates. Calling `upsert_filing` with a subset of fields resets all other fields to defaults. A simple score update could silently mark a filing as not-downloaded and not-analyzed.

🟡 WARN: [203-213] `_conn()` context manager creates a new connection for every operation. PRAGMAs re-executed each time. Significant overhead for batch operations.

🟡 WARN: [166-178] Explicit `conn.commit()` after `conn.executescript(SCHEMA_SQL)` is redundant. `executescript` implicitly commits.

🟡 WARN: [487-503] `save_backtest` serializes `benchmark_return` separately but also includes it in the full `result` dict. Data duplication in storage.

### Style

🔵 STYLE: [19] `import time` is imported but never used.

🔵 STYLE: [691-697] `_row_to_filing_dict` merges `extra` JSON into row dict. If extra contains keys colliding with column names, they silently overwrite column values.

---

## FILE: `monitor.py` (547 lines)

### Bugs

🔴 BUG: [56, 80-81, 152, 165] `SEC_REQUEST_DELAY` imported from `scraper` but does NOT exist in `scraper.py`. Import will raise `ImportError` at runtime. Additionally, `_safe_request` does not accept a `delay` keyword argument, so lines 80-81 and 165 would also raise `TypeError`. Both `search_new_filings_efts()` and `resolve_filing_urls()` are completely broken.

🔴 BUG: [80-81] `_safe_request(session, url, params=params, delay=SEC_REQUEST_DELAY)` — `_safe_request` signature is `(session, url, params=None, limiter=None, retries=3)`. The `delay` kwarg would raise `TypeError: got an unexpected keyword argument 'delay'`.

### Warnings

🟡 WARN: [460-546] `DailyScheduler._run_loop` runs in a daemon thread and accesses `self.monitor_state` and `self.scraper` without locks. If `run_now()` is called while the background loop triggers a scheduled run, both could call `run_daily_check` simultaneously — race conditions on shared data.

🟡 WARN: [240-244] `MonitorState.save()` writes two files non-atomically. If process crashes between writes, state and log will be inconsistent.

🟡 WARN: [321] `path = download_filing(filing, scraper.filings_dir, session)` — if `scraper.filings_dir` is not set, raises `AttributeError`. No guard.

🟡 WARN: [344] `scraper._save_state()` calls a private method. Couples monitor to scraper internals.

🟡 WARN: [39] `EFTS_SEARCH_URL` defined as module constant and also referenced in docstring. Two sources of truth.

---

## FILE: `analyzer.py` (822 lines)

### Warnings

🟡 WARN: [437-443] `print()` statements used for user feedback instead of `logging.getLogger(__name__)`. Cannot be controlled by `--verbose` and pollutes stdout when `--json` output is requested.

🟡 WARN: [609-624] `_merge_flags` deduplication is too loose — single-word overlap like "Risk" in titles could incorrectly merge unrelated flags from different sources.

🟡 WARN: [711-714] Sentiment level determination `max(keys, key=lambda k: scores[k])` breaks ties arbitrarily. The highest enum value wins, which may not reflect actual sentiment.

🟡 WARN: [640-641] Flag detection runs all regex patterns on entire filing. Patterns like `r"going concern"` match safe-harbor boilerplate, causing high false-positive rates.

🟡 WARN: [467-477] `llm_analyze_flags(current_text, prior_text)` called with original text while regex analysis uses lowercased text. Correct now but fragile for future changes.

### Style

🔵 STYLE: [8] `from dataclasses import dataclass, field, asdict` — all three imports are unused. No dataclasses in this module.

🔵 STYLE: [437-443, 467, 477, 489, 493] Six `print()` calls should be `logger.info()` or `logger.debug()` per project conventions.

---

# SUMMARY

## Total Issues by Severity (all 14 modules)

| Severity | Count |
|----------|-------|
| 🔴 BUG   | **53** |
| 🟡 WARN  | **118** |
| 🔵 STYLE | **84** |
| **Total** | **255** |

## Per-File Breakdown

| File | 🔴 BUG | 🟡 WARN | 🔵 STYLE | Total |
|------|--------|---------|----------|-------|
| engine.py | 11 | 16 | 12 | 39 |
| scraper.py | 8 | 12 | 9 | 29 |
| llm_analysis.py | 7 | 12 | 9 | 28 |
| calibration.py | 6 | 11 | 9 | 26 |
| filing_signals.py | 5 | 14 | 7 | 26 |
| cli.py | 5 | 11 | 9 | 25 |
| market_data.py | 4 | 12 | 7 | 23 |
| insider_tracker.py | 3 | 4 | 1 | 8 |
| inflection_detector.py | 1 | 3 | 3 | 7 |
| monitor.py | 2 | 5 | 0 | 7 |
| database.py | 1 | 4 | 2 | 7 |
| analyzer.py | 0 | 5 | 2 | 7 |
| forensics.py | 0 | 4 | 2 | 6 |
| catalyst_extractor.py | 0 | 2 | 2 | 4 |

## Top 5 Most Critical Issues to Fix Immediately

1. **🔴 monitor.py:56/80/152/165 — Entire monitor module is broken at runtime**
   `SEC_REQUEST_DELAY` does not exist in `scraper.py`, and `_safe_request` does not accept a `delay` keyword. Both `search_new_filings_efts()` and `resolve_filing_urls()` will crash with `ImportError`/`TypeError`. The entire daily monitoring pipeline is non-functional.

2. **🔴 llm_analysis.py:277 — Missing `global _last_token_usage` in `_call_anthropic`**
   All Anthropic API token usage is silently lost. Cost tracking is broken for all Anthropic models (Haiku, Sonnet, Opus). One-line fix: add `global _last_token_usage` at the top of `_call_anthropic`.

3. **🔴 engine.py/llm_analysis.py/scraper.py — Score normalization heuristic (<=10 → *10)**
   Appears in 3 files. Any legitimate score of 1-10 on the 0-100 scale is silently inflated 10x. This corrupts the entire scoring pipeline. The heuristic is fundamentally ambiguous and should be replaced with explicit scale metadata from the LLM response.

4. **🔴 scraper.py:1287/1369/1475 — Unhandled `future.result()` in 3 thread pools**
   A single network error crashes the entire pipeline step, losing all progress. Each needs a try/except around `future.result()`.

5. **🔴 engine.py:589-601 — Thread-unsafe `analysis_status` dict + singleton races**
   Multiple threads mutate `analysis_status` without consistent locking. `get_scraper()` and `get_db()` singletons are not thread-safe. Data races during T1/T2 concurrent analysis can corrupt progress tracking.

## Files That Need the Most Work

1. **engine.py** (39 issues) — Thread safety is the dominant concern. The massive code duplication between `run_analysis()` and `prepare_analysis()` (~200+ shared lines) is a maintenance burden. The 40+ bare `except Exception: pass` blocks make debugging nearly impossible.

2. **scraper.py** (29 issues) — Three unprotected `future.result()` calls are the most urgent. Non-atomic state saves risk data corruption. No error handling for corrupt JSON state files.

3. **llm_analysis.py** (28 issues) — The missing `global` declaration silently breaks all Anthropic cost tracking. The filing lookup collision bug can cause wrong filings to be analyzed. Prompt injection from SEC filings is a systemic risk.

4. **calibration.py** (26 issues) — Population vs sample std inflates stability scores, potentially causing overfit weights to be applied. Several dead code variables suggest incomplete implementation.

5. **filing_signals.py** (26 issues) — Performance issues with 4x redundant word list iterations and repeated regex compilation. The naive sentence splitting biases readability metrics.

6. **monitor.py** (7 issues, but 2 are fatal) — The module is completely non-functional at runtime due to importing a non-existent constant and passing invalid keyword arguments. Must be fixed before the daily monitoring feature can be used at all.
