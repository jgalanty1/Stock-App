#!/usr/bin/env python3
"""
SEC Analyzer — Terminal CLI
============================
Run the full SEC filing analysis pipeline from the command line.

Usage:
    python cli.py setup                     # Set API keys
    python cli.py status                    # Pipeline state summary
    python cli.py scan                      # Full pipeline: universe → find → download
    python cli.py analyze                   # Run T1 + T2 analysis (API-based)
    python cli.py prepare                   # Prepare bundles for local Claude Code analysis
    python cli.py import-analysis           # Import Claude Code analysis results
    python cli.py gems                      # Show top picks
    python cli.py potential                 # Show CPS-ranked companies
    python cli.py company TICKER            # Deep dive on one company
    python cli.py backtest                  # Run price backtest
    python cli.py calibrate                 # Run feedback loop
    python cli.py outcomes                  # Collect & show price outcomes
    python cli.py watchlist                 # Manage watchlist
    python cli.py export                    # Export to CSV
    python cli.py backup                    # Backup sec_data/
"""

import argparse
import getpass
import json
import os
import sys
import time
import logging

# Add project dir to path so flat-structure sibling modules (engine, scraper, etc.)
# can be imported by name without requiring a package install or PYTHONPATH setup.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# RICH SETUP — graceful fallback if not installed
# ============================================================================
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

if HAS_RICH:
    console = Console()
    err_console = Console(stderr=True)
else:
    class _FallbackConsole:
        def print(self, *args, **kwargs):
            style = kwargs.pop('style', '')
            for a in args:
                print(str(a))
        def rule(self, title='', **kwargs):
            print(f"\n{'='*60}")
            if title:
                print(f"  {title}")
            print('='*60)
        def __str__(self):
            return '<FallbackConsole>'
    console = _FallbackConsole()
    err_console = _FallbackConsole()


def _color(text, color):
    if HAS_RICH:
        return f"[{color}]{text}[/{color}]"
    return str(text)


def _green(t): return _color(t, 'green')
def _red(t): return _color(t, 'red')
def _yellow(t): return _color(t, 'yellow')
def _cyan(t): return _color(t, 'cyan')
def _bold(t): return _color(t, 'bold')
def _dim(t): return _color(t, 'dim')

# Model short-name to full model-id mapping for CLI --model flag
MODEL_MAP = {
    'haiku': 'claude-haiku',
    'sonnet': 'claude-sonnet',
    'opus': 'claude-opus',
    'gpt4o-mini': 'gpt-4o-mini',
}


# ============================================================================
# SETUP COMMAND
# ============================================================================
def cmd_setup(args):
    """Interactive API key setup."""
    import engine

    console.rule("SEC Analyzer — Setup")
    console.print("")
    keys = engine.get_key_status()

    # FMP
    fmp_status = _green("✓ Set") if keys['fmp']['set'] else _red("✗ Not set")
    console.print(f"FMP API Key:       {fmp_status}  {_dim(keys['fmp']['masked'])}")
    console.print(f"  (Required for universe building. Get at financialmodelingprep.com)")
    resp = getpass.getpass("  Enter FMP key (or press Enter to skip): ").strip()
    if resp:
        try:
            engine.persist_key('FMP_API_KEY', resp)
            console.print(f"  {_green('✓ Saved')}")
        except Exception as e:
            console.print(f"  {_red('✗ Failed to save FMP key')}: {e}")

    # Anthropic
    ant_status = _green("✓ Set") if keys['anthropic']['set'] else _red("✗ Not set")
    console.print(f"\nAnthropic API Key: {ant_status}  {_dim(keys['anthropic']['masked'])}")
    console.print(f"  (Required for LLM analysis. Get at console.anthropic.com)")
    resp = getpass.getpass("  Enter Anthropic key (or press Enter to skip): ").strip()
    if resp:
        try:
            engine.persist_key('ANTHROPIC_API_KEY', resp)
            console.print(f"  {_green('✓ Saved')}")
        except Exception as e:
            console.print(f"  {_red('✗ Failed to save Anthropic key')}: {e}")

    # OpenAI
    oai_status = _green("✓ Set") if keys['openai']['set'] else _red("✗ Not set")
    console.print(f"\nOpenAI API Key:    {oai_status}  {_dim(keys['openai']['masked'])}")
    console.print(f"  (Optional. Alternative LLM provider.)")
    resp = getpass.getpass("  Enter OpenAI key (or press Enter to skip): ").strip()
    if resp:
        try:
            engine.persist_key('OPENAI_API_KEY', resp)
            console.print(f"  {_green('✓ Saved')}")
        except Exception as e:
            console.print(f"  {_red('✗ Failed to save OpenAI key')}: {e}")

    console.print(f"\n{_green('✓')} Setup complete")


# ============================================================================
# STATUS COMMAND
# ============================================================================
def cmd_status(args):
    """Show pipeline status summary."""
    import engine

    status = engine.get_status()
    keys = status.get('keys', {})

    if args.json:
        print(json.dumps(status, indent=2))
        return

    console.rule("SEC Analyzer — Status")

    if HAS_RICH:
        t = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        t.add_column("Key", style="bold")
        t.add_column("Value")

        t.add_row("Universe", f"{status['universe_count']} companies")
        t.add_row("Filings found", str(status['filings_total']))
        t.add_row("Downloaded", str(status['filings_downloaded']))
        t.add_row("Tier 1 analyzed", str(status['tier1_analyzed']))
        t.add_row("Tier 2 analyzed", str(status['tier2_analyzed']))
        t.add_row("", "")
        t.add_row("FMP key", _green("✓") if keys.get('fmp', {}).get('set') else _red("✗"))
        t.add_row("Anthropic key", _green("✓") if keys.get('anthropic', {}).get('set') else _red("✗"))
        t.add_row("OpenAI key", _green("✓") if keys.get('openai', {}).get('set') else _red("✗"))
        console.print(t)
    else:
        console.print(f"  Universe:        {status['universe_count']} companies")
        console.print(f"  Filings found:   {status['filings_total']}")
        console.print(f"  Downloaded:      {status['filings_downloaded']}")
        console.print(f"  Tier 1 analyzed: {status['tier1_analyzed']}")
        console.print(f"  Tier 2 analyzed: {status['tier2_analyzed']}")
        console.print(f"  FMP key:         {'✓' if keys.get('fmp', {}).get('set') else '✗'}")
        console.print(f"  Anthropic key:   {'✓' if keys.get('anthropic', {}).get('set') else '✗'}")
        console.print(f"  OpenAI key:      {'✓' if keys.get('openai', {}).get('set') else '✗'}")


# ============================================================================
# SCAN COMMAND
# ============================================================================
def cmd_scan(args):
    """Run scraper pipeline."""
    import engine

    steps = args.step or ['universe', 'find', 'download', 'insider']

    console.rule(f"SEC Analyzer — Scan [{', '.join(steps)}]")

    # Progress display
    if HAS_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Starting...", total=None)

            def _progress(step, msg, current=0, total=0):
                progress.update(task_id, description=f"[cyan]{step}[/] {msg}",
                                completed=current, total=total or None)

            result = engine.run_pipeline(
                steps=steps,
                min_market_cap=args.min_cap,
                max_market_cap=args.max_cap,
                max_companies=args.max_companies,
                max_downloads=args.max_downloads,
                progress_fn=_progress,
            )
    else:
        def _progress(step, msg, current=0, total=0):
            print(f"  [{step}] {msg}")

        result = engine.run_pipeline(
            steps=steps,
            min_market_cap=args.min_cap,
            max_market_cap=args.max_cap,
            max_companies=args.max_companies,
            max_downloads=args.max_downloads,
            progress_fn=_progress,
        )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    console.print(f"\n{_green('✓')} Scan complete")
    console.print(f"  Universe: {result['universe_count']} companies")
    console.print(f"  Filings:  {result['filings_total']} found, {result['filings_downloaded']} downloaded")


# ============================================================================
# ANALYZE COMMAND
# ============================================================================
def cmd_analyze(args):
    """Run LLM analysis (T1 + T2)."""
    import engine

    console.rule("SEC Analyzer — LLM Analysis")

    t2_model = MODEL_MAP.get(args.model, args.model)

    config_lines = [
        f"  T1 workers:   {args.t1_workers}",
        f"  T2 workers:   {args.t2_workers}",
        f"  T2 model:     {t2_model}",
        f"  T2 top %:     {int(args.tier2_pct * 100)}%",
        f"  Re-analyze:   {'Yes' if args.reanalyze else 'No'}",
    ]
    if args.effort:
        config_lines.append(f"  Effort:       {args.effort}")
    for line in config_lines:
        console.print(line)
    console.print("")

    last_msg = [""]

    def _progress(status):
        msg = status.get('message', '')
        if msg != last_msg[0]:
            last_msg[0] = msg
            tier = status.get('tier', 1)
            prog = status.get('progress', 0)
            total = status.get('total', 0)
            pct = f"{prog}/{total}" if total else ""
            console.print(f"  [T{tier}] {pct} {msg}")

    result = engine.run_analysis(
        tier2_pct=args.tier2_pct,
        reanalyze=args.reanalyze,
        tier2_model=t2_model,
        effort=args.effort or '',
        tier1_workers=args.t1_workers,
        tier2_workers=args.t2_workers,
        progress_fn=_progress,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    console.print(f"\n{_green('✓')} Analysis complete")
    console.print(f"  T1: {_green(result.get('completed_tier1', 0))} OK / {_red(result.get('failed_tier1', 0))} failed")
    console.print(f"  T2: {_green(result.get('completed_tier2', 0))} OK / {_red(result.get('failed_tier2', 0))} failed")
    console.print(f"  Est. cost: ${result.get('estimated_cost_usd', 0):.4f}")

    if result.get('errors'):
        console.print(f"\n{_yellow('Errors')} ({len(result['errors'])} total):")
        for e in result['errors'][:10]:
            console.print(f"  {_red('✗')} {e}")
        if len(result['errors']) > 10:
            console.print(f"  ... and {len(result['errors'])-10} more")


# ============================================================================
# GEMS COMMAND
# ============================================================================
def cmd_gems(args):
    """Show top gem candidates."""
    import engine

    gems = engine.get_gems()
    if not gems:
        console.print(f"{_yellow('No gems found.')} Run: python cli.py analyze")
        return

    limit = args.limit if args.limit is not None else len(gems)
    gems = gems[:limit]

    if args.json:
        print(json.dumps(gems, indent=2))
        return

    console.rule(f"SEC Analyzer — Top Gems ({len(gems)})")

    has_confidence = any(g.get('confidence') for g in gems)

    if HAS_RICH:
        t = Table(box=box.ROUNDED, show_lines=True)
        t.add_column("#", style="dim", width=3)
        t.add_column("Ticker", style="bold cyan")
        t.add_column("Company")
        t.add_column("Gem", justify="right")
        if has_confidence:
            t.add_column("Confidence")
        t.add_column("Conviction")
        t.add_column("Insider")
        t.add_column("Diff")
        t.add_column("Rec")
        t.add_column("One-Liner", max_width=45)

        for i, g in enumerate(gems):
            score = g['gem_score']
            score_color = 'green' if score >= 70 else ('yellow' if score >= 50 else 'red')
            conv = g.get('conviction', '')
            conv_color = 'green' if conv == 'high' else ('yellow' if conv == 'medium' else 'dim')
            ins = g.get('insider_signal', '')
            ins_color = 'green' if ins in ('strong_accumulation', 'accumulation') else 'dim'
            diff = g.get('diff_signal', '') or ''
            diff_color = 'green' if 'improving' in diff else 'dim'

            row = [
                str(i + 1),
                g['ticker'],
                g['company_name'][:30],
                f"[{score_color}]{score:.0f}[/{score_color}]",
            ]
            if has_confidence:
                conf = g.get('confidence', {})
                conf_level = conf.get('confidence_level', '')
                ci = conf.get('confidence_interval')
                win_rate = conf.get('historical_win_rate')
                if ci and win_rate is not None:
                    cl_color = 'green' if conf_level == 'high' else ('yellow' if conf_level == 'medium' else 'dim')
                    row.append(f"[{cl_color}]{ci[0]:+.0f}% to {ci[1]:+.0f}%[/{cl_color}]")
                else:
                    row.append("[dim]—[/dim]")
            row.extend([
                f"[{conv_color}]{conv}[/{conv_color}]",
                f"[{ins_color}]{ins}[/{ins_color}]",
                f"[{diff_color}]{diff}[/{diff_color}]",
                g.get('recommendation', '')[:15],
                g.get('one_liner', '')[:45],
            ])
            t.add_row(*row)
        console.print(t)
    else:
        fmt = "{:>3}  {:8}  {:30}  {:>5}  {:12}  {:20}  {:15}  {:15}  {:45}"
        console.print(fmt.format("#", "Ticker", "Company", "Gem", "Conviction", "Insider", "Diff", "Rec", "One-Liner"))
        console.print("-" * 160)
        for i, g in enumerate(gems):
            console.print(fmt.format(
                i + 1, g['ticker'], g['company_name'][:30],
                f"{g['gem_score']:.0f}", g.get('conviction', ''),
                g.get('insider_signal', ''), g.get('diff_signal', '') or '',
                g.get('recommendation', '')[:15], g.get('one_liner', '')[:45],
            ))


# ============================================================================
# POTENTIAL COMMAND
# ============================================================================
def cmd_potential(args):
    """Show CPS-ranked companies."""
    import engine

    companies = engine.get_potential()
    if not companies:
        console.print(f"{_yellow('No potential data.')} Run: python cli.py analyze")
        return

    limit = args.limit or 30
    companies = companies[:limit]

    if args.json:
        print(json.dumps(companies, indent=2, default=str))
        return

    console.rule(f"SEC Analyzer — Company Potential ({len(companies)})")

    if HAS_RICH:
        t = Table(box=box.ROUNDED)
        t.add_column("#", style="dim", width=3)
        t.add_column("Ticker", style="bold cyan")
        t.add_column("Company")
        t.add_column("CPS", justify="right")
        t.add_column("Catalyst", justify="right")
        t.add_column("Trajectory", justify="right")
        t.add_column("Insider", justify="right")
        t.add_column("Turnaround", justify="right")
        t.add_column("Recency", justify="right")
        t.add_column("Filings", justify="right")

        for i, c in enumerate(companies):
            cps = c['cps']
            cps_color = 'green' if cps >= 60 else ('yellow' if cps >= 40 else 'dim')
            comp = c.get('components', {})
            t.add_row(
                str(i + 1),
                c['ticker'],
                c['company_name'][:25],
                f"[{cps_color}]{cps}[/{cps_color}]",
                f"{comp.get('catalyst_strength', 0):.0f}",
                f"{comp.get('improvement_trajectory', 0):.0f}",
                f"{comp.get('insider_conviction', 0):.0f}",
                f"{comp.get('turnaround_indicators', 0):.0f}",
                f"{comp.get('filing_recency', 0):.0f}",
                str(c.get('filing_count', 0)),
            )
        console.print(t)
    else:
        for i, c in enumerate(companies):
            console.print(f"  {i+1:3}. {c['ticker']:8} CPS={c['cps']:3}  {c['company_name'][:30]}")


# ============================================================================
# COMPANY COMMAND
# ============================================================================
def cmd_company(args):
    """Show detailed info for one company."""
    import engine

    ticker = args.ticker.upper()
    detail = engine.get_company_detail(ticker)

    if not detail.get('filings'):
        console.print(f"{_yellow(f'No data for {ticker}.')} Check the ticker or run scan first.")
        return

    if args.json:
        print(json.dumps(detail, indent=2, default=str))
        return

    console.rule(f"SEC Analyzer — {ticker}")

    company = detail.get('company', {})
    if company:
        console.print(f"  Company:    {company.get('company_name', '?')}")
        console.print(f"  CIK:        {company.get('cik', '?')}")
        console.print(f"  Market Cap: ${company.get('market_cap', 0):,.0f}")
        console.print(f"  Sector:     {company.get('sector', '?')}")

    filings = detail.get('filings', [])
    console.print(f"\n  Filings: {len(filings)}")

    if HAS_RICH:
        t = Table(box=box.SIMPLE)
        t.add_column("Date")
        t.add_column("Form")
        t.add_column("T1")
        t.add_column("Gem")
        t.add_column("T2")
        t.add_column("Conviction")
        t.add_column("Diff")
        t.add_column("Insider")

        for f in filings[:20]:
            t1 = f.get('tier1_score')
            gem = f.get('final_gem_score')
            t2 = '✓' if f.get('tier2_analyzed') else ''
            t.add_row(
                f.get('filing_date', ''),
                f.get('form_type', ''),
                str(t1) if t1 is not None else '—',
                f"[green]{gem}[/green]" if gem and gem >= 70 else str(gem) if gem is not None else '—',
                t2,
                f.get('conviction', '') or '',
                f.get('diff_signal', '') or '',
                f.get('insider_signal', '') or '',
            )
        console.print(t)
    else:
        for f in filings[:20]:
            t2_mark = 'T2' if f.get('tier2_analyzed') else '  '
            conv = f.get('conviction', '') or ''
            diff = f.get('diff_signal', '') or ''
            ins = f.get('insider_signal', '') or ''
            console.print(f"    {f.get('filing_date',''):10}  {f.get('form_type',''):10} "
                          f"T1={str(f.get('tier1_score','—')):>3} "
                          f"Gem={str(f.get('final_gem_score','—')):>3} "
                          f"{t2_mark}  {conv:8} {diff:15} {ins}")

    # Show T2 result details
    t2_results = detail.get('t2_results', [])
    if t2_results:
        console.print(f"\n  Latest T2 Analysis:")
        r = t2_results[0]
        console.print(f"    Score:          {r.get('final_gem_score', '?')}")
        console.print(f"    Conviction:     {r.get('conviction_level', '?')}")
        console.print(f"    Recommendation: {r.get('recommendation', '?')}")
        console.print(f"    One-liner:      {r.get('one_liner', '')}")
        if r.get('thesis'):
            console.print(f"    Thesis:         {r['thesis'][:200]}")


# ============================================================================
# BACKTEST COMMAND
# ============================================================================
def cmd_backtest(args):
    """Run price backtesting."""
    import engine

    include_t1 = getattr(args, 'all_filings', False)
    result = engine.run_backtest(clear_cache=args.clear, include_t1=include_t1)

    if result.get('error'):
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            console.print(f"  {_red(result['error'])}")
        return

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    console.rule("SEC Analyzer — Backtest")

    stats = result.get('stats', {})
    console.print(f"\n  Total tested:    {result['total']}")
    console.print(f"  With price data: {_green(stats.get('with_price_data', 0))}")
    if stats.get('error_count', 0):
        console.print(f"  Errors:          {_red(stats['error_count'])}")

    console.print(f"\n  Avg 30d return:  {stats.get('avg_30d', 0):.1f}%")
    console.print(f"  Avg 60d return:  {stats.get('avg_60d', 0):.1f}%")
    console.print(f"  Avg 90d return:  {stats.get('avg_90d', 0):.1f}%")
    console.print(f"  Avg 180d return: {stats.get('avg_180d', 0):.1f}%")
    console.print(f"  Win rate (30d):  {stats.get('win_rate_30d', 0):.0f}%")

    # Alpha (vs benchmark)
    if stats.get('avg_alpha_90d'):
        console.print(f"\n  {_bold('vs Benchmark (IWC):')}")
        for w in ['30', '60', '90', '180']:
            alpha = stats.get(f'avg_alpha_{w}d', 0)
            awr = stats.get(f'alpha_win_rate_{w}d', 0)
            if alpha or awr:
                color = 'green' if alpha > 0 else 'red'
                console.print(f"  Alpha {w}d: {_color(f'{alpha:+.1f}%', color)}  "
                              f"(beat benchmark: {awr:.0f}%)")

    if stats.get('best_ticker'):
        console.print(f"\n  Best:  {_green(stats['best_ticker'])} +{stats.get('best_return', 0):.1f}%")
        console.print(f"  Worst: {_red(stats['worst_ticker'])} {stats.get('worst_return', 0):.1f}%")

    # Individual results table
    individual = result.get('results', [])
    if individual and HAS_RICH:
        console.print("")
        t = Table(box=box.SIMPLE, title="Individual Results")
        t.add_column("Ticker", style="bold")
        t.add_column("Date")
        t.add_column("Gem", justify="right")
        t.add_column("30d", justify="right")
        t.add_column("60d", justify="right")
        t.add_column("90d", justify="right")
        t.add_column("180d", justify="right")
        t.add_column("Alpha 90d", justify="right")
        t.add_column("Status")

        def _fmt_ret(v):
            if v is None: return "[dim]—[/dim]"
            c = 'green' if v >= 0 else 'red'
            return f"[{c}]{v:.1f}%[/{c}]"

        for r in individual[:30]:
            err = r.get('error', '')
            status = f"[red]{err}[/red]" if err else (
                "[green]Win[/green]" if r.get('return_90d') is not None and r['return_90d'] > 0
                else "[red]Loss[/red]" if r.get('return_90d') is not None else "[dim]—[/dim]"
            )

            t.add_row(
                r['ticker'], r.get('filing_date', ''),
                str(r.get('gem_score', '—')),
                _fmt_ret(r.get('return_30d')),
                _fmt_ret(r.get('return_60d')),
                _fmt_ret(r.get('return_90d')),
                _fmt_ret(r.get('return_180d')),
                _fmt_ret(r.get('alpha_90d')),
                status,
            )
        console.print(t)


# ============================================================================
# CALIBRATE COMMAND
# ============================================================================
def cmd_calibrate(args):
    """Run feedback loop calibration."""
    import engine

    multi_window = args.multi_window if hasattr(args, 'multi_window') else False

    report = engine.run_calibration(
        return_window=args.window,
        auto_apply=args.apply,
        multi_window=multi_window,
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return

    console.rule("SEC Analyzer — Calibration")
    console.print(f"  Return window: {args.window} days")
    console.print(f"  Auto-apply:    {'Yes' if args.apply else 'No'}")
    if multi_window:
        console.print(f"  Multi-window:  Yes (30/60/90/180d)")
    console.print("")

    if report.get('status') in ('insufficient_data', 'insufficient_backtest_data'):
        console.print(f"  {_yellow(report.get('message', 'Insufficient data'))}")
        n_have = report.get('companies_with_returns', report.get('companies_analyzed', 0))
        n_need = 10
        console.print(f"  {_dim(f'{n_need - n_have} more filings with returns needed for calibration')}")
        return

    console.print(f"  {_green(report.get('message', 'Calibration complete'))}")

    # Show correlations
    corrs = report.get('correlations', {})
    if corrs and HAS_RICH:
        t = Table(box=box.SIMPLE, title="Signal Correlations")
        t.add_column("Signal")
        t.add_column("Correlation", justify="right")
        t.add_column("Strength")
        t.add_column("Spread", justify="right")
        for sig_name, sig_data in sorted(corrs.items(), key=lambda x: abs(x[1].get('correlation', 0)), reverse=True):
            corr = sig_data.get('correlation', 0)
            strength = sig_data.get('strength', '')
            spread = sig_data.get('spread', 0)
            c = 'green' if corr > 0.1 else ('red' if corr < -0.1 else 'dim')
            t.add_row(sig_name, f"[{c}]{corr:.3f}[/{c}]", strength,
                      f"{spread:+.1f}%")
        console.print(t)

    # Walk-forward results
    wf = report.get('walk_forward', {})
    if wf and wf.get('fold_results'):
        risk = wf.get('overfitting_risk', '?')
        stability = wf.get('stability_score', 0)
        risk_color = 'green' if risk == 'low' else ('yellow' if risk == 'medium' else 'red')
        console.print(f"\n  {_bold('Walk-Forward Validation:')}")
        console.print(f"    Folds:           {wf.get('n_folds', len(wf['fold_results']))}")
        console.print(f"    Stability:       {stability:.2f}")
        console.print(f"    Overfitting risk: {_color(risk, risk_color)}")
    elif wf and wf.get('message'):
        console.print(f"\n  Walk-Forward: {_dim(wf['message'])}")

    # Multi-window correlation matrix
    mw_corrs = report.get('multi_window_correlations', {})
    if mw_corrs and HAS_RICH:
        console.print(f"\n  {_bold('Multi-Window Signal Performance:')}")
        perf_matrix = report.get('signal_performance_matrix', {}).get('matrix', {})
        if perf_matrix:
            t = Table(box=box.SIMPLE, title="Signal x Horizon Correlations")
            t.add_column("Signal")
            for w in ['30', '60', '90', '180']:
                t.add_column(f"{w}d", justify="right")
            t.add_column("Best", justify="center")

            for sig_name in sorted(perf_matrix.keys(),
                                    key=lambda s: abs(perf_matrix[s].get('best_correlation', 0)),
                                    reverse=True)[:20]:
                row_data = perf_matrix[sig_name]
                cells = [sig_name]
                for w in ['30', '60', '90', '180']:
                    w_data = row_data.get(f"{w}d", {})
                    corr = w_data.get('correlation', 0)
                    c = 'green' if corr > 0.15 else ('red' if corr < -0.15 else 'dim')
                    cells.append(f"[{c}]{corr:.2f}[/{c}]")
                cells.append(f"{row_data.get('best_horizon', '?')}d")
                t.add_row(*cells)
            console.print(t)

    # Show suggested weights
    opt = report.get('weight_optimization', {})
    suggested = opt.get('suggested_weights', {})
    if suggested:
        console.print(f"\n  {_bold('Suggested Weights:')}")
        current = opt.get('current_weights', {})
        for k, v in suggested.items():
            change = v - current.get(k, 0)
            change_str = f"({change:+d})" if change != 0 else ""
            console.print(f"    {k:30} = {v:3}  {_dim(change_str)}")
        if opt.get('confidence'):
            console.print(f"    Confidence: {opt['confidence']}")
        if report.get('weights_applied'):
            console.print(f"\n  {_green('✓ Weights applied automatically')}")
        else:
            console.print(f"\n  To apply: python cli.py calibrate --apply")


# ============================================================================
# WATCHLIST COMMAND
# ============================================================================
def cmd_watchlist(args):
    """Manage watchlist."""
    import engine

    if args.add:
        engine.add_to_watchlist(args.add)
        console.print(f"  {_green('✓')} Added {args.add.upper()} to watchlist")
    if args.remove:
        engine.remove_from_watchlist(args.remove)
        console.print(f"  {_green('✓')} Removed {args.remove.upper()} from watchlist")

    wl = engine.get_watchlist()

    if args.json:
        gems = {g['ticker']: g for g in engine.get_gems()}
        watchlist_data = []
        for ticker in wl:
            g = gems.get(ticker)
            if g:
                watchlist_data.append({
                    'ticker': ticker,
                    'gem_score': g.get('gem_score'),
                    'conviction': g.get('conviction', ''),
                    'one_liner': g.get('one_liner', ''),
                })
            else:
                watchlist_data.append({'ticker': ticker})
        print(json.dumps(watchlist_data, indent=2))
        return

    console.rule(f"Watchlist ({len(wl)} tickers)")
    if wl:
        # Show detail for each
        gems = {g['ticker']: g for g in engine.get_gems()}
        for ticker in wl:
            g = gems.get(ticker)
            if g:
                score_str = f"Gem={g['gem_score']:.0f}"
                conv = g.get('conviction', '')
                ticker_str = _cyan(ticker)
                console.print(f"  {ticker_str}  {score_str:10}  {conv:10}  {g.get('one_liner', '')[:50]}")
            else:
                console.print(f"  {_cyan(ticker)}  {_dim('(no analysis data)')}")
    else:
        console.print(f"  {_dim('Empty. Add with: python cli.py watchlist --add TICKER')}")


# ============================================================================
# OUTCOMES COMMAND
# ============================================================================
def cmd_outcomes(args):
    """Collect and display price outcome status for analyzed filings."""
    import engine
    from datetime import datetime as _dt, timedelta as _td
    from market_data import BACKTEST_WINDOWS

    if args.refresh:
        try:
            from market_data import clear_price_cache
            clear_price_cache()
        except (ImportError, AttributeError):
            pass

    result = engine.auto_collect_outcomes(include_t1=True)

    if not args.json:
        console.rule("SEC Analyzer — Outcomes")
        console.print(f"\n  Filings tracked:   {result.get('collected', 0)}")
        console.print(f"  With returns:      {_green(result.get('with_returns', 0))}")
        console.print(f"  Pending windows:   {_yellow(result.get('pending_windows', 0))}")

    # NOTE: Reaching into scraper.filings_index directly breaks the engine abstraction.
    # Ideally engine would provide a method to return per-filing maturation data.
    scraper = engine.get_scraper()
    now = _dt.now()

    complete = []    # All 4 windows matured
    partial = []     # Some windows matured
    pending = []     # No windows matured yet

    for f in scraper.filings_index:
        if not f.get('filing_date'):
            continue
        if not (f.get('llm_analyzed') or f.get('tier2_analyzed')):
            continue

        try:
            fd = _dt.strptime(f['filing_date'][:10], '%Y-%m-%d')
        except ValueError:
            continue

        matured_windows = sum(1 for w in BACKTEST_WINDOWS if (fd + _td(days=w)) <= now)
        entry = {
            'ticker': f.get('ticker', ''),
            'filing_date': f['filing_date'][:10],
            'form_type': f.get('form_type', ''),
            'tier2': f.get('tier2_analyzed', False),
            'gem_score': f.get('final_gem_score'),
            'matured_windows': matured_windows,
        }
        # Next maturation date
        for w in BACKTEST_WINDOWS:
            mat_date = fd + _td(days=w)
            if mat_date > now:
                entry['next_maturation'] = mat_date.strftime('%Y-%m-%d')
                entry['next_window'] = f"{w}d"
                break

        if matured_windows == len(BACKTEST_WINDOWS):
            complete.append(entry)
        elif matured_windows > 0:
            partial.append(entry)
        else:
            pending.append(entry)

    if args.pending:
        # Show only pending/partial filings
        to_show = pending + partial
        label = "Pending/Partial"
    else:
        to_show = complete + partial + pending
        label = "All"

    if args.json:
        print(json.dumps({
            'summary': result,
            'complete': len(complete),
            'partial': len(partial),
            'pending': len(pending),
        }, indent=2))
        return

    if to_show and HAS_RICH:
        console.print("")
        t = Table(box=box.SIMPLE, title=f"{label} Outcomes ({len(to_show)} filings)")
        t.add_column("Ticker", style="bold")
        t.add_column("Date")
        t.add_column("Form")
        t.add_column("Tier", justify="center")
        t.add_column("Gem", justify="right")
        t.add_column("Windows", justify="center")
        t.add_column("Next Maturation")

        for entry in to_show[:50]:
            tier = "T2" if entry['tier2'] else "T1"
            gem = str(entry['gem_score']) if entry['gem_score'] is not None else "—"
            mw = entry['matured_windows']
            total_w = len(BACKTEST_WINDOWS)
            w_color = 'green' if mw == total_w else ('yellow' if mw > 0 else 'dim')
            next_mat = entry.get('next_maturation', '—')
            next_win = entry.get('next_window', '')
            t.add_row(
                entry['ticker'],
                entry['filing_date'],
                entry['form_type'],
                tier,
                gem,
                f"[{w_color}]{mw}/{total_w}[/{w_color}]",
                f"{next_mat} ({next_win})" if next_win else "Complete",
            )
        console.print(t)
    elif to_show:
        for entry in to_show[:30]:
            console.print(f"  {entry['ticker']:8} {entry['filing_date']}  "
                          f"{entry['matured_windows']}/{len(BACKTEST_WINDOWS)} windows")


# ============================================================================
# PREPARE COMMAND
# ============================================================================
def cmd_prepare(args):
    """Prepare analysis bundles for local Claude Code analysis."""
    import engine

    tier = str(args.tier) if args.tier else 'auto'
    def _progress(msg):
        if not args.json:
            console.print(f"  {msg}")

    result = engine.prepare_analysis(
        tier=tier,
        tier2_pct=args.tier2_pct,
        reanalyze=args.reanalyze,
        progress_fn=_progress,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    console.rule("SEC Analyzer — Prepare Analysis Bundles")
    console.print(f"  Tier:        {tier}")
    console.print(f"  T2 top %:    {int(args.tier2_pct * 100)}%")
    console.print(f"  Re-prepare:  {'Yes' if args.reanalyze else 'No'}")

    if result.get('message') and result.get('t1_count', 0) == 0 and result.get('t2_count', 0) == 0:
        console.print(f"\n  {_yellow(result['message'])}")
        return

    console.print(f"\n{_green('✓')} Preparation complete")
    console.print(f"  T1 bundles: {result.get('t1_count', 0)}")
    console.print(f"  T2 bundles: {result.get('t2_count', 0)}")
    console.print(f"\n  Bundles written to: {_cyan('prepared/')}")

    if result.get('t1_count', 0) > 0:
        console.print(f"\n  {_bold('Next step (T1):')}")
        console.print(f"  Ask Claude Code to analyze T1 bundles in prepared/t1/")
        console.print(f"  Results should be written to prepared/t1_results/")
        console.print(f"  Then: python cli.py import-analysis --tier 1")

    if result.get('t2_count', 0) > 0:
        console.print(f"\n  {_bold('Next step (T2):')}")
        console.print(f"  Ask Claude Code to analyze T2 bundles in prepared/t2/")
        console.print(f"  Results should be written to prepared/t2_results/")
        console.print(f"  Then: python cli.py import-analysis --tier 2")


# ============================================================================
# IMPORT-ANALYSIS COMMAND
# ============================================================================
def cmd_import_analysis(args):
    """Import analysis results from Claude Code back into the system."""
    import engine

    tier = str(args.tier) if args.tier else 'auto'

    if not args.json:
        console.rule("SEC Analyzer — Import Analysis Results")
        console.print(f"  Tier:     {tier}")
        console.print(f"  Source:   prepared/")
        console.print("")

    def _progress(msg):
        if not args.json:
            console.print(f"  {msg}")

    result = engine.import_analysis_results(
        tier=tier,
        progress_fn=_progress,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    if result.get('message') and result.get('t1_imported', 0) == 0 and result.get('t2_imported', 0) == 0:
        console.print(f"\n  {_yellow(result['message'])}")
        return

    console.print(f"\n{_green('✓')} Import complete")
    console.print(f"  T1 imported: {_green(result.get('t1_imported', 0))}")
    console.print(f"  T2 imported: {_green(result.get('t2_imported', 0))}")

    if result.get('errors'):
        console.print(f"\n  {_yellow('Errors')} ({len(result['errors'])}):")
        for e in result['errors'][:10]:
            console.print(f"    {_red('✗')} {e}")
        if len(result['errors']) > 10:
            console.print(f"    ... and {len(result['errors'])-10} more")

    if result.get('t2_imported', 0) > 0:
        console.print(f"\n  View results: python cli.py gems")


# ============================================================================
# PORTFOLIO COMMAND
# ============================================================================
def cmd_portfolio(args):
    """Portfolio risk analysis for top picks."""
    import engine

    result = engine.analyze_portfolio(top_n=args.top, max_sector_pct=args.max_sector)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    console.print(f"\n  Analyzing portfolio (top {args.top} picks)...\n")

    if result.get('error'):
        console.print(f"  {_red('Error')}: {result['error']}")
        return

    # Positions table
    console.print(f"  {_green('Portfolio Positions')} ({result['position_count']} holdings)\n")
    if HAS_RICH:
        t = Table(show_header=True, header_style="bold")
        t.add_column("#", style="dim", width=3)
        t.add_column("Ticker", width=8)
        t.add_column("Company", width=25)
        t.add_column("Score", justify="right", width=6)
        t.add_column("Conviction", width=10)
        t.add_column("Sector", width=18)
        t.add_column("Weight %", justify="right", width=8)
        for i, p in enumerate(result['positions'], 1):
            conv_color = {"high": "green", "medium": "yellow", "low": "white"}.get(p['conviction'], "white")
            t.add_row(
                str(i), p['ticker'], p['company_name'][:25],
                str(p['gem_score']),
                f"[{conv_color}]{p['conviction']}[/]",
                p['sector'][:18],
                f"{p['weight_pct']:.1f}%",
            )
        console.print(t)
    else:
        for i, p in enumerate(result['positions'], 1):
            console.print(f"  {i:2d}. {p['ticker']:8s} {p['company_name'][:25]:25s} "
                          f"Score={p['gem_score']:3.0f}  {p['conviction']:6s}  "
                          f"{p['sector'][:18]:18s}  {p['weight_pct']:.1f}%")

    # Sector exposure
    console.print(f"\n  {_green('Sector Exposure')}:")
    for sector, pct in result.get('sector_exposure', {}).items():
        bar = '#' * int(pct / 2)
        console.print(f"    {sector:20s} {pct:5.1f}% {bar}")

    # Diversification score
    div_score = result.get('diversification_score', 0)
    div_label = "Excellent" if div_score >= 70 else "Good" if div_score >= 50 else "Fair" if div_score >= 30 else "Poor"
    console.print(f"\n  Diversification Score: {div_score}/100 ({div_label})")
    console.print(f"  Herfindahl Index: {result.get('herfindahl_index', 0):.4f}")
    console.print(f"  Unique Sectors: {result.get('unique_sectors', 0)}")

    # Warnings
    warnings = result.get('concentration_warnings', [])
    if warnings:
        console.print(f"\n  {_red('Concentration Warnings')}:")
        for w in warnings:
            console.print(f"    ! {w}")

    corr_pairs = result.get('high_correlation_pairs', [])
    if corr_pairs:
        console.print(f"\n  {_red('High Similarity Pairs')} (signal overlap > 85%):")
        for cp in corr_pairs:
            console.print(f"    {cp['pair']} — similarity: {cp['similarity']:.3f}")

    console.print()


def cmd_export(args):
    """Export results to CSV."""
    import engine

    output = args.output or 'sec_data/export.csv'
    path = engine.export_csv(output)
    try:
        size = os.path.getsize(path)
    except FileNotFoundError:
        size = 0

    if args.json:
        print(json.dumps({'path': path, 'size_bytes': size}, indent=2))
        return

    console.print(f"  {_green('✓')} Exported to {path} ({size:,} bytes)")


# ============================================================================
# BACKUP COMMAND
# ============================================================================
def cmd_backup(args):
    """Backup sec_data/ to a zip file."""
    import engine

    output = args.output
    path = engine.backup_data(output)
    try:
        size = os.path.getsize(path)
    except FileNotFoundError:
        size = 0

    if args.json:
        print(json.dumps({'path': path, 'size_bytes': size}, indent=2))
        return

    console.print(f"  {_green('✓')} Backup saved: {path} ({size / 1024 / 1024:.1f} MB)")


# ============================================================================
# MAIN — ARGPARSE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        prog='sec-analyzer',
        description='SEC Filing Analyzer — Terminal CLI',
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    sub = parser.add_subparsers(dest='command')

    # setup
    sub.add_parser('setup', help='Set API keys interactively')

    # status
    p = sub.add_parser('status', help='Show pipeline state')
    p.add_argument('--json', action='store_true')

    # scan
    p = sub.add_parser('scan', help='Run scraper pipeline')
    p.add_argument('--step', nargs='+', choices=['universe', 'find', 'download', 'insider'],
                   help='Run specific steps (default: all)')
    p.add_argument('--min-cap', type=int, default=None, help='Min market cap (default: 20M)')
    p.add_argument('--max-cap', type=int, default=None, help='Max market cap (default: 100M)')
    p.add_argument('--max-companies', type=int, default=None, help='Limit companies to process')
    p.add_argument('--max-downloads', type=int, default=None, help='Limit filing downloads')
    p.add_argument('--json', action='store_true')

    # analyze
    p = sub.add_parser('analyze', help='Run LLM analysis (T1 + T2)')
    p.add_argument('--t1-workers', type=int, default=6, help='Tier 1 parallel workers (default: 6)')
    p.add_argument('--t2-workers', type=int, default=3, help='Tier 2 parallel workers (default: 3)')
    p.add_argument('--model', default='haiku',
                   help='T2 model: haiku, sonnet, opus, gpt4o-mini (default: haiku)')
    p.add_argument('--tier2-pct', type=float, default=0.25,
                   help='Top %% of companies for T2 (default: 0.25)')
    p.add_argument('--effort', choices=['low', 'medium', 'high', 'max'], default=None,
                   help='Analysis effort level (affects thinking budget)')
    p.add_argument('--reanalyze', action='store_true', help='Re-analyze already processed filings')
    p.add_argument('--json', action='store_true')

    # gems
    p = sub.add_parser('gems', help='Show top gem candidates')
    p.add_argument('--limit', type=int, default=None, help='Max results to show')
    p.add_argument('--json', action='store_true')

    # potential
    p = sub.add_parser('potential', help='Show CPS-ranked companies')
    p.add_argument('--limit', type=int, default=30, help='Max results (default: 30)')
    p.add_argument('--json', action='store_true')

    # company
    p = sub.add_parser('company', help='Deep dive on one company')
    p.add_argument('ticker', help='Stock ticker')
    p.add_argument('--json', action='store_true')

    # backtest
    p = sub.add_parser('backtest', help='Run price backtesting')
    p.add_argument('--clear', action='store_true', help='Clear cached results first')
    p.add_argument('--all-filings', action='store_true', help='Include T1-only filings (not just T2)')
    p.add_argument('--json', action='store_true')

    # calibrate
    p = sub.add_parser('calibrate', help='Run feedback loop calibration')
    p.add_argument('--window', type=int, default=90, help='Return window in days (default: 90)')
    p.add_argument('--apply', action='store_true', help='Auto-apply suggested weights')
    p.add_argument('--multi-window', action='store_true', help='Analyze all return windows (30/60/90/180d)')
    p.add_argument('--json', action='store_true')

    # outcomes
    p = sub.add_parser('outcomes', help='Collect and show price outcome status')
    p.add_argument('--refresh', action='store_true', help='Re-fetch all price data')
    p.add_argument('--pending', action='store_true', help='Show only filings with immature windows')
    p.add_argument('--json', action='store_true')

    # prepare
    p = sub.add_parser('prepare', help='Prepare analysis bundles for local Claude Code analysis')
    p.add_argument('--tier', type=int, choices=[1, 2], default=None,
                   help='Force specific tier (default: auto-detect)')
    p.add_argument('--tier2-pct', type=float, default=0.25,
                   help='Top %% of companies for T2 (default: 0.25)')
    p.add_argument('--reanalyze', action='store_true', help='Re-prepare already processed filings')
    p.add_argument('--json', action='store_true')

    # import-analysis
    p = sub.add_parser('import-analysis', help='Import Claude Code analysis results')
    p.add_argument('--tier', type=int, choices=[1, 2], default=None,
                   help='Import specific tier (default: auto-detect)')
    p.add_argument('--json', action='store_true')

    # portfolio
    p = sub.add_parser('portfolio', help='Portfolio risk analysis for top picks')
    p.add_argument('--top', type=int, default=10, help='Number of top picks (default: 10)')
    p.add_argument('--max-sector', type=float, default=0.35, help='Max sector weight (default: 0.35)')
    p.add_argument('--json', action='store_true')

    # watchlist
    p = sub.add_parser('watchlist', help='Manage watchlist')
    p.add_argument('--add', type=str, help='Add ticker')
    p.add_argument('--remove', type=str, help='Remove ticker')
    p.add_argument('--json', action='store_true')

    # export
    p = sub.add_parser('export', help='Export results to CSV')
    p.add_argument('--output', type=str, help='Output file path')
    p.add_argument('--json', action='store_true')

    # backup
    p = sub.add_parser('backup', help='Backup sec_data/')
    p.add_argument('--output', type=str, help='Output zip path')
    p.add_argument('--json', action='store_true')

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(asctime)s %(name)s %(levelname)s %(message)s')

    if not args.command:
        parser.print_help()
        return

    # Dispatch
    commands = {
        'setup': cmd_setup,
        'status': cmd_status,
        'scan': cmd_scan,
        'analyze': cmd_analyze,
        'prepare': cmd_prepare,
        'import-analysis': cmd_import_analysis,
        'gems': cmd_gems,
        'potential': cmd_potential,
        'company': cmd_company,
        'backtest': cmd_backtest,
        'calibrate': cmd_calibrate,
        'outcomes': cmd_outcomes,
        'portfolio': cmd_portfolio,
        'watchlist': cmd_watchlist,
        'export': cmd_export,
        'backup': cmd_backup,
    }
    fn = commands.get(args.command)
    if fn:
        try:
            fn(args)
        except KeyboardInterrupt:
            console.print(f"\n{_yellow('Interrupted')}")
            sys.exit(1)
        except Exception as e:
            console.print(f"\n{_red('Error')}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
