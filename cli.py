#!/usr/bin/env python3
"""
SEC Analyzer — Terminal CLI
============================
Run the full SEC filing analysis pipeline from the command line.

Usage:
    python cli.py setup                     # Set API keys
    python cli.py status                    # Pipeline state summary
    python cli.py scan                      # Full pipeline: universe → find → download
    python cli.py analyze                   # Run T1 + T2 analysis
    python cli.py gems                      # Show top picks
    python cli.py potential                 # Show CPS-ranked companies
    python cli.py company TICKER            # Deep dive on one company
    python cli.py backtest                  # Run price backtest
    python cli.py calibrate                 # Run feedback loop
    python cli.py watchlist                 # Manage watchlist
    python cli.py export                    # Export to CSV
    python cli.py backup                    # Backup sec_data/
"""

import argparse
import json
import os
import sys
import time
import logging

# Add project dir to path
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
    resp = input("  Enter FMP key (or press Enter to skip): ").strip()
    if resp:
        engine.persist_key('FMP_API_KEY', resp)
        console.print(f"  {_green('✓ Saved')}")

    # Anthropic
    ant_status = _green("✓ Set") if keys['anthropic']['set'] else _red("✗ Not set")
    console.print(f"\nAnthropic API Key: {ant_status}  {_dim(keys['anthropic']['masked'])}")
    console.print(f"  (Required for LLM analysis. Get at console.anthropic.com)")
    resp = input("  Enter Anthropic key (or press Enter to skip): ").strip()
    if resp:
        engine.persist_key('ANTHROPIC_API_KEY', resp)
        console.print(f"  {_green('✓ Saved')}")

    # OpenAI
    oai_status = _green("✓ Set") if keys['openai']['set'] else _red("✗ Not set")
    console.print(f"\nOpenAI API Key:    {oai_status}  {_dim(keys['openai']['masked'])}")
    console.print(f"  (Optional. Alternative LLM provider.)")
    resp = input("  Enter OpenAI key (or press Enter to skip): ").strip()
    if resp:
        engine.persist_key('OPENAI_API_KEY', resp)
        console.print(f"  {_green('✓ Saved')}")

    console.print(f"\n{_green('✓')} Keys saved to config.json")


# ============================================================================
# STATUS COMMAND
# ============================================================================
def cmd_status(args):
    """Show pipeline status summary."""
    import engine

    status = engine.get_status()
    keys = status.get('keys', {})

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

    if args.json:
        print(json.dumps(status, indent=2))


# ============================================================================
# SCAN COMMAND
# ============================================================================
def cmd_scan(args):
    """Run scraper pipeline."""
    import engine

    steps = args.step if args.step else ['universe', 'find', 'download', 'insider']

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

    console.print(f"\n{_green('✓')} Scan complete")
    console.print(f"  Universe: {result['universe_count']} companies")
    console.print(f"  Filings:  {result['filings_total']} found, {result['filings_downloaded']} downloaded")

    if args.json:
        print(json.dumps(result, indent=2))


# ============================================================================
# ANALYZE COMMAND
# ============================================================================
def cmd_analyze(args):
    """Run LLM analysis (T1 + T2)."""
    import engine

    console.rule("SEC Analyzer — LLM Analysis")

    model_map = {
        'haiku': 'claude-haiku-4-5-20251001',
        'sonnet': 'claude-sonnet-4-5-20250929',
        'gpt4o-mini': 'gpt-4o-mini',
        'gpt4o': 'gpt-4o',
    }
    t2_model = model_map.get(args.model, args.model)

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

    if args.json:
        print(json.dumps(result, indent=2, default=str))


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

    limit = args.limit or len(gems)
    gems = gems[:limit]

    console.rule(f"SEC Analyzer — Top Gems ({len(gems)})")

    if HAS_RICH:
        t = Table(box=box.ROUNDED, show_lines=True)
        t.add_column("#", style="dim", width=3)
        t.add_column("Ticker", style="bold cyan")
        t.add_column("Company")
        t.add_column("Gem", justify="right")
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
            diff = g.get('diff_signal', '')
            diff_color = 'green' if 'improving' in diff else 'dim'

            t.add_row(
                str(i + 1),
                g['ticker'],
                g['company_name'][:30],
                f"[{score_color}]{score:.0f}[/{score_color}]",
                f"[{conv_color}]{conv}[/{conv_color}]",
                f"[{ins_color}]{ins}[/{ins_color}]",
                f"[{diff_color}]{diff}[/{diff_color}]",
                g.get('recommendation', '')[:15],
                g.get('one_liner', '')[:45],
            )
        console.print(t)
    else:
        fmt = "{:>3}  {:8}  {:30}  {:>5}  {:12}  {:20}  {:15}"
        console.print(fmt.format("#", "Ticker", "Company", "Gem", "Conviction", "Insider", "Diff"))
        console.print("-" * 100)
        for i, g in enumerate(gems):
            console.print(fmt.format(
                i + 1, g['ticker'], g['company_name'][:30],
                f"{g['gem_score']:.0f}", g.get('conviction', ''),
                g.get('insider_signal', ''), g.get('diff_signal', ''),
            ))

    if args.json:
        print(json.dumps(gems, indent=2))


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

    if args.json:
        print(json.dumps(companies, indent=2, default=str))


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
            console.print(f"    {f.get('filing_date','')}  {f.get('form_type',''):10} T1={f.get('tier1_score','—')} Gem={f.get('final_gem_score','—')}")

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

    if args.json:
        print(json.dumps(detail, indent=2, default=str))


# ============================================================================
# BACKTEST COMMAND
# ============================================================================
def cmd_backtest(args):
    """Run price backtesting."""
    import engine

    console.rule("SEC Analyzer — Backtest")

    if args.clear:
        console.print(f"  {_yellow('Clearing cached results...')}")

    console.print("  Fetching price data (this may take a minute)...")
    result = engine.run_backtest(clear_cache=args.clear)

    if result.get('error'):
        console.print(f"  {_red(result['error'])}")
        return

    stats = result.get('stats', {})
    console.print(f"\n  Total tested:    {result['total']}")
    console.print(f"  With price data: {_green(stats.get('with_price_data', 0))}")
    if stats.get('error_count', 0):
        console.print(f"  Errors:          {_red(stats['error_count'])}")

    console.print(f"\n  Avg 30d return:  {stats.get('avg_30d', 0):.1f}%")
    console.print(f"  Avg 60d return:  {stats.get('avg_60d', 0):.1f}%")
    console.print(f"  Avg 90d return:  {stats.get('avg_90d', 0):.1f}%")
    console.print(f"  Win rate (30d):  {stats.get('win_rate_30d', 0):.0f}%")

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
        t.add_column("Status")

        for r in individual[:30]:
            def _fmt_ret(v):
                if v is None: return "[dim]—[/dim]"
                c = 'green' if v >= 0 else 'red'
                return f"[{c}]{v:.1f}%[/{c}]"

            err = r.get('error', '')
            status = f"[red]{err}[/red]" if err else (
                "[green]✅ Win[/green]" if r.get('return_90d') is not None and r['return_90d'] > 0
                else "[red]❌ Loss[/red]" if r.get('return_90d') is not None else "[dim]—[/dim]"
            )

            t.add_row(
                r['ticker'], r.get('filing_date', ''),
                str(r.get('gem_score', '—')),
                _fmt_ret(r.get('return_30d')),
                _fmt_ret(r.get('return_60d')),
                _fmt_ret(r.get('return_90d')),
                status,
            )
        console.print(t)

    if args.json:
        print(json.dumps(result, indent=2, default=str))


# ============================================================================
# CALIBRATE COMMAND
# ============================================================================
def cmd_calibrate(args):
    """Run feedback loop calibration."""
    import engine

    console.rule("SEC Analyzer — Calibration")
    console.print(f"  Return window: {args.window} days")
    console.print(f"  Auto-apply:    {'Yes' if args.apply else 'No'}")
    console.print("")

    report = engine.run_calibration(
        return_window=args.window,
        auto_apply=args.apply,
    )

    if report.get('status') in ('insufficient_data', 'insufficient_backtest_data'):
        console.print(f"  {_yellow(report.get('message', 'Insufficient data'))}")
        return

    console.print(f"  {_green(report.get('message', 'Calibration complete'))}")

    # Show correlations
    corrs = report.get('correlations', {})
    if corrs and HAS_RICH:
        t = Table(box=box.SIMPLE, title="Signal Correlations")
        t.add_column("Signal")
        t.add_column("Correlation", justify="right")
        t.add_column("Strength")
        for sig_name, sig_data in sorted(corrs.items(), key=lambda x: abs(x[1].get('correlation', 0)), reverse=True):
            corr = sig_data.get('correlation', 0)
            strength = sig_data.get('strength', '')
            c = 'green' if corr > 0.1 else ('red' if corr < -0.1 else 'dim')
            t.add_row(sig_name, f"[{c}]{corr:.3f}[/{c}]", strength)
        console.print(t)

    # Show suggested weights
    opt = report.get('weight_optimization', {})
    suggested = opt.get('suggested_weights', {})
    if suggested:
        console.print(f"\n  Suggested weights:")
        for k, v in suggested.items():
            console.print(f"    {k:30} = {v}")
        if report.get('weights_applied'):
            console.print(f"\n  {_green('✓ Weights applied automatically')}")
        else:
            console.print(f"\n  To apply: python cli.py calibrate --apply")

    if args.json:
        print(json.dumps(report, indent=2, default=str))


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
    console.rule(f"Watchlist ({len(wl)} tickers)")
    if wl:
        # Show detail for each
        gems = {g['ticker']: g for g in engine.get_gems()}
        for ticker in wl:
            g = gems.get(ticker)
            if g:
                score_str = f"Gem={g['gem_score']:.0f}"
                conv = g.get('conviction', '')
                console.print(f"  {_cyan(ticker):8}  {score_str:10}  {conv:10}  {g.get('one_liner', '')[:50]}")
            else:
                console.print(f"  {_cyan(ticker):8}  {_dim('(no analysis data)')}")
    else:
        console.print(f"  {_dim('Empty. Add with: python cli.py watchlist --add TICKER')}")


# ============================================================================
# EXPORT COMMAND
# ============================================================================
def cmd_export(args):
    """Export results to CSV."""
    import engine

    output = args.output or 'sec_data/export.csv'
    path = engine.export_csv(output)
    size = os.path.getsize(path)
    console.print(f"  {_green('✓')} Exported to {path} ({size:,} bytes)")


# ============================================================================
# BACKUP COMMAND
# ============================================================================
def cmd_backup(args):
    """Backup sec_data/ to a zip file."""
    import engine

    output = args.output
    console.print("  Backing up sec_data/...")
    path = engine.backup_data(output)
    size = os.path.getsize(path)
    console.print(f"  {_green('✓')} Backup saved: {path} ({size / 1024 / 1024:.1f} MB)")


# ============================================================================
# MAIN — ARGPARSE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        prog='sec-analyzer',
        description='SEC Filing Analyzer — Terminal CLI',
    )
    parser.add_argument('--json', action='store_true', help='Output raw JSON')
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
                   help='T2 model: haiku, sonnet, gpt4o-mini, gpt4o (default: haiku)')
    p.add_argument('--tier2-pct', type=float, default=0.25,
                   help='Top %% of companies for T2 (default: 0.25)')
    p.add_argument('--effort', choices=['low', 'medium', 'high'], default=None,
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
    p.add_argument('--json', action='store_true')

    # calibrate
    p = sub.add_parser('calibrate', help='Run feedback loop calibration')
    p.add_argument('--window', type=int, default=90, help='Return window in days (default: 90)')
    p.add_argument('--apply', action='store_true', help='Auto-apply suggested weights')
    p.add_argument('--json', action='store_true')

    # watchlist
    p = sub.add_parser('watchlist', help='Manage watchlist')
    p.add_argument('--add', type=str, help='Add ticker')
    p.add_argument('--remove', type=str, help='Remove ticker')
    p.add_argument('--json', action='store_true')

    # export
    p = sub.add_parser('export', help='Export results to CSV')
    p.add_argument('--output', type=str, help='Output file path')

    # backup
    p = sub.add_parser('backup', help='Backup sec_data/')
    p.add_argument('--output', type=str, help='Output zip path')

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
        'gems': cmd_gems,
        'potential': cmd_potential,
        'company': cmd_company,
        'backtest': cmd_backtest,
        'calibrate': cmd_calibrate,
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
