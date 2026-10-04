"""Typer CLI — the primary user surface for the assistant.

Commands are thin: each one builds a GraphState with the right intent, invokes
the LangGraph, and renders the result. New commands plug in by setting a new
intent and (if needed) adding a route in graph.py.
"""

from __future__ import annotations

import subprocess
import uuid
import webbrowser
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from pydantic import ValidationError

from sqlalchemy import func, select

from assistant.config import get_config
from assistant.graph import get_graph
from assistant.rag.filters import MetadataFilters
from assistant.storage.domains import sync_domains_from_config
from assistant.storage import (
    Chunk,
    CriteriaVersion,
    Domain,
    Interaction,
    Paper,
    get_engine,
    session_scope,
)

app = typer.Typer(
    name="assistant",
    help="Personal AI research assistant: track domains, curate papers, answer questions.",
    no_args_is_help=True,
)
criteria_app = typer.Typer(help="Manage curation criteria for a domain.")
domain_app = typer.Typer(help="Manage tracked research domains.")
monitor_app = typer.Typer(help="Run the arxiv/venue monitor.")
config_app = typer.Typer(help="Inspect configuration.")
app.add_typer(criteria_app, name="criteria")
app.add_typer(domain_app, name="domain")
app.add_typer(monitor_app, name="monitor")
app.add_typer(config_app, name="config")

console = Console()


def _invoke(state: dict) -> dict:
    """Invoke the graph with a fresh thread_id per CLI call.

    The checkpointer requires a thread_id; a per-invocation UUID is fine for
    one-shot commands and keeps state from leaking across runs.
    """
    return get_graph().invoke(
        state, config={"configurable": {"thread_id": str(uuid.uuid4())}}
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question."),
    min_year: int | None = typer.Option(None, help="Earliest publication year."),
    max_year: int | None = typer.Option(None, help="Latest publication year."),
    venue: str | None = typer.Option(None, help="Exact publication venue (case-insensitive)."),
    section: list[str] | None = typer.Option(None, help="Section type; repeat to include several."),
    domain: str | None = typer.Option(None, help="Limit retrieval to a topic."),
    paper: str | None = typer.Option(None, help="Limit retrieval to a paper ID."),
) -> None:
    """Ask the assistant a question over the curated knowledge base."""
    if domain and paper:
        raise typer.BadParameter("Choose either --domain or --paper, not both")
    try:
        filters = MetadataFilters(min_year=min_year, max_year=max_year, venue=venue, section_types=section or [])
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    state: dict = {"intent": "ask", "question": question}
    if domain:
        state["domain"] = domain
    if paper:
        state["paper_ids"] = [paper]
    if filters.model_dump(exclude_defaults=True):
        state["scratch"] = {"metadata_filters": filters.model_dump(exclude_defaults=True)}
    result = _invoke(state)
    console.print(result.get("answer", "[no answer]"))
    citations = result.get("citations") or []
    if citations:
        console.print(f"\n[dim]Citations:[/dim] {', '.join(citations)}")


@app.command()
def ingest(source: str = typer.Argument(..., help="arxiv ID or path to a PDF.")) -> None:
    """Ingest a paper directly (bypasses the curator)."""
    result = _invoke({"intent": "ingest", "scratch": {"source": source}})
    ingested = result.get("ingested") or []
    if ingested:
        console.print(f"[green]Ingested:[/green] {', '.join(ingested)}")
        return
    errors = (result.get("scratch") or {}).get("ingest_errors") or []
    if errors:
        for error in errors:
            console.print(f"[red]Failed to ingest {error['source']}:[/red] {error['error']}")
        raise typer.Exit(1)
    console.print("[yellow]Nothing ingested.[/yellow]")


@criteria_app.command("add")
def criteria_add(
    domain: str = typer.Argument(..., help="Domain name."),
    comment: str = typer.Argument(..., help="Free-text feedback on curation."),
) -> None:
    """Add a user comment that updates this domain's curation criteria."""
    result = _invoke(
        {"intent": "criteria_update", "domain": domain, "user_comment": comment}
    )
    v = result.get("new_criteria_version")
    if v is not None:
        console.print(f"[green]Criteria for {domain!r} updated to version {v}.[/green]")
    else:
        console.print("[yellow]Criteria update is a stub (Phase 5).[/yellow]")


@criteria_app.command("show")
def criteria_show(domain: str = typer.Argument(...)) -> None:
    """Show the active criteria version for a domain."""
    from assistant.memory.criteria_store import get_active

    snap = get_active(domain)
    if snap is None:
        console.print(f"[yellow]Domain {domain!r} not found.[/yellow]")
        return
    console.print(f"[bold]{domain}[/bold] — active version {snap.version}")
    console.print("\n[bold]structured_rules:[/bold]")
    console.print_json(data=snap.structured_rules)
    console.print("\n[bold]nl_addendum:[/bold]")
    console.print(snap.nl_addendum or "[dim](none)[/dim]")
    if snap.source_comment:
        console.print("\n[bold]source comment:[/bold]")
        console.print(snap.source_comment)


@domain_app.command("add")
def domain_add(
    name: str = typer.Argument(...),
    categories: str = typer.Option("", help="Comma-separated arxiv categories."),
    venues: str = typer.Option("", help="Comma-separated venue names."),
) -> None:
    """Track a new research domain (writes directly to the DB)."""
    cats = [c.strip() for c in categories.split(",") if c.strip()]
    vs = [v.strip() for v in venues.split(",") if v.strip()]
    with session_scope() as s:
        existing = s.execute(select(Domain).where(Domain.name == name)).scalar_one_or_none()
        if existing is None:
            s.add(Domain(name=name, arxiv_categories=cats, venues=vs))
            console.print(f"[green]Added domain {name!r}.[/green]")
        else:
            existing.arxiv_categories = cats
            existing.venues = vs
            console.print(f"[green]Updated domain {name!r}.[/green]")


@domain_app.command("list")
def domain_list() -> None:
    """List tracked domains (from the DB)."""
    with session_scope() as s:
        rows = s.execute(select(Domain).order_by(Domain.name)).scalars().all()
    if not rows:
        console.print(
            "[dim]No domains in the DB. Edit config.yaml then run `assistant domain sync`.[/dim]"
        )
        return
    t = Table(title="Tracked domains")
    t.add_column("name")
    t.add_column("arxiv categories")
    t.add_column("venues")
    for d in rows:
        t.add_row(d.name, ", ".join(d.arxiv_categories or []), ", ".join(d.venues or []))
    console.print(t)


@domain_app.command("sync")
def domain_sync() -> None:
    """Sync domains from config.yaml into the DB (insert-or-update by name)."""
    added, updated = sync_domains_from_config()
    console.print(f"[green]Synced {added} new, {updated} updated.[/green]")


@monitor_app.command("tick")
def monitor_tick(domain: str | None = typer.Option(None, help="Restrict to one domain.")) -> None:
    """Run one curation cycle: fetch recent papers -> judge -> ingest accepted."""
    result = _invoke({"intent": "monitor_tick", "domain": domain})
    candidates = result.get("candidates") or []
    accepted = result.get("accepted") or []
    ingested = result.get("ingested") or []
    console.print(
        f"candidates={len(candidates)}  accepted={len(accepted)}  ingested={len(ingested)}"
    )


@monitor_app.command("run")
def monitor_run() -> None:
    """Start the scheduler and tick on the configured interval (blocks)."""
    from assistant.scheduler import MonitorScheduler

    sched = MonitorScheduler()
    if not sched.configure():
        console.print(
            "[yellow]Monitor is disabled. Set monitor.enabled: true in config.yaml.[/yellow]"
        )
        return
    sched.run_forever()


@config_app.command("show")
def config_show() -> None:
    """Print the active configuration."""
    cfg = get_config()
    console.print_json(cfg.model_dump_json(indent=2))


@app.command("init")
def init_storage() -> None:
    """Create the SQLite schema and data directories. Safe to re-run."""
    cfg = get_config()
    cfg.storage.ensure_dirs()
    get_engine()  # triggers create_all
    console.print(
        f"[green]Initialized.[/green]\n"
        f"  sqlite: {cfg.storage.sqlite_path}\n"
        f"  qdrant: {cfg.storage.qdrant_path}\n"
        f"  pdfs:   {cfg.storage.pdf_dir}"
    )


@app.command("status")
def status() -> None:
    """Show row counts for each table."""
    with session_scope() as s:
        counts = {
            "domains": s.scalar(select(func.count()).select_from(Domain)),
            "papers": s.scalar(select(func.count()).select_from(Paper)),
            "chunks": s.scalar(select(func.count()).select_from(Chunk)),
            "criteria_versions": s.scalar(select(func.count()).select_from(CriteriaVersion)),
            "interactions": s.scalar(select(func.count()).select_from(Interaction)),
        }
    t = Table(title="Storage status")
    t.add_column("table")
    t.add_column("rows", justify="right")
    for k, v in counts.items():
        t.add_row(k, str(v))
    console.print(t)


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", help="Address to bind."),
    port: int = typer.Option(8000, help="Port to bind."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the UI."),
    build: bool = typer.Option(False, help="Install and build the React app first."),
) -> None:
    """Start the local web UI and API in one server process."""
    root = Path(__file__).resolve().parent.parent
    web_dir = root / "web"
    static_index = Path(__file__).resolve().parent / "web" / "static" / "index.html"

    if build:
        if not web_dir.is_dir():
            raise typer.BadParameter(f"Frontend directory not found: {web_dir}")
        if not (web_dir / "node_modules").is_dir():
            console.print("[dim]Installing frontend dependencies...[/dim]")
            subprocess.run(["npm", "install"], cwd=web_dir, check=True, shell=True)
        console.print("[dim]Building frontend...[/dim]")
        subprocess.run(["npm", "run", "build"], cwd=web_dir, check=True, shell=True)

    if not static_index.is_file():
        console.print(
            "[red]Web UI has not been built.[/red] Run `assistant serve --build` "
            "or `cd web; npm install; npm run build`."
        )
        raise typer.Exit(1)
    if host not in {"127.0.0.1", "localhost", "::1"}:
        console.print(
            "[yellow]Warning: the server is intended for localhost. "
            "The API still rejects non-local Host headers.[/yellow]"
        )

    url_host = "localhost" if host in {"0.0.0.0", "::"} else host
    url = f"http://{url_host}:{port}"
    if open_browser:
        webbrowser.open(url)
    console.print(f"[green]Serving research assistant at {url}[/green]")

    import uvicorn

    uvicorn.run("assistant.web.app:create_app", factory=True, host=host, port=port, workers=1)


@config_app.command("roles")
def config_roles() -> None:
    """List configured LLM roles and the provider/model assigned to each."""
    cfg = get_config()
    t = Table(title="LLM roles")
    t.add_column("role")
    t.add_column("provider")
    t.add_column("model")
    for role, spec in cfg.llm.roles.items():
        t.add_row(role, spec.provider, spec.model)
    console.print(t)


if __name__ == "__main__":
    app()
