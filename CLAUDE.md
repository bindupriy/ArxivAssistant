# CLAUDE.md

Guidance for Claude Code (and other AI coding agents) working in this repository. Read this first before making non-trivial changes.

## TL;DR

- Research assistant tracking domains, curating arXiv papers, and answering with citations. The current localhost implementation is single-user; multi-user access is a requirement but needs authentication, authorization, and data isolation (see `PLANNER.md`). Do not equate concurrent requests with multi-user support.
- **Config-driven** is the core principle: provider/model is chosen per-agent-role in `config.yaml`, RAG mode (vanilla/agentic) is a config toggle, MCP servers are added by YAML edit. Don't hard-code these.
- **All nine agents are real LangGraph nodes** from day 1. Some are deeper than others. Do not delete or merge nodes — they're the public topology of the system.
- The approved design and build order live at `~/.claude/plans/i-want-to-develop-enchanted-wozniak.md`. The narrative ("why this shape?") is there.

## Repo layout

```
assistant/
  config.py          # Pydantic config (AppConfig + Settings) loaded from config.yaml + .env
  state.py           # LangGraph GraphState TypedDict — the shared state schema
  graph.py           # Wires all 9 agents into a LangGraph; SqliteSaver checkpointer
  cli.py             # Typer CLI — graph-backed commands use _invoke(); serve hosts the web app
  scheduler.py       # APScheduler wrapper around `monitor_tick` (disabled by default)

  web/
    app.py           # localhost FastAPI routes + SPA serving
    runner.py        # graph streaming + bounded concurrent in-memory jobs
    api_client.py    # loopback-only HTTP client for optional Streamlit UI
  streamlit_app.py   # optional Streamlit frontend; uses FastAPI, not embedded stores
    static/          # generated Vite build; gitignored

  agents/            # One file per node — all are real, some stubs are intentional
    orchestrator.py  # intent dispatch
    qa.py            # claim generation + evidence verification; low-conf info_gatherer detour
    retrieval.py     # vanilla vs agentic, paper/topic/metadata scope, query rewrite
    curator.py       # per-topic LLM judge; preserves accepted and rejected results
    ingestion.py     # delegates to rag.ingest.ingest_source
    info_gatherer.py # binds MCP tools to the info_gatherer LLM
    criteria.py      # comment + active criteria -> new versioned criteria
    memory.py        # writes interactions + curation decisions; consolidate() is a stub
    monitor.py       # arxiv polling with library/inbox deduplication

  llm/
    router.py        # LLMRouter.chat/embeddings/reranker — only place agents touch the model layer
    providers.py     # one factory per (provider, modality)

  rag/
    arxiv_fetch.py   # arxiv ID -> PDF + metadata
    parser.py        # PyMuPDF + heuristic section detection -> ParsedPaper
    chunker.py       # section-aware chunking, 15% overlap
    embedder.py      # thin wrapper over LLMRouter.embeddings
    retriever.py     # hybrid dense+BM25 with RRF fusion; cache invalidated on ingest/removal
    reranker.py      # optional cross-encoder via sentence-transformers (lazy import)
    agentic.py       # decompose -> retrieve -> critique loop -> rerank
    compress.py      # extractive answer-context windows; retains full evidence text
    filters.py       # validated per-question publication/section filters
    ingest.py        # end-to-end: source -> chunks -> embeddings -> stores

  storage/
    schema.py        # SQLAlchemy models incl. papers, chats, decisions, criteria, memory
    domains.py       # shared config.yaml -> database topic synchronization
    sqlite_store.py  # engine + session_scope() context manager
    qdrant_store.py  # client + upsert + search + ensure_collection (lazy on first insert)

  memory/
    episodic.py      # record_interaction / set_feedback / recent
    criteria_store.py# get_active / append_version
    conversations.py # saved fixed-scope chats + history
    curation_store.py# curator inbox decisions and statuses

  mcp/
    client.py        # MultiServerMCPClient from config; aget_tools / get_tools_sync

  web/                 # React/Vite/TypeScript source; builds into assistant/web/static
```

`data/` is gitignored and holds the Qdrant store, SQLite DB, and downloaded PDFs.

## Conventions

### State management

- One `GraphState` TypedDict, declared in `state.py`. Add fields here when an agent needs to share something.
- Each agent **only writes the fields it owns** (see `ARCHITECTURE.md` for the field-by-agent map). Returning extra junk pollutes state.
- For multi-turn data that's not part of the public contract, use `state.scratch: dict[str, Any]`.

### LLM access

- **Never instantiate a model client directly.** Always go through `get_router().chat(role)` (or `.embeddings(role)` / `.reranker(role)`).
- **Never hardcode a role name** outside of `config.yaml` and the call site. If you find yourself adding a new role, add it to the YAML's example block and to the per-role default in `config.yaml`.
- Roles are arbitrary strings — `qa`, `citation_verifier`, `conversation_title`, `retrieval`, `curator_judge`, `summarizer`, `criteria`, `info_gatherer`, `embedder`, `reranker`, `orchestrator` exist today. Add more by editing config; the router lazily resolves on first use.

### Answer attribution

- QA generates validated, single-line claim blocks with per-claim source numbers and exact supporting quotes. General knowledge is allowed; do not force every claim to cite a paper.
- A paper citation requires both quote occurrence in the retrieved chunk (whitespace-normalized) and a positive whole-claim verdict from a separate verifier call. The verifier uses `citation_verifier`, falling back to `qa` for older configs. It is a service call inside the existing QA node, not a new graph node.
- When prompt compression is enabled, the quote must also occur in the exact excerpt the generator saw. Keep the full chunk in state and verifier context; never verify a model-generated summary as paper evidence.
- The application renders citation markers only from verified evidence. Do not reintroduce raw model-supplied markers or trust a standalone citations list. Missing/invalid evidence and verifier failures must withhold citations.
- Claims without accepted citations receive the `LLM knowledge` label. This means unverified against paper sources, not proven correct. External tool context and prior-turn citations are not evidence for current paper citations.
- Keep the public `answer`, `citations` (chunk IDs), `confidence`, and `retrieved_chunks` contract unchanged. Run `python -m unittest discover -s tests -p test_qa.py -v` after citation-policy changes; the tests mock models and do not establish real-model citation accuracy.

### Storage

- All DB access goes through `with session_scope() as s:` — never create sessions ad hoc.
- Qdrant collections are created lazily on first insert (see `ensure_collection`). Don't pre-create them or assume a vector dimension. The first upsert sets it.
- Embedded Qdrant calls must remain under the store's process lock because web chat and background jobs use different threads.
- The BM25 index is in-memory and cached via `@lru_cache`. **Call `invalidate_bm25_cache()` after any operation that inserts, updates, or deletes chunks**, as ingestion and paper removal already do.
- `Paper.reading_status` (`new`, `reviewing`, `read`) and `is_favorite` are user-controlled, separate from ingestion `status`. Defaults are `new` and `False`; re-ingestion must preserve both preferences.
- Paper removal deletes the SQLite paper and its cascading chunks, calls `delete_paper_vectors()` for both Qdrant collections, and invalidates BM25 after the database commit. `delete_by_paper()` only removes chunk vectors and is still used by re-ingestion. Preserve source PDFs and saved chats.

### Config

- Pydantic models in `config.py` are the source of truth for shape. If you add a field to `config.yaml`, add it to the corresponding model.
- API keys, base URLs → `.env` (Settings). Everything else → `config.yaml` (AppConfig).
- `get_config()` and `get_settings()` are `lru_cache`'d. They're fine to call from anywhere.

### CLI

- Every command should build a state dict and call `_invoke(state)` rather than calling `get_graph().invoke()` directly — `_invoke` attaches a fresh `thread_id` for the checkpointer.
- Use Rich for output (`console.print`, `Table`). Don't `print()`.

### Web

- Run one server process because embedded Qdrant is process-local. Qdrant calls are protected by the store lock.
- All `/api` writes require `X-Requested-With: assistant-ui`; keep localhost TrustedHost restrictions intact.
- Conversations have a fixed scope selected at creation. New chat in the main sidebar opens a draft; the composer dropdown chooses its scope before the first message creates the conversation. Existing conversations show disabled scope controls reflecting their saved scope. Load prior turns from the conversation store and pass them via `scratch.history`; do not reuse LangGraph state across turns.
- Auto-name only chats still titled `New conversation`. `generate_conversation_title()` uses the first saved question and `conversation_title` (fallback `qa`), calls the model outside the database transaction, and conditionally updates the placeholder. Preserve descriptive titles and deleted chats. Emit the title as a `conversation` event after the final answer; failures must not discard the answer. Run `python -m unittest discover -s tests -p test_conversations.py -v` after naming changes.
- Paper PATCH requests update only explicitly supplied fields (`model_fields_set`); changing a star or reading status must not clear the topic or reset other preferences.
- Frontend source lives in `web/`; `npm run build` writes generated assets to `assistant/web/static/`.
- The optional Streamlit UI in `assistant/streamlit_app.py` talks only to local FastAPI through `assistant/web/api_client.py`. Never open a second embedded Qdrant client or use graph/storage directly from Streamlit; install via `pip install -e '.[streamlit]'`.
- The API streams chat events as NDJSON. Keep event shapes compatible with `web/src/api.ts`.
- The UI must keep reading after `final` for the optional title update, update the matching sidebar item, and change the active header only if that same conversation is still selected.
- Keep `ChatPage` mounted and visible in `Shell` beside the active Library, Topics, or Inbox workspace. Workspace changes refresh chat lists and scope pickers, not the pending transcript. Drafts, turns, and progress are keyed by conversation ID; stream callbacks must update their originating session and ignore deleted sessions. Do not overwrite cached turns with incomplete saved history or treat workspace navigation as a full page reload.
- New chat, history selection, and paper/topic chat actions must preserve the open workspace. Only the workspace close button or Chat navigation returns to chat-only mode. Keep workspace drawers/dialogs inside their pane; use the stacked responsive layout on narrow screens rather than hiding chat.
- `ChatPage` renders New chat and history into `Shell`'s persistent sidebar slot through a portal. Keep those controls available from other tabs, below the page navigation separator; do not restore a second conversation rail. Scope controls belong inside the composer, including before any conversation exists.
- Background ingest and monitor jobs use a configurable, bounded thread pool (two workers by default). Index writes and same-paper operations are coordinated in process; jobs remain in memory.

### Frontend

- Run `npm run lint` and `npm run build` from `web/` after changes; the build includes TypeScript checking.
- Run `npx prettier --check src/App.tsx src/api.ts` for the main frontend source files.
- Preserve fixed conversation scope (`library`, `topic`, or `paper`). Library status and Favorites filters affect browsing, not chat retrieval scope.
- Resolve citation `[n]` through `turn.citations[n - 1]`, then find the matching `chunk_id` in the API-provided `sources` array. Never index `sources` by citation number: deleted chunks are omitted. Disable citations whose source is unavailable.
- The Library detail pane shows metadata and actions, not raw chunks or a Sections list. Keep chunk content available through chat citation drawers.

## Common tasks

### Add a new agent

1. Create `agents/<name>.py` with a `<name>_node(state) -> partial_state` callable.
2. Register it in `graph.py`: `g.add_node("<name>", <name>_node)` + the edges that connect it.
3. If the agent needs new state fields, add them to `GraphState` in `state.py`.
4. If the agent calls an LLM, pick a role name and add it to `config.yaml`'s `llm.roles` block (with a sensible default provider/model).
5. Update `ARCHITECTURE.md`'s agent table.

### Add a new MCP server

Just edit `config.yaml`:

```yaml
mcp:
  servers:
    - name: my_server
      transport: stdio
      command: ["uvx", "some-mcp-server"]
      env: { SOME_KEY: "$SOME_ENV_VAR" }
```

No code changes. The Info Gatherer will pick it up.

### Swap a model

Edit one line in `config.yaml`:

```yaml
llm:
  roles:
    qa: { provider: ollama, model: qwen2.5:14b }  # was claude-sonnet-4-6
```

Done. No code changes. Test with `assistant config roles` then `assistant ask "..."`.

### Replace the PDF parser

`rag/parser.py:parse_pdf(pdf_path) -> ParsedPaper` is the swap point. Drop in Marker, GROBID, or anything else that returns the same shape (`full_text`, `sections: list[Section]`, `abstract`). Nothing else needs to change.

### Run a smoke test by hand

```powershell
assistant init
assistant ingest 2401.05566
assistant status                 # confirm papers/chunks rows > 0
assistant ask "summarize the method"
assistant serve --build
```

If embeddings fail, check Ollama is running (`ollama list`) or switch the `embedder` role to OpenAI.

### Run library regression tests

From the repository root, using the project's Python environment:

```powershell
python -m unittest discover -s tests -p test_library.py -v
```

The tests use temporary SQLite databases and in-memory Qdrant. They cover preferences, combined filters, write protection, database upgrades, re-ingestion, and deletion cleanup without touching the user's library.

## Gotchas

- **`langgraph-checkpoint-sqlite` is required for state persistence.** If it's not installed, `graph.py` silently falls back to `MemorySaver`. That's fine for dev but loses state across runs.
- **`sentence-transformers` is intentionally not in `pyproject.toml`.** Reranking degrades gracefully without it. Add it (`pip install sentence-transformers`) when you want better top-k quality.
- **Marker, the recommended better parser, is also not in deps** — it's heavy. PyMuPDF gets us to working end-to-end; swap when needed (see "Replace the PDF parser").
- **Manual CLI ingest has no topic option.** The shared ingest function and web UI support topic attribution; a future CLI `--domain` flag can thread it through `scratch`.
- **Web background jobs are in memory.** They run on a bounded worker pool but disappear on server restart; keep one server process for embedded Qdrant and in-process locks.
- **There is no general migration framework yet.** `get_engine()` lazily calls `create_all`, then adds missing `reading_status` and `is_favorite` columns with idempotent SQLite ALTER statements. Existing papers default to New and unstarred. Restart the API after upgrading; other changes to existing tables still need an explicit migration strategy.
- **The BM25 index doesn't persist** — it's rebuilt per process from the SQLite chunks table. For very large libraries this becomes slow; the planned upgrade is Qdrant's native sparse vectors.
- **`config.yaml` keys with no default in `AppConfig` will error on startup.** If you add an optional field, give it a `Field(default_factory=...)`.
- **The QA -> info_gatherer detour can only fire once per invocation** (guarded by `state.did_gather`). Don't remove that guard or you'll loop.

## What NOT to do

- Don't create new agent files outside `agents/` and expect them to be wired automatically — the graph is built explicitly in `graph.py`.
- Don't add provider-specific code paths inside agents. If you need provider-specific behavior, push it into `llm/providers.py`.
- Don't import from `langchain` directly in agents when LangChain abstractions already wrap it (`get_router().chat(...)` returns a `BaseChatModel`; use `.invoke()`).
- Don't write user-facing docs into the code (long docstrings explaining what the project does). Code comments should explain *why*, not *what*. README/ARCHITECTURE/PLANNER are the user-facing docs.
- Don't commit anything in `data/` — it's gitignored.

## References

- **README.md** — user-facing intro + quickstart.
- **ARCHITECTURE.md** — full system architecture.
- **PLANNER.md** — assumptions, pending work, future requirements.
- **`~/.claude/plans/i-want-to-develop-enchanted-wozniak.md`** — the approved design plan that produced this codebase.
