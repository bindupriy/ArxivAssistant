# PLANNER.md

Living document of what we assumed, what is built, what's intentionally pending, and what's known to be a future improvement. Update this whenever the answer to "wait, why isn't X done?" or "what about Y later?" comes up.

---

## Assumptions

These are the premises the design rests on. If any of them changes, parts of the architecture should be revisited.

| # | Assumption | Where it bites if violated |
|---|---|---|
| A1 | **Multiple users are a product requirement, not a supported feature today.** The current localhost MVP has no login, authorization, or per-user data isolation. | Before admitting multiple users, choose whether papers/topics are private, shared, or both; then scope SQLite records, Qdrant search, BM25, conversations, jobs, and access to PDFs by identity. Concurrent requests alone do not provide multi-user safety. |
| A2 | **Initial workload is modest, but multi-user capacity is unmeasured.** The current design targets a small paper library and light query traffic. | As users and jobs grow, benchmark the in-memory BM25 cache, one-process embedded Qdrant, SQLite write contention, and local-model memory/latency. |
| A3 | **The user is comfortable editing `config.yaml`.** Configuration is the primary UX for changing behavior; CLI surfaces a subset for convenience. | Need a config-management UI / `config set` CLI for less technical use. |
| A4 | **Python 3.10+** on Linux or Windows. The package declares `requires-python = ">=3.10"` and uses modern union syntax with postponed annotations. | Lower versions need conditional type-hint imports; some deps may raise the practical floor over time. |
| A5 | **Network access for discovery and hosted roles.** arXiv fetch and any configured hosted model/MCP servers need connectivity; locally cached Ollama models and local PDFs can work without it. | Offline-first discovery would require more caching and degraded paths. |
| A6 | **Ollama is the default local model host** when a role is set to `provider: ollama`. Expected at `http://localhost:11434` unless `OLLAMA_BASE_URL` overrides. | Other local hosts (LM Studio, vLLM) need a new provider factory. |
| A7 | **arXiv is the authoritative monitored source for now.** Conference proceedings are not directly scraped, and the current `arxiv_fetch.fetch()` does not populate `Paper.venue` even though storage and retrieval support that field. | Conference-only papers are missed; venue-filtered retrieval may be empty until metadata is enriched. |
| A8 | **English-oriented pipeline.** Prompts, section heuristics, and curation criteria are written for English-language papers; the default embedder is `nomic-embed-text`. | Non-English papers may parse or retrieve poorly. |
| A9 | **Per-role LLM mapping is what the user wants** — not a single "smart default" or auto-routing. Cost/quality tuning is manual. | If the user prefers "just pick a model," we'd want a `provider_profile: cheap|balanced|premium` shortcut. |
| A10 | **Local-model memory is limited.** The current Ollama chat role uses `qwen2.5:3b`, embeddings use `nomic-embed-text`, and `concurrency.background_workers` defaults to two. The local cross-encoder reranker is optional and not installed by default. | Running multiple Ollama calls alongside indexing on an 8GB machine may swap or be slow; tune workers/models and benchmark on Raspberry Pi. |
| A11 | **Curation criteria are domain-scoped, not global.** Each domain has its own version chain. | If the user wants global rules ("always exclude surveys"), we'd need a `global` pseudo-domain or a criteria-merge step in the curator. |

---

## Intentionally pending (clear implementation boundaries)

These have named entry points or established extension points so future work is localized.

| Item | Where | Why deferred | Acceptance criteria |
|---|---|---|---|
| **Multi-user authentication and isolation** | `web/app.py`, `storage/schema.py`, `memory/`, `rag/retriever.py`, Qdrant payloads, and `web/` | User identity, private/shared library policy, and an authentication approach have not been specified or implemented. The current request header is not authentication. | Users can sign in; server-side authorization applies to every read, write, upload, PDF, job, and stream; private data cannot cross user boundaries in SQL, BM25, dense/graph retrieval, conversations, or citations; integration tests prove this. Decide the sharing model before choosing schema and deployment changes. |
| **Memory consolidation** | `agents/memory.py:consolidate()` | Nightly job that promotes durable facts from `interactions` → `profile_facts` and biases chunk-ranking from feedback. Not needed until enough interaction data accumulates. | Has an extractor that turns N recent interactions into ≤ K profile-fact rows; retrieval ranker reads `user_feedback` to up/down-rank chunks. |
| **Auto-scheduler in monitor** | `scheduler.py` + `config.monitor.enabled` | Wired but `enabled: false` by default. Manual `assistant monitor tick` works. Keep off until the curation criteria are tuned enough to trust unattended ingestion. | User flips the flag, `assistant monitor run` runs for a week without producing low-quality ingests. |
| **Marker PDF parser** | `rag/parser.py:parse_pdf` | PyMuPDF is "good enough" and avoids a heavy install. Marker handles equations and multi-column layouts much better. | Drop-in replacement returning `ParsedPaper`; same shape, higher fidelity, no other code changes. |
| **Reranker dep in pyproject** | `pyproject.toml` | `sentence-transformers` is heavy; reranker degrades gracefully without it. Install on demand. | Either add as optional extra (`pip install -e .[rerank]`) or just document the manual install — TBD. |
| **MCP tool execution** | `agents/info_gatherer.py` | Servers can be configured and their tools bound, but the current node records requested calls without invoking them or returning results. | Execute allowed calls with bounded retries/timeouts; send actual tool output into QA and test failure and prompt-injection handling. |
| **Venue enrichment** | `rag/arxiv_fetch.py` + ingestion | `Paper.venue` and exact venue filters exist, but current arXiv fetch leaves the venue unset. | Resolve trustworthy venue metadata (or expose explicit entry), persist it on first ingest/re-ingest, and test venue-filtered retrieval. |
| **CLI topic selection on manual ingest** | `cli.py:ingest` | The shared ingest path supports a topic and the web UI exposes it, but the CLI command has no `--domain` flag. | Add the flag and thread it through `scratch`. |
| **CLI multi-turn chat** | `cli.py:_invoke` | Saved, scoped multi-turn conversations exist in the web UI; CLI calls remain one-shot. | Add `assistant chat` over the conversation store. |
| **Feedback-biased retrieval** | `rag/retriever.py` + `memory/episodic.py` | Episodic memory captures `user_feedback` per interaction with cited chunks; retrieval doesn't yet read it. | Hybrid score blends a feedback prior per chunk; an A/B comparison shows uplift on questions whose chunks have prior +/-1 ratings. |
| **Better citation rendering in CLI** | `cli.py:ask` | Citations print as raw chunk IDs. Better: paper title + section. | `assistant ask` resolves chunk IDs to "[<paper title>, §<section>]" via SQLite lookup. |

---

## Tracked requirements

Items explicitly discussed in the design brainstorm. Their status is recorded here so implemented work is not mistaken for backlog.

1. **Local application: built.** FastAPI + React/Vite provides saved fixed-scope chat with automatic LLM-generated names, citation drawers, library/topic/inbox workflows, feedback, PDF/arXiv ingestion, and bounded concurrent background jobs. `assistant serve --build` hosts the API and static app. Placeholder chats are named from the first question after a successful answer; descriptive titles are preserved, and older unnamed chats are handled on their next successful reply rather than through a bulk backfill.
2. **Deployment on a Raspberry Pi** (Pi 5 / 8GB). The current local MVP uses embedded Qdrant, SQLite, and Ollama; performance and safe access for multiple users have not been established. Outstanding work:
   - **Benchmark the current all-local setup**: `qwen2.5:3b` is used for chat roles and `nomic-embed-text` for embeddings; two concurrent background jobs may exceed the useful memory/CPU budget.
   - **Optional hosted-heavy-role profile**: if local QA/curation is too slow, move those roles to a configured API provider while keeping embeddings local (requires API keys and connectivity).
   - **Keep the optional reranker off if too slow**: the current `cross-encoder/ms-marco-MiniLM-L6-v2` only loads when `sentence-transformers` is installed; the default falls back to fused results without it.
   - **Packaging**: Dockerfile + `docker-compose.yml` with Ollama and the assistant; or systemd unit for the scheduler.
   - **Confirm Marker is feasible** on ARM; otherwise stick with PyMuPDF.
3. **Modular MCP configuration: partially built.** A YAML entry is enough to discover/bind an MCP server's tools, but actual tool-call execution and returning tool output are pending (see Intentionally pending). Possible additions after that work:
   - Semantic Scholar MCP — citation graph, influential-citation counts.
   - Zotero MCP — pull user's existing library / push curated papers.
   - Obsidian MCP — write daily notes / paper summaries.
4. **Memory-first behavior: partially built.** The implemented seven-layer model covers working state, episodic interactions, semantic paper knowledge, procedural criteria, saved conversations, curation decisions, and profile facts. Versioned criteria and feedback capture are built; feedback-biased retrieval and consolidation remain pending.
5. **Local-vs-API choice via config.** Built. Per-role mapping is in place; switching any role is a single YAML line.
6. **Vanilla vs agentic RAG via config.** Built. Toggle is `rag.mode`.
7. **Library organization and removal: built.** Papers have manually selected New/Reviewing/Read status, a combinable status filter, an Interesting star, and a Favorites tab. New and existing papers default to New and unstarred; re-ingestion preserves user preferences. Confirmed removal deletes library metadata, chunks, and search vectors while retaining PDFs and saved chats. The paper detail pane no longer displays raw chunks under Sections; chat citation drawers remain available for retained sources.
8. **Selective citation verification: built.** QA produces claim-level evidence, checks exact quotes against retrieved chunks, and runs a separate whole-claim support verifier before emitting paper citations. General explanations and other claims without accepted citations are marked LLM knowledge. Verification failures withhold citations; malformed generation gets one schema-constrained retry before failing closed. This does not guarantee factual accuracy, especially with small local models, and saved answers are not retroactively verified.
9. **Page-aware chunking and citation graph: built.** New PDF ingests keep page-local section chunks and cite the correct physical page. BM25 and dense search were already fused; optional graph retrieval follows explicit arXiv links between indexed papers in the configured scope, reserving a bounded number of answer-context slots. Existing chunks need re-ingestion for pages and graph edges; re-ingestion replaces chunk IDs, so old chat citations can become unavailable. This is not full entity/relationship GraphRAG.
10. **Question rewriting, metadata filters, and extractive compression: built.** Optional first-turn/follow-up rewrite uses the `retrieval` role with safe fallback; explicit year/venue/section filters intersect fixed conversation scope in both BM25 and dense/graph/agentic retrieval and are exposed in web chat/CLI. Bounded source excerpts reduce QA prompt length while full original chunks remain available; paper citations still require a quote visible in the excerpt and the original text plus verifier approval. This is not LLM summarization or automatic extraction of publication constraints from natural language.
11. **Multi-user support: required, not implemented.** The application can process multiple simultaneous requests within one server process, but it has no user accounts, authorization, private data boundaries, or sharing policy. Do not deploy it for distinct users until the pending authentication/isolation work and cross-user leakage tests are complete.
12. **Streamlit UI: built as an optional local client.** A Streamlit interface provides saved scoped chat, citation details, arXiv topic/title/ID search with explicit one-click PDF ingestion, library/PDF upload, topic criteria, inbox actions, and background job status through FastAPI HTTP/NDJSON only. The API remains the sole owner of embedded Qdrant/SQLite and must be running first. Streamlit is not an authentication or multi-user solution. Asking a question does not silently download papers; the user previews and selects what to index.

---

## Good-to-haves (not yet committed)

The list of "yes, eventually" items that didn't make MVP. Treat as a backlog, not a roadmap.

### RAG / quality

- **Citation precision evaluation** with real configured models and a labeled set of supported, partially supported, irrelevant, and contradictory claim/passage pairs. Current unit tests prove verification gating and labeling behavior using mocked verdicts, not empirical citation accuracy.
- **Qdrant native sparse vectors** to replace in-memory BM25 (scales beyond a few thousand chunks).
- **HyDE-style query expansion** for very short questions.
- **Multi-hop citation traversal** beyond the built one-hop, explicit-arXiv citation graph (with careful relevance and scope evaluation).
- **Entity/relationship GraphRAG** beyond the built explicit-arXiv citation graph, with extraction, entity resolution, and evaluation before trusting inferred edges.
- **Automatic filter extraction** from natural-language questions (e.g. “after 2022”), with validation so inferred constraints never silently override explicit year/venue/section selections.
- **Per-section weighting** (e.g. prefer Method/Results over Related Work for technique questions).
- **Better summary generation**: today the paper-level vector comes from the metadata/parsed abstract, falling back to the first chunk or title; a generated summary (via the `summarizer` role) could be richer.

### Ingestion

- **Conference scrapers** beyond arxiv: OpenReview, ACL Anthology, ICML/NeurIPS proceedings pages.
- **Newsletter/RSS sources** (Sebastian Raschka, Lilian Weng, AK on X) — same `candidates` shape, different fetcher.
- **Durable background jobs** that survive a server restart; current web jobs run concurrently within one process but remain in memory.

### UX

- **CLI sessions** (`assistant chat`); saved web conversations and inline feedback are built.
- **`config set` CLI** so config tweaks don't require a YAML editor.
- **Better PDF browsing**: highlight the precise cited passage within the page; current citation drawers open the source PDF at its physical page when available.

### Ops / deployment

- **Dockerfile + compose** including Ollama.
- **Systemd unit** for the monitor scheduler.
- **Backup/export**: a single command that bundles SQLite + Qdrant + PDFs.
- **Versioned migrations**: SQLAlchemy + Alembic for general schema evolution. Current engine initialization uses `create_all` plus idempotent SQLite upgrades for paper reading-status/favorite and chunk-page columns; it is not a general migration system.
- **Tracing**: LangSmith or local OTel for agent step-level visibility.

### Memory

- **Auto-promotion to profile facts** from repeated user behavior (consolidation job).
- **Feedback-biased ranking** (see pending list).
- **Conflict detection across paper claims** — flag chunks that contradict each other for the same question.
- **CLI / granular forget interface**: `assistant forget <chunk-id-or-paper-id>` remains pending. Whole-paper removal is built in the web Library; it does not erase saved chats or source PDFs, and individual-chunk removal is not exposed.

### Multi-agent

- **Replan loop**: an explicit re-plan node when QA confidence stays low even after MCP gathering.
- **Tool-use logging** to a separate `mcp_invocations` table.
- **Per-domain agent customization** (e.g. a domain can override the `qa` system prompt with a domain-specific preamble).

---

## Notes on resolved decisions (don't relitigate)

These came up in the brainstorm and were settled. Recording them here so future sessions don't reopen them without a reason.

- **LangGraph over LangChain agents / CrewAI / AutoGen.** Chosen for stateful graphs, checkpointers, and conditional edges that fit a multi-intent assistant.
- **Qdrant over Chroma / Weaviate.** Embedded mode, hybrid search, metadata filters, RPi-friendly.
- **PyMuPDF first, Marker later.** Lean install for MVP; clear upgrade path.
- **SQLAlchemy + SQLite over plain sqlite3.** Typed models pay off as schema grows; Alembic when migrations matter.
- **Typer + Rich over Click / argparse.** Better DX, less boilerplate.
- **MCP via `langchain-mcp-adapters`.** Lets MCP tools be bound to LangChain chat models with `bind_tools`; tool invocation and result handling remain pending.
- **Per-role LLM mapping over single-global or two-tier.** User picked it explicitly.
- **LLM-judge curation with versioned criteria over embedding-similarity-to-seed-set.** User picked it explicitly.
- **CLI plus a localhost FastAPI/React MVP.** The current web implementation is one-process and has no authentication; this is an implementation limit, not a decision to exclude future multi-user support.
- **Persistent chat beside the workspace.** Library, Topics, and Inbox open to the left of chat with the majority of desktop content width and independent scrolling. Smaller screens stack the panes. Chat remains visible, and starting or selecting a conversation does not dismiss the workspace.
- **Fixed conversation scope at creation.** New chat and history sit below the page tabs in the main sidebar. New chat opens a draft, and the composer selector chooses library, topic, or paper scope before the first message creates the conversation. Existing chats display their fixed scope with disabled controls. Scope limits library retrieval, not the model's general knowledge or MCP results.
- **Manual reading preferences.** Reading status and Interesting stars are independent of ingestion status, do not change automatically, and survive re-ingestion. Favorites and status filters organize the Library view, not the chat's retrieval scope.
- **Library removal is not a full memory purge.** Source PDFs and saved chats are retained. Citations to deleted chunks become unavailable rather than pointing to another source.
- **Citation relevance over citation volume.** Do not require paper references for every claim. Paper citations must pass quote and support checks; uncited claims are explicitly labeled LLM knowledge, which is an attribution boundary rather than an assurance of truth.
- **Multi-user is in scope.** Its identity and library-sharing policy remain to be decided; request concurrency is already built, but access control and tenant isolation are not.

---

## How to use this file

- When a question comes up like "should we add X now?" — check Good-to-haves first.
- When something looks half-done or stubbed — check Intentionally pending; it's probably deliberate.
- When you're about to assume something — check Assumptions and either confirm or update.
- When a design decision is made — add a row to "Resolved decisions" so it's not relitigated.
