# Architecture

This document describes how the AI Research Assistant is structured. For a user-facing overview, see [README.md](README.md). For open questions and pending work, see [PLANNER.md](PLANNER.md).

## High-level view

```
┌───────────────────────────────────────────────────────────────┐
│     CLI (Typer) / FastAPI + React / Streamlit API client      │
└───────────────────────────────┬───────────────────────────────┘
                                │
                ┌───────────────▼────────────────┐
                │     LangGraph Orchestrator     │
                │  (state, checkpointer, router) │
                └──┬──────┬──────┬──────┬────────┘
                   │      │      │      │
        ┌──────────▼┐ ┌───▼──┐ ┌─▼────┐ ┌▼──────────┐
        │ QA / RAG  │ │Curator│ │Info │ │ Criteria  │
        │  Agent    │ │Agent  │ │Gath.│ │ Manager   │
        └─────┬─────┘ └───┬───┘ └──┬──┘ └─────┬─────┘
              │           │        │           │
   ┌──────────▼───────────▼────────▼───────────▼─────────┐
   │            Shared Services Layer                     │
   │  LLMRouter | MCPClient | Embedder | Reranker        │
   └──┬───────────┬────────────┬──────────────┬──────────┘
      │           │            │              │
  ┌───▼───┐   ┌───▼───┐    ┌───▼────┐    ┌────▼─────┐
  │Qdrant │   │SQLite │    │ Files  │    │  MCP     │
  │(vecDB)│   │(meta+ │    │(PDFs)  │    │ Servers  │
  │       │   │memory)│    │        │    │(arxiv,…) │
  └───────┘   └───────┘    └────────┘    └──────────┘
```

## Design principles

- **Config-driven.** Provider/model per agent role, RAG mode, MCP servers, storage paths — all in `config.yaml`. No code change should be required to swap any LLM, switch RAG mode, or add an MCP tool.
- **Multi-user target, single-user implementation.** The system currently has no identity or tenant-scoped data model. Concurrent requests in one process are not equivalent to safe multi-user access; [PLANNER.md](PLANNER.md) tracks authentication, sharing decisions, and isolation as required work.
- **Single shared state.** A LangGraph `GraphState` TypedDict flows through every node. Each agent reads what it needs and writes the fields it owns — no other agent reaches into private state.
- **Selective, verified citations.** Numbered references require an exact supporting quote and a positive whole-claim verification verdict. Claims without accepted references are labeled `LLM knowledge`. State and storage retain stable `chunk_id` citations; model-based verification improves grounding but does not guarantee correctness.
- **Versioned, never overwritten.** Curation criteria are append-only — every user comment becomes a new version, and the prior version remains queryable. Same idea will extend to profile facts later.
- **Stubs are real nodes.** Every agent in the design lives in the LangGraph from day 1, even if it returns trivially. This keeps the topology stable and makes "deepening" any agent a localized change.

## The multi-agent layer

All agents are LangGraph nodes — callables taking `GraphState` and returning a partial state update.

| Node | File | Role |
|---|---|---|
| `orchestrator` | `agents/orchestrator.py` | Reads `state.intent`; routes to the right branch via a conditional edge. |
| `qa` | `agents/qa.py` | Top-level QA. Retrieves chunks, generates structured claims and supporting quotes, verifies candidate evidence, and renders citations or LLM-knowledge labels. Detours through `info_gatherer` once when confidence is low. |
| `retrieval` | `agents/retrieval.py` | Resolves paper/topic and per-question metadata scope, optionally rewrites first-turn/follow-up search queries, and picks vanilla vs agentic RAG. |
| `curator` | `agents/curator.py` | LLM-as-judge over candidate abstracts. Loads criteria per candidate topic and preserves every decision. |
| `ingestion` | `agents/ingestion.py` | Calls `rag.ingest.ingest_source()` for every source in state (single from CLI, list from curator output). |
| `info_gatherer` | `agents/info_gatherer.py` | Binds configured MCP tools to an LLM on a low-confidence QA detour. Currently records requested tool names/arguments or a model summary; it does not execute returned tool calls. |
| `criteria` | `agents/criteria.py` | Takes a user comment + active criteria, asks the `criteria` LLM for an updated structured_rules + nl_addendum, appends a new version. |
| `memory` | `agents/memory.py` | Writes Q&A interactions and final curation decision statuses; consolidation is stubbed. |
| `monitor` | `agents/monitor.py` | Polls arxiv per domain and skips papers already ingested or judged. |

### Graph topology

```
START -> orchestrator -> {
    ask              -> qa
                          ├─if chunks missing─> retrieval_node inline
                          │                     └─vanilla hybrid or agentic loop
                          ├─if low confidence─> info_gatherer -> qa (once)
                          └─otherwise─────────> memory -> END
    ingest           -> ingestion -> memory -> END
    criteria_update  -> criteria  -> memory -> END
    monitor_tick     -> monitor -> curator -> ingestion -> memory -> END
}
```

Wired in `graph.py`. Conditional edges live in `orchestrator.route` (intent dispatch) and `_route_after_qa` (low-confidence detour). The `retrieval` callable is also registered as a public graph node, but the current ask path enters `qa`, which invokes retrieval when state has no chunks.

## Shared services

### `LLMRouter` (`llm/router.py`)

The only place agents touch the model layer. Three methods:

- `chat(role)` — returns a `BaseChatModel` for a chat-style role.
- `embeddings(role="embedder")` — returns an `Embeddings` object.
- `reranker(role="reranker")` — returns a local CrossEncoder (lazy import) or OpenRouter rerank client.

Adding a provider = a factory in `llm/providers.py` + a branch in the router. Adding a role = a YAML entry under `llm.roles`.

### `MCPClient` (`mcp/client.py`)

Builds a `MultiServerMCPClient` from `config.mcp.servers`. `get_tools_sync()` returns LangChain `BaseTool` objects bindable to any LLM. Degrades gracefully when the package or servers are missing — agents must check for empty. Configuring a server does not yet make tool **results** available to QA; the Info Gatherer has no call-execution loop.

### `Embedder` (`rag/embedder.py`)

Thin wrapper over `LLMRouter.embeddings("embedder")`. Used by ingestion (chunk + paper-summary vectors) and retrieval (query vector).

### Reranker (`rag/reranker.py`)

Optional layer between hybrid retrieval and the QA prompt. Supports a local cross-encoder via `sentence-transformers` or OpenRouter's rerank API. If unavailable, returns input order with a logged warning.

## RAG flow

The question-answering path keeps the search query and answer question separate:

```
original question + fixed scope + per-turn filters
    -> optional rewrite (search query only)
    -> intersect paper/topic with year/venue in SQLite
    -> BM25 + Qdrant dense search (same section filter) -> weighted RRF
    -> optional citation-graph expansion / agentic loop / rerank
    -> retrieved chunks (original text retained)
    -> optional extractive context windows -> QA with original question
    -> quote in displayed window AND original chunk -> whole-claim verifier
    -> verified [n] chunk IDs, otherwise LLM knowledge
```

### Vanilla (default)

1. BM25 over SQLite chunks (`rank-bm25`, in-memory index, cached and invalidated on ingest or paper removal) → top-`candidate_k`; an empty index returns no results.
2. `Embedder.embed_query(search_question)` → dense Qdrant search → top-`candidate_k` chunks with text in payload.
3. Weighted Reciprocal Rank Fusion (`k=60`) merges the two lists using `rag.hybrid.dense_weight` and `sparse_weight`.
4. When available, the optional reranker scores `(search_question, chunk_text)` pairs and returns top-`config.rag.top_k`; otherwise the fused order is kept. `rag.graph.enabled` selects the graph expansion path described below, which may return fused top-k directly if no citation neighbors exist.

With `rag.graph.enabled`, `rag/graph_retriever.py` first finds hybrid seed passages,
then follows one hop of explicit arXiv citations in either direction between *ingested*
papers (`paper_citations` in SQLite). It retrieves relevant passages from up to
`graph.max_neighbors` linked papers, reserving at most `graph.graph_slots` of the
top-k candidate slots; final optional reranking still applies. Scope filtering
also applies to graph neighbors. There is no entity/relationship extraction or
global knowledge-graph summarization; absent links or unindexed older papers
fall back to the base hybrid results.

### Agentic (`config.rag.mode: agentic`)

1. Decompose: `retrieval` LLM splits the question into 2–4 sub-queries.
2. For each sub-query, run hybrid retrieval (with citation-graph expansion when enabled) and accumulate results.
3. Critique: LLM inspects accumulated chunks vs the *original* question; returns "gaps" (additional sub-queries) or empty.
4. If gaps and `iter < max_iters`, loop with the new sub-queries.
5. Final reranker pass against the original question → top-`top_k`.

Both paths accept the same paper-ID scope. Topic scope resolves to paper IDs in SQLite, and sparse results are filtered as strictly as dense results. `qa_node` labels retrieved chunks `[1]..[n]` for the model and includes recent saved turns, but only accepted evidence becomes a citation in the final answer.

`retrieval_node` optionally rewrites first-turn or follow-up questions using the `retrieval` role, falling back to the original on malformed or failed output and rejecting rewritten queries that drop explicit arXiv IDs or quoted phrases. The QA answer still uses the original question. Per-question `MetadataFilters` intersects any fixed paper/topic scope with SQLite paper year/venue constraints; section types are applied to **both** dense Qdrant and sparse BM25 candidates and are forwarded through agentic and citation-graph retrieval. Venue comparison is case-insensitive exact match; missing metadata does not match a requested filter. Web messages pass an optional `filters` object independently per turn (validated before streaming), stored in `scratch.metadata_filters`; the CLI exposes equivalent `--min-year`, `--max-year`, `--venue`, repeatable `--section`, `--domain`, and `--paper` options. Question filters do not change the conversation's fixed scope or limit uncited model knowledge.

When enabled, `rag/compress.py` extracts bounded contiguous windows from each retrieved chunk for the QA prompt; it does not summarize or change stored text. The full text stays in `retrieved_chunks`, and the verifier requires exact quote occurrence in both the displayed window and the original passage before running whole-claim support verification. This avoids treating invisible or invented evidence as cited.

### Claim attribution

1. The `qa` model returns a validated JSON draft: `claims` with single-line Markdown `text` and `evidence` entries (`source`, `quote`), plus `confidence`. It may use general knowledge with an empty evidence list; no chunks is not an automatic refusal. The prompt forbids invented paper-specific findings.
2. Candidate evidence must name an existing retrieved source and contain an exact quote found in both the displayed context (when compressed) and the original chunk after whitespace normalization. Nonexistent sources and fabricated quotations are discarded before verification.
3. One separate chat call batches remaining claim/quote/passage candidates. The `citation_verifier` role (or `qa` if absent) judges whether each quoted passage supports the entire associated claim, rejecting mere topic overlap, partial support, and unsupported qualifications or numbers. Malformed, incomplete, duplicate, or failed verification responses withhold citations.
4. Application code appends numbered markers only for accepted evidence, deduplicates chunk IDs in first-use order, and labels claims without accepted evidence `LLM knowledge`. Draft citation links are rejected and raw numeric markers are stripped. The public state and web response remain `answer`, `citations`, `confidence`, and `retrieved_chunks`.
5. Rejected proposed evidence lowers confidence for the existing low-confidence detour. Intentionally uncited general knowledge alone does not lower confidence. An invalid draft gets one schema-constrained generation retry; failure produces a message without citations instead of bypassing verification.

The verifier is part of the QA implementation, not an additional LangGraph node. It does not verify the truth of uncited knowledge, does not turn external tool summaries or prior-turn citations into paper evidence, and does not revisit saved answers. The exact-quote check is deterministic; semantic support remains an LLM judgment and needs real-model precision evaluation. `tests/test_qa.py` covers the verification and rendering contract with mocked model responses.

## Memory layers

Seven distinct layers, each with a clear store:

| Layer | Store | What it holds |
|---|---|---|
| **Working** | LangGraph state + SqliteSaver checkpointer | One run's intermediate state; resumable per `thread_id`. |
| **Episodic** | `interactions` table | Every Q&A turn: question, answer, cited chunk IDs, RAG mode, confidence, feedback. |
| **Semantic (paper KB)** | Qdrant `chunks` + `papers` + SQLite `papers`/`chunks`/`paper_citations` | Chunks for retrieval, abstract vectors for papers, explicit arXiv links for one-hop expansion. |
| **Procedural (criteria)** | `criteria_versions` | Versioned per domain. Active = `is_active=True` + max version. |
| **Conversation** | `conversations` + scoped `interactions` | Fixed-scope saved chats; interaction `extra` links each turn. |
| **Curation inbox** | `curation_decisions` | Every judge result and its ingest/dismiss status. |
| **User profile** | `profile_facts` | Manual or consolidated; consumed by future ranking biases. |

Episodic writes happen in `memory_node` for `ask` intent. Feedback (`+1/0/-1`) is captured per interaction; bias on retrieval scoring is a future hook.

## Data model

SQLAlchemy declarative models in `storage/schema.py`:

- **Domain**: `name`, `arxiv_categories`, `venues`, `seed_papers`.
- **Paper**: `id` (arXiv ID or local PDF filename stem), title, authors, abstract, venue, year, pdf_path, summary, `domain_id`, `accepted_score`, ingestion `status`, user-controlled `reading_status` (`new`, `reviewing`, `read`), `is_favorite`, extra JSON.
- **Chunk**: `id` (UUID matching Qdrant point), `paper_id`, `section_type` (abstract/intro/method/...), `section_title`, `order`, `text`, `char_start/end`, nullable 1-based `page_number`.
- **PaperCitation**: explicit arXiv reference from a source paper to a target arXiv ID (the target may not be ingested). Source edges are replaced on re-ingestion and cascade on paper removal.
- **CriteriaVersion**: `domain_id`, `version` (per-domain monotonic), `structured_rules` JSON, `nl_addendum`, `source_comment`, `is_active`.
- **Interaction**: `question`, `answer`, `cited_chunk_ids`, `rag_mode`, `confidence`, `user_feedback`, extra JSON.
- **Conversation**: UUID, title, fixed scope type/target, timestamps.
- **CurationDecision**: topic + base arXiv ID, paper metadata, score, reason, status, error.
- **ProfileFact**: `key`, `value`, `source` (manual/consolidated).

Reading status defaults to `new` and favorites to `False`. These preferences are independent of ingestion status and remain unchanged during re-ingestion. On first database access, `storage/sqlite_store.py:get_engine()` creates missing tables and adds missing preference and nullable chunk-page columns with idempotent SQLite ALTER statements. Existing chunks have no page number or extracted citation-graph edges until re-ingested. Re-ingestion generates new chunk IDs, so old saved-answer citations may become unavailable. There is no general versioned migration framework yet.

Qdrant collections:

- `papers` — one point per paper; payload = `{title, authors, year, arxiv_id}`; vector = embedding of metadata abstract, parsed abstract, first chunk, or title (in that order of availability). Generated summaries are not used yet.
- `chunks` — one point per chunk; payload = `{paper_id, section_type, section_title, order, text, page_number}`; vector = embedding of chunk text.

Vector dimensions are inferred from the first upsert (so the embedder can change without migrations as long as collections are empty or recreated).

## Configuration

`config.yaml` is loaded into typed Pydantic models in `config.py`:

- `AppConfig.llm.roles: dict[str, RoleSpec]` — per-role provider/model.
- `AppConfig.rag: RAGConfig` — mode, top_k, history_turns, low_confidence_threshold, hybrid weights, agentic max_iters, citation-graph, query rewrite, and compression options.
- `AppConfig.domains: list[DomainConfig]` — name + arxiv categories + venues + seed papers.
- `AppConfig.mcp.servers: list[MCPServerSpec]` — name, transport, command/url.
- `AppConfig.storage: StorageConfig` — paths for Qdrant, SQLite, PDFs.
- `AppConfig.curation: CurationConfig` — accept threshold.
- `AppConfig.monitor: MonitorConfig` — enabled flag + interval.
- `AppConfig.concurrency: ConcurrencyConfig` — bounded background-job worker count (two by default).

`Settings` (Pydantic-settings) pulls API keys, the Ollama base URL, and optional OpenRouter endpoint/attribution settings from `.env`.

## Ingestion pipeline

`rag/ingest.py:ingest_source(source, domain=None, accepted_score=None)`:

1. **Resolve source.** arxiv ID → `arxiv_fetch.fetch()` downloads PDF + metadata. PDF path → use directly.
2. **Parse.** `parser.parse_pdf()` uses PyMuPDF to extract per-page text and offsets, then heuristically detects headings to produce `Section` objects with `section_type` (abstract / intro / method / experiments / results / conclusion / etc.).
3. **Chunk.** `chunker.chunk_paper()` splits first at physical page and section boundaries, then uses up to 900 characters per chunk with 135 characters of within-page overlap. Skips `references` and `acknowledgments`.
4. **Embed.** Chunk vectors in a batch; paper-level vector from abstract (or title fallback).
5. **Persist.** SQLite: upsert `Paper` + clear/rewrite `Chunk` and explicit arXiv reference edges for idempotency, preserving an existing paper's reading status and favorite flag. Qdrant: upsert into `chunks` and `papers` collections (collections auto-created on first insert with the observed vector size).
6. **Invalidate BM25 cache** so newly-ingested chunks are searchable in the same process.

## MCP integration

`MultiServerMCPClient` from `langchain-mcp-adapters` is built from the config's server list. The Info Gatherer node binds its tools to the `info_gatherer` LLM via `bind_tools()` and lets the model decide which to call.

The QA detour: when `qa_node` returns `confidence < rag.low_confidence_threshold` AND `get_mcp_client() is not None`, the conditional edge routes through `info_gatherer`, which writes `mcp_results` + sets `did_gather=True`. The edge `info_gatherer → qa` then re-enters QA, which augments its context with the model's summary or recorded tool-call arguments. **Requested MCP calls are not executed yet**, so their arguments are not verified external findings. `did_gather` prevents infinite loops.

## Scheduling

`scheduler.py:MonitorScheduler` wraps APScheduler's `BackgroundScheduler`. Each job invokes the monitor graph with a fresh checkpoint `thread_id`. Disabled by default — set `monitor.enabled: true` and `monitor.interval_minutes: N` to use it. `assistant monitor run` blocks and ticks on the configured interval.

## Local web layer

`assistant/web/app.py` exposes the library, topics, inbox, jobs, conversations, feedback, PDF, and streaming chat routes. `runner.py` gives each graph request a fresh checkpointer thread ID, streams multiple chats independently as NDJSON, and runs ingest/monitor jobs on a bounded `concurrency.background_workers` pool (two by default). SQLite uses WAL and a busy timeout; a read/write index lock allows parallel searches while serializing SQLite/Qdrant/BM25 index updates. Same-paper ingestion/removal, same-topic monitor ticks, and criteria updates are keyed separately. Keep one server process: the locks and embedded Qdrant are process-local. FastAPI serves the Vite build from `assistant/web/static`; frontend source lives in `web/`.

The optional Streamlit app (`assistant/streamlit_app.py`) is another **frontend**, not a graph runner. Its loopback-only HTTP client (`assistant/web/api_client.py`) uses the FastAPI routes and required write header, reads NDJSON through `final` for the optional conversation-title update, and resolves source details by citation chunk ID. Chat and Library use `GET /api/arxiv/search?q=…` to preview bounded arXiv title/topic/ID results and annotate already-indexed papers; `POST /api/papers/arxiv` explicitly queues PDF download/embedding after a user click. Search itself never ingests. Streamlit can start as a separate process because it never opens SQLite or embedded Qdrant directly. The optional `streamlit` package extra is not needed for CLI, API, or React. The Streamlit process must also stay bound to localhost; it does not add accounts or user isolation.

The React app presents a persistent Chat surface and three routed workspace panels:

- **Chat** creates, loads, and deletes fixed-scope conversations; automatically names placeholder chats after a successful answer; renders Markdown answers and clickable numbered citations; and records feedback.
- **Library** combines title/ID search, topic and reading-status filters, and All papers/Favorites tabs; queues arXiv/PDF ingestion; changes topic, reading status, and Interesting stars; removes papers with confirmation; and starts paper-scoped chats. The detail pane shows metadata and actions, not raw chunks or a Sections list.
- **Topics** merges config and database topic state, syncs config, starts topic-scoped chats, filters the library, runs monitors, and refines criteria.
- **Inbox** filters curator decisions and supports ingest, dismiss, and disagree/criteria-update actions.

Chat uses `application/x-ndjson`: an initial conversation event, custom progress events (`retrieving`, `answering`, `gathering`), then a final result enriched with citation details and the persisted interaction ID. An optional second `conversation` event follows the answer when automatic naming succeeds; the frontend continues reading until the stream closes. Conversations do not reuse LangGraph checkpoint state; earlier turns are loaded from SQLite into `scratch.history` for each fresh invocation. Streams for different conversations run concurrently; requests for the same conversation are ordered and reload history after the previous response finishes.

`memory/conversations.py:generate_conversation_title()` names only rows still titled `New conversation`. It uses the earliest saved user question (or the current question if history is empty), invokes the `conversation_title` role with `qa` as a backward-compatible fallback, and stores a single-line title of at most 80 characters. The model call happens outside the transaction. A conditional update preserves a competing title change or deletion. Naming runs after the answer has been yielded, does not add a graph node, and never changes conversation scope. Failures retain the placeholder for retry on a later successful turn; existing descriptive titles are not regenerated. `tests/test_conversations.py` covers persistence, history, naming failures, concurrency, and streamed event order with mocked models.

`Shell` keeps `ChatPage` mounted and visible in a persistent conversation pane. Library, Topics, and Inbox route into a separate workspace pane to its left, using a roughly 60/40 desktop split with a minimum chat width and independent scrolling. At narrow widths the panes stack while keeping the composer on-screen. Workspace drawers and dialogs stay within the workspace. A close button or Chat navigation restores chat-only mode; new-chat, history, and paper/topic chat actions retain the active workspace and update conversation selection separately.

A portal places New chat and history in the main sidebar below a separator after the page tabs. Browser-local sessions retain drafts, turns, and progress by conversation ID, and each stream updates only its originating session. Workspace changes refresh conversation and scope-picker lists without replacing locally cached turns with incomplete saved history. This is browser-session state, not a durable stream-resumption mechanism across full page reloads.

New chat starts an unsaved draft. Scope controls inside the message composer select library, topic, or paper scope; sending the first message creates the saved conversation with that scope. For an existing chat, the same controls are disabled and show its saved scope. Paper/topic chat actions elsewhere can still create scoped conversations directly. Library scope permits retrieval across all papers, topic scope resolves the topic's current paper IDs on each invocation, and paper scope selects one paper ID. Reading-status and Favorites filters only affect Library browsing. Scope constrains library retrieval, not the model's general knowledge or MCP tool results.

Citation `[n]` resolves to `turn.citations[n - 1]`, then to the matching `chunk_id` in `sources`. The API omits deleted chunks from source details, so indexing `sources` by citation number would misattribute later citations. The UI disables unavailable citations while preserving saved answer text and access to remaining sources.

Write routes require `X-Requested-With: assistant-ui`, and `TrustedHostMiddleware` accepts localhost only. Neither provides authentication or prevents one user from reading another's data; the current API must not be exposed as a multi-user service. Embedded Qdrant calls are guarded by a process lock, and `assistant serve` always runs one Uvicorn worker. Multi-user design must address identity, authorization, storage/retrieval isolation, and whether papers/topics may be shared, separately from request concurrency.

### Paper library API

These routes operate directly on storage; they do not invoke the agent graph:

- `GET /api/papers` combines `q`, `domain`, `reading_status`, and `is_favorite` filters. List and detail responses include reading status and the favorite flag; detail responses no longer include a `sections` payload.
- `PATCH /api/papers/{paper_id}` updates only supplied fields: `domain`, `reading_status`, and/or `is_favorite`. Omitted fields are preserved; an explicit null domain clears the topic. Invalid reading statuses or null preference values are rejected.
- `DELETE /api/papers/{paper_id}` acquires the same-paper and search-index write locks, removes chunk and paper vectors under the embedded Qdrant process lock using `delete_paper_vectors()`, then deletes the SQLite paper with cascading chunk and outgoing graph-edge deletion. After the database commit, it invalidates BM25 and returns 204. The separate `delete_by_paper()` helper only removes chunk vectors for re-ingestion. Source PDFs, saved conversations, interactions, and curation decisions are retained.

`tests/test_library.py` exercises defaults, partial updates, combined filters, validation and write protection, deletion from both search indexes, failure handling, idempotent database upgrades, and preservation of preferences on re-ingestion and PDFs/chats on removal. Tests use temporary SQLite databases and in-memory Qdrant without model calls.

## Checkpointer

LangGraph is compiled with `SqliteSaver` (from `langgraph-checkpoint-sqlite`, falling back to `MemorySaver`). CLI and web requests generate a fresh graph `thread_id`; web multi-turn context is loaded from persisted interactions and passed explicitly, preventing stale retrieved chunks from leaking between turns.

## Extension points (where to plug things in)

- **New LLM provider** → factory in `llm/providers.py`, branch in `llm/router.py`.
- **New agent** → file in `agents/`, node + edges in `graph.py`, state fields in `state.py`.
- **New MCP server** → entry under `mcp.servers` in `config.yaml`. No code.
- **New ingestion source** → `is_arxiv_id`-style branch in `rag/ingest.py:ingest_source()`.
- **Better PDF parser** → replacement for `rag/parser.py:parse_pdf()` returning `ParsedPaper`; preserve `page_ranges` and section text offsets to retain page-accurate chunks and PDF links.
- **New retrieval strategy** → function in `rag/`, branch in `agents/retrieval.py`.
- **New CLI command** → `@app.command` in `cli.py` that builds a state dict and calls `_invoke()`.
- **New API route** → endpoint in `assistant/web/app.py`; write routes inherit the required request-header middleware.
- **New web workflow** → API types/client in `web/src/api.ts`, routed UI in `web/src/App.tsx`, then `npm run lint && npm run build`.
- **New Streamlit workflow** → UI in `assistant/streamlit_app.py`, HTTP wrapper in `assistant/web/api_client.py`; never import the graph or open embedded stores from the Streamlit process.
