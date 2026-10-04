# AI Research Assistant

A research assistant that **tracks named research domains** (e.g. speech synthesis, RAG, attention mechanisms), curates relevant arXiv papers into a local knowledge base, and answers questions over that knowledge base with citations. Conference venues can be recorded as paper metadata; conference proceedings are not scraped separately. The current localhost app is a single-user MVP; **multiple users are a requirement but accounts and data isolation are not implemented yet**.

Built as a config-driven multi-agent system on top of **LangGraph + Qdrant + SQLite + MCP**. Any LLM role can be backed by Anthropic, OpenAI, OpenRouter, or local Ollama — the choice is one YAML edit per role.

## What it does

- **Track domains** you care about. Each domain has its own arxiv categories, target venues, and (importantly) its own versioned curation criteria.
- **Curate papers.** A manual or optionally scheduled monitor polls arXiv, an LLM judge scores each abstract against the active topic criteria, and accepted papers are ingested while every decision remains reviewable. Scheduling is off by default.
- **Guide the curator with plain English.** Type `assistant criteria add rag "skip survey papers"` and a Criteria Manager turns that comment into a new versioned rule set the judge will use going forward.
- **Answer questions** with hybrid retrieval (BM25 + dense), optional cross-encoder reranking, and an agentic loop (decompose → iterate → critique → synthesize) toggleable in config.
- **Refine searches.** Optional first-turn/follow-up query rewriting expands search terms without changing the question used for the answer. Per-question year, venue, and section filters apply to both BM25 and dense retrieval (and citation-graph expansion).
- **Compress answer context safely.** The answer prompt sees bounded, extractive passage windows; full original chunks remain available for source drawers and citation verification. A quote must occur in the displayed window and the original chunk before the verifier may accept it.
- **Keep PDF page provenance.** New ingests split at physical page and section boundaries; citation drawers show page numbers and open PDFs at the cited page. Previously indexed papers need re-ingestion to gain page numbers and citation-graph links.
- **Explore citation links.** Optional graph expansion follows explicit arXiv references between papers already in your library after BM25 + embedding retrieval. This is a citation graph, not an entity-extraction knowledge graph; it cannot find un-ingested papers or links without explicit IDs.
- **Configure MCP integrations.** MCP servers listed in `config.yaml` expose tools to the Info Gatherer. The current node binds tools and records model-requested calls but does **not yet execute those calls or return their results**; `mcp.servers` is empty by default.
- **Remember.** Every Q&A turn is logged to episodic memory with citations; criteria are versioned; the user profile is a structured store that the consolidation job (stub) will populate over time.
- **Work in a local web UI.** Chat with paper/topic scope, browse the library, manage topics, review curator decisions, upload PDFs, and leave answer feedback from one localhost-only app.
- **Use an optional Streamlit UI.** A second, Python-only interface provides chat, citations, arXiv title/topic search with one-click PDF indexing, library management, topics, inbox, and job tracking through the same local FastAPI server. It does not open a second Qdrant client.
- **Handle concurrent requests.** Multiple chats can stream independently; `concurrency.background_workers` controls parallel ingest/monitor jobs (two by default). Search-index writes are coordinated, but embedded Qdrant still requires one server process.

## Quickstart

```bash
# 1. Setup (Linux/macOS, from the repository root)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
# API keys are needed only if you switch roles to hosted providers.

# 2. Start Ollama separately if it is not already running: ollama serve
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

# 3. Initialize storage
assistant init                  # creates ./data, SQLite schema, etc.

# 4. Configure a domain in config.yaml under `domains:`, then:
assistant domain sync           # write config-yaml domains into the DB

# 5. Try it out
assistant ingest 2401.05566     # ingest a paper directly
assistant ask "what method does the paper propose?"
assistant monitor tick          # one curation cycle: fetch -> judge -> ingest
assistant criteria add rag "prioritize papers with reproducible code"
assistant criteria show rag     # see the new versioned criteria

# 6. Start the local web UI (Node.js is required for the first build)
assistant serve --build         # opens http://127.0.0.1:8000
```

On Windows, activate the virtual environment with `.\.venv\Scripts\Activate.ps1` instead. The default [config.yaml](config.yaml) uses Ollama for chat and embeddings, so no API-key file is needed for this quickstart. If `python3 -m venv` reports that `ensurepip` is missing on Debian/Ubuntu, install the matching `python3-venv` system package first. Ollama must be running before ingestion or Q&A. The optional local reranker requires `sentence-transformers`; without it, retrieval keeps the BM25+dense fused order.

## CLI

| Command | What it does |
|---|---|
| `assistant init` | Create data directories + SQLite schema |
| `assistant status` | Row counts for the core domain, paper, chunk, criteria, and interaction tables |
| `assistant ask "<question>"` | Run QA; supports `--min-year`, `--max-year`, `--venue`, repeatable `--section`, `--domain`, or `--paper` retrieval filters |
| `assistant ingest <arxiv-id\|pdf-path>` | Manually ingest a paper (bypasses curator) |
| `assistant domain add <name> --categories cs.CL,cs.IR --venues ACL,EMNLP` | Track a new domain (DB direct) |
| `assistant domain list` | List tracked domains |
| `assistant domain sync` | Pull domains from `config.yaml` into the DB |
| `assistant criteria add <domain> "<comment>"` | Append a new criteria version from a comment |
| `assistant criteria show <domain>` | Print active criteria for a domain |
| `assistant monitor tick [--domain <name>]` | One curation cycle |
| `assistant monitor run` | Start the scheduler (blocks; requires `monitor.enabled: true`) |
| `assistant config show` | Dump active config |
| `assistant config roles` | Per-role LLM table |
| `assistant serve [--build] [--no-open]` | Build and/or run the local FastAPI + React app |

For example, `assistant ask "How does retrieval work?" --domain rag --min-year 2021 --section method` searches only Method chunks from papers dated 2021 or later in that topic. `--domain` and `--paper` cannot be combined. Venue matching is case-insensitive and exact; missing year/venue metadata does not satisfy a requested filter. The current arXiv fetch does not populate `Paper.venue`, so a venue filter may return nothing until another ingestion path provides venue metadata. These filters constrain paper retrieval, not the model's general knowledge.

## Web UI

`assistant serve` runs one Uvicorn process on `127.0.0.1:8000`. Use `--build` after frontend changes; it installs `web/` dependencies when needed and builds into `assistant/web/static/`. The app provides:

- fixed-scope, persisted conversations with numbered inline citations, source drawers, deletion, and feedback;
- a searchable paper library with arXiv/PDF ingestion, topic assignment, manual New/Reviewing/Read status, status filtering, starred Favorites, removal, and paper-scoped chat;
- topic status, topic-scoped chat, paper filtering, criteria refinement, and manual monitor runs;
- a curation inbox for accepted, rejected, failed, ingested, and dismissed decisions.

Chats created as "New conversation" receive a short LLM-generated name after their first successful answer. The name is based on the first user message, saved to SQLite, and streamed to the sidebar and chat header. Existing descriptive names are preserved; older unnamed chats are named on their next successful reply. Naming failures leave the answer intact and can retry on a later turn. The `conversation_title` role selects the naming model, with `qa` as a fallback for older configs.

New chat and conversation history live in the main sidebar below the page tabs. Select Whole library, One topic, or One paper inside the message composer before sending the first message, which creates the conversation. Existing conversations display their fixed scope in the same controls; start a new chat to choose a different scope. Different conversations can stream at once; messages to the same conversation run in order so each follow-up sees its predecessor.

The composer also has optional **Search filters** for publication-year range, exact venue, and section. These apply only to the current question, within the conversation's fixed paper/topic/library scope. Unknown publication years or venues do not match those filters. API clients can pass the same fields as `filters` in a message request.

Library, Topics, and Inbox open beside the always-visible conversation. On desktop, the workspace gets roughly 60% of the available content width and chat gets the rest, with independent scrolling. Narrow screens stack the workspace above chat while keeping the composer visible. Close the workspace pane or select Chat for a full-width conversation. Starting a chat, selecting history, or chatting about a paper/topic keeps the workspace open. Drafts and pending responses remain attached to their original conversation; a full page reload still does not resume an in-flight response, though completed turns remain saved.

The API accepts write requests only with the UI's `X-Requested-With` header and rejects foreign Host headers. These checks are **not authentication**. The current app is intended for one trusted user on localhost; do not expose it to multiple users or a network until authentication, authorization, and a private/shared library policy are implemented (see [PLANNER.md](PLANNER.md)). Concurrent chats and background jobs do not provide user isolation.

### Optional Streamlit interface

Keep FastAPI running on `127.0.0.1:8000` (`assistant serve --no-open` or the **Run FastAPI API** VS Code task), then start a second terminal:

```bash
python -m pip install -e '.[streamlit]'
python -m streamlit run assistant/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Open http://127.0.0.1:8501. Alternatively, run the **Run Streamlit UI** VS Code task after installing the optional dependency. In **Chat** or **Library**, expand *Find and add a paper from arXiv*, search by title/topic or paste an ID, preview the results, and select *Download PDF and index paper*. When the job finishes, use *Start paper chat* to ask grounded questions. Searching only previews metadata; downloading and embedding start after you click the indexing button. You can still add an ID directly or upload a PDF from Library. Streamlit is an **API client**, not a second assistant server: it calls FastAPI for conversations, streamed answers and verified citations, uploads, topic criteria, curation inbox, and job status. It uses the same stored data as the React UI. Set `ASSISTANT_API_URL` to another local HTTP loopback origin if FastAPI runs on a different port. Keep both processes on the same trusted machine; neither UI provides multi-user authentication. Background jobs update in Streamlit when you refresh their status or the Jobs page.

New papers start as New and unstarred. Reading status changes only when you select it; re-ingestion preserves status and favorites. Existing databases are upgraded automatically on startup, with existing papers defaulting to New and unstarred. Restart the API after upgrading.

Removing a paper requires confirmation and removes its library entry, chunks, and search vectors. The original PDF and saved chats are retained; citations to removed chunks become unavailable. Re-ingestion also replaces chunk IDs, so citations in older saved answers may become unavailable. The paper detail pane shows metadata and actions, not raw retrieval chunks.

### Answer attribution

Paper citations are selective: general background does not need a paper reference. For each proposed citation, the answer model must provide an exact supporting quote from a retrieved chunk. The application checks that the quote occurs in that chunk (ignoring whitespace differences), then a separate verification call checks whether the evidence supports the entire claim. Only accepted references become numbered citations.

When context compression is enabled, the proposed quote must also occur in the actual extractive snippet shown to the answer model. The verifier receives the unchanged full passage; compression cannot create a new paper citation by paraphrasing a source.

Claims without an accepted paper citation are marked **LLM knowledge**, including general explanations and claims whose proposed evidence could not be verified. This label means the claim is not verified against a paper; it is not a guarantee of correctness. Answers can use general knowledge even when no paper chunks are retrieved. Unsupported paper-specific findings must not be invented.

Web chat displays this attribution as a compact **LK** chip styled like numbered citations, with the full meaning in its tooltip. Confidence remains available internally but is not shown as a badge below answers.

The verifier uses the `citation_verifier` role in `config.yaml`, falling back to `qa` when that role is absent. Verification adds one model call when candidate evidence exists. Invalid quotes, rejected evidence, and verification failures do not produce paper citations. Verification is model-based and can still make mistakes; stronger verifier models may improve accuracy. Existing saved answers are not re-verified automatically.

For frontend development, run the API with `assistant serve --no-open` and then:

```powershell
cd web
npm install
npm run dev       # Vite proxies /api to 127.0.0.1:8000
npm run lint
npm run build     # type-checks and writes assistant/web/static/
```

## Configuration

The single source of truth is [config.yaml](config.yaml). The key idea is **per-role model mapping**. The current local-first configuration uses Ollama for chat and embeddings; the reranker is an optional local cross-encoder, not an Ollama model:

```yaml
llm:
  roles:
    orchestrator:  { provider: ollama, model: qwen2.5:3b }
    qa:            { provider: ollama, model: qwen2.5:3b }
    citation_verifier: { provider: ollama, model: qwen2.5:3b }
    conversation_title: { provider: ollama, model: qwen2.5:3b }
    retrieval:     { provider: ollama, model: qwen2.5:3b }
    curator_judge: { provider: ollama, model: qwen2.5:3b }
    summarizer:    { provider: ollama, model: qwen2.5:3b }
    criteria:      { provider: ollama, model: qwen2.5:3b }
    info_gatherer: { provider: ollama, model: qwen2.5:3b }
    embedder:      { provider: ollama, model: nomic-embed-text }
    reranker:      { provider: local, model: cross-encoder/ms-marco-MiniLM-L6-v2 }

rag:
  mode: vanilla        # or "agentic"
  top_k: 8
  history_turns: 4
  hybrid: { dense_weight: 0.7, sparse_weight: 0.3 }
  query_rewrite: { enabled: true, first_turn: true, max_chars: 300 }
  compression: { enabled: true, max_chars_per_chunk: 400, max_total_chars: 3200 }
  graph:
    enabled: true       # one-hop arXiv citation links; false for BM25 + dense only
    max_neighbors: 6
    graph_slots: 2      # maximum answer-context slots reserved for linked papers

concurrency:
  background_workers: 2  # increase cautiously on machines with enough memory
```

Switch any role to a different supported provider/model by editing its YAML entry; no agent code changes are needed. The default reranker is skipped (with a warning) unless `sentence-transformers` is installed. Local `qwen2.5:3b` can still produce malformed structured answers or weak verification: mocked unit tests do not establish real-model citation accuracy.

To use OpenRouter, set `OPENROUTER_API_KEY` in `.env` and assign any role a model from the [OpenRouter catalog](https://openrouter.ai/models):

```yaml
llm:
  roles:
    qa:       { provider: openrouter, model: anthropic/claude-sonnet-4.6 }
    embedder: { provider: openrouter, model: openai/text-embedding-3-small }
```

Model IDs are passed through unchanged, so provider-prefixed slugs, variants, and aliases supported by OpenRouter do not require code changes. Use a chat-capable model for agent roles and an embedding model for `embedder`. Optional `OPENROUTER_SITE_URL`, `OPENROUTER_APP_NAME`, and `OPENROUTER_BASE_URL` settings are shown in `.env.example`.

If using hosted providers, put their API keys in `.env`; no key is required for the shipped Ollama configuration. Ollama's base URL defaults to `http://localhost:11434`. A model switch that changes embedding dimensions requires rebuilding the existing Qdrant vectors, not merely editing YAML. Restart the API after changing configuration because it is cached per process.

## Project layout

```
assistant/
  agents/        # 9 LangGraph nodes (orchestrator, qa, retrieval, curator, ...)
  llm/           # Per-role LLM router + provider factories
  rag/           # PDF/page chunking, BM25+dense/graph retrieval, filters, compression
  storage/       # SQLAlchemy schema, domain sync + locked Qdrant client
  memory/        # episodic, criteria, conversation + curation stores
  mcp/           # MCP client (loads servers from config)
  web/           # FastAPI routes, job runner, built React assets
  config.py, state.py, graph.py, cli.py, scheduler.py
web/              # Vite + React + TypeScript source
```

## Further reading

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — layers, agents, data model, RAG flow
- **[PLANNER.md](PLANNER.md)** — assumptions, pending work, future requirements
- **[CLAUDE.md](CLAUDE.md)** — guidance for AI coding agents working in this repo
