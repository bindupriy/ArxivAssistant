import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  NavLink,
  Route,
  Routes,
  useLocation,
  useNavigate,
  useSearchParams,
} from "react-router-dom";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import {
  Archive,
  BookOpen,
  Bot,
  Check,
  ChevronRight,
  CirclePlus,
  ExternalLink,
  FileText,
  Inbox,
  Library,
  LoaderCircle,
  MessageSquare,
  PanelLeft,
  Play,
  Search,
  Send,
  Settings2,
  Star,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import {
  api,
  type Decision,
  type Conversation,
  type MetadataFilters,
  type Paper,
  type Source,
  type Status,
  streamMessage,
  type Topic,
  type Turn,
  uploadPdf,
  write,
} from "./api";
import "./chat-filters.css";

const progressLabels: Record<string, string> = {
  retrieving: "Searching the library",
  answering: "Composing an answer",
  gathering: "Gathering external context",
};

function useLoad<T>(path: string, initial: T, refresh = 0, pollMs = 0) {
  const [data, setData] = useState(initial);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    const load = () =>
      api<T>(path)
        .then((result) => {
          if (active) {
            setData(result);
            setError("");
          }
        })
        .catch((err: Error) => {
          if (active) setError(err.message);
        });
    load();
    const timer = pollMs ? window.setInterval(load, pollMs) : undefined;
    return () => {
      active = false;
      if (timer) window.clearInterval(timer);
    };
  }, [path, refresh, pollMs]);
  return { data, setData, error };
}

function SourceDrawer({
  source,
  close,
  chat,
}: {
  source?: Source;
  close: () => void;
  chat: (paperId: string) => void;
}) {
  if (!source) return null;
  return (
    <aside className="drawer source-drawer">
      <header>
        <div>
          <span className="eyebrow">Cited passage</span>
          <h2>{source.title}</h2>
        </div>
        <button className="icon-button" onClick={close} title="Close">
          <X />
        </button>
      </header>
      <p className="section-label">
        {source.section}
        {source.page_number ? ` · Page ${source.page_number}` : ""}
      </p>
      <blockquote>{source.text}</blockquote>
      <div className="drawer-actions">
        {source.arxiv_url && (
          <a
            className="button secondary"
            href={source.arxiv_url}
            target="_blank"
          >
            arXiv <ExternalLink />
          </a>
        )}
        {source.pdf_url && (
          <a className="button secondary" href={source.pdf_url} target="_blank">
            Open PDF <FileText />
          </a>
        )}
        <button className="button" onClick={() => chat(source.paper_id)}>
          Chat about paper <MessageSquare />
        </button>
      </div>
    </aside>
  );
}

function Answer({
  turn,
  onSource,
}: {
  turn: Turn;
  onSource: (source: Source) => void;
}) {
  const marked = turn.answer.replace(/\[(\d+)]/g, "[$1](citation:$1)");
  return (
    <article className="answer">
      <div className="answer-mark">
        <Bot />
      </div>
      <div className="answer-body">
        <ReactMarkdown
          urlTransform={(url) =>
            url.startsWith("citation:") ? url : defaultUrlTransform(url)
          }
          components={{
            em: ({ children }) =>
              typeof children === "string" &&
              /^\(LLM knowledge\)$/i.test(children.trim()) ? (
                <span
                  className="citation-chip"
                  title="LLM knowledge: not verified against a paper"
                  aria-label="LLM knowledge: not verified against a paper"
                  tabIndex={0}
                >
                  LK
                </span>
              ) : (
                <em>{children}</em>
              ),
            a: ({ href, children }) =>
              href?.startsWith("citation:") ? (
                <button
                  className="citation-chip"
                  disabled={
                    !turn.sources.some(
                      (source) =>
                        source.chunk_id ===
                        turn.citations[Number(href.split(":")[1]) - 1],
                    )
                  }
                  title={
                    turn.sources.some(
                      (source) =>
                        source.chunk_id ===
                        turn.citations[Number(href.split(":")[1]) - 1],
                    )
                      ? "View source"
                      : "Source no longer available"
                  }
                  onClick={() => {
                    const chunkId =
                      turn.citations[Number(href.split(":")[1]) - 1];
                    const source = turn.sources.find(
                      (item) => item.chunk_id === chunkId,
                    );
                    if (source) onSource(source);
                  }}
                >
                  {children}
                </button>
              ) : (
                <a href={href} target="_blank">
                  {children}
                </a>
              ),
          }}
        >
          {marked}
        </ReactMarkdown>
        {turn.id && (
          <div className="feedback">
            <button
              title="Helpful"
              onClick={() =>
                write(`/interactions/${turn.id}/feedback`, "POST", { score: 1 })
              }
            >
              <ThumbsUp />
            </button>
            <button
              title="Not helpful"
              onClick={() =>
                write(`/interactions/${turn.id}/feedback`, "POST", {
                  score: -1,
                })
              }
            >
              <ThumbsDown />
            </button>
          </div>
        )}
      </div>
    </article>
  );
}

type ChatSession = {
  turns: Turn[];
  message: string;
  stage: string;
};

function ChatPage({
  refreshKey,
  sidebar,
  openChat,
}: {
  refreshKey: number;
  sidebar: HTMLDivElement | null;
  openChat: () => void;
}) {
  const { data: conversations, setData: setConversations } = useLoad<
    Conversation[]
  >("/conversations", [], refreshKey);
  const { data: topics } = useLoad<Topic[]>("/topics", [], refreshKey);
  const { data: papers } = useLoad<Paper[]>("/papers", [], refreshKey);
  const [active, setActive] = useState<Conversation>();
  const activeConversationId = useRef<string | undefined>(undefined);
  const [sessions, setSessions] = useState<Record<string, ChatSession>>({});
  const [draftMessage, setDraftMessage] = useState("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState("");
  const session = active ? sessions[active.id] : undefined;
  const turns = session?.turns ?? [];
  const message = active ? (session?.message ?? "") : draftMessage;
  const stage = session?.stage ?? "";
  const [source, setSource] = useState<Source>();
  const [scope, setScope] = useState<"library" | "topic" | "paper">("library");
  const [target, setTarget] = useState("");
  const [filters, setFilters] = useState<MetadataFilters>({});
  const selectedScope = active?.scope_type ?? scope;
  const selectedTarget = active?.scope_target ?? target;
  const [searchParams] = useSearchParams();
  const requestedConversation = searchParams.get("conversation");
  useEffect(() => {
    let cancelled = false;
    if (
      requestedConversation &&
      requestedConversation !== activeConversationId.current
    )
      api<Conversation>(`/conversations/${requestedConversation}`).then(
        (loaded) => {
          if (cancelled) return;
          activeConversationId.current = loaded.id;
          setActive(loaded);
          setSource(undefined);
          setSessions((current) => ({
            ...current,
            [loaded.id]: current[loaded.id] ?? {
              turns: loaded.turns ?? [],
              message: "",
              stage: "",
            },
          }));
        },
      );
    return () => {
      cancelled = true;
    };
  }, [requestedConversation]);
  async function selectConversation(item: Conversation) {
    openChat();
    if (item.id === activeConversationId.current) return;
    const loaded = await api<Conversation>(`/conversations/${item.id}`);
    activeConversationId.current = loaded.id;
    setActive(loaded);
    setSource(undefined);
    setCreateError("");
    setSessions((current) => ({
      ...current,
      [loaded.id]: current[loaded.id] ?? {
        turns: loaded.turns ?? [],
        message: "",
        stage: "",
      },
    }));
  }
  function updateSession(
    conversationId: string,
    update: (current: ChatSession) => ChatSession,
  ) {
    setSessions((current) =>
      current[conversationId]
        ? { ...current, [conversationId]: update(current[conversationId]) }
        : current,
    );
  }
  async function deleteChat(item: Conversation) {
    await write(`/conversations/${item.id}`, "DELETE");
    setConversations((current) =>
      current.filter((conversation) => conversation.id !== item.id),
    );
    setSessions((current) => {
      const remaining = { ...current };
      delete remaining[item.id];
      return remaining;
    });
    if (activeConversationId.current === item.id) {
      activeConversationId.current = undefined;
      setActive(undefined);
      setSource(undefined);
    }
  }
  async function newChat(forPaper?: string) {
    const scopeType = forPaper ? "paper" : scope;
    const scopeTarget =
      forPaper || (scopeType === "library" ? undefined : target);
    const created = await write<Conversation>("/conversations", "POST", {
      title: "New conversation",
      scope_type: scopeType,
      scope_target: scopeTarget,
    });
    setConversations((current) => [created, ...current]);
    activeConversationId.current = created.id;
    setActive(created);
    setSource(undefined);
    setSessions((current) => ({
      ...current,
      [created.id]: { turns: [], message: "", stage: "" },
    }));
    openChat();
    return created;
  }
  function startNewChat() {
    activeConversationId.current = undefined;
    setActive(undefined);
    setSource(undefined);
    setDraftMessage("");
    setCreateError("");
    openChat();
  }
  async function send() {
    if (
      !message.trim() ||
      stage ||
      creating ||
      (!active && scope !== "library" && !target)
    )
      return;
    const question = message.trim();
    let conversation = active;
    if (!conversation) {
      setCreating(true);
      setCreateError("");
      try {
        conversation = await newChat();
        setDraftMessage("");
      } catch (error) {
        setCreateError(
          error instanceof Error ? error.message : "Could not create chat",
        );
        return;
      } finally {
        setCreating(false);
      }
    }
    const conversationId = conversation.id;
    updateSession(conversationId, (current) => ({
      ...current,
      message: "",
      stage: "retrieving",
      turns: [
        ...current.turns,
        { question, answer: "", citations: [], sources: [] },
      ],
    }));
    try {
      await streamMessage(
        conversationId,
        question,
        (event) => {
          if (event.type === "conversation") {
            const updated = event.conversation as Conversation;
            if (updated.title !== "New conversation") {
              setConversations((current) =>
                current.map((item) =>
                  item.id === updated.id
                    ? { ...item, title: updated.title }
                    : item,
                ),
              );
              setActive((current) =>
                current?.id === updated.id
                  ? { ...current, title: updated.title }
                  : current,
              );
            }
          }
          if (event.type === "progress") {
            updateSession(conversationId, (current) => ({
              ...current,
              stage: String(event.stage),
            }));
          }
          if (event.type === "final") {
            const result = event.result as Record<string, unknown>;
            const turn: Turn = {
              id: result.interaction_id as number | undefined,
              question,
              answer: String(result.answer ?? ""),
              citations: (result.citations as string[]) ?? [],
              sources: (result.sources as Source[]) ?? [],
              confidence: Number(result.confidence ?? 0),
            };
            updateSession(conversationId, (current) => ({
              ...current,
              turns: [...current.turns.slice(0, -1), turn],
              stage: "",
            }));
          }
          if (event.type === "error") {
            updateSession(conversationId, (current) => ({
              ...current,
              turns: [
                ...current.turns.slice(0, -1),
                {
                  question,
                  answer: String(event.message),
                  citations: [],
                  sources: [],
                },
              ],
              stage: "",
            }));
          }
        },
        filters,
      );
    } catch (err) {
      updateSession(conversationId, (current) => ({
        ...current,
        stage: "",
        turns: [
          ...current.turns.slice(0, -1),
          {
            question,
            answer: (err as Error).message,
            citations: [],
            sources: [],
          },
        ],
      }));
    }
  }
  return (
    <div className="chat-layout">
      {sidebar &&
        createPortal(
          <section className="sidebar-chats" aria-label="Conversations">
            <button
              className="button sidebar-new-chat"
              onClick={startNewChat}
              disabled={creating}
            >
              <CirclePlus /> New chat
            </button>
            <h2>History</h2>
            <div className="conversation-list">
              {conversations.map((item) => (
                <div className="conversation-item" key={item.id}>
                  <button
                    className={active?.id === item.id ? "active" : ""}
                    title={item.title}
                    onClick={() => selectConversation(item)}
                    disabled={creating}
                  >
                    <MessageSquare />
                    <span>{item.title}</span>
                    <ChevronRight />
                  </button>
                  <button
                    className="delete-chat"
                    onClick={() => deleteChat(item)}
                    title="Delete conversation"
                    aria-label={`Delete conversation: ${item.title}`}
                    disabled={creating}
                  >
                    <X />
                  </button>
                </div>
              ))}
            </div>
          </section>,
          sidebar,
        )}
      <section className="chat-main">
        {!active ? (
          <div className="empty-state">
            <div className="empty-glyph">
              <MessageSquare />
            </div>
            <h1>New chat</h1>
          </div>
        ) : (
          <>
            <header className="page-header compact">
              <div>
                <span className="eyebrow">
                  {active.scope_type}
                  {active.scope_target ? ` / ${active.scope_target}` : ""}
                </span>
                <h1>{active.title}</h1>
              </div>
            </header>
            <div className="transcript">
              {turns.map((turn, index) => (
                <div key={index}>
                  <div className="question">{turn.question}</div>
                  {turn.answer ? (
                    <Answer turn={turn} onSource={setSource} />
                  ) : (
                    <div className="progress">
                      <LoaderCircle className="spin" />{" "}
                      {progressLabels[stage] ?? stage}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
        <div className="composer">
          <div
            className="composer-scope"
            title={
              active
                ? "Scope is fixed for this conversation"
                : "Scope for the new conversation"
            }
          >
            <select
              aria-label="Chat scope"
              value={selectedScope}
              disabled={!!active || creating}
              onChange={(event) => {
                setScope(event.target.value as typeof scope);
                setTarget("");
              }}
            >
              <option value="library">Whole library</option>
              <option value="topic">One topic</option>
              <option value="paper">One paper</option>
            </select>
            {selectedScope === "topic" && (
              <select
                className="scope-target"
                aria-label="Topic"
                value={selectedTarget}
                disabled={!!active || creating}
                onChange={(event) => setTarget(event.target.value)}
              >
                <option value="">Choose topic</option>
                {selectedTarget &&
                  !topics.some((item) => item.name === selectedTarget) && (
                    <option value={selectedTarget}>{selectedTarget}</option>
                  )}
                {topics.map((item) => (
                  <option key={item.name}>{item.name}</option>
                ))}
              </select>
            )}
            {selectedScope === "paper" && (
              <select
                className="scope-target"
                aria-label="Paper"
                value={selectedTarget}
                disabled={!!active || creating}
                onChange={(event) => setTarget(event.target.value)}
              >
                <option value="">Choose paper</option>
                {selectedTarget &&
                  !papers.some((item) => item.id === selectedTarget) && (
                    <option value={selectedTarget}>{selectedTarget}</option>
                  )}
                {papers.map((item) => (
                  <option value={item.id} key={item.id}>
                    {item.title}
                  </option>
                ))}
              </select>
            )}
          </div>
          <details className="composer-filters">
            <summary>Search filters</summary>
            <div className="composer-filter-fields">
              <label>
                From year
                <input
                  type="number"
                  min="1800"
                  max="2100"
                  value={filters.min_year ?? ""}
                  onChange={(event) =>
                    setFilters((current) => ({
                      ...current,
                      min_year: event.target.value
                        ? Number(event.target.value)
                        : undefined,
                    }))
                  }
                />
              </label>
              <label>
                To year
                <input
                  type="number"
                  min="1800"
                  max="2100"
                  value={filters.max_year ?? ""}
                  onChange={(event) =>
                    setFilters((current) => ({
                      ...current,
                      max_year: event.target.value
                        ? Number(event.target.value)
                        : undefined,
                    }))
                  }
                />
              </label>
              <label>
                Venue
                <input
                  type="text"
                  maxLength={128}
                  placeholder="e.g. NeurIPS"
                  value={filters.venue ?? ""}
                  onChange={(event) =>
                    setFilters((current) => ({
                      ...current,
                      venue: event.target.value || undefined,
                    }))
                  }
                />
              </label>
              <label>
                Section
                <select
                  value={filters.section_types?.[0] ?? ""}
                  onChange={(event) =>
                    setFilters((current) => ({
                      ...current,
                      section_types: event.target.value
                        ? [event.target.value]
                        : [],
                    }))
                  }
                >
                  <option value="">Any section</option>
                  {[
                    "abstract",
                    "intro",
                    "background",
                    "related",
                    "method",
                    "experiments",
                    "results",
                    "discussion",
                    "analysis",
                    "conclusion",
                    "limitations",
                    "appendix",
                    "body",
                  ].map((section) => (
                    <option key={section} value={section}>
                      {section}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </details>
          <textarea
            value={message}
            disabled={creating}
            onChange={(event) => {
              const value = event.target.value;
              if (active) {
                updateSession(active.id, (current) => ({
                  ...current,
                  message: value,
                }));
              } else {
                setDraftMessage(value);
              }
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                send();
              }
            }}
            placeholder="Ask a research question..."
            aria-label="Message"
          />
          <button
            className="send-button"
            onClick={send}
            disabled={
              !message.trim() ||
              !!stage ||
              creating ||
              (!active && scope !== "library" && !target)
            }
            title="Send"
          >
            {creating ? <LoaderCircle className="spin" /> : <Send />}
          </button>
          {createError && (
            <p className="error" role="alert">
              {createError}
            </p>
          )}
        </div>
      </section>
      <SourceDrawer
        source={source}
        close={() => setSource(undefined)}
        chat={(paperId) => {
          newChat(paperId);
          setSource(undefined);
        }}
      />
    </div>
  );
}

const readingStatuses = {
  new: "New",
  reviewing: "Reviewing",
  read: "Read",
} as const;

function LibraryPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [query, setQuery] = useState("");
  const [topic, setTopic] = useState(searchParams.get("domain") ?? "");
  const [readingStatus, setReadingStatus] = useState("");
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const { data: topics } = useLoad<Topic[]>("/topics", [], refresh);
  const { data: papers, error: loadError } = useLoad<Paper[]>(
    `/papers?q=${encodeURIComponent(query)}&domain=${encodeURIComponent(topic)}${readingStatus ? `&reading_status=${readingStatus}` : ""}${favoritesOnly ? "&is_favorite=true" : ""}`,
    [],
    refresh,
  );
  const [selected, setSelected] = useState<Paper>();
  const [pending, setPending] = useState("");
  const [error, setError] = useState("");
  const [adding, setAdding] = useState(false);
  const [arxiv, setArxiv] = useState("");
  const [addTopic, setAddTopic] = useState("");
  const [file, setFile] = useState<File>();
  async function openPaper(paper: Paper) {
    setPending(paper.id);
    setError("");
    try {
      setSelected(await api<Paper>(`/papers/${encodeURIComponent(paper.id)}`));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load paper");
    } finally {
      setPending("");
    }
  }
  async function updatePaper(
    paper: Paper,
    changes: Partial<Pick<Paper, "reading_status" | "is_favorite">> & {
      domain?: string | null;
    },
  ) {
    setPending(paper.id);
    setError("");
    try {
      const updated = await write<Paper>(
        `/papers/${encodeURIComponent(paper.id)}`,
        "PATCH",
        changes,
      );
      setSelected((current) =>
        current?.id === updated.id ? updated : current,
      );
      setRefresh((value) => value + 1);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not update paper");
    } finally {
      setPending("");
    }
  }
  async function removePaper(paper: Paper) {
    if (
      !window.confirm(
        `Remove "${paper.title}" from the library?\n\nThe paper and its searchable chunks will be removed. The original PDF and saved chats will be kept.`,
      )
    )
      return;
    setPending(paper.id);
    setError("");
    try {
      await write(`/papers/${encodeURIComponent(paper.id)}`, "DELETE");
      setSelected((current) =>
        current?.id === paper.id ? undefined : current,
      );
      setRefresh((value) => value + 1);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not remove paper");
    } finally {
      setPending("");
    }
  }
  async function addPaper() {
    if (file) await uploadPdf(file, addTopic || undefined);
    else
      await write("/papers/arxiv", "POST", {
        arxiv_id: arxiv,
        domain: addTopic || undefined,
      });
    setAdding(false);
    setRefresh((x) => x + 1);
  }
  async function chatAboutPaper(paper: Paper) {
    const conversation = await write<Conversation>("/conversations", "POST", {
      title: paper.title,
      scope_type: "paper",
      scope_target: paper.id,
    });
    const next = new URLSearchParams(searchParams);
    next.set("conversation", conversation.id);
    navigate({ search: next.toString() });
    setSelected(undefined);
  }
  return (
    <div className="page library-page">
      <header className="page-header">
        <div>
          <span className="eyebrow">Knowledge base</span>
          <h1>Library</h1>
          <p>
            {papers.length} {papers.length === 1 ? "paper" : "papers"}
          </p>
        </div>
        <button className="button" onClick={() => setAdding(true)}>
          <CirclePlus /> Add paper
        </button>
      </header>
      <div className="library-tabs" role="tablist" aria-label="Library views">
        <button
          role="tab"
          aria-selected={!favoritesOnly}
          aria-controls="library-papers"
          onClick={() => setFavoritesOnly(false)}
        >
          <Library /> All papers
        </button>
        <button
          role="tab"
          aria-selected={favoritesOnly}
          aria-controls="library-papers"
          onClick={() => setFavoritesOnly(true)}
        >
          <Star /> Favorites
        </button>
      </div>
      <div className="toolbar">
        <label className="search">
          <Search />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search title or arXiv ID"
          />
        </label>
        <select
          aria-label="Filter by topic"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
        >
          <option value="">All topics</option>
          {topics.map((item) => (
            <option key={item.name}>{item.name}</option>
          ))}
        </select>
        <select
          aria-label="Filter by reading status"
          value={readingStatus}
          onChange={(event) => setReadingStatus(event.target.value)}
        >
          <option value="">All statuses</option>
          {Object.entries(readingStatuses).map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </div>
      {(error || loadError) && (
        <p className="error" role="alert">
          {error || loadError}
        </p>
      )}
      <div
        className="paper-table"
        id="library-papers"
        role="tabpanel"
        aria-label={favoritesOnly ? "Favorites" : "All papers"}
      >
        <div className="table-head">
          <span />
          <span>Paper</span>
          <span>Status</span>
          <span>Topic</span>
          <span>Year</span>
          <span />
        </div>
        {papers.map((paper) => (
          <div
            className="paper-row"
            key={paper.id}
            aria-busy={pending === paper.id}
          >
            <button
              className="icon-button favorite-button"
              aria-pressed={paper.is_favorite}
              title={
                paper.is_favorite
                  ? "Unmark as interesting"
                  : "Mark as interesting"
              }
              aria-label={`${paper.is_favorite ? "Unmark as interesting" : "Mark as interesting"}: ${paper.title}`}
              disabled={!!pending}
              onClick={() =>
                updatePaper(paper, { is_favorite: !paper.is_favorite })
              }
            >
              <Star />
            </button>
            <button
              className="paper-title"
              disabled={!!pending}
              onClick={() => openPaper(paper)}
            >
              <strong>{paper.title}</strong>
              <small>
                {paper.authors.slice(0, 3).join(", ") || paper.arxiv_id}
              </small>
            </button>
            <select
              className="reading-status"
              aria-label={`Reading status for ${paper.title}`}
              value={paper.reading_status}
              disabled={!!pending}
              onChange={(event) =>
                updatePaper(paper, {
                  reading_status: event.target.value as Paper["reading_status"],
                })
              }
            >
              {Object.entries(readingStatuses).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
            <span className="paper-topic">
              <i className="topic-dot" />
              {paper.domain || "Unsorted"}
            </span>
            <span className="paper-year">{paper.year || "—"}</span>
            <button
              className="icon-button remove-paper"
              title="Remove from library"
              aria-label={`Remove from library: ${paper.title}`}
              disabled={!!pending}
              onClick={() => removePaper(paper)}
            >
              <Trash2 />
            </button>
          </div>
        ))}
        {papers.length === 0 && (
          <div className="data-empty">
            {favoritesOnly
              ? "No favorites match this view."
              : "No papers match this view."}
          </div>
        )}
      </div>
      {selected && (
        <aside className="drawer">
          <header>
            <div>
              <span className="eyebrow">{selected.arxiv_id}</span>
              <h2>{selected.title}</h2>
            </div>
            <button
              className="icon-button"
              title="Close paper"
              aria-label="Close paper"
              onClick={() => setSelected(undefined)}
            >
              <X />
            </button>
          </header>
          <p className="authors">{selected.authors.join(", ")}</p>
          <p>{selected.abstract}</p>
          <div className="paper-preferences">
            <label>
              Reading status
              <select
                value={selected.reading_status}
                disabled={!!pending}
                onChange={(event) =>
                  updatePaper(selected, {
                    reading_status: event.target
                      .value as Paper["reading_status"],
                  })
                }
              >
                {Object.entries(readingStatuses).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <button
              className="icon-button favorite-button"
              aria-pressed={selected.is_favorite}
              title={
                selected.is_favorite
                  ? "Unmark as interesting"
                  : "Mark as interesting"
              }
              aria-label={
                selected.is_favorite
                  ? "Unmark as interesting"
                  : "Mark as interesting"
              }
              disabled={!!pending}
              onClick={() =>
                updatePaper(selected, { is_favorite: !selected.is_favorite })
              }
            >
              <Star />
            </button>
          </div>
          <label>
            Topic
            <select
              value={selected.domain || ""}
              disabled={!!pending}
              onChange={(event) =>
                updatePaper(selected, { domain: event.target.value || null })
              }
            >
              <option value="">Unsorted</option>
              {topics.map((item) => (
                <option key={item.name}>{item.name}</option>
              ))}
            </select>
          </label>
          <div className="drawer-actions">
            {selected.has_pdf && (
              <a
                className="button secondary"
                href={`/api/papers/${selected.id}/pdf`}
                target="_blank"
              >
                Open PDF <FileText />
              </a>
            )}
            {selected.arxiv_id && (
              <a
                className="button secondary"
                href={`https://arxiv.org/abs/${selected.arxiv_id}`}
                target="_blank"
              >
                arXiv <ExternalLink />
              </a>
            )}
            <button className="button" onClick={() => chatAboutPaper(selected)}>
              Chat about this paper <MessageSquare />
            </button>
          </div>
          <div className="paper-removal">
            <button
              className="button secondary remove-paper"
              disabled={!!pending}
              onClick={() => removePaper(selected)}
            >
              <Trash2 />{" "}
              {pending === selected.id ? "Working..." : "Remove from library"}
            </button>
            {error && (
              <p className="error" role="alert">
                {error}
              </p>
            )}
          </div>
        </aside>
      )}
      {adding && (
        <div className="modal-backdrop">
          <form
            className="modal"
            onSubmit={(e) => {
              e.preventDefault();
              addPaper();
            }}
          >
            <header>
              <h2>Add to library</h2>
              <button
                type="button"
                className="icon-button"
                onClick={() => setAdding(false)}
              >
                <X />
              </button>
            </header>
            <label>
              arXiv ID
              <input
                value={arxiv}
                onChange={(e) => setArxiv(e.target.value)}
                placeholder="2401.05566"
                disabled={!!file}
              />
            </label>
            <div className="divider">or</div>
            <label className="upload-field">
              <Upload />
              <span>{file?.name || "Choose a PDF up to 50 MB"}</span>
              <input
                type="file"
                accept="application/pdf"
                onChange={(e) => setFile(e.target.files?.[0])}
              />
            </label>
            <label>
              Topic
              <select
                value={addTopic}
                onChange={(e) => setAddTopic(e.target.value)}
              >
                <option value="">Unsorted</option>
                {topics.map((item) => (
                  <option key={item.name}>{item.name}</option>
                ))}
              </select>
            </label>
            <button className="button full" disabled={!arxiv && !file}>
              Queue ingestion
            </button>
          </form>
        </div>
      )}
    </div>
  );
}

function TopicsPage() {
  const navigate = useNavigate();
  const [refresh, setRefresh] = useState(0);
  const { data: topics } = useLoad<Topic[]>("/topics", [], refresh);
  const [commenting, setCommenting] = useState<Topic>();
  const [comment, setComment] = useState("");
  async function chatAboutTopic(topic: Topic) {
    const conversation = await write<Conversation>("/conversations", "POST", {
      title: topic.name,
      scope_type: "topic",
      scope_target: topic.name,
    });
    navigate({
      search: new URLSearchParams({ conversation: conversation.id }).toString(),
    });
  }
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <span className="eyebrow">Research map</span>
          <h1>Topics</h1>
          <p>Config-defined areas and their active curation criteria</p>
        </div>
        <button
          className="button secondary"
          onClick={async () => {
            await write("/topics/sync", "POST");
            setRefresh((x) => x + 1);
          }}
        >
          <Settings2 /> Sync from config
        </button>
      </header>
      <div className="topic-grid">
        {topics.map((topic) => (
          <article className="topic-card" key={topic.name}>
            <header>
              <div>
                <h2>{topic.name}</h2>
                <span className={topic.synced ? "status ready" : "status"}>
                  {topic.synced ? "Synced" : "Not synced"}
                </span>
              </div>
              <strong>
                {topic.paper_count}
                <small> papers</small>
              </strong>
            </header>
            <p>
              {topic.criteria?.nl_addendum ||
                "No natural-language curation guidance yet."}
            </p>
            <div className="tag-row">
              {topic.arxiv_categories.map((item) => (
                <span key={item}>{item}</span>
              ))}
              {topic.venues.map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
            <footer>
              <span>Criteria v{topic.criteria?.version ?? 0}</span>
              <div>
                <button
                  title="Chat about topic"
                  onClick={() => chatAboutTopic(topic)}
                >
                  <MessageSquare />
                </button>
                <button
                  title="View papers"
                  onClick={() =>
                    navigate(
                      `/library?domain=${encodeURIComponent(topic.name)}`,
                    )
                  }
                >
                  <Library />
                </button>
                <button
                  title="Run monitor"
                  onClick={() => write(`/topics/${topic.name}/monitor`, "POST")}
                >
                  <Play />
                </button>
                <button
                  title="Refine criteria"
                  onClick={() => {
                    setCommenting(topic);
                    setComment("");
                  }}
                >
                  <Settings2 />
                </button>
              </div>
            </footer>
          </article>
        ))}
        {topics.length === 0 && (
          <div className="data-empty">
            No topics are configured. Add one under domains in config.yaml, then
            sync.
          </div>
        )}
      </div>
      {commenting && (
        <div className="modal-backdrop">
          <form
            className="modal"
            onSubmit={async (e) => {
              e.preventDefault();
              await write(`/topics/${commenting.name}/criteria`, "POST", {
                comment,
              });
              setCommenting(undefined);
              setRefresh((x) => x + 1);
            }}
          >
            <header>
              <div>
                <span className="eyebrow">Refine criteria</span>
                <h2>{commenting.name}</h2>
              </div>
              <button
                type="button"
                className="icon-button"
                onClick={() => setCommenting(undefined)}
              >
                <X />
              </button>
            </header>
            <label>
              What should the curator learn?
              <textarea
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                rows={6}
              />
            </label>
            <button className="button full" disabled={!comment.trim()}>
              Create criteria version
            </button>
          </form>
        </div>
      )}
    </div>
  );
}

function InboxPage() {
  const [topic, setTopic] = useState("");
  const [status, setStatus] = useState("");
  const [refresh, setRefresh] = useState(0);
  const { data: topics } = useLoad<Topic[]>("/topics", []);
  const { data: decisions } = useLoad<Decision[]>(
    `/inbox?domain=${encodeURIComponent(topic)}&status=${encodeURIComponent(status)}`,
    [],
    refresh,
  );
  const [disagree, setDisagree] = useState<Decision>();
  const [comment, setComment] = useState("");
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <span className="eyebrow">Curation review</span>
          <h1>Inbox</h1>
          <p>Review what the curator accepted, rejected, or failed to ingest</p>
        </div>
      </header>
      <div className="toolbar">
        <select value={topic} onChange={(e) => setTopic(e.target.value)}>
          <option value="">All topics</option>
          {topics.map((item) => (
            <option key={item.name}>{item.name}</option>
          ))}
        </select>
        <select value={status} onChange={(e) => setStatus(e.target.value)}>
          <option value="">All statuses</option>
          {["accepted", "rejected", "ingested", "failed", "dismissed"].map(
            (item) => (
              <option key={item}>{item}</option>
            ),
          )}
        </select>
      </div>
      <div className="inbox-list">
        {decisions.map((item) => (
          <article className="decision" key={item.id}>
            <div className="decision-score">
              <strong>{Math.round(item.score * 100)}</strong>
              <span>%</span>
              <i
                style={
                  { "--score": `${item.score * 100}%` } as React.CSSProperties
                }
              />
            </div>
            <div className="decision-body">
              <header>
                <div>
                  <span className="eyebrow">
                    {item.domain} / {item.arxiv_id}
                  </span>
                  <h2>{item.title}</h2>
                </div>
                <span className={`status ${item.status}`}>{item.status}</span>
              </header>
              <p>{item.judge_reason}</p>
              {item.error && <p className="error">{item.error}</p>}
              <footer>
                <span>{item.authors.slice(0, 3).join(", ")}</span>
                <div>
                  {["rejected", "failed"].includes(item.status) && (
                    <button
                      className="button small"
                      onClick={async () => {
                        await write(`/inbox/${item.id}/ingest`, "POST");
                        setRefresh((x) => x + 1);
                      }}
                    >
                      <Archive /> Ingest
                    </button>
                  )}
                  <button
                    className="button secondary small"
                    onClick={async () => {
                      await write(`/inbox/${item.id}/dismiss`, "POST");
                      setRefresh((x) => x + 1);
                    }}
                  >
                    <Check /> Dismiss
                  </button>
                  <button
                    className="button ghost small"
                    onClick={() => {
                      setDisagree(item);
                      setComment(`For ${item.title}: `);
                    }}
                  >
                    Disagree
                  </button>
                </div>
              </footer>
            </div>
          </article>
        ))}
        {decisions.length === 0 && (
          <div className="data-empty">
            No curation decisions match these filters. Run a topic monitor to
            fill the inbox.
          </div>
        )}
      </div>
      {disagree && (
        <div className="modal-backdrop">
          <form
            className="modal"
            onSubmit={async (e) => {
              e.preventDefault();
              await write(`/inbox/${disagree.id}/disagree`, "POST", {
                comment,
              });
              setDisagree(undefined);
              setRefresh((x) => x + 1);
            }}
          >
            <header>
              <h2>Teach the curator</h2>
              <button
                type="button"
                className="icon-button"
                onClick={() => setDisagree(undefined)}
              >
                <X />
              </button>
            </header>
            <label>
              Criteria comment
              <textarea
                rows={6}
                value={comment}
                onChange={(e) => setComment(e.target.value)}
              />
            </label>
            <button className="button full">Update criteria</button>
          </form>
        </div>
      )}
    </div>
  );
}

function Shell() {
  const { pathname } = useLocation();
  const [searchParams, setSearchParams] = useSearchParams();
  const { data: status } = useLoad<Status>(
    "/status",
    { paper_count: 0, rag_mode: "—", roles: {} },
    0,
    5000,
  );
  const { data: jobs } = useLoad<{ status: string }[]>("/jobs", [], 0, 2000);
  const [open, setOpen] = useState(false);
  const [chatSidebar, setChatSidebar] = useState<HTMLDivElement | null>(null);
  const navigate = useNavigate();
  const activeJobs = jobs.filter((job) =>
    ["queued", "running"].includes(job.status),
  ).length;
  const nav = [
    { to: "/chat", label: "Chat", icon: MessageSquare },
    { to: "/library", label: "Library", icon: Library },
    { to: "/topics", label: "Topics", icon: BookOpen },
    { to: "/inbox", label: "Inbox", icon: Inbox },
  ];
  const workspace = nav.find(
    (item) => item.to === pathname && item.to !== "/chat",
  );
  return (
    <div className="shell">
      <aside className={`sidebar ${open ? "open" : ""}`}>
        <div className="brand">
          <div className="brand-mark">
            <Archive />
          </div>
          <div>
            <strong>Fieldnotes</strong>
            <span>research assistant</span>
          </div>
        </div>
        <nav>
          {nav.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} onClick={() => setOpen(false)}>
              <Icon />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-chat-slot" ref={setChatSidebar} />
        <footer>
          <div className="system-line">
            <span className={activeJobs ? "pulse" : ""} />
            {activeJobs ? `${activeJobs} job running` : "System ready"}
          </div>
          <dl>
            <div>
              <dt>Papers</dt>
              <dd>{status.paper_count}</dd>
            </div>
            <div>
              <dt>RAG</dt>
              <dd>{status.rag_mode}</dd>
            </div>
            <div>
              <dt>QA</dt>
              <dd title={status.roles.qa?.model}>
                {status.roles.qa?.model?.split("/").pop() || "—"}
              </dd>
            </div>
          </dl>
        </footer>
      </aside>
      <button
        className="mobile-menu"
        onClick={() => setOpen(!open)}
        aria-label="Toggle navigation"
        aria-expanded={open}
      >
        <PanelLeft />
      </button>
      <main className={`workbench${workspace ? " workbench-split" : ""}`}>
        <section
          className="workspace-pane"
          hidden={!workspace}
          aria-label={`${workspace?.label ?? "Workspace"} workspace`}
        >
          <div className="workspace-toolbar">
            <button
              className="icon-button"
              title={`Close ${workspace?.label ?? "workspace"}`}
              aria-label={`Close ${workspace?.label ?? "workspace"}`}
              onClick={() => navigate("/chat")}
            >
              <X />
            </button>
          </div>
          <div className="workspace-body">
            <Routes>
              <Route path="/" element={null} />
              <Route path="/chat" element={null} />
              <Route path="/library" element={<LibraryPage />} />
              <Route path="/topics" element={<TopicsPage />} />
              <Route path="/inbox" element={<InboxPage />} />
            </Routes>
          </div>
        </section>
        <section className="chat-pane" aria-label="Conversation">
          <ChatPage
            refreshKey={nav.findIndex((item) => item.to === pathname)}
            sidebar={chatSidebar}
            openChat={() => {
              if (searchParams.has("conversation")) {
                const next = new URLSearchParams(searchParams);
                next.delete("conversation");
                setSearchParams(next, { replace: true });
              }
              setOpen(false);
            }}
          />
        </section>
      </main>
    </div>
  );
}

export default Shell;
