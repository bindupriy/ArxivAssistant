"""Optional Streamlit frontend; all data and model work stays in FastAPI."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import streamlit as st

from assistant.web.api_client import AssistantAPI, AssistantAPIError


st.set_page_config(page_title="Research Desk", page_icon="📚", layout="wide")
st.markdown(
    """<style>
    .block-container { max-width: 1150px; padding-top: 2rem; }
    [data-testid="stSidebar"] { background: #122321; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #f1f5f0; }
    [data-testid="stChatMessage"] { border-radius: 12px; border: 1px solid #dde8e1; }
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def _client(url: str) -> AssistantAPI:
    return AssistantAPI(url)


def _act(operation, *, success: str = "Saved") -> bool:
    try:
        operation()
    except AssistantAPIError as exc:
        st.error(str(exc))
        return False
    st.toast(success)
    return True


def _pdf_link(api: AssistantAPI, source: dict) -> str | None:
    path = source.get("pdf_url")
    return f"{api.base_url}{path}" if isinstance(path, str) and path.startswith("/api/papers/") else None


def _answer(turn: dict, api: AssistantAPI) -> None:
    st.markdown(turn.get("answer") or "_No answer received._")
    cited = list(turn.get("citations") or [])
    sources = {source["chunk_id"]: source for source in turn.get("sources") or []}
    if cited:
        with st.expander(f"📎 Paper sources ({len(cited)})"):
            for number, chunk_id in enumerate(cited, start=1):
                source = sources.get(chunk_id)
                if source is None:
                    st.caption(f"[{number}] Source no longer available")
                    continue
                location = source.get("section") or "Paper"
                if source.get("page_number"):
                    location += f" · p. {source['page_number']}"
                st.markdown(f"**[{number}] {source.get('title') or source['paper_id']}** · {location}")
                st.code(source.get("text") or "", language=None)
                links = st.columns(2)
                pdf = _pdf_link(api, source)
                if pdf:
                    links[0].link_button("Open PDF at page", pdf)
                if source.get("arxiv_url"):
                    links[1].link_button("arXiv", source["arxiv_url"])
    interaction_id = turn.get("id")
    if interaction_id:
        left, right, _ = st.columns([1, 1, 10])
        for column, score, icon in ((left, 1, "👍"), (right, -1, "👎")):
            if column.button(icon, key=f"feedback-{interaction_id}-{score}"):
                _act(
                    lambda: api.post(f"/interactions/{interaction_id}/feedback", {"score": score}),
                    success="Feedback saved",
                )


def _filters() -> dict[str, Any]:
    with st.expander("🔎 Search filters (this question)"):
        left, middle, right = st.columns([1, 1, 2])
        start = left.text_input("From year", placeholder="2020", key="min_year")
        end = middle.text_input("Through year", placeholder="2026", key="max_year")
        venue = right.text_input("Exact venue", placeholder="NeurIPS", key="venue")
        section = st.selectbox(
            "Section", ["Any", "abstract", "intro", "background", "related", "method",
                        "experiments", "results", "discussion", "analysis", "conclusion",
                        "limitations", "appendix", "body"], key="section",
        )
    result: dict[str, Any] = {}
    for value, field in ((start, "min_year"), (end, "max_year")):
        if value.strip():
            if not value.strip().isdigit() or not 1800 <= int(value) <= 2100:
                raise ValueError(f"{field} must be a year from 1800 through 2100")
            result[field] = int(value)
    if result.get("min_year", 0) > result.get("max_year", 2100):
        raise ValueError("From year cannot exceed through year")
    if venue.strip():
        result["venue"] = venue.strip()
    if section != "Any":
        result["section_types"] = [section]
    return result


def _open_chat(scope: str, target: str) -> None:
    st.session_state["pending_chat_scope"] = (scope, target)
    st.rerun()


def _arxiv_search(api: AssistantAPI, *, key: str, domain: str | None = None) -> None:
    """Search arXiv from either workspace, then explicitly queue PDF ingestion."""
    with st.form(f"arxiv-search-{key}"):
        query = st.text_input("Find an arXiv paper by title, topic, or ID", key=f"arxiv-query-{key}",
                              placeholder="RoPE rotary position embeddings")
        submitted = st.form_submit_button("Search arXiv")
    results_key = f"arxiv-results-{key}"
    if submitted:
        st.session_state[results_key] = []
        if query.strip():
            try:
                with st.spinner("Searching arXiv metadata…"):
                    st.session_state[results_key] = api.get("/arxiv/search", q=query.strip())
            except AssistantAPIError as exc:
                st.error(str(exc))
    results = st.session_state.get(results_key) or []
    if not results:
        if submitted:
            st.info("No matching papers. Try a title, shorter keywords, or an arXiv ID.")
        return
    by_id = {item["arxiv_id"]: item for item in results}
    selected = st.selectbox("Search results", list(by_id),
                            format_func=lambda paper_id: f"{by_id[paper_id]['title']} ({paper_id})",
                            key=f"arxiv-choice-{key}")
    paper = by_id[selected]
    st.caption(f"{paper.get('year') or 'Year unknown'} · {', '.join(paper['authors'][:3])}")
    st.write(paper["abstract"])
    if paper.get("paper_id"):
        st.success("Already indexed in your library.")
        if st.button("Chat about this paper", key=f"arxiv-chat-{key}"):
            _open_chat("paper", paper["paper_id"])
    elif st.button("Download PDF and index paper", key=f"arxiv-ingest-{key}"):
        try:
            job = api.post("/papers/arxiv", {"arxiv_id": selected, "domain": domain})
            st.session_state[f"arxiv-job-{key}"] = job["id"]
            st.toast("Paper queued — check progress below or in Jobs")
        except AssistantAPIError as exc:
            st.error(str(exc))

    job_id = st.session_state.get(f"arxiv-job-{key}")
    if job_id:
        if st.button("Refresh indexing status", key=f"arxiv-refresh-{key}"):
            st.rerun()
        jobs = api.get("/jobs")
        job = next((item for item in jobs if item["id"] == job_id), None)
        if job is None:
            st.info("Job no longer available (the API may have restarted). Search again to check the library.")
        elif job["status"] == "failed":
            st.error(job.get("error") or "Indexing failed")
        elif job["status"] == "completed":
            ingested = (job.get("result") or {}).get("ingested") or []
            if ingested:
                st.success("PDF downloaded and indexed. You can now ask about this paper.")
                if st.button("Start paper chat", key=f"arxiv-ready-{key}"):
                    _open_chat("paper", ingested[0])
            else:
                errors = ((job.get("result") or {}).get("scratch") or {}).get("ingest_errors") or []
                st.error(str(errors[0].get("error")) if errors else "No paper was indexed")
        else:
            st.info(f"Indexing job: {job['status']}. Refresh in a moment.")


def _chat(api: AssistantAPI) -> None:
    conversations = api.get("/conversations")
    with st.sidebar:
        st.divider()
        st.subheader("Conversations")
        if st.button("＋ New conversation", use_container_width=True):
            st.session_state["active_id"] = None
            st.rerun()
        for conversation in conversations[:35]:
            label = conversation["title"]
            if st.button(label, key=f"chat-{conversation['id']}", use_container_width=True):
                st.session_state["active_id"] = conversation["id"]
                st.rerun()

    active_id = st.session_state.get("active_id")
    active = api.get(f"/conversations/{active_id}") if active_id else None
    st.title("💬 Ask your research library")
    if active:
        st.caption(f"{active['title']} · {active['scope_type']}" +
                   (f" / {active['scope_target']}" if active.get("scope_target") else ""))
        for turn in active.get("turns") or []:
            with st.chat_message("user"):
                st.markdown(turn["question"])
            with st.chat_message("assistant"):
                _answer(turn, api)
        with st.expander("Delete conversation"):
            confirmed = st.checkbox("I understand this removes the saved chat and its turns")
            if st.button("Delete this conversation", disabled=not confirmed, type="secondary"):
                if _act(lambda: api.delete(f"/conversations/{active_id}"), success="Conversation deleted"):
                    st.session_state["active_id"] = None
                    st.rerun()
    else:
        topics = api.get("/topics")
        papers = api.get("/papers")
        st.info("Choose what to search before the first message. The conversation keeps this scope.")
        scope = st.selectbox("Search scope", ["library", "topic", "paper"], key="scope")
        target = None
        if scope == "topic":
            names = [item["name"] for item in topics]
            if not names:
                st.warning("No topics yet. Create one under Topics or choose the whole library.")
            else:
                target = st.selectbox("Topic", names, key="target_topic")
        elif scope == "paper":
            ids = [item["id"] for item in papers]
            if not ids:
                st.warning("No papers yet. Search arXiv below or upload a PDF in Library.")
            else:
                by_id = {item["id"]: item["title"] for item in papers}
                target = st.selectbox("Paper", ids, format_func=lambda paper_id: by_id[paper_id], key="target_paper")
    with st.expander("Find and add a paper from arXiv"):
        topic = active.get("scope_target") if active and active["scope_type"] == "topic" else (
            target if not active and scope == "topic" else None
        )
        _arxiv_search(api, key="chat", domain=topic)
    try:
        filters = _filters()
    except ValueError as exc:
        st.warning(str(exc))
        filters = None

    question = st.chat_input("Ask about a method, result, or comparison…", disabled=filters is None)
    if not question:
        return
    try:
        if active is None:
            if scope != "library" and not target:
                st.warning("Choose a topic or paper first.")
                return
            active = api.post("/conversations", {"scope_type": scope, "scope_target": target})
            active_id = active["id"]
            st.session_state["active_id"] = active_id
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            result = st.empty()
            status = st.status("Searching the library…", expanded=False)
            complete = False
            try:
                for event in api.stream_message(active_id, question, filters):
                    kind = event.get("type")
                    if kind == "progress":
                        stage = {"retrieving": "Retrieving evidence", "answering": "Checking the answer",
                                 "gathering": "Gathering more context"}.get(event.get("stage"), "Working")
                        status.update(label=stage)
                    elif kind == "final":
                        turn = event.get("result") or {}
                        result.markdown(turn.get("answer") or "_No answer received._")
                        complete = True
                    elif kind == "error":
                        raise AssistantAPIError(str(event.get("message") or "Answer failed"))
                status.update(label="Answer ready", state="complete")
            except AssistantAPIError:
                status.update(label="Answer failed", state="error")
                raise
        if complete:
            st.rerun()  # reload saved turns, citation sources, and the optional updated title
    except AssistantAPIError as exc:
        st.error(str(exc))


def _library(api: AssistantAPI) -> None:
    st.title("📚 Paper library")
    topics = api.get("/topics")
    names = [topic["name"] for topic in topics]
    with st.expander("Add a paper", expanded=True):
        with st.form("arxiv-ingest"):
            arxiv_id = st.text_input("arXiv ID", placeholder="2104.09864")
            topic = st.selectbox("Topic (optional)", ["No topic", *names], key="arxiv-topic")
            submitted = st.form_submit_button("Queue arXiv paper")
        if submitted and arxiv_id.strip():
            if _act(lambda: api.post("/papers/arxiv", {
                "arxiv_id": arxiv_id.strip(), "domain": None if topic == "No topic" else topic,
            }), success="Paper queued — check Jobs for progress"):
                st.rerun()
        _arxiv_search(api, key="library", domain=None if topic == "No topic" else topic)
        upload = st.file_uploader("Or upload a PDF", type="pdf")
        if upload and st.button("Queue PDF upload"):
            if _act(lambda: api.upload_pdf(upload.name, upload.getvalue(),
                                            None if topic == "No topic" else topic),
                    success="PDF queued — check Jobs for progress"):
                st.rerun()

    search, topic_col, reading_col, favorite_col = st.columns([3, 2, 2, 1])
    query = search.text_input("Search titles or IDs", key="paper-search")
    domain = topic_col.selectbox("Topic", ["All topics", *names], key="library-topic")
    reading = reading_col.selectbox("Reading status", ["Any", "new", "reviewing", "read"])
    favorites = favorite_col.checkbox("Favorites only")
    papers = api.get("/papers", q=query or None,
                     domain=None if domain == "All topics" else domain,
                     reading_status=None if reading == "Any" else reading,
                     is_favorite=True if favorites else None)
    st.caption(f"{len(papers)} papers")
    if not papers:
        st.info("No matching papers. Add an arXiv paper or upload a PDF above.")
        return
    titles = {paper["id"]: paper["title"] for paper in papers}
    chosen = st.selectbox("View paper", list(titles), format_func=lambda paper_id: titles[paper_id])
    paper = api.get(f"/papers/{quote(chosen, safe='')}")
    st.subheader(paper["title"])
    st.caption(" · ".join(paper.get("authors") or []) +
               (f" · {paper['year']}" if paper.get("year") else ""))
    st.write(paper.get("abstract") or "No abstract available.")
    left, center, right = st.columns(3)
    if left.button("☆ Unstar" if paper["is_favorite"] else "★ Favorite"):
        if _act(lambda: api.patch(f"/papers/{quote(chosen, safe='')}",
                                  {"is_favorite": not paper["is_favorite"]})):
            st.rerun()
    status = center.selectbox("Set reading status", ["new", "reviewing", "read"],
                              index=["new", "reviewing", "read"].index(paper["reading_status"]))
    if center.button("Save reading status"):
        if _act(lambda: api.patch(f"/papers/{quote(chosen, safe='')}", {"reading_status": status})):
            st.rerun()
    if right.button("Chat about this paper"):
        _open_chat("paper", chosen)
    assignment_options = ["No topic", *names]
    saved_topic = paper.get("domain") or "No topic"
    if saved_topic not in assignment_options:
        assignment_options.append(saved_topic)
    assigned = st.selectbox(
        "Paper topic", assignment_options,
        index=assignment_options.index(saved_topic), key=f"paper-topic-{chosen}",
    )
    if st.button("Save paper topic"):
        if _act(lambda: api.patch(f"/papers/{quote(chosen, safe='')}",
                                  {"domain": None if assigned == "No topic" else assigned})):
            st.rerun()
    if paper.get("has_pdf"):
        st.link_button("Open PDF", api.base_url + f"/api/papers/{quote(chosen, safe='')}/pdf")
    with st.expander("Remove from library"):
        confirmed = st.checkbox("I understand the paper and its search vectors will be removed")
        if st.button("Delete paper", disabled=not confirmed, type="secondary"):
            if _act(lambda: api.delete(f"/papers/{quote(chosen, safe='')}"), success="Paper removed"):
                st.rerun()


def _topics(api: AssistantAPI) -> None:
    st.title("🗂 Topics")
    if st.button("Sync topics from config"):
        if _act(lambda: api.post("/topics/sync"), success="Topics synchronized"):
            st.rerun()
    topics = api.get("/topics")
    if not topics:
        st.info("Add a topic under domains in the assistant configuration, then sync it here.")
        return
    chosen = st.selectbox("Topic", [topic["name"] for topic in topics])
    topic = next(item for item in topics if item["name"] == chosen)
    st.caption(f"{topic['paper_count']} papers · {', '.join(topic['arxiv_categories'])}")
    if st.button("Chat about this topic"):
        _open_chat("topic", chosen)
    criteria = topic.get("criteria") or {}
    with st.expander("Active selection criteria", expanded=True):
        st.caption(f"Version {criteria.get('version', 0)}")
        st.json(criteria.get("structured_rules") or {})
        if criteria.get("nl_addendum"):
            st.write(criteria["nl_addendum"])
    with st.form("criteria-comment"):
        comment = st.text_area("Refine curation criteria", placeholder="Skip survey papers")
        submitted = st.form_submit_button("Save criteria")
    if submitted and comment.strip():
        if _act(lambda: api.post(f"/topics/{quote(chosen, safe='')}/criteria",
                                 {"comment": comment.strip()}), success="Criteria updated"):
            st.rerun()
    if st.button("Monitor this topic for recent papers"):
        if _act(lambda: api.post(f"/topics/{quote(chosen, safe='')}/monitor"),
                success="Monitor queued — check Jobs for progress"):
            st.rerun()


def _inbox(api: AssistantAPI) -> None:
    st.title("📥 Curation inbox")
    status = st.selectbox("Decision status", ["All", "rejected", "failed", "ingested", "dismissed"])
    decisions = api.get("/inbox", status=None if status == "All" else status)
    if not decisions:
        st.info("No curator decisions yet. Run a topic monitor to review results.")
        return
    by_id = {item["id"]: item for item in decisions}
    selected = st.selectbox("Review paper", list(by_id),
                            format_func=lambda decision_id: by_id[decision_id]["title"])
    decision = by_id[selected]
    st.subheader(decision["title"])
    st.caption(f"{decision['domain']} · {decision['status']} · score {decision['score']:.2f}")
    st.write(decision.get("abstract") or "No abstract available.")
    st.info(decision.get("judge_reason") or "No judge explanation provided.")
    if decision.get("error"):
        st.error(decision["error"])
    one, two = st.columns(2)
    if one.button("Ingest paper"):
        if _act(lambda: api.post(f"/inbox/{selected}/ingest"), success="Ingestion queued"):
            st.rerun()
    if two.button("Dismiss"):
        if _act(lambda: api.post(f"/inbox/{selected}/dismiss"), success="Decision dismissed"):
            st.rerun()
    with st.form("inbox-feedback"):
        feedback = st.text_input("Disagree with the curator? Explain why")
        disagree = st.form_submit_button("Update this topic's criteria")
    if disagree and feedback.strip():
        if _act(lambda: api.post(f"/inbox/{selected}/disagree", {"comment": feedback.strip()}),
                success="Criteria updated"):
            st.rerun()


def _jobs(api: AssistantAPI) -> None:
    st.title("⏳ Background jobs")
    if st.button("Refresh jobs"):
        st.rerun()
    jobs = api.get("/jobs")
    if not jobs:
        st.info("No jobs in this server session yet.")
    for job in jobs:
        with st.expander(f"{job['label']} · {job['status']}", expanded=job["status"] == "failed"):
            st.caption(f"{job['kind']} · created {job['created_at']}")
            if job.get("error"):
                st.error(job["error"])
            elif job.get("result"):
                st.json(job["result"])


def main() -> None:
    pending = st.session_state.pop("pending_chat_scope", None)
    if pending:
        scope, target = pending
        st.session_state["active_id"] = None
        st.session_state["scope"] = scope
        st.session_state[f"target_{scope}"] = target
        st.session_state["view"] = "Chat"  # before the sidebar widget is created
    with st.sidebar:
        st.title("Research Desk")
        st.caption("Local research assistant · Streamlit")
        view = st.radio("Workspace", ["Chat", "Library", "Topics", "Inbox", "Jobs"], key="view")
        st.caption("Local MVP: no multi-user authentication yet.")
    try:
        api = _client(os.getenv("ASSISTANT_API_URL", "http://127.0.0.1:8000"))
        status = api.get("/status")
        with st.sidebar:
            st.caption(f"{status['paper_count']} papers · {status['rag_mode']} RAG")
        {"Chat": _chat, "Library": _library, "Topics": _topics,
         "Inbox": _inbox, "Jobs": _jobs}[view](api)
    except (AssistantAPIError, ValueError) as exc:
        st.error(str(exc))
        st.info("Start FastAPI on 127.0.0.1:8000, then reload this page.")


if __name__ == "__main__":
    main()