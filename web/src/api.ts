export type Status = {
  paper_count: number;
  rag_mode: string;
  roles: Record<string, { provider: string; model: string }>;
};
export type Source = {
  chunk_id: string;
  paper_id: string;
  title: string;
  arxiv_id?: string;
  section: string;
  text: string;
  page_number?: number | null;
  arxiv_url?: string;
  pdf_url?: string;
};
export type MetadataFilters = {
  min_year?: number;
  max_year?: number;
  venue?: string;
  section_types?: string[];
};
export type Turn = {
  id?: number;
  question: string;
  answer: string;
  citations: string[];
  sources: Source[];
  confidence?: number;
  feedback?: number;
};
export type Conversation = {
  id: string;
  title: string;
  scope_type: "library" | "topic" | "paper";
  scope_target?: string;
  turns?: Turn[];
};
export type Paper = {
  id: string;
  title: string;
  authors: string[];
  abstract?: string;
  year?: number;
  arxiv_id?: string;
  domain?: string;
  accepted_score?: number;
  has_pdf: boolean;
  reading_status: "new" | "reviewing" | "read";
  is_favorite: boolean;
};
export type Topic = {
  name: string;
  arxiv_categories: string[];
  venues: string[];
  paper_count: number;
  monitored: boolean;
  synced: boolean;
  criteria?: {
    version: number;
    structured_rules: Record<string, unknown>;
    nl_addendum: string;
  };
};
export type Decision = {
  id: number;
  arxiv_id: string;
  title: string;
  authors: string[];
  abstract: string;
  domain: string;
  score: number;
  judge_reason: string;
  status: string;
  error?: string;
};

const writeHeaders = {
  "Content-Type": "application/json",
  "X-Requested-With": "assistant-ui",
};

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init);
  if (!response.ok) {
    const detail = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail || response.statusText);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const write = <T>(path: string, method: string, body?: unknown) =>
  api<T>(path, {
    method,
    headers: writeHeaders,
    body: body === undefined ? undefined : JSON.stringify(body),
  });

export async function uploadPdf(file: File, domain?: string): Promise<unknown> {
  const body = new FormData();
  body.append("file", file);
  if (domain) body.append("domain", domain);
  return api("/papers/upload", {
    method: "POST",
    headers: { "X-Requested-With": "assistant-ui" },
    body,
  });
}

export async function streamMessage(
  conversationId: string,
  message: string,
  onEvent: (event: Record<string, unknown>) => void,
  filters?: MetadataFilters,
): Promise<void> {
  const response = await fetch(
    `/api/conversations/${conversationId}/messages`,
    {
      method: "POST",
      headers: writeHeaders,
      body: JSON.stringify({ message, filters: filters ?? {} }),
    },
  );
  if (!response.ok || !response.body) throw new Error(await response.text());
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) if (line.trim()) onEvent(JSON.parse(line));
    if (done) break;
  }
}
