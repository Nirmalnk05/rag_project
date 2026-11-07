"""Agent tools bound to the loaded Retriever + chunk store.

The registry exposes provider-neutral ToolSpecs and executes tool calls. It keeps
a running list of retrieved sources across calls so `[#n]` citations in the final
answer map to real chunks (fed to verify_citations).

`search` goes through hybrid_retrieve and inherits every retrieval gate. The
direct-fetch tools read the chunk store by name and inherit nothing, so they
run the same checks themselves — see `ToolRegistry._visible`.
"""
from __future__ import annotations

import logging

from src.config import Settings, get_settings
from src.guardrails import screen_chunks
from src.observability import log_event
from src.providers.base import ToolSpec

_logger = logging.getLogger("rag.agent.tools")


def _coerce_top_k(value) -> int | None:
    """Tool arguments come from the model's tool call, not a type-checked
    caller — the declared JSON-schema type ("integer") isn't guaranteed at
    runtime, and local/smaller tool-calling models in particular sometimes
    pass e.g. top_k="5" instead of an int. Coerce leniently rather than let
    list slicing raise; ignore anything that can't be parsed as a positive
    int so search just returns unclipped results instead of erroring."""
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


TOOL_SPECS = [
    ToolSpec(
        name="search",
        description="Search the maintenance knowledge base with hybrid retrieval. "
                    "Returns numbered source snippets. Use for any factual lookup.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search query"},
                "filters": {"type": "object", "description":
                            "optional exact-match filters: source_file, doc_title, "
                            "section, element_type (text|table|figure), page"},
                "top_k": {"type": "integer", "description": "max results (default 5)"},
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="lookup_fault_code",
        description="Look up a fault/alarm/error code in the tabular fault matrices. "
                    "Searches only table content.",
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string", "description": "the fault code, e.g. 4F, E-24"}},
            "required": ["code"],
        },
    ),
    ToolSpec(
        name="get_document_section",
        description="Fetch the full text of a named section of a document when a "
                    "search snippet is not enough.",
        parameters={
            "type": "object",
            "properties": {
                "source_file": {"type": "string"},
                "section": {"type": "string"},
            },
            "required": ["source_file", "section"],
        },
    ),
    ToolSpec(
        name="list_documents",
        description="List the available documents (title and file) to orient a search.",
        parameters={"type": "object", "properties": {}},
    ),
]

CMMS_TOOL_SPEC = ToolSpec(
    name="query_cmms",
    description="Look up live asset status and open work orders from the CMMS "
                "(maintenance system of record) for an asset id, e.g. P-100.",
    parameters={
        "type": "object",
        "properties": {"asset_id": {"type": "string", "description": "asset tag, e.g. P-100"}},
        "required": ["asset_id"],
    },
)


class ToolRegistry:
    def __init__(self, retriever, settings: Settings | None = None, access=None,
                 cmms=None) -> None:
        self._r = retriever
        self._s = settings or get_settings()
        self._access = access
        self.sources: list[dict] = []   # accumulated across the agent run
        self._cmms = cmms
        if self._cmms is None and self._s.enable_cmms:
            from src.agent.cmms import get_cmms_connector
            self._cmms = get_cmms_connector(self._s)

    @property
    def specs(self) -> list[ToolSpec]:
        specs = list(TOOL_SPECS)
        if self._cmms is not None:
            specs.append(CMMS_TOOL_SPEC)
        return specs

    @property
    def num_sources(self) -> int:
        return len(self.sources)

    async def execute(self, name: str, args: dict) -> str:
        log_event(_logger, "tool_call", tool=name, args=args)
        try:
            if name == "search":
                return await self._search(args.get("query", ""),
                                          args.get("filters"), args.get("top_k"))
            if name == "lookup_fault_code":
                return await self._search(args.get("code", ""),
                                          {"element_type": "table"}, None)
            if name == "get_document_section":
                return self._get_section(args.get("source_file", ""), args.get("section", ""))
            if name == "list_documents":
                return self._list_documents()
            if name == "query_cmms":
                return self._query_cmms(args.get("asset_id", ""))
            return f"Error: unknown tool '{name}'"
        except Exception as e:  # noqa: BLE001 — return error to the model, don't crash the loop
            _logger.warning("tool_failed", exc_info=True)
            return f"Tool error: {e}"

    # ---------- implementations ----------

    async def _search(self, query: str, filters: dict | None, top_k) -> str:
        if not query.strip():
            return "Error: empty query."
        results = await self._r.hybrid_retrieve(query, filters or None, self._access)
        clean = self._screened([r.to_dict() for r in results])
        top_k = _coerce_top_k(top_k)
        if top_k:
            clean = clean[:top_k]
        if not clean:
            return "No results found."
        return self._format_and_record(clean)

    def _visible(self, chunks: list[dict]) -> list[dict]:
        """Every gate a direct read of the chunk store would otherwise skip:
        ACL, freshness, and live metadata while this process is behind the index.

        Content screening is NOT here: it belongs to the paths that return chunk
        text, and a flagged chunk has to be distinguishable from a missing one.
        """
        from src.access import acl_visible
        from src.retrieval import is_fresh

        live: dict[str, dict] = {}
        moved = getattr(self._r, "index_has_moved", None)
        if callable(moved) and moved():
            live = self._r.live_metadata([c.get("id", "") for c in chunks])
            # the snapshot's labels are not authoritative while it is behind:
            # a chunk the index no longer holds is dropped, not served
            chunks = [c for c in chunks if c.get("id") in live]

        out = []
        for c in chunks:
            meta = live.get(c.get("id", ""), c.get("metadata", {}))
            if not acl_visible(meta, self._access):
                continue
            if not is_fresh(meta, self._s.retrieve_superseded):
                continue
            out.append(c)
        return out

    def _screened(self, chunks: list[dict]) -> list[dict]:
        clean, flagged = screen_chunks(chunks)
        if flagged:
            log_event(_logger, "chunk_flagged_in_tool", count=len(flagged))
        return clean

    def _get_section(self, source_file: str, section: str) -> str:
        matched = self._visible([c for c in self._chunks()
                                 if c.get("metadata", {}).get("source_file") == source_file
                                 and c.get("section") == section])
        if not matched:
            return f"No section '{section}' found in {source_file}."
        clean = self._screened(matched)
        if not clean:
            # "not found" would send the agent hunting for another name
            return (f"Section '{section}' in {source_file} was withheld: its "
                    "content was flagged by prompt-injection screening.")
        return self._format_and_record(clean)

    def _list_documents(self) -> str:
        seen = {}
        for c in self._visible(self._chunks()):
            sf = c.get("metadata", {}).get("source_file", "?")
            seen[sf] = c.get("doc_title", "?")
        if not seen:
            return "No documents are available to you."
        return "\n".join(f"- {title} ({sf})" for sf, title in sorted(seen.items()))

    def _query_cmms(self, asset_id: str) -> str:
        if self._cmms is None:
            return "CMMS is not enabled."
        asset = self._cmms.get_asset(asset_id)
        if not asset:
            return f"No asset '{asset_id}' found in the CMMS."
        wos = self._cmms.get_work_orders(asset_id)
        lines = [f"Asset {asset_id}: {asset.get('name','?')} — status "
                 f"{asset.get('status','?')}, last service {asset.get('last_service','?')} "
                 f"({asset.get('hours_since_service','?')}h ago)."]
        if wos:
            lines.append("Open work orders:")
            lines += [f"  - {w['id']} ({w['type']}): {w['summary']}" for w in wos]
        else:
            lines.append("No open work orders.")
        return "\n".join(lines)

    # ---------- helpers ----------

    def _chunks(self) -> list[dict]:
        return getattr(self._r, "_chunks", [])

    @staticmethod
    def _rev_tag(meta: dict) -> str:
        ver, date = meta.get("doc_version"), meta.get("doc_date")
        if ver and date:
            return f", rev {ver} {date}"
        if ver:
            return f", rev {ver}"
        if date:
            return f", {date}"
        return ""

    def _format_and_record(self, chunks: list[dict]) -> str:
        lines = []
        for c in chunks:
            n = len(self.sources) + 1
            self.sources.append(c)
            rev = self._rev_tag(c.get("metadata", {}))
            lines.append(f"[#{n}] ({c.get('doc_title','?')} > {c.get('section','?')}{rev})\n{c['text']}")
        out = "\n\n".join(lines)
        limit = self._s.agent_max_tool_results_chars
        return out[:limit] + ("\n…(truncated)" if len(out) > limit else "")
