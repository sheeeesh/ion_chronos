# tools/web_search.py
"""
Simple web search tool using DDGS (DuckDuckGo) with optional LLM summarization.
No API key required.

Usage from the agent:
    web_search("your query")
    web_search("your query", llm=some_llm, max_results=5)

Notes
- Prefers the modern `ddgs` package. Falls back to `duckduckgo_search` if present.
- Returns readable Markdown; if an LLM is supplied it will produce a short summary
  and then list sources.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any

_BACKEND: Optional[str] = None
_IMPORT_ERR: Optional[Exception] = None

# Prefer modern package
try:
    from ddgs import DDGS  # pip install ddgs
    _BACKEND = "ddgs"
except Exception as e1:  # Fallback to legacy package (still common in the wild)
    try:
        from ddgs import DDGS  # pip install duckduckgo_search
        _BACKEND = "duckduckgo_search"
    except Exception as e2:
        _IMPORT_ERR = e1 if e1 is not None else e2
        DDGS = None  # type: ignore


def _sanitize_int(x: Any, default: int, lo: int, hi: int) -> int:
    try:
        v = int(x)
    except Exception:
        return default
    return max(lo, min(hi, v))


def _format_results_md(query: str, results: List[Dict[str, Any]]) -> str:
    lines: List[str] = [f"**Web search for:** `{query}`", ""]
    for i, r in enumerate(results, 1):
        title = r.get("title") or "Untitled"
        snippet = r.get("body") or r.get("snippet") or ""
        url = r.get("href") or r.get("url") or ""
        if url:
            lines.append(f"{i}. [{title}]({url})")
        else:
            lines.append(f"{i}. {title}")
        if snippet:
            lines.append(f"   - {snippet}")
    return "\n".join(lines)


def web_search(
    query: str,
    llm: Optional[Any] = None,
    max_results: int = 5,
    *,
    region: str = "wt-wt",              # worldwide
    safesearch: str = "moderate",       # "off" | "moderate" | "strict"
    timelimit: Optional[str] = None,    # "d", "w", "m", "y" (legacy supports "d","w","m","y")
) -> str:
    """
    Run a DuckDuckGo-style search and return a concise Markdown render of results.
    If `llm` is provided, returns a short bullet summary followed by sources.

    Args
    ----
    query: search string
    llm:   optional LLM with `.invoke(prompt)` that returns an object with `.content`
    max_results: number of results to fetch (1..20)
    region: DDG region code (e.g., "us-en", "uk-en", "wt-wt")
    safesearch: "off" | "moderate" | "strict"
    timelimit: optional time filter (e.g., "d","w","m","y")

    Returns
    -------
    Markdown string with either a summary + sources or just sources.
    """
    query = (query or "").strip()
    if not query:
        return "Usage: `web_search(\"your query\", max_results=5)`"

    if DDGS is None or _BACKEND is None:
        err = (
            "[web_search] Could not import a DDG search backend.\n\n"
            "Install one of the following:\n"
            "  • `pip install ddgs`  (preferred)\n"
            "  • `pip install duckduckgo_search`  (legacy)\n"
        )
        if _IMPORT_ERR:
            err += f"\nUnderlying import error: `{_IMPORT_ERR!r}`"
        return err

    max_results = _sanitize_int(max_results, default=5, lo=1, hi=20)

    # Fetch results
    try:
        # You can pass a custom UA via headers if needed
        # with DDGS(headers={"User-Agent": "IonChronos/1.0"}) as ddgs:
        with DDGS() as ddgs:
            # Signatures differ slightly between packages, but these kwargs
            # are supported by both recent versions.
            results: List[Dict[str, Any]] = list(
                ddgs.text(
                    query,
                    max_results=max_results,
                    region=region,
                    safesearch=safesearch,
                    timelimit=timelimit,
                )
            )
    except Exception as e:
        return f"[web_search] error: {e}"

    if not results:
        return f"[web_search] no results for: `{query}`"

    # Plain Markdown if no LLM
    sources_md = _format_results_md(query, results)

    if llm is None:
        # Nudge if running on legacy package
        if _BACKEND == "duckduckgo_search":
            return (
                sources_md
                + "\n\n> Note: using legacy `duckduckgo_search`. "
                + "For best results, consider `pip install ddgs`."
            )
        return sources_md

    # Summarize with LLM (short, cautious)
    try:
        joined = _format_results_md(query, results)
        prompt = (
            "Summarize the key points from these search results in 5–8 concise bullet points. "
            "Be neutral and note any disagreements. Then list 3–6 sources by title with links.\n\n"
            + joined
        )
        summary_obj = llm.invoke(prompt)
        summary_text = getattr(summary_obj, "content", str(summary_obj))
        return f"### Summary\n{summary_text}\n\n### Sources\n{joined}"
    except Exception:
        # If the LLM call fails for any reason, fall back to sources list
        return sources_md
