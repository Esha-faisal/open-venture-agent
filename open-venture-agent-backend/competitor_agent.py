import os
import re
from difflib import SequenceMatcher
from typing import Literal, TypedDict
from urllib.parse import urlparse

from dotenv import load_dotenv
from gemini_config import GEMINI_MODEL
from gemini_retry import invoke_with_retry
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

FindingCategory = Literal[
    "competitor",
    "pricing_intelligence",
    "vendor_technology",
    "directory_source",
]

load_dotenv()

if not os.getenv("TAVILY_API_KEY"):
    raise RuntimeError("TAVILY_API_KEY is not set")
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY is not set")

# Only these hosts count as "same account" identities (profile handles).
_SOCIAL_DOMAINS = frozenset(
    {
        "tiktok.com",
        "instagram.com",
        "facebook.com",
        "fb.com",
        "youtube.com",
        "twitter.com",
        "x.com",
        "linkedin.com",
    }
)

_CONTENT_SEGMENTS = frozenset(
    {
        "reel",
        "reels",
        "p",
        "post",
        "posts",
        "video",
        "videos",
        "watch",
        "status",
        "share",
        "embed",
        "explore",
        "tag",
        "tags",
        "hashtag",
        "t",
        "s",
        "stories",
        "tv",
        "live",
        "about",
        "photos",
        "public",
        "feed",
    }
)

# High bar — only near-identical names (typos / minor suffix differences).
_NAME_SIMILARITY_THRESHOLD = 0.88


class CompetitorFinding(BaseModel):
    competitor_name: str | None = Field(
        default=None,
        description=(
            "Rival business name, pricing-site label, or vendor/tool name. "
            "Must be null when category is directory_source."
        ),
    )
    finding: str
    source: str
    category: FindingCategory = Field(
        description=(
            "competitor = sells same/overlapping products/services to the same "
            "customers in the same area; pricing_intelligence = source reports "
            "real observed local prices; vendor_technology = POS/software/tool "
            "providers; directory_source = listing/directory/aggregator or "
            "'best places' article that is not itself a business."
        )
    )


class CompetitorFindingsList(BaseModel):
    findings: list[CompetitorFinding] = Field(min_length=3, max_length=5)


class CompetitorState(TypedDict):
    business_idea: str
    search_results: str
    findings: list[dict]


def _normalize_host(netloc: str) -> str:
    host = netloc.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m."):
        host = host[2:]
    return host


def _is_social_domain(host: str) -> bool:
    return any(host == domain or host.endswith("." + domain) for domain in _SOCIAL_DOMAINS)


def _social_account_identity(source: str) -> tuple[str, str] | None:
    """Return (domain, handle) only for social profile URLs — not shared article pages."""
    if not source or not str(source).strip():
        return None
    try:
        parsed = urlparse(str(source).strip())
    except ValueError:
        return None

    host = _normalize_host(parsed.netloc or "")
    if not host or not _is_social_domain(host):
        return None

    domain_key = host
    for domain in _SOCIAL_DOMAINS:
        if host == domain or host.endswith("." + domain):
            domain_key = domain
            break

    parts = [p for p in (parsed.path or "").split("/") if p]
    if not parts:
        return None

    if parts[0].lower() in {"channel", "c", "user", "users", "in"} and len(parts) >= 2:
        handle = parts[1].lower().lstrip("@")
        return (domain_key, handle) if handle else None

    if parts[0].startswith("@"):
        handle = parts[0].lower().lstrip("@")
        return (domain_key, handle) if handle else None

    # Content URLs like /reel/<id> have no profile handle.
    if parts[0].lower() in _CONTENT_SEGMENTS:
        return None

    handle = parts[0].lower().lstrip("@")
    return (domain_key, handle) if handle else None


def _normalize_competitor_name(name: str) -> str:
    text = name.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _names_are_similar(left: str, right: str) -> bool:
    a = _normalize_competitor_name(left)
    b = _normalize_competitor_name(right)
    if not a or not b:
        return False
    if a == b:
        return True
    # One name fully contains the other (e.g. "Royal Beauty" vs "Royal Beauty Salon")
    if a in b or b in a:
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if len(shorter) >= 6 and len(shorter) / len(longer) >= 0.6:
            return True
    return SequenceMatcher(None, a, b).ratio() >= _NAME_SIMILARITY_THRESHOLD


def _is_true_competitor(item: dict) -> bool:
    """True rivals only; missing category treated as competitor for older payloads."""
    category = item.get("category")
    return category is None or category == "competitor"


def _dedupe_exact_findings(findings: list[dict]) -> list[dict]:
    """Drop exact duplicates with the same source URL and finding text."""
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for item in findings:
        key = (
            str(item.get("source") or "").strip(),
            str(item.get("finding") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _normalize_finding_fields(findings: list[dict]) -> list[dict]:
    """Enforce category-specific field rules after model dump."""
    normalized: list[dict] = []
    for item in findings:
        row = dict(item)
        if row.get("category") == "directory_source":
            row["competitor_name"] = None
        elif not str(row.get("competitor_name") or "").strip():
            row["competitor_name"] = None
        else:
            row["competitor_name"] = str(row["competitor_name"]).strip()
        normalized.append(row)
    return normalized


def _flag_possible_duplicates(findings: list[dict]) -> list[dict]:
    """Flag likely same-business pairs; never flag solely for sharing a source URL."""
    flagged = [dict(item) for item in findings]
    pairs: set[tuple[int, int]] = set()
    competitor_indices = [i for i, item in enumerate(flagged) if _is_true_competitor(item)]

    # 1) Same social profile handle (e.g. tiktok.com/@sania_komal)
    identity_to_indices: dict[tuple[str, str], list[int]] = {}
    for idx in competitor_indices:
        identity = _social_account_identity(str(flagged[idx].get("source") or ""))
        if identity is None:
            continue
        identity_to_indices.setdefault(identity, []).append(idx)
    for indices in identity_to_indices.values():
        if len(indices) < 2:
            continue
        for i, left in enumerate(indices):
            for right in indices[i + 1 :]:
                pairs.add((left, right))

    # 2) Highly similar rival names (independent of source URL)
    for ii, i in enumerate(competitor_indices):
        name_i = str(flagged[i].get("competitor_name") or "")
        for j in competitor_indices[ii + 1 :]:
            name_j = str(flagged[j].get("competitor_name") or "")
            if _names_are_similar(name_i, name_j):
                pairs.add((i, j))

    dup_names: dict[int, list[str]] = {}
    for i, j in pairs:
        name_j = flagged[j].get("competitor_name")
        name_i = flagged[i].get("competitor_name")
        if name_j:
            dup_names.setdefault(i, []).append(str(name_j))
        if name_i:
            dup_names.setdefault(j, []).append(str(name_i))

    for idx, names in dup_names.items():
        unique = list(dict.fromkeys(names))
        if unique:
            flagged[idx]["possible_duplicate_of"] = ", ".join(unique)

    return flagged


def search_competitors(state: CompetitorState) -> dict:
    tool = TavilySearch(max_results=5)
    business_idea = state["business_idea"]
    query = f"{business_idea} reviews pricing local named competitors"
    result = tool.invoke({"query": query})
    return {"search_results": str(result)}


def summarize_competitors(state: CompetitorState) -> dict:
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(CompetitorFindingsList)

    prompt = (
        "You are a competitive intelligence analyst. From the search results below, "
        "extract 3-5 specific entries relevant to this business idea.\n\n"
        "STRICT COMPETITOR CRITERIA — category \"competitor\" ONLY if the entity is an "
        "actual business that sells the same or overlapping products/services to the "
        "same customers in the same area (e.g. another grocery store, supermarket, "
        "cash & carry, bakery, gym). Topical relatedness is NOT enough.\n\n"
        "DIRECTORY / LISTING SOURCES — mandatory:\n"
        "- If a source is a business directory, listing aggregator, yellow-pages style "
        "site, or a \"best places to shop\" / \"top stores in X\" article that names "
        "many businesses but is NOT itself a retail/service rival (e.g. Lookup.pk, "
        "BusinessList.pk), do NOT invent a competitor_name for the directory site.\n"
        "- Classify those as category \"directory_source\" with competitor_name null. "
        "The finding may summarize useful listed rivals or note that the page is a "
        "directory — but the directory site itself is not a competitor.\n"
        "- If the directory page clearly names specific local rival businesses, prefer "
        "extracting those named businesses as separate category \"competitor\" entries "
        "(with their real business names) when the evidence supports it, instead of "
        "labeling the directory as a competitor.\n\n"
        "EXPLICITLY EXCLUDE from category \"competitor\" (keep them if useful, but "
        "categorize correctly):\n"
        "- software / POS / billing / inventory vendors and SaaS tools "
        "(category: vendor_technology)\n"
        "- sources that report real observed local prices / market rates "
        "(category: pricing_intelligence) — only when the source describes actual "
        "price data (e.g. observed bazaar rates), NOT mere shopping-guide fluff "
        "without concrete prices\n"
        "- business directories / listing aggregators / multi-business listicles "
        "(category: directory_source; competitor_name must be null)\n"
        "- generic how-to articles or news pieces without a named local rival "
        "business\n"
        "- tool and technology providers that sell TO businesses, not TO the same "
        "end customers (category: vendor_technology)\n\n"
        "Prioritize local and regional rival businesses near the stated location over "
        "well-known international chains — only include global brands as competitors "
        "if they actually operate in that specific area.\n\n"
        "For each entry, provide:\n"
        "- competitor_name: rival business name, pricing-site label, or vendor name; "
        "MUST be null for directory_source. This field does NOT imply a business "
        "rival unless category is \"competitor\".\n"
        "- finding: a concise insight (pricing, strengths, weaknesses, reviews, "
        "or why this source matters)\n"
        "- source: a URL from the search results supporting this finding\n"
        "- category: one of \"competitor\", \"pricing_intelligence\", "
        "\"vendor_technology\", \"directory_source\"\n\n"
        "Prefer true local competitors when available. If search results mainly "
        "surface directories, pricing sites, or vendors, include them with the "
        "correct category rather than mislabeling them as competitors.\n\n"
        f"Business idea: {state['business_idea']}\n\n"
        f"Search results:\n{state['search_results']}"
    )
    result = invoke_with_retry(
        lambda: structured_llm.invoke(prompt),
        label="competitor",
    )
    findings = _normalize_finding_fields(
        [item.model_dump() for item in result.findings]
    )
    findings = _dedupe_exact_findings(findings)
    return {"findings": _flag_possible_duplicates(findings)}


def build_competitor_graph():
    graph = StateGraph(CompetitorState)
    graph.add_node("search", search_competitors)
    graph.add_node("summarize", summarize_competitors)
    graph.add_edge(START, "search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


competitor_agent = build_competitor_graph()


def run_competitor_research(business_idea: str) -> list[dict]:
    result = competitor_agent.invoke({"business_idea": business_idea})
    return result["findings"]


if __name__ == "__main__":
    findings = run_competitor_research("gym in the middle of Jhelum")
    for item in findings:
        print(item)
