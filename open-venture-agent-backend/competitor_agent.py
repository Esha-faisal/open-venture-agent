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
from text_sanitize import sanitize_display_text

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
    operates_in_target_location: bool = Field(
        description=(
            "True ONLY when the source content clearly confirms relevance to "
            "the specific target city/area from the business idea. Cross-city "
            "name matches (same brand/name in a different city) must be false. "
            "If the source does not clearly confirm the target location, set "
            "false — do not assume branches or related outlets. For "
            "location-agnostic vendor tooling only, true is allowed. False "
            "entries must NOT appear in the findings list at all — omit them "
            "entirely rather than explaining why they were rejected."
        )
    )


class CompetitorFindingsList(BaseModel):
    findings: list[CompetitorFinding] = Field(
        min_length=0,
        max_length=5,
        description=(
            "0-5 geographically relevant entries only. Omit cross-city or "
            "unconfirmed-location sources entirely — never pad with rejected "
            "items or explanations of what was filtered out."
        ),
    )


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
            row["competitor_name"] = sanitize_display_text(str(row["competitor_name"]))
        row["finding"] = sanitize_display_text(str(row.get("finding") or ""))
        normalized.append(row)
    return normalized


def _filter_geographic_mismatches(findings: list[dict]) -> list[dict]:
    """Drop every finding that failed geographic relevance — no explanations kept.

    The LLM sets operates_in_target_location during extraction. Anything false
    or missing is omitted entirely so the frontend never sees rejected
    cross-city / off-target sources. Empty results are intentional.
    """
    kept: list[dict] = []
    for item in findings:
        if not item.get("operates_in_target_location"):
            continue
        row = dict(item)
        row.pop("operates_in_target_location", None)
        kept.append(row)
    return kept


def _looks_like_rejection_note(finding: str) -> bool:
    """True when the finding mainly explains why a source was irrelevant."""
    text = finding.lower().strip()
    if not text:
        return True
    markers = (
        "not relevant",
        "not located in",
        "not in the target",
        "different city",
        "another city",
        "outside the target",
        "does not operate in",
        "doesn't operate in",
        "only operates in",
        "unrelated to the target",
        "wrong city",
        "cross-city",
        "excluded because",
        "rejected because",
        "should not be counted",
        "cannot confirm it operates",
        "no evidence it operates in",
    )
    return any(marker in text for marker in markers)


def _filter_rejection_notes(findings: list[dict]) -> list[dict]:
    """Strip leftover 'why we rejected this' writeups if the model still emits them."""
    return [
        item
        for item in findings
        if not _looks_like_rejection_note(str(item.get("finding") or ""))
    ]


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


def _parse_business_idea(business_idea: str) -> tuple[str, str | None]:
    """Split 'gym in Gulberg, Lahore' into category + location."""
    match = re.search(r"\bin\s+(.+)$", business_idea, flags=re.IGNORECASE)
    if match:
        category = business_idea[: match.start()].strip() or business_idea
        location = match.group(1).strip()
        return category, location
    return business_idea, None


_QUERY_FILLER = re.compile(
    r"\b("
    r"i\s+want\s+to|i\s+wanna|i'?d\s+like\s+to|looking\s+to|want\s+to|wanna|"
    r"start|open|build|launch|sell|selling|run|running|create|creating|"
    r"an?|the|my|our|online|remotely|remote|digital|ecommerce|e-commerce|"
    r"business|shop|store|venture|idea|company"
    r")\b",
    flags=re.IGNORECASE,
)


def _niche_for_query(business_idea: str) -> tuple[str, str]:
    """Return (niche, place_hint) for competitor search queries."""
    category, location = _parse_business_idea(business_idea)
    niche = _QUERY_FILLER.sub(" ", category or business_idea)
    niche = re.sub(r"[^\w\s&/+-]", " ", niche)
    niche = re.sub(r"\s+", " ", niche).strip(" -/") or business_idea.strip()
    place = ""
    if location:
        place = re.sub(r"\s+", " ", location.replace(",", " ")).strip()
    return niche, place


def _build_competitor_queries(business_idea: str) -> list[str]:
    """General web search plus social-platform query for informal sellers."""
    niche, place = _niche_for_query(business_idea)
    social_core = f"{niche} {place}".strip() if place else niche
    return [
        f"{business_idea} reviews pricing local named competitors",
        (
            f"{social_core} site:instagram.com OR site:facebook.com "
            f"OR site:tiktok.com"
        ),
    ]


def search_competitors(state: CompetitorState) -> dict:
    tool = TavilySearch(max_results=5)
    business_idea = state["business_idea"]
    queries = _build_competitor_queries(business_idea)
    chunks: list[str] = []
    for query in queries:
        print(f"[competitor] search query: {query}")
        result = tool.invoke({"query": query})
        chunks.append(f"Query: {query}\nResults: {result}")
    return {"search_results": "\n\n".join(chunks)}


def summarize_competitors(state: CompetitorState) -> dict:
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(CompetitorFindingsList)

    prompt = (
        "You are a competitive intelligence analyst. From the search results below, "
        "extract 3-5 specific entries relevant to this business idea.\n\n"
        "First, infer the TARGET LOCATION (city and/or neighborhood/area) from the "
        "business idea. Every local rival you keep must be verified against that "
        "specific place — not a different city that happens to share a business name.\n\n"
        "STRICT COMPETITOR CRITERIA — category \"competitor\" ONLY if the entity is an "
        "actual business that sells the same or overlapping products/services to the "
        "same customers in the same area (e.g. another grocery store, supermarket, "
        "cash & carry, bakery, gym). Topical relatedness is NOT enough.\n\n"
        "GEOGRAPHIC VALIDATION — mandatory for every entry you keep:\n"
        "- Before including anything, check that the SOURCE CONTENT clearly "
        "confirms it is about the target city/area from the business idea — "
        "not merely that a name matches something you searched.\n"
        "- If a source is about a different city/region (cross-city name "
        "match), or does not clearly place the business in the target "
        "location: OMIT IT ENTIRELY from findings. Do not include it under "
        "any category. Do not write a finding that explains why it is "
        "irrelevant, why it was rejected, or that it is \"in Lahore not "
        "Islamabad\" etc. The output list must contain ONLY kept, relevant "
        "results — never a padded list of rejections.\n"
        "- Example: idea in Islamabad, source only describes \"Fasana Cafe\" "
        "in Lahore → do not emit that entry at all.\n"
        "- Same-name chains only count when the source explicitly places that "
        "outlet in the target city/area.\n"
        "- Set operates_in_target_location=true only for entries you keep. "
        "Never emit an entry with operates_in_target_location=false.\n"
        "- Mention the confirmed location in the finding text when you keep a "
        "rival.\n"
        "- Returning 0 findings is acceptable when nothing is geographically "
        "relevant.\n\n"
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
        "(with their real business names) when the evidence supports it AND the source "
        "places them in the target location, instead of labeling the directory as a "
        "competitor.\n\n"
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
        "end customers (category: vendor_technology)\n"
        "- any named business whose source only confirms operations in a different "
        "city/area than the target location\n\n"
        "Prioritize local and regional rival businesses near the stated location over "
        "well-known international chains — only include global brands as competitors "
        "if they actually operate in that specific area (source must say so).\n\n"
        "SOCIAL / INFORMAL SELLERS — treat seriously:\n"
        "- Instagram, Facebook, and TikTok shop/seller pages can be real rivals for "
        "handmade, home-based, or informal niches. If a social profile clearly sells "
        "overlapping products to the same market (city/country), classify it as "
        "category \"competitor\" with the shop/page name.\n"
        "- Do not dismiss social sellers just because they lack a traditional "
        "website. Prefer naming the seller/page when the source supports it.\n\n"
        "For each entry, provide:\n"
        "- competitor_name: rival business name, pricing-site label, or vendor name; "
        "MUST be null for directory_source. This field does NOT imply a business "
        "rival unless category is \"competitor\".\n"
        "- finding: a concise insight (pricing, strengths, weaknesses, reviews, "
        "or why this source matters); for competitors, include where they operate "
        "when the source states it\n"
        "- source: a URL from the search results supporting this finding\n"
        "- category: one of \"competitor\", \"pricing_intelligence\", "
        "\"vendor_technology\", \"directory_source\"\n"
        "- operates_in_target_location: boolean geographic check described above\n\n"
        "Prefer true local competitors when available. If search results mainly "
        "surface directories, pricing sites, or vendors that are relevant to the "
        "target area, include them with the correct category rather than "
        "mislabeling them as competitors. Never pad the list with off-target "
        "or rejected sources. Fewer (or zero) relevant findings is better than "
        "including cross-city name matches or \"here's what we rejected\" notes.\n\n"
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
    findings = _filter_geographic_mismatches(findings)
    findings = _filter_rejection_notes(findings)
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
