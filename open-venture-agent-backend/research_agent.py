import os
import re
from typing import Literal, TypedDict

from dotenv import load_dotenv
from gemini_config import GEMINI_MODEL
from gemini_retry import invoke_with_retry
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

if not os.getenv("TAVILY_API_KEY"):
    raise RuntimeError("TAVILY_API_KEY is not set")
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY is not set")


class Finding(BaseModel):
    finding: str
    source: str


class FindingsList(BaseModel):
    findings: list[Finding] = Field(min_length=3, max_length=5)


class HiringPosting(BaseModel):
    role: str = Field(
        description=(
            "Short job title / role name only, e.g. 'Event Manager', "
            "'Pastry Chef', 'Front Desk Associate'. No full sentence."
        )
    )
    detail: str = Field(
        description=(
            "Everything else about the posting: location, volume, employer "
            "context, and what the role focuses on — not repeating the role "
            "name as a full sentence subject."
        )
    )


class HiringExtraction(BaseModel):
    postings_found: list[HiringPosting] = Field(
        default_factory=list,
        description=(
            "Relevant job postings, each split into a concise role title and "
            "a separate detail string. Never blend role + detail into one "
            "sentence in either field."
        ),
    )
    employers: list[str] = Field(
        default_factory=list,
        description="Distinct employer or business names hiring for this category/location.",
    )


class ResearchState(TypedDict, total=False):
    business_idea: str
    search_results: str
    hiring_search_results: str
    hiring_query: str
    findings: list[dict]
    hiring_signal: dict


def _hiring_confidence(employer_count: int) -> Literal["strong", "weak", "none"]:
    if employer_count >= 2:
        return "strong"
    if employer_count == 1:
        return "weak"
    return "none"


def _parse_business_idea(business_idea: str) -> tuple[str, str | None]:
    """Split 'gym in Gulberg, Lahore' into category + location."""
    match = re.search(r"\bin\s+(.+)$", business_idea, flags=re.IGNORECASE)
    if match:
        category = business_idea[: match.start()].strip() or business_idea
        location = match.group(1).strip()
        return category, location
    return business_idea, None


def _build_market_queries(business_idea: str) -> list[str]:
    """Short, direct queries that favor local listings over industry reports."""
    category, location = _parse_business_idea(business_idea)
    if location:
        place = location.replace(",", " ")
        place = re.sub(r"\s+", " ", place).strip()
        # Plural-ish "near" query tends to surface maps/directories; second adds competition.
        return [
            f"{category}s near {place}",
            f"{category} {place} market competition demand",
        ]
    return [f"{business_idea} market competition demand"]


def _build_hiring_query(business_idea: str) -> str:
    """
    Natural-language hiring query (no Boolean OR).
    Boolean operators often confuse Tavily and return empty/irrelevant hits.
    """
    category, location = _parse_business_idea(business_idea)
    if location:
        place = location.replace(",", " ")
        place = re.sub(r"\s+", " ", place).strip()
        return f"{category} jobs hiring {place}"
    return f"{business_idea} jobs hiring"


def search_market(state: ResearchState) -> dict:
    tool = TavilySearch(max_results=5)
    queries = _build_market_queries(state["business_idea"])
    chunks: list[str] = []
    for query in queries:
        print(f"[research] market search query: {query}")
        result = tool.invoke({"query": query})
        chunks.append(f"Query: {query}\nResults: {result}")
    return {"search_results": "\n\n".join(chunks)}


def search_hiring(state: ResearchState) -> dict:
    tool = TavilySearch(max_results=8)
    query = _build_hiring_query(state["business_idea"])
    print(f"[research] hiring search query: {query}")
    result = tool.invoke({"query": query})
    return {
        "hiring_query": query,
        "hiring_search_results": str(result),
    }


def summarize_findings(state: ResearchState) -> dict:
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    findings_llm = llm.with_structured_output(FindingsList)
    hiring_llm = llm.with_structured_output(HiringExtraction)

    _, location = _parse_business_idea(state["business_idea"])
    location_hint = location or "the stated location in the business idea"

    findings_prompt = (
        "You are a market research analyst. From the search results below, "
        "produce 3-5 concise findings focused on the SPECIFIC location "
        f"({location_hint}). Prefer city/neighborhood/regional data over "
        "global industry statistics. Only use worldwide figures when no "
        "local or regional data appears in the results — and if you must, "
        "clearly label them as global, not local.\n"
        "If a search hit is clearly about a different city/region than "
        f"{location_hint}, omit it entirely — do not include a finding that "
        "only explains why that source is irrelevant.\n"
        "Each finding must include a short 'finding' insight and a 'source' URL "
        "taken from the search results.\n\n"
        "EVIDENCE DISCIPLINE — follow exactly:\n"
        "- State only what the source directly supports as fact.\n"
        "- Do not make causal leaps (e.g. presence of a business ≠ proof of "
        "demand; one review ≠ proof of a trend; a hiring post ≠ proof of "
        "market growth).\n"
        "- When a source suggests something but does not prove a bigger claim, "
        "keep the wording modest. Vary how you flag limits — do NOT reuse one "
        "fixed template (especially avoid repeating "
        "\"although the source alone does not establish…\" on nearly every "
        "finding). Sometimes fold the caveat into the sentence; sometimes use "
        "a different phrasing; and skip an explicit hedge when the finding is "
        "already carefully limited to what the source shows.\n\n"
        "Example — overclaimed (DO NOT write like this):\n"
        "\"A supermarket operates in this area, which indicates strong regional "
        "demand for retail shopping.\"\n\n"
        "Examples — appropriately modest (vary across findings; pick what fits):\n"
        "1. \"A supermarket is listed in this area, which shows retail presence "
        "locally — not proof of overall regional demand.\"\n"
        "2. \"Reviews praise the store's hours; that speaks to customer "
        "satisfaction for this shop, without saying the whole category is "
        "booming.\"\n"
        "3. \"Hiring posts for cashiers appear nearby, suggesting some local "
        "operators are staffing up, though that alone isn't a market-growth "
        "signal.\"\n"
        "4. \"One directory lists three bakeries in the neighborhood — useful "
        "as a snapshot of who is open, not a full competitive census.\"\n\n"
        f"Business idea: {state['business_idea']}\n\n"
        f"Search results:\n{state['search_results']}"
    )
    findings_result = invoke_with_retry(
        lambda: findings_llm.invoke(findings_prompt),
        label="research-findings",
    )

    hiring_prompt = (
        "You are analyzing job/hiring signals for a local business opportunity. "
        "From the hiring search results below, extract job postings related to "
        "this business category or nearby businesses of the same type "
        f"(location focus: {location_hint}). "
        "Include gym/fitness/hospitality/retail staff roles when relevant. "
        "Be inclusive: if a posting is in the same city/area and same industry, "
        "include it.\n\n"
        "For EVERY posting, return structured fields — never one blended sentence:\n"
        "- role: the job title / role name only (e.g. \"Event Manager\", "
        "\"Baker\", \"Fitness Trainer\").\n"
        "- detail: the rest — open-job counts, location, focus areas, employer "
        "context (e.g. \"120 open jobs in Lahore Cantonment focusing on event "
        "planning, budgeting, and sponsorship development.\").\n"
        "Do not put the full description into role. Do not start detail with "
        "\"The Event Manager role…\" — detail should read as the supporting "
        "facts after the title.\n"
        "Also list distinct employer names. Only return empty lists if the "
        "results contain no job-related content at all.\n\n"
        f"Business idea: {state['business_idea']}\n"
        f"Search query used: {state.get('hiring_query', '')}\n\n"
        f"Hiring search results:\n{state.get('hiring_search_results', '')}"
    )
    hiring_result = invoke_with_retry(
        lambda: hiring_llm.invoke(hiring_prompt),
        label="research-hiring",
    )

    employers = list(
        dict.fromkeys(name.strip() for name in hiring_result.employers if name.strip())
    )
    employer_count = len(employers)
    postings_found = []
    for item in hiring_result.postings_found:
        role = (item.role or "").strip()
        detail = (item.detail or "").strip()
        if not role and not detail:
            continue
        postings_found.append({"role": role, "detail": detail})
    hiring_signal = {
        "postings_found": postings_found,
        "employer_count": employer_count,
        "confidence": _hiring_confidence(employer_count),
        "query_used": state.get("hiring_query", ""),
    }

    return {
        "findings": [
            {"finding": item.finding, "source": item.source}
            for item in findings_result.findings
        ],
        "hiring_signal": hiring_signal,
    }


def build_research_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("search", search_market)
    graph.add_node("search_hiring", search_hiring)
    graph.add_node("summarize", summarize_findings)
    graph.add_edge(START, "search")
    graph.add_edge("search", "search_hiring")
    graph.add_edge("search_hiring", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


research_agent = build_research_graph()


def run_research(business_idea: str) -> dict:
    result = research_agent.invoke({"business_idea": business_idea})
    # Research findings are finding + source only — never competitor_name.
    findings = []
    for item in result.get("findings") or []:
        findings.append(
            {
                "finding": item["finding"],
                "source": item["source"],
            }
        )
    return {
        "findings": findings,
        "hiring_signal": result.get("hiring_signal")
        or {
            "postings_found": [],
            "employer_count": 0,
            "confidence": "none",
            "query_used": "",
        },
    }


if __name__ == "__main__":
    idea = "gym in Gulberg, Lahore"
    print("Market queries:", _build_market_queries(idea))
    print("Hiring query:", _build_hiring_query(idea))
    result = run_research(idea)
    print("Findings:")
    for item in result["findings"]:
        print(item)
    print("Hiring signal:")
    print(result["hiring_signal"])
