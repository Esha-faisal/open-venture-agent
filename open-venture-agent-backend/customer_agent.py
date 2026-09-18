import os
import re
from typing import Literal, TypedDict
from urllib.parse import urlparse

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


class CustomerSegment(BaseModel):
    segment_name: str = Field(
        description="Specific customer group grounded in the search evidence."
    )
    evidence: str = Field(
        description="Short quote or paraphrase from the search results supporting this segment."
    )
    needs: list[str] = Field(
        description="Concrete needs for this segment that appear in the search results."
    )
    source: str = Field(description="URL from the search results backing this segment.")


class CustomerPainPoint(BaseModel):
    pain_point: str = Field(
        description="A specific customer complaint, frustration, or unmet need from the results."
    )
    source: str = Field(description="URL from the search results backing this pain point.")


class CustomerExtraction(BaseModel):
    customer_segments: list[CustomerSegment] = Field(default_factory=list)
    customer_pain_points: list[CustomerPainPoint] = Field(default_factory=list)
    no_local_data_reason: str | None = Field(
        default=None,
        description=(
            "Required when both lists are empty: short explanation that no "
            "location-specific customer evidence was found."
        ),
    )


class CustomerState(TypedDict, total=False):
    business_idea: str
    delivery_mode: str | None
    reviews_query: str
    needs_query: str
    reviews_search_results: str
    needs_search_results: str
    customer_segments: list[dict]
    customer_pain_points: list[dict]
    confidence: str
    no_local_data_reason: str | None


def _customer_confidence(source_count: int) -> Literal["strong", "weak", "none"]:
    if source_count >= 2:
        return "strong"
    if source_count == 1:
        return "weak"
    return "none"


_REMOTE_NO_CUSTOMER_REASON = (
    "Searched for general customer feedback about this type of product "
    "rather than local reviews; little usable customer voice turned up."
)


def _default_no_local_reason(
    location_label: str | None,
    *,
    delivery_mode: str | None = None,
) -> str:
    if (delivery_mode or "").lower() == "remote" or not location_label:
        return _REMOTE_NO_CUSTOMER_REASON
    return (
        f"No customer reviews or discussions specific to {location_label} were "
        "found in search results; only generic/non-local content was available."
    )


def _parse_business_idea(business_idea: str) -> tuple[str, str | None]:
    """Split 'shoe shop in Packages Mall, Lahore' into category + location."""
    match = re.search(r"\bin\s+(.+)$", business_idea, flags=re.IGNORECASE)
    if match:
        category = business_idea[: match.start()].strip() or business_idea
        location = match.group(1).strip()
        return category, location
    return business_idea, None


def _normalize_place(location: str) -> str:
    place = location.replace(",", " ")
    return re.sub(r"\s+", " ", place).strip()


def _build_reviews_query(business_idea: str) -> str:
    category, location = _parse_business_idea(business_idea)
    if location:
        place = _normalize_place(location)
        return f"{category} reviews complaints {place}"
    return f"{business_idea} reviews complaints feedback"


def _build_needs_query(business_idea: str) -> str:
    category, location = _parse_business_idea(business_idea)
    if location:
        place = _normalize_place(location)
        return f"what people look for buying {category} {place}"
    return f"what people look for buying {business_idea}"


def _normalize_source_key(source: str) -> str:
    """Normalize URLs so the same page counts once for confidence."""
    text = (source or "").strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        path = (parsed.path or "").rstrip("/")
        if host:
            return f"{host}{path}"
    except ValueError:
        pass
    return text.lower()


def search_reviews(state: CustomerState) -> dict:
    tool = TavilySearch(max_results=5)
    query = _build_reviews_query(state["business_idea"])
    print(f"[customer] reviews search query: {query}")
    result = tool.invoke({"query": query})
    return {
        "reviews_query": query,
        "reviews_search_results": str(result),
    }


def search_needs(state: CustomerState) -> dict:
    tool = TavilySearch(max_results=5)
    query = _build_needs_query(state["business_idea"])
    print(f"[customer] needs search query: {query}")
    result = tool.invoke({"query": query})
    return {
        "needs_query": query,
        "needs_search_results": str(result),
    }


def extract_customer_insights(state: CustomerState) -> dict:
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(CustomerExtraction)

    _, location = _parse_business_idea(state["business_idea"])
    delivery_mode = (state.get("delivery_mode") or "").lower() or None
    is_remote = delivery_mode == "remote" or not location

    if is_remote:
        location_hint = "general / non-local product or service feedback"
        gate_block = (
            "EVIDENCE GATE (remote / no specific place named):\n"
            "- Prefer customer reviews, complaints, and needs about this type of "
            "product or service in general — not a specific city.\n"
            "- Extract segments and pain points only when the source clearly "
            "discusses buyers of this kind of offering.\n"
            "- When ZERO usable items are found, return empty lists and set "
            "no_local_data_reason to a short honest sentence like: "
            f"\"{_REMOTE_NO_CUSTOMER_REASON}\"\n"
            "- Do NOT say \"the stated location\" or invent a city.\n"
            "- When at least one item is usable, leave no_local_data_reason null.\n\n"
        )
    else:
        location_hint = location
        gate_block = (
            "LOCAL EVIDENCE GATE — mandatory:\n"
            f"- The target location is: {location_hint}.\n"
            "- ONLY extract a pain_point or customer_segment if the supporting source "
            "explicitly mentions this location (city/neighborhood/area name) OR clear "
            "local context (a named local business, local landmark, local market, or "
            "clearly area-specific forum/review thread).\n"
            "- If a result is only topically related (generic 'how to start a grocery "
            "store', national industry blogs, US/UK lifestyle articles, worldwide "
            "buyer personas) and does NOT name the actual area or local context, "
            "DO NOT extract it — and do not emit a segment/pain point whose only "
            "purpose is to explain that the source was irrelevant or off-location.\n"
            "- When ZERO items pass this gate, return empty customer_segments and "
            "empty customer_pain_points, set no_local_data_reason to a short honest "
            "sentence like: \"No customer reviews or discussions specific to "
            f"{location_hint} were found in search results; only generic/non-local "
            "content was available.\", and do not invent fillers.\n"
            "- When at least one item is local, leave no_local_data_reason null.\n\n"
        )

    prompt = (
        "You are a customer-research analyst. From the search results below, extract "
        "customer segments and pain points for this business idea.\n\n"
        f"{gate_block}"
        "OTHER RULES:\n"
        "- ONLY extract segments and pain points clearly backed by the search "
        "results (reviews, complaints, forums, local articles, customer comments).\n"
        "- Every segment must include concrete evidence text from the results and "
        "a source URL that appears in the results.\n"
        "- Every pain point must include a source URL from the results.\n"
        "- Do NOT invent generic personas such as 'young professionals', 'families', "
        "or 'middle-class buyers' unless the search results explicitly support them "
        "with evidence.\n"
        "- needs must be specific and grounded in the evidence, not marketing "
        "boilerplate.\n\n"
        f"Business idea: {state['business_idea']}\n"
        f"Delivery mode: {delivery_mode or 'unknown'}\n"
        f"Reviews query: {state.get('reviews_query', '')}\n"
        f"Needs query: {state.get('needs_query', '')}\n\n"
        f"Reviews / complaints search results:\n{state.get('reviews_search_results', '')}\n\n"
        f"Customer needs / preferences search results:\n{state.get('needs_search_results', '')}"
    )

    result = invoke_with_retry(
        lambda: structured_llm.invoke(prompt),
        label="customer-insights",
    )

    segments = [
        {
            "segment_name": item.segment_name,
            "evidence": item.evidence,
            "needs": list(item.needs),
            "source": item.source,
        }
        for item in result.customer_segments
    ]
    pain_points = [
        {"pain_point": item.pain_point, "source": item.source}
        for item in result.customer_pain_points
    ]

    if not segments and not pain_points:
        reason = (result.no_local_data_reason or "").strip() or _default_no_local_reason(
            location,
            delivery_mode=delivery_mode,
        )
        if is_remote and (
            "stated location" in reason.lower()
            or (location is None and "specific to" in reason.lower())
        ):
            reason = _REMOTE_NO_CUSTOMER_REASON
        return {
            "customer_segments": [],
            "customer_pain_points": [],
            "confidence": "none",
            "no_local_data_reason": reason,
        }

    source_keys = {
        _normalize_source_key(item["source"])
        for item in [*segments, *pain_points]
        if _normalize_source_key(item.get("source", ""))
    }

    return {
        "customer_segments": segments,
        "customer_pain_points": pain_points,
        "confidence": _customer_confidence(len(source_keys)),
        "no_local_data_reason": None,
    }


def build_customer_graph():
    graph = StateGraph(CustomerState)
    graph.add_node("search_reviews", search_reviews)
    graph.add_node("search_needs", search_needs)
    graph.add_node("extract", extract_customer_insights)
    graph.add_edge(START, "search_reviews")
    graph.add_edge("search_reviews", "search_needs")
    graph.add_edge("search_needs", "extract")
    graph.add_edge("extract", END)
    return graph.compile()


customer_agent = build_customer_graph()


def run_customer_research(
    business_idea: str,
    *,
    delivery_mode: str | None = None,
) -> dict:
    result = customer_agent.invoke(
        {
            "business_idea": business_idea,
            "delivery_mode": delivery_mode,
        }
    )
    return {
        "customer_segments": result.get("customer_segments") or [],
        "customer_pain_points": result.get("customer_pain_points") or [],
        "confidence": result.get("confidence") or "none",
        "no_local_data_reason": result.get("no_local_data_reason"),
    }


if __name__ == "__main__":
    import json

    idea = "shoe shop in Packages Mall, Lahore"
    print(f"Business idea: {idea}")
    print(f"Reviews query: {_build_reviews_query(idea)}")
    print(f"Needs query: {_build_needs_query(idea)}")

    output = run_customer_research(idea)

    print("\n=== confidence ===")
    print(output.get("confidence"))
    print("\n=== no_local_data_reason ===")
    print(output.get("no_local_data_reason"))

    print("\n=== customer_segments ===")
    segments = output.get("customer_segments") or []
    if not segments:
        print("(none)")
    else:
        for i, segment in enumerate(segments, start=1):
            print(f"\n[{i}] {segment.get('segment_name')}")
            print(f"    evidence: {segment.get('evidence')}")
            print(f"    needs: {segment.get('needs')}")
            print(f"    source: {segment.get('source')}")

    print("\n=== customer_pain_points ===")
    pain_points = output.get("customer_pain_points") or []
    if not pain_points:
        print("(none)")
    else:
        for i, item in enumerate(pain_points, start=1):
            print(f"\n[{i}] {item.get('pain_point')}")
            print(f"    source: {item.get('source')}")

    print("\n=== raw json ===")
    print(json.dumps(output, indent=2, ensure_ascii=False))
