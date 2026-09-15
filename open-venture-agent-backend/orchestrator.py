import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from competitor_agent import run_competitor_research
from customer_agent import run_customer_research
from gemini_config import GEMINI_MODEL
from gemini_retry import invoke_with_retry
from location_agent import run_location_analysis
from research_agent import run_research
from strategy_agent import run_strategy
from task_planner import generate_task_plan
from venture_report import generate_final_report

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY is not set")

DeliveryMode = Literal["remote", "onsite", "hybrid"]

_PARALLEL_RESEARCH_NODES = (
    "research_node",
    "competitor_node",
    "customer_node",
)


class DeliveryModeResult(BaseModel):
    delivery_mode: DeliveryMode = Field(
        description=(
            "How the business primarily delivers value: "
            "'remote' for fully online/digital with no physical premises, "
            "'onsite' for a physical location customers visit, "
            "'hybrid' for both physical and remote delivery."
        )
    )
    rationale: str = Field(
        description="Brief reason for choosing this delivery mode."
    )


class VentureOrchestratorState(TypedDict, total=False):
    business_idea: str
    delivery_mode: DeliveryMode
    delivery_mode_rationale: str
    research_findings: list[dict]
    hiring_signal: dict
    competitor_findings: list[dict]
    customer_segments: list[dict]
    customer_pain_points: list[dict]
    customer_confidence: str
    no_local_data_reason: str | None
    location_data: dict | None
    strategy: dict
    task_plan: list[dict]
    final_report: dict


def infer_delivery_mode(state: VentureOrchestratorState) -> dict:
    """Infer delivery_mode and write it into shared graph state."""
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(DeliveryModeResult)

    prompt = (
        "Classify how this business idea primarily delivers value to customers.\n"
        "Choose exactly one of: remote, onsite, hybrid.\n"
        "- remote: fully online/digital, no physical premises needed "
        "(e.g. SaaS, online tutoring, e-commerce dropshipping).\n"
        "- onsite: customers visit a physical location "
        "(e.g. cafe, gym, restaurant, retail shop).\n"
        "- hybrid: meaningful mix of physical and remote delivery "
        "(e.g. clinic with telehealth, bakery with local delivery + storefront).\n\n"
        f"Business idea: {state['business_idea']}"
    )
    result = invoke_with_retry(
        lambda: structured_llm.invoke(prompt),
        label="delivery-mode",
    )
    return {
        "delivery_mode": result.delivery_mode,
        "delivery_mode_rationale": result.rationale,
    }


def route_after_delivery_mode(
    state: VentureOrchestratorState,
) -> Literal["location"] | list[str]:
    """Gate Location Agent; remote fans out directly to parallel research nodes."""
    if state.get("delivery_mode") in {"onsite", "hybrid"}:
        return "location"
    return list(_PARALLEL_RESEARCH_NODES)


def run_location(state: VentureOrchestratorState) -> dict:
    """Run Location Agent unchanged; only invoked when routed here."""
    analysis = run_location_analysis(state["business_idea"])
    return {"location_data": analysis}


def research_node(state: VentureOrchestratorState) -> dict:
    """Market research branch — only writes research keys."""
    research = run_research(state["business_idea"])
    return {
        "research_findings": list(research.get("findings") or []),
        "hiring_signal": research.get("hiring_signal")
        or {"postings_found": [], "employer_count": 0, "confidence": "none"},
    }


def competitor_node(state: VentureOrchestratorState) -> dict:
    """Competitor branch — only writes competitor_findings."""
    return {
        "competitor_findings": list(run_competitor_research(state["business_idea"]) or []),
    }


def customer_node(state: VentureOrchestratorState) -> dict:
    """Customer branch — only writes customer research keys."""
    customer = run_customer_research(state["business_idea"])
    return {
        "customer_segments": customer.get("customer_segments") or [],
        "customer_pain_points": customer.get("customer_pain_points") or [],
        "customer_confidence": customer.get("confidence") or "none",
        "no_local_data_reason": customer.get("no_local_data_reason"),
    }


def merge_research(state: VentureOrchestratorState) -> dict:
    """Fan-in: dedupe research against competitors after all parallel branches finish."""
    research_findings = list(state.get("research_findings") or [])
    competitor_findings = list(state.get("competitor_findings") or [])

    competitor_sources = {
        str(item.get("source") or "")
        for item in competitor_findings
        if item.get("source")
    }
    deduped_research = [
        item
        for item in research_findings
        if str(item.get("source") or "") not in competitor_sources
    ]
    removed_by_url = len(research_findings) - len(deduped_research)
    if removed_by_url:
        print(
            f"[orchestrator] removed {removed_by_url} research finding(s) with source URLs "
            "also present in competitor_findings"
        )

    competitor_names = [
        str(item.get("competitor_name") or "").strip()
        for item in competitor_findings
        if str(item.get("competitor_name") or "").strip()
        and (item.get("category") is None or item.get("category") == "competitor")
    ]
    name_filtered: list[dict] = []
    removed_by_name = 0
    for item in deduped_research:
        finding_text = str(item.get("finding") or "").lower()
        mentions_competitor = any(
            name.lower() in finding_text for name in competitor_names
        )
        if mentions_competitor:
            removed_by_name += 1
            continue
        name_filtered.append(item)
    if removed_by_name:
        print(
            f"[orchestrator] removed {removed_by_name} research finding(s) that mention "
            "a competitor_name already found by Competitor Agent"
        )

    updates: dict = {"research_findings": name_filtered}
    # Remote ventures never set location_data — keep it explicitly null.
    if "location_data" not in state:
        updates["location_data"] = None
    return updates


# Map-data competitor counts at or below this are "very low" for conflict checks.
_LOW_NEARBY_COMPETITOR_THRESHOLD = 1


def _flag_location_competitor_conflict(
    location_data: dict | None,
    competitor_findings: list[dict],
) -> dict | None:
    """Downgrade location data_confidence when map density conflicts with named competitors."""
    if location_data is None:
        return None

    updated = dict(location_data)
    updated.setdefault("data_confidence", "high")
    updated.setdefault("note", None)

    nearby = updated.get("competitor_count_nearby")
    named = [
        str(item.get("competitor_name") or "").strip()
        for item in competitor_findings
        if str(item.get("competitor_name") or "").strip()
        and (item.get("category") is None or item.get("category") == "competitor")
    ]
    if (
        nearby is not None
        and nearby <= _LOW_NEARBY_COMPETITOR_THRESHOLD
        and len(named) >= 2
    ):
        preview = ", ".join(named[:5])
        updated["data_confidence"] = "low"
        updated["note"] = (
            f"Location map data reports {nearby} direct competitor(s) nearby, but "
            f"Competitor Agent found {len(named)} named competitors ({preview}). "
            "Treat the location score as uncertain until verified on the ground."
        )
    return updated


def run_strategy_node(state: VentureOrchestratorState) -> dict:
    location_data = _flag_location_competitor_conflict(
        state.get("location_data"),
        state.get("competitor_findings") or [],
    )
    # Strategy accepts a dict; pass an empty marker when location was skipped.
    strategy_location_input = (
        location_data
        if location_data is not None
        else {
            "status": "skipped",
            "reason": "delivery_mode is remote — location analysis not applicable",
        }
    )
    strategy = run_strategy(
        state["business_idea"],
        state.get("research_findings") or [],
        state.get("competitor_findings") or [],
        strategy_location_input,
        state.get("hiring_signal"),
        customer_segments=state.get("customer_segments") or [],
        customer_pain_points=state.get("customer_pain_points") or [],
        customer_confidence=state.get("customer_confidence") or "none",
        no_local_data_reason=state.get("no_local_data_reason"),
    )
    # Persist conflict flag back into pipeline state for API/DB.
    return {"strategy": strategy, "location_data": location_data}


def run_task_planning(state: VentureOrchestratorState) -> dict:
    strategy = state.get("strategy") or {}
    return {"task_plan": generate_task_plan(strategy)}


def run_final_report(state: VentureOrchestratorState) -> dict:
    venture_data = {
        "business_idea": state.get("business_idea"),
        "delivery_mode": state.get("delivery_mode"),
        "delivery_mode_rationale": state.get("delivery_mode_rationale"),
        "research_findings": state.get("research_findings") or [],
        "competitor_findings": state.get("competitor_findings") or [],
        "customer_research": {
            "customer_segments": state.get("customer_segments") or [],
            "customer_pain_points": state.get("customer_pain_points") or [],
            "confidence": state.get("customer_confidence") or "none",
            "no_local_data_reason": state.get("no_local_data_reason"),
        },
        "hiring_signal": state.get("hiring_signal"),
        "location_data": state.get("location_data"),
        "strategy": state.get("strategy"),
        "task_plan": state.get("task_plan") or [],
    }
    return {"final_report": generate_final_report(venture_data)}


def build_orchestrator_graph():
    graph = StateGraph(VentureOrchestratorState)
    graph.add_node("infer_delivery_mode", infer_delivery_mode)
    graph.add_node("location", run_location)
    graph.add_node("research_node", research_node)
    graph.add_node("competitor_node", competitor_node)
    graph.add_node("customer_node", customer_node)
    graph.add_node("merge_research", merge_research)
    graph.add_node("strategy", run_strategy_node)
    graph.add_node("task_planning", run_task_planning)
    graph.add_node("final_report", run_final_report)

    graph.add_edge(START, "infer_delivery_mode")
    graph.add_conditional_edges(
        "infer_delivery_mode",
        route_after_delivery_mode,
        {
            "location": "location",
            "research_node": "research_node",
            "competitor_node": "competitor_node",
            "customer_node": "customer_node",
        },
    )
    # Onsite/hybrid: location then fan-out to the three research agents.
    graph.add_edge("location", "research_node")
    graph.add_edge("location", "competitor_node")
    graph.add_edge("location", "customer_node")
    # Fan-in: wait for all three parallel branches, then merge/dedupe.
    graph.add_edge(list(_PARALLEL_RESEARCH_NODES), "merge_research")
    graph.add_edge("merge_research", "strategy")
    graph.add_edge("strategy", "task_planning")
    graph.add_edge("task_planning", "final_report")
    graph.add_edge("final_report", END)
    return graph.compile()


venture_orchestrator = build_orchestrator_graph()


def run_venture_pipeline(business_idea: str) -> dict:
    """Run the full venture pipeline; location is gated by delivery_mode."""
    result = venture_orchestrator.invoke({"business_idea": business_idea})
    return {
        "delivery_mode": result.get("delivery_mode"),
        "delivery_mode_rationale": result.get("delivery_mode_rationale"),
        "research_findings": result.get("research_findings") or [],
        "hiring_signal": result.get("hiring_signal"),
        "competitor_findings": result.get("competitor_findings") or [],
        "customer_segments": result.get("customer_segments") or [],
        "customer_pain_points": result.get("customer_pain_points") or [],
        "customer_confidence": result.get("customer_confidence") or "none",
        "no_local_data_reason": result.get("no_local_data_reason"),
        "location_data": result.get("location_data"),  # may be None for remote
        "strategy": result.get("strategy"),
        "task_plan": result.get("task_plan") or [],
        "final_report": result.get("final_report"),
    }


if __name__ == "__main__":
    import json

    result = run_venture_pipeline("gym in Gulberg, Lahore")

    print("=== delivery_mode ===")
    print(result.get("delivery_mode"))
    print("\n=== delivery_mode_rationale ===")
    print(result.get("delivery_mode_rationale"))
    print("\n=== research_findings ===")
    print(json.dumps(result.get("research_findings"), indent=2, ensure_ascii=False))
    print("\n=== hiring_signal ===")
    print(json.dumps(result.get("hiring_signal"), indent=2, ensure_ascii=False))
    print("\n=== competitor_findings ===")
    print(json.dumps(result.get("competitor_findings"), indent=2, ensure_ascii=False))
    print("\n=== customer_segments ===")
    print(json.dumps(result.get("customer_segments"), indent=2, ensure_ascii=False))
    print("\n=== customer_pain_points ===")
    print(json.dumps(result.get("customer_pain_points"), indent=2, ensure_ascii=False))
    print("\n=== customer_confidence ===")
    print(result.get("customer_confidence"))
    print("\n=== no_local_data_reason ===")
    print(result.get("no_local_data_reason"))
    print("\n=== location_data ===")
    print(json.dumps(result.get("location_data"), indent=2, ensure_ascii=False))
    print("\n=== strategy ===")
    strategy = result.get("strategy") or {}
    print(json.dumps(strategy, indent=2, ensure_ascii=False))
    print("\n=== strategy.opportunity_tier ===")
    print(strategy.get("opportunity_tier"))
    print("\n=== strategy.first_call_question ===")
    print(strategy.get("first_call_question"))
    print("\n=== task_plan ===")
    print(json.dumps(result.get("task_plan"), indent=2, ensure_ascii=False))
    print("\n=== final_report ===")
    report = result.get("final_report") or {}
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "full_data"},
            indent=2,
            ensure_ascii=False,
        )
    )
