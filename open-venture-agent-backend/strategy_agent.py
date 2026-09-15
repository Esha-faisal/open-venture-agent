import json
import os
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from gemini_config import GEMINI_MODEL
from gemini_retry import invoke_with_retry
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY is not set")

OpportunityTier = Literal[
    "strong_opportunity",
    "viable_but_contested",
    "insufficient_evidence",
    "not_recommended_as_is",
]


class StrategyRecommendation(BaseModel):
    positioning: str
    target_customers: str
    pricing_strategy: str
    key_differentiators: list[str]
    confidence: int = Field(ge=0, le=100)
    assumptions: list[str]
    verification_needed: list[str]
    opportunity_tier: OpportunityTier = Field(
        description=(
            "Must follow the tiering rules in the prompt exactly — "
            "do not invent a tier outside those conditions."
        )
    )
    first_call_question: str = Field(
        description=(
            "One specific in-person verification question tailored to the chosen "
            "opportunity_tier."
        )
    )


class StrategyState(TypedDict):
    business_idea: str
    research_findings: list[dict]
    competitor_findings: list[dict]
    location_data: dict
    hiring_signal: dict
    customer_segments: list[dict]
    customer_pain_points: list[dict]
    customer_confidence: str
    no_local_data_reason: str | None
    strategy: dict


def _format_input_section(title: str, data: Any) -> str:
    return f"{title}:\n{json.dumps(data, indent=2)}"


def generate_strategy(state: StrategyState) -> dict:
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(StrategyRecommendation)

    prompt = (
        "You are a venture strategy advisor. Synthesize the inputs below into a "
        "coherent go-to-market recommendation for this business idea.\n\n"
        "Important: reconcile contradictions across sources explicitly. If the "
        "Location Agent reports zero nearby competitors but the Competitor Agent "
        "names specific competitors, or if market research conflicts with location "
        "data, flag these as discrepancies in verification_needed rather than "
        "silently ignoring them. Lower confidence when sources disagree.\n"
        "If location data has status 'skipped' (remote delivery mode), do not invent "
        "physical location insights — focus on digital/remote go-to-market instead.\n"
        "Use hiring_signal as supporting demand evidence (strong/weak/none), not as "
        "the sole reason for a tier.\n"
        "CUSTOMER EVIDENCE — use customer_segments and customer_pain_points as the "
        "primary basis for target_customers. Prefer evidenced segment names and needs "
        "over invented personas. If customer_confidence is 'none' or segments are "
        "empty, say so in assumptions/verification_needed and keep target_customers "
        "cautious rather than guessing generic groups like 'young professionals'. "
        "If no_local_data_reason is present, treat it as an explicit signal that "
        "local customer evidence was unavailable — do not invent local personas.\n\n"
        "OPPORTUNITY TIER — pick exactly one using these rules (do not guess freely):\n"
        "1. strong_opportunity — use ONLY when competitor findings show low/no "
        "contested named competitors in this niche AND there are no unresolved "
        "contradictions between agents (research, competitor, location, hiring, "
        "customer).\n"
        "2. viable_but_contested — use when there are named, established competitors "
        "already operating in this space, but a plausible gap or differentiation "
        "still exists.\n"
        "3. insufficient_evidence — use when findings are sparse/thin OR agents "
        "contradict each other with no clear resolution (e.g. location says 0 "
        "competitors while competitor agent names several).\n"
        "4. not_recommended_as_is — use ONLY when dominant, well-established "
        "competitors clearly already dominate this exact niche with no visible gap.\n\n"
        "FIRST CALL QUESTION — write one specific question the founder should go "
        "verify in person, tailored to the tier:\n"
        "- insufficient_evidence: ask them to directly verify the unresolved gap "
        "(e.g. walk the street / count stores / ask locals).\n"
        "- viable_but_contested: ask them to find a specific competitive gap "
        "(e.g. pricing, hours, niche service missing).\n"
        "- strong_opportunity: ask them to confirm demand with a concrete local check.\n"
        "- not_recommended_as_is: ask them what would have to be true for a wedge "
        "to exist despite the dominant players.\n\n"
        f"Business idea: {state['business_idea']}\n\n"
        f"{_format_input_section('Research findings', state['research_findings'])}\n\n"
        f"{_format_input_section('Hiring signal', state['hiring_signal'])}\n\n"
        f"{_format_input_section('Competitor findings', state['competitor_findings'])}\n\n"
        f"{_format_input_section('Customer segments', state['customer_segments'])}\n\n"
        f"{_format_input_section('Customer pain points', state['customer_pain_points'])}\n\n"
        f"{_format_input_section('Customer confidence', state['customer_confidence'])}\n\n"
        f"{_format_input_section('No local customer data reason', state.get('no_local_data_reason'))}\n\n"
        f"{_format_input_section('Location data', state['location_data'])}\n\n"
        "Produce a structured recommendation with:\n"
        "- positioning: how the venture should position itself\n"
        "- target_customers: who to serve first — grounded in customer_segments / "
        "pain_points when available\n"
        "- pricing_strategy: how to price and why\n"
        "- key_differentiators: 3-5 concrete differentiators\n"
        "- confidence: 0-100 score reflecting data quality and agreement across sources\n"
        "- assumptions: key assumptions you are making\n"
        "- verification_needed: items to validate before committing, including any "
        "cross-source discrepancies\n"
        "- opportunity_tier: one of the four tiers above, chosen by the rules\n"
        "- first_call_question: one tier-specific in-person verification question"
    )
    result = invoke_with_retry(
        lambda: structured_llm.invoke(prompt),
        label="strategy",
    )
    return {"strategy": result.model_dump()}


def build_strategy_graph():
    graph = StateGraph(StrategyState)
    graph.add_node("generate", generate_strategy)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", END)
    return graph.compile()


strategy_agent = build_strategy_graph()


def run_strategy(
    business_idea: str,
    research_findings: list[dict],
    competitor_findings: list[dict],
    location_data: dict | None,
    hiring_signal: dict | None = None,
    customer_segments: list[dict] | None = None,
    customer_pain_points: list[dict] | None = None,
    customer_confidence: str | None = None,
    no_local_data_reason: str | None = None,
) -> dict:
    result = strategy_agent.invoke(
        {
            "business_idea": business_idea,
            "research_findings": research_findings,
            "competitor_findings": competitor_findings,
            "location_data": location_data or {"status": "skipped"},
            "hiring_signal": hiring_signal
            or {
                "postings_found": [],
                "employer_count": 0,
                "confidence": "none",
            },
            "customer_segments": customer_segments or [],
            "customer_pain_points": customer_pain_points or [],
            "customer_confidence": customer_confidence or "none",
            "no_local_data_reason": no_local_data_reason,
        }
    )
    return result["strategy"]


if __name__ == "__main__":
    sample_research = [
        {
            "finding": "Fitness and gym membership demand is rising among young professionals in Lahore's commercial districts.",
            "source": "https://example.com/lahore-fitness-market",
        },
        {
            "finding": "Gulberg is a high-footfall commercial hub with strong spending power for premium wellness services.",
            "source": "https://example.com/gulberg-commercial",
        },
    ]
    sample_competitors = [
        {
            "competitor_name": "Hexa Gym",
            "finding": "Located in Gulberg, Lahore; full-service gym with floor staff and front-desk operations — currently hiring a female receptionist.",
            "source": "https://example.com/hexa-gym-gulberg",
        },
        {
            "competitor_name": "Structure Health and Fitness",
            "finding": "Operates in Gulberg/Johar Town, Lahore; offers floor instruction and fitness staffing — hiring female floor instructors and fitness staff.",
            "source": "https://example.com/structure-health-fitness",
        },
        {
            "competitor_name": "Activo Fitness Studio",
            "finding": "Studio-style fitness in DHA Lahore (serving Lahore's premium fitness market); offers yoga and full front-desk/manager roles — hiring Gym Manager, Front Desk Officer, and Yoga Trainer.",
            "source": "https://example.com/activo-fitness-studio",
        },
    ]
    sample_location = {
        "location_name": "Gulberg, Lahore, Punjab, Pakistan",
        "coordinates": {"latitude": 31.5204, "longitude": 74.3587},
        "competitor_count_nearby": 0,
        "score": 78,
        "explanation": "Within 1500m: 0 direct competitors and 22 commercial POIs nearby.",
    }
    sample_hiring = {
        "postings_found": [
            "Female receptionist for Hexa Gym in Gulberg, Lahore",
            "Female floor instructor and fitness staff at Structure Health and Fitness in Gulberg/Johar Town, Lahore",
            "Gym Manager, Front Desk Officer, Yoga Trainer at Activo Fitness Studio in DHA Lahore",
        ],
        "employer_count": 4,
        "confidence": "strong",
    }
    strategy = run_strategy(
        "gym in Gulberg, Lahore",
        sample_research,
        sample_competitors,
        sample_location,
        sample_hiring,
    )
    print(strategy)
