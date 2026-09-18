import json
import logging
import os
import re
from typing import Any, Literal, TypedDict

logger = logging.getLogger(__name__)

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
    verification_needed: list[str] = Field(
        description=(
            "Pure risk / discrepancy statements only — what is uncertain or "
            "conflicting. Do NOT embed action language (no 'go check', 'walk "
            "around', 'you should…'). State the concern and stop."
        )
    )
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


def _known_competitor_names(competitor_findings: list[dict]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for item in competitor_findings:
        name = str(item.get("competitor_name") or "").strip()
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            names.append(name)
    return names


def _format_allowed_competitor_names(names: list[str]) -> str:
    if not names:
        return "none — refer generically to 'local competitors' or 'named rivals in the findings' only"
    return ", ".join(names)


# Capitalized tokens that are usually strategy prose, not business names.
_NAME_STOPWORDS = frozenset(
    {
        "The",
        "What",
        "How",
        "If",
        "Verify",
        "Confirm",
        "Ask",
        "Walk",
        "Visit",
        "Consumers",
        "Other",
        "Local",
        "Named",
        "Specific",
        "Whether",
        "Day",
        "One",
        "First",
        "Before",
        "After",
        "During",
        "When",
        "Where",
        "Which",
        "This",
        "That",
        "These",
        "Those",
        "Pakistan",
        "Lahore",
        "Islamabad",
        "Gujrat",
        "Gulberg",
        "Bahria",
        "Town",
        "Phase",
        "Street",
        "Market",
        "Bakery",
        "Gym",
        "Competitor",
        "Competitors",
        "Customer",
        "Customers",
        "Founder",
        "Staff",
        "Service",
        "Quality",
        "Pricing",
        "Location",
        "Tier",
    }
)


def _business_idea_tokens(business_idea: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z]{3,}", business_idea.lower())
    return set(tokens)


def _extract_name_candidates(text: str) -> list[str]:
    """Pull capitalized business-name-like terms from strategy prose."""
    candidates: list[str] = []
    seen: set[str] = set()

    def push(value: str) -> None:
        key = value.lower()
        if key in seen or value in _NAME_STOPWORDS:
            return
        seen.add(key)
        candidates.append(value)

    for match in re.finditer(
        r"\b[A-Z][a-z]+(?:'[sS])?(?:[ \t]+[A-Z][a-z]+(?:'[sS])?)+\b",
        text,
    ):
        push(match.group())

    for match in re.finditer(r"\b[A-Z][a-z]{2,}\b", text):
        push(match.group())

    return candidates


def _is_grounded_competitor_name(candidate: str, known_names: list[str]) -> bool:
    needle = candidate.lower().strip()
    if len(needle) < 3:
        return True
    for known in known_names:
        haystack = known.lower()
        if needle == haystack or needle in haystack or haystack in needle:
            return True
    return False


def _strategy_fields_for_name_scan(strategy: dict) -> list[str]:
    texts: list[str] = []
    for field in ("verification_needed", "assumptions"):
        for item in strategy.get(field) or []:
            text = str(item).strip()
            if text:
                texts.append(text)
    first_call = str(strategy.get("first_call_question") or "").strip()
    if first_call:
        texts.append(first_call)
    return texts


def _warn_hallucinated_competitor_names(
    strategy: dict,
    competitor_findings: list[dict],
    *,
    business_idea: str,
) -> None:
    known_names = _known_competitor_names(competitor_findings)
    idea_tokens = _business_idea_tokens(business_idea)
    field_texts = _strategy_fields_for_name_scan(strategy)
    if not field_texts:
        return

    hallucinated: list[str] = []
    seen_hallucinated: set[str] = set()
    for field_text in field_texts:
        for candidate in _extract_name_candidates(field_text):
            if candidate.lower() in idea_tokens:
                continue
            if _is_grounded_competitor_name(candidate, known_names):
                continue
            key = candidate.lower()
            if key not in seen_hallucinated:
                seen_hallucinated.add(key)
                hallucinated.append(candidate)

    if hallucinated:
        logger.warning(
            "[strategy] hallucinated competitor name(s) not in competitor_findings: %s "
            "(allowed names: %s)",
            ", ".join(hallucinated),
            ", ".join(known_names) if known_names else "(none provided)",
        )


def generate_strategy(state: StrategyState) -> dict:
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(StrategyRecommendation)
    allowed_competitor_names = _known_competitor_names(state["competitor_findings"])

    prompt = (
        "You are a venture strategy advisor. Synthesize the inputs below into a "
        "coherent go-to-market recommendation for this business idea.\n\n"
        "COMPETITOR NAMES — mandatory grounding:\n"
        "- You must only reference competitor names that appear in the Competitor "
        "findings provided below.\n"
        "- Do NOT introduce, assume, or recall any business names from general "
        "knowledge or memory.\n"
        "- Allowed named competitors for this venture: "
        f"{_format_allowed_competitor_names(allowed_competitor_names)}.\n"
        "- If you need to refer to rivals but do not have enough named competitors, "
        "say so generically (e.g. 'other local competitors', 'existing bakeries in "
        "the area') rather than inventing a specific business name.\n"
        "- This applies everywhere in your output, especially verification_needed, "
        "assumptions, and first_call_question.\n\n"
        "USER-FACING LANGUAGE — mandatory for every field a founder will read "
        "(positioning, target_customers, pricing_strategy, key_differentiators, "
        "assumptions, verification_needed, first_call_question):\n"
        "- Write as if explaining it to a friend who is thinking about opening this "
        "business — conversational, concrete, and plain.\n"
        "- Never mention internal agent/system names (research agent, location "
        "agent, competitor agent, customer agent, orchestrator, pipeline, etc.).\n"
        "- Do not make \"data,\" \"sources,\" \"evidence,\" \"reports,\" \"signals,\" "
        "or \"findings\" the subject of the sentence. Talk about the real-world "
        "situation (what's on the street, what customers complain about, what "
        "rivals seem to offer).\n"
        "- Prefer \"you,\" \"locals,\" \"nearby shops,\" \"customers,\" "
        "\"competitors\" over analyst jargon.\n\n"
        "VERIFICATION_NEEDED vs FIRST_CALL_QUESTION — keep them distinct:\n"
        "- verification_needed: risk/discrepancy STATEMENTS only. Describe what "
        "is unclear, conflicting, or unconfirmed. STOP there — no \"go do X\", "
        "\"walk around\", \"ask locals\", \"you should check\", or other action "
        "language tacked on.\n"
        "  GOOD: \"Maps show dozens of nearby food spots, but research didn't "
        "name specific seafood rivals — how much direct competition there really "
        "is remains unclear.\"\n"
        "  BAD: \"…so walk around and count the seafood places yourself.\"\n"
        "- first_call_question: the ONE place for in-person action language — "
        "a specific thing to go verify.\n"
        "- assumptions: beliefs you are leaning on (not tasks, not risks).\n\n"
        "Before → after for mixed research language (follow this pattern for "
        "prose fields; for verification_needed use the GOOD statement form "
        "above — concern only, no action coda):\n"
        "1. BEFORE: \"Map data reports 2 nearby competitors while local customer "
        "evidence is entirely missing.\"\n"
        "   AFTER (verification_needed): \"There look to be a couple of similar "
        "shops nearby, but we couldn't find real customer reviews for this exact "
        "area — so demand and dissatisfaction are still unclear.\"\n"
        "2. BEFORE: \"Online research didn't surface much about this area, but map "
        "data shows 2 nearby competitors.\"\n"
        "   AFTER (verification_needed): \"The internet is quiet about this "
        "neighborhood, yet there still seem to be a couple of rivals nearby — "
        "who they are and how busy they get is unresolved.\"\n"
        "3. BEFORE: \"Hiring signals indicate demand for bakery staff, suggesting "
        "market activity.\"\n"
        "   AFTER (assumption / supporting prose): \"Local bakeries appear to be "
        "hiring, which usually means the category is active.\"\n\n"
        "Important: still reconcile contradictions across the inputs. If maps show "
        "few nearby shops but named local rivals appear in the competitor list, or "
        "if online coverage clashes with what's nearby, put that in "
        "verification_needed as a plain risk statement — do not silently "
        "ignore it, and do not turn it into a to-do. Lower confidence when the "
        "picture is mixed.\n"
        "If location data has status 'skipped' (remote delivery mode), do not invent "
        "physical location insights — focus on digital/remote go-to-market instead.\n"
        "Use hiring_signal as supporting demand context (strong/weak/none), not as "
        "the sole reason for a tier.\n"
        "CUSTOMER EVIDENCE — use customer_segments and customer_pain_points as the "
        "primary basis for target_customers. Prefer evidenced segment names and needs "
        "over invented personas. If customer_confidence is 'none' or segments are "
        "empty, say so in assumptions/verification_needed in plain language (e.g. "
        "\"we couldn't find real customer reviews for this neighborhood\") and keep "
        "target_customers cautious rather than guessing generic groups like 'young "
        "professionals'. If no_local_data_reason is present, treat it as a signal "
        "that local customer voice was unavailable — do not invent local personas.\n\n"
        "OPPORTUNITY TIER — pick exactly one using these rules (do not guess freely):\n"
        "1. strong_opportunity — use ONLY when competitor findings show low/no "
        "contested named competitors in this niche AND there are no unresolved "
        "contradictions across the research inputs (online research, named "
        "competitors, map data, hiring, customer evidence).\n"
        "2. viable_but_contested — use when there are named, established competitors "
        "already operating in this space, but a plausible gap or differentiation "
        "still exists.\n"
        "3. insufficient_evidence — use when findings are sparse/thin OR sources "
        "contradict each other with no clear resolution (e.g. map data says 0 "
        "nearby competitors while named local rivals appear in the findings).\n"
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
        "- verification_needed: pure risk/discrepancy statements (what is "
        "uncertain or conflicting) — NO action/instructions language; include "
        "cross-source discrepancies when present\n"
        "- opportunity_tier: one of the four tiers above, chosen by the rules\n"
        "- first_call_question: one tier-specific in-person verification question"
    )
    result = invoke_with_retry(
        lambda: structured_llm.invoke(prompt),
        label="strategy",
    )
    strategy = result.model_dump()
    _warn_hallucinated_competitor_names(
        strategy,
        state["competitor_findings"],
        business_idea=state["business_idea"],
    )
    return {"strategy": strategy}


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
