import json
import logging
import os
import re
from typing import Any, Literal, TypedDict

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from gemini_config import GEMINI_MODEL
from gemini_retry import invoke_with_retry
from plain_language import PLAIN_LANGUAGE_BLOCK
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


class ProsConsItem(BaseModel):
    point: str = Field(
        description=(
            "Short interpretive headline — what this means for the founder "
            "(not a restated research finding)."
        )
    )
    reasoning: str = Field(
        description=(
            "2–4 sentences of synthesized 'why this matters for you' insight. "
            "Combine or interpret multiple pieces of evidence when possible. "
            "Do NOT copy or lightly rephrase a single finding's wording. "
            "Stay grounded in the inputs — invent nothing."
        )
    )
    source: str = Field(
        description=(
            "Transparency citation: one or more source URLs/labels from the "
            "inputs that support this takeaway (comma-separated if several). "
            "Prefer naming the actual sources used over a vague placeholder."
        )
    )


class ProsAndCons(BaseModel):
    pros: list[ProsConsItem] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "3–4 strong, distinct advantages (fewer only if evidence is truly "
            "thin). Each must be a synthesized takeaway, not a relabeled finding."
        ),
    )
    cons: list[ProsConsItem] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "3–4 strong, distinct drawbacks (fewer only if evidence is truly "
            "thin). Each must be a synthesized takeaway, not a relabeled finding."
        ),
    )


class StrategyRecommendation(BaseModel):
    positioning: str
    target_customers: str
    pricing_strategy: str
    estimated_budget: str = Field(
        description=(
            "Rough startup cost breakdown for this business type and location "
            "(rent range, initial inventory/equipment, staffing, marketing, etc.). "
            "Clearly labeled as an estimate requiring local verification — not a "
            "precise sourced figure."
        )
    )
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
    pros_and_cons: ProsAndCons = Field(
        description=(
            "3–4 synthesized pros and 3–4 synthesized cons. Each item interprets "
            "combined evidence into a 'why this matters for you' takeaway — never "
            "a one-to-one copy of a single research finding."
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
    operational_pain_points: list[dict]
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


_NON_RIVAL_CATEGORIES = {
    "vendor_technology",
    "pricing_intelligence",
    "directory_source",
}


def _named_rival_count(competitor_findings: list[dict]) -> int:
    """Count named direct rivals (exclude vendors / pricing / directory noise)."""
    count = 0
    seen: set[str] = set()
    for item in competitor_findings or []:
        if item.get("category") in _NON_RIVAL_CATEGORIES:
            continue
        name = str(item.get("competitor_name") or "").strip()
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            count += 1
    return count


def _competitor_entry_count(competitor_findings: list[dict] | None) -> int:
    return len(competitor_findings or [])


def _open_market_framing_notes(state: StrategyState) -> str:
    """Guidance when competitor/customer/hiring research is empty or thin."""
    notes: list[str] = []
    entry_count = _competitor_entry_count(state.get("competitor_findings"))
    rival_count = _named_rival_count(state.get("competitor_findings") or [])
    if entry_count < 2 or rival_count < 2:
        notes.append(
            "- Competitor findings are empty or thin for THIS search run — that is "
            "incomplete coverage, NOT proof the market is empty. NEVER claim "
            "\"nobody else is doing this\", \"no competition exists\", \"zero "
            "competitors\", or that the niche is a guaranteed open market.\n"
            "  Required phrasing (adapt wording, keep the meaning): \"We couldn't "
            "find direct competitors in our search, though that doesn't guarantee "
            "none exist — especially informal or social-media-based sellers who "
            "may not show up in web results.\"\n"
            "  Lower confidence for thin competitor coverage. Prefer "
            "insufficient_evidence (or a cautious viable_but_contested) over "
            "strong_opportunity when competitor_findings has fewer than 2 entries. "
            "Put the search-gap caveat in assumptions and/or verification_needed."
        )

    segments = state.get("customer_segments") or []
    pains = state.get("customer_pain_points") or []
    operational = state.get("operational_pain_points") or []
    customer_confidence = str(state.get("customer_confidence") or "none").lower()
    if customer_confidence in {"none", ""} or (
        not segments and not pains and not operational
    ):
        notes.append(
            "- Customer research is empty/thin. For a niche or new idea, frame this "
            "as an early market (few public reviews yet) — e.g. \"This looks like "
            "an early market where customer voice online is still thin, which is "
            "normal for a new niche rather than a red flag.\" Do NOT default to "
            "\"no data / couldn't find reviews\" as a negative risk. Keep "
            "target_customers cautious but forward-looking; do not invent fake "
            "local personas."
        )

    hiring = state.get("hiring_signal") or {}
    hiring_confidence = str(hiring.get("confidence") or "none").lower()
    postings = hiring.get("postings_found") or []
    if hiring_confidence in {"none", ""} or not postings:
        notes.append(
            "- Hiring signal is empty/none. Frame as an early/nascent category "
            "(few employers posting related roles yet) — not as proof the market "
            "is weak or that labor data is \"missing.\" Absence of hiring alone "
            "must not make the opportunity look negative."
        )

    if not notes:
        return ""
    return (
        "SPARSE RESEARCH FRAMING — mandatory for this run because some research "
        "buckets are empty or thin:\n"
        + "\n".join(notes)
        + "\n"
        "For thin competitor coverage, do NOT celebrate an \"open market.\" For "
        "thin customer/hiring on a niche idea, early-market wording is fine. Only "
        "put an extra concern in verification_needed when something actively "
        "conflicts (e.g. maps show many nearby shops while named rivals are "
        "empty).\n\n"
    )


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


def _categorical_signal_score(level: str | None) -> float:
    normalized = str(level or "none").strip().lower()
    return {
        "strong": 100.0,
        "weak": 62.0,
        "none": 48.0,
    }.get(normalized, 48.0)


def _location_component_score(location_data: dict | None) -> float:
    """Map location quality into 0–100; remote/skipped gets a neutral mid score."""
    if not location_data:
        return 72.0
    status = str(
        location_data.get("data_status") or location_data.get("status") or ""
    ).lower()
    if status in {"skipped"}:
        return 72.0
    if status == "overpass_unavailable":
        return 40.0
    if status == "location_not_found":
        return 35.0
    raw = location_data.get("score")
    if raw is None:
        return 55.0
    try:
        return float(max(0, min(100, int(raw))))
    except (TypeError, ValueError):
        return 55.0


_CONTRADICTION_MARKERS = (
    "contradict",
    "conflict",
    "discrepan",
    "disagree",
    "mismatch",
    "inconsist",
    "unresolved",
)


def _contradictions_present(
    strategy: dict,
    location_data: dict | None,
) -> bool:
    if location_data and str(location_data.get("data_confidence") or "").lower() == "low":
        return True
    if location_data and location_data.get("note"):
        note = str(location_data.get("note") or "").lower()
        if any(marker in note for marker in _CONTRADICTION_MARKERS):
            return True
    for item in strategy.get("verification_needed") or []:
        text = str(item).lower()
        if any(marker in text for marker in _CONTRADICTION_MARKERS):
            return True
    return False


def _idea_score_label(score: int) -> str:
    if score >= 85:
        return "Strong foundation"
    if score >= 70:
        return "Promising, but needs validation"
    if score >= 55:
        return "Mixed signals — dig deeper"
    if score >= 40:
        return "Weak evidence so far"
    return "High risk as framed"


def _competitor_coverage_score(competitor_findings: list[dict] | None) -> float:
    """Sparse competitor lists are a coverage gap — not proof of an open market."""
    n = _competitor_entry_count(competitor_findings)
    if n >= 2:
        return 100.0
    if n == 1:
        return 45.0
    return 25.0


def compute_idea_score(
    *,
    confidence: int | float | None,
    customer_confidence: str | None,
    hiring_signal: dict | None,
    location_data: dict | None,
    competitor_findings: list[dict] | None = None,
    strategy: dict | None = None,
) -> tuple[int, str]:
    """Deterministic 0–100 idea score from weighted research signals."""
    strategy = strategy or {}
    try:
        confidence_score = float(max(0, min(100, int(confidence or 0))))
    except (TypeError, ValueError):
        confidence_score = 0.0

    contradiction_score = (
        30.0 if _contradictions_present(strategy, location_data) else 100.0
    )
    customer_score = _categorical_signal_score(customer_confidence)
    hiring = hiring_signal or {}
    hiring_score = _categorical_signal_score(hiring.get("confidence"))
    location_score = _location_component_score(location_data)
    competitor_score = _competitor_coverage_score(competitor_findings)

    # Weighted blend — always sums to 1.0.
    blended = (
        0.30 * confidence_score
        + 0.15 * contradiction_score
        + 0.15 * customer_score
        + 0.125 * hiring_score
        + 0.125 * location_score
        + 0.15 * competitor_score
    )
    score = int(round(max(0.0, min(100.0, blended))))
    return score, _idea_score_label(score)


def _apply_sparse_competitor_confidence_penalty(
    strategy: dict,
    competitor_findings: list[dict] | None,
) -> None:
    """Lower model confidence when competitor coverage is thin (< 2 entries)."""
    n = _competitor_entry_count(competitor_findings)
    if n >= 2:
        return
    try:
        conf = int(strategy.get("confidence") or 0)
    except (TypeError, ValueError):
        conf = 0
    penalty = 20 if n == 0 else 12
    strategy["confidence"] = max(0, min(100, conf - penalty))


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
        "(positioning, target_customers, pricing_strategy, estimated_budget, "
        "key_differentiators, assumptions, verification_needed, first_call_question, "
        "pros_and_cons):\n"
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
        f"{PLAIN_LANGUAGE_BLOCK}\n"
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
        "the sole reason for a tier. When hiring is empty/none for a niche idea, "
        "treat it as early-market labor quietness (neutral-to-positive), not a "
        "failed search.\n"
        "CUSTOMER EVIDENCE — use customer_segments and customer_pain_points as the "
        "primary basis for target_customers. Prefer evidenced segment names and needs "
        "over invented personas. Use operational_pain_points for industry/operator "
        "problems a venture could solve or should acknowledge — do NOT treat those "
        "as end-customer frustrations when writing target_customers. If "
        "customer_confidence is 'none' or segments/customer pain points are empty "
        "for a niche or new idea, frame that as an early market with little public "
        "customer chatter yet (opportunity to define the category) — not as a "
        "negative \"no data\" failure. Keep target_customers cautious rather than "
        "guessing generic groups like 'young professionals'. If "
        "no_local_data_reason is present, still avoid inventing local personas, "
        "but prefer early-market wording over sounding like the research broke.\n\n"
        "EMPTY / SPARSE RESEARCH — careful framing (do NOT invent certainty):\n"
        "- Empty or thin competitor_findings (< 2 entries) → search coverage gap, "
        "NOT proof of zero competition or an open market. NEVER say \"nobody else "
        "is doing this\", \"no competition exists\", or similar absolutes. Say: "
        "\"We couldn't find direct competitors in our search, though that doesn't "
        "guarantee none exist, especially for informal/social-media-based sellers.\" "
        "Lower confidence. Prefer insufficient_evidence over strong_opportunity "
        "when competitor coverage is this thin.\n"
        "- Empty customer research on a niche idea → early market, NOT a red flag.\n"
        "- Empty hiring signal on a niche idea → early/nascent category, NOT "
        "proof of weak demand.\n"
        "- Do NOT choose strong_opportunity solely because competitors came back "
        "empty. Do NOT choose insufficient_evidence solely because customer or "
        "hiring came back empty when nothing else contradicts an early-market "
        "read.\n\n"
        f"{_open_market_framing_notes(state)}"
        "OPPORTUNITY TIER — pick exactly one using these rules (do not guess freely):\n"
        "1. strong_opportunity — use when competitor findings show a plausible "
        "lightly contested niche WITH enough named evidence to trust the picture "
        "(typically 2+ relevant competitor entries, or clear map/directory "
        "evidence of low rivalry) AND there are no unresolved contradictions "
        "across the research inputs. Do NOT pick this tier merely because "
        "competitor_findings is empty.\n"
        "2. viable_but_contested — use when there are named, established competitors "
        "already operating in this space, but a plausible gap or differentiation "
        "still exists.\n"
        "3. insufficient_evidence — use when sources contradict each other with "
        "no clear resolution (e.g. map data says 0 nearby competitors while named "
        "local rivals appear in the findings), when competitor_findings is empty "
        "or has fewer than 2 entries (thin web coverage — especially possible "
        "informal/social sellers), or when the overall picture is too mixed to "
        "trust.\n"
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
        f"{_format_input_section('Operational pain points', state.get('operational_pain_points') or [])}\n\n"
        f"{_format_input_section('Customer confidence', state['customer_confidence'])}\n\n"
        f"{_format_input_section('No local customer data reason', state.get('no_local_data_reason'))}\n\n"
        f"{_format_input_section('Location data', state['location_data'])}\n\n"
        "Produce a structured recommendation with:\n"
        "- positioning: how the venture should position itself\n"
        "- target_customers: who to serve first — grounded in customer_segments / "
        "customer_pain_points when available (not operational_pain_points)\n"
        "- pricing_strategy: how to price and why\n"
        "- estimated_budget: a rough startup cost breakdown for this business "
        "type and location — cover rent/lease range (if physical), initial "
        "inventory or equipment, early staffing, and basic marketing. Fold in "
        "any concrete price points from competitor/research findings when "
        "available; otherwise use grounded general knowledge for the category "
        "and place. Use ranges, not fake precision. ALWAYS open or close with a "
        "clear disclaimer that this is an estimate that needs local verification "
        "(quotes, site visits) — it is NOT sourced the same way as research "
        "findings.\n"
        "- key_differentiators: 3-5 concrete differentiators\n"
        "- confidence: 0-100 score reflecting data quality and agreement across "
        "sources — lower this when competitor_findings is empty or has fewer "
        "than 2 entries (incomplete competitive coverage)\n"
        "- assumptions: key assumptions you are making\n"
        "- verification_needed: pure risk/discrepancy statements (what is "
        "uncertain or conflicting) — NO action/instructions language; include "
        "cross-source discrepancies when present\n"
        "- opportunity_tier: one of the four tiers above, chosen by the rules\n"
        "- first_call_question: one tier-specific in-person verification question\n"
        "- pros_and_cons: object with pros[] and cons[] — STRICTLY 3–4 items "
        "each (quality over quantity; no near-duplicates).\n"
        "  How to write each item:\n"
        "  - point: short interpretive headline (what it means for THIS idea).\n"
        "  - reasoning: synthesize / connect multiple pieces of evidence into "
        "one \"why this matters for you\" insight. Do NOT paste or lightly "
        "rephrase a single finding as the pro/con. Interpret what the evidence "
        "implies for the founder's decision.\n"
        "  - source: cite the supporting source URL(s) or labels from the "
        "inputs for transparency (comma-separated if several).\n"
        "  BAD (relabeled finding): point=\"Bundtly launched in Bahria Town\", "
        "reasoning=\"A new dessert spot opened offering cakes and bundts\" "
        "(this just restates one research line).\n"
        "  GOOD (synthesized takeaway): point=\"People here already buy "
        "celebration desserts locally\", reasoning=\"A few new dessert spots "
        "and established bakeries both show up nearby, which usually means "
        "locals are already shopping for cakes and sweets close to home — so "
        "you would be joining demand that exists, not inventing it from "
        "scratch.\" source=\"…instagram…, …directory…\"\n"
        "  BAD (relabeled finding): point=\"14 competitors nearby\", "
        "reasoning=\"Location score shows 14 direct competitors within 1500m.\"\n"
        "  GOOD (synthesized takeaway): point=\"You'll have to stand out "
        "quickly\", reasoning=\"Maps show a crowded bakery strip nearby while "
        "named shops already cover everyday sweets — so a me-too bakery will "
        "struggle, but a clear specialty (hours, custom cakes, delivery) could "
        "still win room.\"\n"
        "  Ground every item in actual inputs — invent nothing. If evidence is "
        "thin, write 3 strong items rather than padding with weak restatements."
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
    _apply_sparse_competitor_confidence_penalty(
        strategy,
        state.get("competitor_findings"),
    )
    idea_score, idea_score_label = compute_idea_score(
        confidence=strategy.get("confidence"),
        customer_confidence=state.get("customer_confidence"),
        hiring_signal=state.get("hiring_signal"),
        location_data=state.get("location_data"),
        competitor_findings=state.get("competitor_findings"),
        strategy=strategy,
    )
    strategy["idea_score"] = idea_score
    strategy["idea_score_label"] = idea_score_label
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
    operational_pain_points: list[dict] | None = None,
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
            "operational_pain_points": operational_pain_points or [],
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
