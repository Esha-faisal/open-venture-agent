import json
import os
import re
from difflib import SequenceMatcher

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from gemini_config import GEMINI_MODEL
from gemini_retry import invoke_with_retry
from plain_language import PLAIN_LANGUAGE_BLOCK
from text_sanitize import sanitize_display_text

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY is not set")


class HeadlineResult(BaseModel):
    headline: str = Field(
        description=(
            "One plain-language sentence summarizing the opportunity. "
            "No jargon, no buzzwords, no multi-sentence paragraphs."
        )
    )


class FullAnalysisResult(BaseModel):
    full_analysis: str = Field(
        description=(
            "A multi-paragraph written analyst report that synthesizes market "
            "context, named competitors, customer complaints, location tradeoffs, "
            "and why the opportunity_tier verdict makes sense. Connected prose, "
            "not a bullet list."
        )
    )


def _customer_bundle(venture_data: dict) -> dict:
    """Normalize customer fields whether nested or flat."""
    nested = venture_data.get("customer_research")
    if isinstance(nested, dict):
        return {
            "customer_segments": nested.get("customer_segments") or [],
            "customer_pain_points": nested.get("customer_pain_points") or [],
            "operational_pain_points": nested.get("operational_pain_points") or [],
            "confidence": nested.get("confidence") or "none",
            "no_local_data_reason": nested.get("no_local_data_reason"),
        }
    return {
        "customer_segments": venture_data.get("customer_segments") or [],
        "customer_pain_points": venture_data.get("customer_pain_points") or [],
        "operational_pain_points": venture_data.get("operational_pain_points") or [],
        "confidence": venture_data.get("customer_confidence") or "none",
        "no_local_data_reason": venture_data.get("no_local_data_reason"),
    }


def _is_true_competitor_finding(item: dict) -> bool:
    category = item.get("category")
    return category not in {
        "vendor_technology",
        "pricing_intelligence",
        "directory_source",
    }


def _format_competitor_finding(item: dict) -> str | None:
    if not _is_true_competitor_finding(item):
        return None
    name = item.get("competitor_name")
    finding = item.get("finding")
    if name and finding:
        return f"Competitor {name}: {finding}"
    if finding:
        return str(finding)
    return None


def _merged_competitor_key_lines(competitor_findings: list[dict]) -> list[str]:
    """One bullet per competitor_name; merge multiple findings into one line."""
    by_name: dict[str, tuple[str, list[str]]] = {}
    order: list[str] = []
    unnamed: list[str] = []

    for item in competitor_findings:
        if not _is_true_competitor_finding(item):
            continue
        name = str(item.get("competitor_name") or "").strip()
        finding = str(item.get("finding") or "").strip()
        if name:
            key = name.lower()
            if key not in by_name:
                by_name[key] = (name, [])
                order.append(key)
            findings = by_name[key][1]
            if finding and finding not in findings:
                findings.append(finding)
        elif finding:
            unnamed.append(finding)

    lines: list[str] = []
    for key in order:
        name, findings = by_name[key]
        if not findings:
            lines.append(f"Competitor {name}")
        elif len(findings) == 1:
            lines.append(f"Competitor {name}: {findings[0]}")
        else:
            lines.append(f"Competitor {name}: {'; '.join(findings)}")
    lines.extend(unnamed)
    return lines


def _competitor_name_in_highlights(name: str, highlights: list[str]) -> bool:
    needle = name.strip().lower()
    if not needle:
        return False
    for line in highlights:
        lowered = line.lower()
        if lowered.startswith(f"competitor {needle}:") or lowered == f"competitor {needle}":
            return True
    return False


def _strategy_mention_text(strategy: dict) -> str:
    """Combined strategy prose checked for explicit competitor name mentions."""
    parts: list[str] = []
    for item in strategy.get("verification_needed") or []:
        text = str(item).strip()
        if text:
            parts.append(text)
    for item in strategy.get("assumptions") or []:
        text = str(item).strip()
        if text:
            parts.append(text)
    first_call = str(strategy.get("first_call_question") or "").strip()
    if first_call:
        parts.append(first_call)
    return "\n".join(parts)


def _strategy_referenced_competitors(
    strategy: dict,
    competitor_findings: list[dict],
) -> list[dict]:
    haystack = _strategy_mention_text(strategy).lower()
    if not haystack:
        return []

    referenced: list[dict] = []
    seen_names: set[str] = set()
    for item in competitor_findings:
        if not _is_true_competitor_finding(item):
            continue
        name = str(item.get("competitor_name") or "").strip()
        if not name or name.lower() in seen_names:
            continue
        if name.lower() in haystack:
            referenced.append(item)
            seen_names.add(name.lower())
    return referenced


def _normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _is_duplicate_of_top_risks(text: str, top_risks: list[str]) -> bool:
    """True when text matches (or nearly matches) an item already in top_risks."""
    candidate = _normalize_for_dedupe(text)
    if not candidate:
        return False
    for risk in top_risks:
        normalized = _normalize_for_dedupe(risk)
        if not normalized:
            continue
        if candidate == normalized:
            return True
        shorter, longer = (
            (candidate, normalized)
            if len(candidate) <= len(normalized)
            else (normalized, candidate)
        )
        if len(shorter) >= 24 and (shorter in longer or longer in shorter):
            return True
        if SequenceMatcher(None, candidate, normalized).ratio() >= 0.88:
            return True
    return False


_MAX_TOP_RISKS = 8

_MAP_DATA_LIMITATION = (
    "Map data services were temporarily unavailable during this search — "
    "location scoring may be less reliable than usual."
)

_TOOL_FAILURE_MARKERS = (
    "temporarily unavailable",
    "service outage",
    "services were unavailable",
    "service was unavailable",
    "timed out",
    "timeout",
    "map-data service",
    "map data service",
    "map data services",
    "could not be verified due to",
    "could not reach",
    "api failed",
    "api error",
    "tool failed",
    "search failed",
    "overpass",
    "data source failed",
    "incomplete data due to",
    "returned incomplete",
    "failed to fetch",
    "failed to retrieve",
)


def _is_tool_failure_text(text: str) -> bool:
    """True when the text describes a tool/API failure, not a business risk."""
    lowered = text.lower()
    return any(marker in lowered for marker in _TOOL_FAILURE_MARKERS)


def _specificity_score(text: str) -> int:
    """Prefer bullets with numbers, prices, named tools, and concrete detail."""
    score = 0
    if re.search(r"\d", text):
        score += 3
    if re.search(r"(?:[$₹€£]|Rs\.?|PKR|USD|AED|%\b)", text, flags=re.IGNORECASE):
        score += 2
    if re.search(
        r"\b(?:POS|Shopify|WhatsApp|Instagram|TikTok|Facebook)\b",
        text,
        flags=re.IGNORECASE,
    ):
        score += 2
    if len(text) >= 90:
        score += 1
    if len(text) >= 140:
        score += 1
    # Penalize very vague short blurbs.
    if len(text) < 40:
        score -= 1
    return score


def _sort_by_specificity(items: list[str]) -> list[str]:
    return sorted(
        items,
        key=lambda text: (_specificity_score(text), len(text)),
        reverse=True,
    )


def _filter_category_items(
    items: list[str],
    *,
    risks_for_dedupe: list[str],
) -> list[str]:
    """Dedupe, drop risk duplicates, keep specificity order — no length trim."""
    selected: list[str] = []
    for text in _sort_by_specificity(list(dict.fromkeys(items))):
        cleaned = sanitize_display_text(text).strip()
        if (
            cleaned
            and cleaned not in selected
            and not _is_duplicate_of_top_risks(cleaned, risks_for_dedupe)
        ):
            selected.append(cleaned)
    return selected


def _format_customer_gap_reason(reason: str, delivery_mode: str | None) -> str:
    cleaned = sanitize_display_text(reason or "").strip()
    if (delivery_mode or "").lower() == "remote" or "stated location" in cleaned.lower():
        return (
            "Customer voice gap: searched for general customer feedback about "
            "this type of product rather than local reviews."
        )
    if cleaned:
        return f"Customer voice gap: {cleaned}"
    return (
        "Customer voice gap: little usable customer feedback turned up for "
        "this idea."
    )


def _pricing_intelligence_lines(competitor_findings: list[dict]) -> list[str]:
    lines: list[str] = []
    for item in competitor_findings or []:
        if item.get("category") != "pricing_intelligence":
            continue
        finding = sanitize_display_text(str(item.get("finding") or "")).strip()
        if not finding:
            continue
        name = sanitize_display_text(str(item.get("competitor_name") or "")).strip()
        if name:
            lines.append(f"Pricing intel ({name}): {finding}")
        else:
            lines.append(f"Pricing intel: {finding}")
    return lines


def _extract_key_findings(
    venture_data: dict,
    *,
    top_risks: list[str] | None = None,
) -> dict[str, list[str]]:
    """Build categorized, detail-rich highlight bullets by source.

    Returns:
        {
          market_research, competitors, customer_pain_points,
          operational_pain_points, customer_segments, hiring_signal
        }
    Each non-empty source contributes its items (specificity-sorted, no
    cross-category crowding or arbitrary global trim).
    """
    strategy = venture_data.get("strategy") or {}
    risks_for_dedupe = (
        top_risks
        if top_risks is not None
        else _extract_top_risks(strategy)
    )
    competitor_findings = venture_data.get("competitor_findings") or []
    customer = _customer_bundle(venture_data)
    hiring = venture_data.get("hiring_signal") or {}
    delivery_mode = venture_data.get("delivery_mode")

    market_items: list[str] = []
    for item in venture_data.get("research_findings") or []:
        text = sanitize_display_text(str(item.get("finding") or "")).strip()
        if text:
            market_items.append(text)

    competitor_items = list(_merged_competitor_key_lines(competitor_findings))
    competitor_items.extend(_pricing_intelligence_lines(competitor_findings))

    # Strategy cross-check: never drop a named competitor the strategy cited.
    merged_by_name: dict[str, str] = {}
    for line in _merged_competitor_key_lines(competitor_findings):
        if line.lower().startswith("competitor "):
            name_part = line[len("Competitor ") :].split(":", 1)[0].strip().lower()
            if name_part:
                merged_by_name[name_part] = line
    for item in _strategy_referenced_competitors(strategy, competitor_findings):
        name = str(item.get("competitor_name") or "").strip()
        line = merged_by_name.get(name.lower())
        cleaned = sanitize_display_text(line or "").strip()
        if (
            cleaned
            and cleaned not in competitor_items
            and not _competitor_name_in_highlights(name, competitor_items)
        ):
            competitor_items.append(cleaned)

    pain_items: list[str] = []
    operational_items: list[str] = []
    reason = customer.get("no_local_data_reason")
    has_customer_content = bool(
        customer.get("customer_segments")
        or customer.get("customer_pain_points")
        or customer.get("operational_pain_points")
    )
    if reason and not has_customer_content:
        pain_items.append(_format_customer_gap_reason(str(reason), delivery_mode))
    for pain in customer.get("customer_pain_points") or []:
        pain_text = sanitize_display_text(str(pain.get("pain_point") or "")).strip()
        if pain_text:
            pain_items.append(pain_text)
    for pain in customer.get("operational_pain_points") or []:
        pain_text = sanitize_display_text(str(pain.get("pain_point") or "")).strip()
        if pain_text:
            operational_items.append(pain_text)

    segment_items: list[str] = []
    for segment in customer.get("customer_segments") or []:
        name = sanitize_display_text(str(segment.get("segment_name") or "")).strip()
        needs = [
            sanitize_display_text(str(n)).strip()
            for n in (segment.get("needs") or [])
            if str(n).strip()
        ]
        if name and needs:
            keep_n = 5 if sum(len(n) for n in needs) > 80 else 3
            segment_items.append(
                f"'{name}' needs: {', '.join(needs[:keep_n])}"
            )
        elif name:
            segment_items.append(name)

    hiring_items: list[str] = []
    if hiring.get("confidence") in {"strong", "weak"}:
        employers = hiring.get("employer_count")
        postings = hiring.get("postings_found") or []
        line = (
            f"Hiring signal is {hiring['confidence']} "
            f"({employers} employer(s) posting related roles)"
        )
        sample: list[str] = []
        for posting in postings[:2]:
            if isinstance(posting, dict):
                role = sanitize_display_text(str(posting.get("role") or "")).strip()
                detail = sanitize_display_text(str(posting.get("detail") or "")).strip()
                if role and detail:
                    sample.append(f"{role}: {detail}")
                elif role or detail:
                    sample.append(role or detail)
            else:
                text = sanitize_display_text(str(posting)).strip()
                if text:
                    sample.append(text)
        if sample:
            line = f"{line}: e.g. {'; '.join(sample)}"
        else:
            line = f"{line}."
        hiring_items.append(line)

    return {
        "market_research": _filter_category_items(
            market_items, risks_for_dedupe=risks_for_dedupe
        ),
        "competitors": _filter_category_items(
            competitor_items, risks_for_dedupe=risks_for_dedupe
        ),
        "customer_pain_points": _filter_category_items(
            pain_items, risks_for_dedupe=risks_for_dedupe
        ),
        "operational_pain_points": _filter_category_items(
            operational_items, risks_for_dedupe=risks_for_dedupe
        ),
        "customer_segments": _filter_category_items(
            segment_items, risks_for_dedupe=risks_for_dedupe
        ),
        "hiring_signal": _filter_category_items(
            hiring_items, risks_for_dedupe=risks_for_dedupe
        ),
    }


def _extract_data_limitations(venture_data: dict) -> list[str]:
    """Collect tool/API failure notes — distinct from business top_risks."""
    limitations: list[str] = []
    seen_norm: list[str] = []

    def try_add(text: str | None) -> None:
        cleaned = sanitize_display_text(text or "").strip()
        if not cleaned:
            return
        norm = _normalize_for_dedupe(cleaned)
        if any(
            norm == prior
            or (len(norm) >= 24 and (norm in prior or prior in norm))
            or SequenceMatcher(None, norm, prior).ratio() >= 0.88
            for prior in seen_norm
        ):
            return
        limitations.append(cleaned)
        seen_norm.append(norm)

    location = venture_data.get("location_data") or {}
    location_status = str(
        location.get("data_status") or location.get("status") or ""
    ).lower()
    if location_status == "overpass_unavailable":
        # Prefer a clear user-facing note over internal service wording.
        try_add(_MAP_DATA_LIMITATION)

    strategy = venture_data.get("strategy") or {}
    for item in strategy.get("verification_needed") or []:
        text = sanitize_display_text(str(item)).strip()
        if not text or not _is_tool_failure_text(text):
            continue
        # Already covered by the canonical map-outage note.
        if location_status == "overpass_unavailable" and any(
            token in text.lower()
            for token in ("map", "overpass", "location scor", "competitor density")
        ):
            continue
        try_add(text)

    return limitations


def _extract_top_risks(strategy: dict, *, max_items: int = _MAX_TOP_RISKS) -> list[str]:
    """Surface distinct business verification risks (exclude tool/API failures)."""
    risks: list[str] = []
    seen_norm: list[str] = []
    for item in strategy.get("verification_needed") or []:
        text = sanitize_display_text(str(item)).strip()
        if not text or _is_tool_failure_text(text):
            continue
        norm = _normalize_for_dedupe(text)
        if any(
            norm == prior
            or (len(norm) >= 24 and (norm in prior or prior in norm))
            or SequenceMatcher(None, norm, prior).ratio() >= 0.88
            for prior in seen_norm
        ):
            continue
        risks.append(text)
        seen_norm.append(norm)
        if len(risks) >= max_items:
            break
    return risks


def _extract_immediate_next_steps(task_plan: list[dict]) -> list[dict]:
    """P0 tasks only, preserving planner order."""
    steps: list[dict] = []
    for task in task_plan or []:
        if str(task.get("priority") or "").upper() != "P0":
            continue
        step = {
            "task_id": task.get("task_id"),
            "phase": task.get("phase"),
            "title": task.get("title"),
            "priority": task.get("priority"),
            "source": task.get("source"),
            "status": task.get("status") or "pending",
        }
        depends_on = task.get("depends_on")
        if depends_on:
            step["depends_on"] = depends_on
        steps.append(step)
    return steps


def _generate_headline(
    venture_data: dict,
    strategy: dict,
    key_findings: dict[str, list[str]],
) -> str:
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(HeadlineResult)

    prompt = (
        "Write exactly ONE plain-language sentence summarizing this venture opportunity.\n"
        "Rules:\n"
        "- One sentence only.\n"
        "- No jargon, no buzzwords, no markdown.\n"
        "- Mention the business idea and the overall opportunity read "
        "(e.g. promising but contested, unclear, strong, not recommended).\n"
        "- Do not invent facts beyond the inputs.\n\n"
        f"{PLAIN_LANGUAGE_BLOCK}\n"
        f"Business idea: {venture_data.get('business_idea')}\n"
        f"Delivery mode: {venture_data.get('delivery_mode')}\n"
        f"Opportunity tier: {strategy.get('opportunity_tier')}\n"
        f"Strategy confidence: {strategy.get('confidence')}\n"
        f"Positioning: {strategy.get('positioning')}\n"
        f"Key findings:\n{json.dumps(key_findings, indent=2)}"
    )
    result = invoke_with_retry(
        lambda: structured_llm.invoke(prompt),
        label="venture-report-headline",
    )
    headline = (result.headline or "").strip()
    if not headline:
        idea = venture_data.get("business_idea") or "this venture"
        tier = strategy.get("opportunity_tier") or "unclear"
        return f"The opportunity for {idea} currently looks {tier.replace('_', ' ')}."
    return sanitize_display_text(headline)


def _recommended_strategy(strategy: dict) -> dict:
    """Core go-to-market recommendation fields, copied as-is from strategy."""
    differentiators = strategy.get("key_differentiators") or []
    if not isinstance(differentiators, list):
        differentiators = [str(differentiators)]
    return {
        "positioning": strategy.get("positioning") or "",
        "target_customers": strategy.get("target_customers") or "",
        "pricing_strategy": strategy.get("pricing_strategy") or "",
        "estimated_budget": strategy.get("estimated_budget") or "",
        "key_differentiators": [str(item) for item in differentiators],
    }


def _analysis_research_bundle(venture_data: dict) -> dict:
    """Compact research payload for the narrative analysis prompt."""
    customer = _customer_bundle(venture_data)
    competitor_findings = venture_data.get("competitor_findings") or []
    true_competitors = [
        {
            "competitor_name": item.get("competitor_name"),
            "finding": item.get("finding"),
            "category": item.get("category"),
        }
        for item in competitor_findings
        if isinstance(item, dict) and _is_true_competitor_finding(item)
    ]
    research_findings = [
        {"finding": item.get("finding"), "source": item.get("source")}
        for item in (venture_data.get("research_findings") or [])
        if isinstance(item, dict) and item.get("finding")
    ]
    location = venture_data.get("location_data") or {}
    hiring = venture_data.get("hiring_signal") or {}
    strategy = venture_data.get("strategy") or {}

    return {
        "business_idea": venture_data.get("business_idea"),
        "delivery_mode": venture_data.get("delivery_mode"),
        "research_findings": research_findings,
        "competitor_findings": true_competitors,
        "customer_research": {
            "customer_segments": customer.get("customer_segments") or [],
            "customer_pain_points": customer.get("customer_pain_points") or [],
            "operational_pain_points": customer.get("operational_pain_points") or [],
            "confidence": customer.get("confidence"),
            "no_local_data_reason": customer.get("no_local_data_reason"),
        },
        "location_data": {
            "location_name": location.get("location_name"),
            "competitor_count_nearby": location.get("competitor_count_nearby"),
            "score": location.get("score"),
            "explanation": location.get("explanation"),
            "data_confidence": location.get("data_confidence"),
            "note": location.get("note"),
            "status": location.get("status"),
        },
        "hiring_signal": {
            "confidence": hiring.get("confidence"),
            "employer_count": hiring.get("employer_count"),
            "postings_found": hiring.get("postings_found") or [],
        },
        "strategy_context": {
            "opportunity_tier": strategy.get("opportunity_tier"),
            "confidence": strategy.get("confidence"),
            "positioning": strategy.get("positioning"),
            "target_customers": strategy.get("target_customers"),
            "pricing_strategy": strategy.get("pricing_strategy"),
            "estimated_budget": strategy.get("estimated_budget"),
            "key_differentiators": strategy.get("key_differentiators") or [],
            "assumptions": strategy.get("assumptions") or [],
            "verification_needed": strategy.get("verification_needed") or [],
            "first_call_question": strategy.get("first_call_question"),
        },
    }


def _generate_full_analysis(
    venture_data: dict,
    *,
    opportunity_tier: str | None,
    key_findings: dict[str, list[str]],
    top_risks: list[str],
) -> str:
    """Multi-paragraph Gemini writeup synthesizing the full research picture."""
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3)
    structured_llm = llm.with_structured_output(FullAnalysisResult)
    bundle = _analysis_research_bundle(venture_data)
    tier_label = (opportunity_tier or "unclear").replace("_", " ")

    prompt = (
        "You are writing a short analyst memo for someone considering this "
        "business idea. Produce a multi-paragraph written report (full_analysis) "
        "that synthesizes ALL of the research into connected reasoning — not a "
        "bullet list, not isolated one-liners.\n\n"
        "What the writeup must cover, in flowing prose:\n"
        "1. Market context — what is going on in this space locally / online.\n"
        "2. Real competitors — name the actual businesses and what they offer, "
        "and what they seem to be missing.\n"
        "3. Customers — what people are actually complaining about or asking for "
        "(use pain points and segments; if local voice is missing, say so plainly).\n"
        "4. Location tradeoffs — density, score, and what that means on the ground "
        "(skip physical location inventiveness if delivery is remote / location "
        "was skipped).\n"
        "5. Hiring / labor context briefly if it adds color, not as the whole story.\n"
        "6. Close by explaining why the opportunity verdict of "
        f"\"{tier_label}\" makes sense given everything above — connect the dots.\n\n"
        "VOICE (mandatory):\n"
        "- Write as if explaining it to a friend who is thinking about opening "
        "this business — conversational, concrete, and plain.\n"
        "- Never mention internal agent/system names (research agent, location "
        "agent, competitor agent, customer agent, orchestrator, pipeline, etc.).\n"
        "- Do not make \"data,\" \"sources,\" \"evidence,\" \"reports,\" "
        "\"signals,\" or \"findings\" the subject of the sentence. Talk about the "
        "real-world situation (what's on the street, what customers complain "
        "about, what rivals seem to offer).\n"
        "- Prefer \"you,\" \"locals,\" \"nearby shops,\" \"customers,\" "
        "\"competitors\" over analyst jargon.\n"
        "- No markdown headings, no bullet lists, no numbered lists. Plain "
        "paragraphs only (roughly 4–7 paragraphs).\n"
        "- Do not invent facts beyond the inputs. If something is thin or "
        "missing, say so honestly.\n\n"
        f"Opportunity tier verdict to justify: {opportunity_tier}\n"
        f"Structured key findings (for grounding, do not copy as a list):\n"
        f"{json.dumps(key_findings, indent=2)}\n"
        f"Top risks (weave in naturally where relevant):\n"
        f"{json.dumps(top_risks, indent=2)}\n\n"
        f"Full research inputs:\n{json.dumps(bundle, indent=2, ensure_ascii=False)}"
    )

    result = invoke_with_retry(
        lambda: structured_llm.invoke(prompt),
        label="venture-report-full-analysis",
    )
    text = (result.full_analysis or "").strip()
    if text:
        return text

    idea = venture_data.get("business_idea") or "this venture"
    return (
        f"Here is the short version on {idea}: the picture so far points to a "
        f"{tier_label} opportunity, but the detailed writeup could not be "
        "generated. Use the key findings and risks below as your checklist "
        "until a fuller memo is available."
    )


def generate_final_report(venture_data: dict) -> dict:
    """Build a consolidated summary report from full pipeline output."""
    strategy = venture_data.get("strategy") or {}
    task_plan = venture_data.get("task_plan") or []

    data_limitations = _extract_data_limitations(venture_data)
    top_risks = _extract_top_risks(strategy)
    key_findings = _extract_key_findings(venture_data, top_risks=top_risks)
    immediate_next_steps = _extract_immediate_next_steps(task_plan)
    headline = _generate_headline(venture_data, strategy, key_findings)
    recommended_strategy = _recommended_strategy(strategy)
    full_analysis = _generate_full_analysis(
        venture_data,
        opportunity_tier=strategy.get("opportunity_tier"),
        key_findings=key_findings,
        top_risks=top_risks,
    )

    return {
        "headline": headline,
        "opportunity_tier": strategy.get("opportunity_tier"),
        "confidence": strategy.get("confidence"),
        "idea_score": strategy.get("idea_score"),
        "idea_score_label": strategy.get("idea_score_label"),
        "recommended_strategy": recommended_strategy,
        "key_findings": key_findings,
        "top_risks": top_risks,
        "data_limitations": data_limitations,
        "immediate_next_steps": immediate_next_steps,
        "full_analysis": full_analysis,
        "full_data": venture_data,
    }


if __name__ == "__main__":
    # Real pipeline data from venture id 33: "bakery in Bahria Town, Islamabad"
    sample = {
        "business_idea": "bakery in Bahria Town, Islamabad",
        "delivery_mode": "onsite",
        "research_findings": [
            {
                "finding": (
                    "New artisanal dessert spots are actively entering the market, "
                    "such as Bundtly launching in Bahria Town Phase 4 to offer custom "
                    "cakes, freshly baked bundts, and signature cupcake bouquets."
                ),
                "source": "https://www.instagram.com/reel/DYbyOMDM1Us",
            },
            {
                "finding": (
                    "Major regional chains like Cakes n Bakes are expanding into the "
                    "area with new bakery experiences, notably launching a branch at "
                    "Bahria Safari."
                ),
                "source": (
                    "https://www.facebook.com/cakesnbakespk/posts/bahria-safari-get-ready-"
                    "to-feast-the-wait-is-officially-overcakes-n-bakes-is-lau/1471312728367157"
                ),
            },
            {
                "finding": (
                    "AMU'S Bakeshop located in the Civic Centre is recognized for "
                    "providing hygienic food items, reasonable pricing, and promotional "
                    "discounts."
                ),
                "source": "https://www.tiktok.com/@axeempasha/video/7322468676446293253",
            },
        ],
        "competitor_findings": [
            {
                "competitor_name": "Al Fazal Bakers and Sweets",
                "finding": (
                    "Located in Bahria Safari Valley, offering traditional sweets and "
                    "bakery items locally."
                ),
                "source": "https://mybahria.com.pk/directory-list/bakeries-and-cake-shop",
            },
            {
                "competitor_name": "AMU'S Bakeshop",
                "finding": (
                    "Located in Civic Center, Bahria Town, known for being hygienic, "
                    "offering a variety of bakery items with reasonable prices and "
                    "occasional discounts."
                ),
                "source": "https://mybahria.com.pk/directory-list/bakeries-and-cake-shop",
            },
            {
                "competitor_name": "Cravings By Anza",
                "finding": (
                    "Operating out of Phase 7, Bahria Town, catering to local residential "
                    "customers as a specialized bakery service."
                ),
                "source": "https://mybahria.com.pk/directory-list/bakeries-and-cake-shop",
            },
            {
                "competitor_name": "Bundtly",
                "finding": (
                    "An upcoming specialty bakery opening in Bahria Town Phase 4 focusing "
                    "on custom cakes, fresh bundts, and signature cupcake bouquets."
                ),
                "source": "https://www.instagram.com/reel/DX2GeJIMAee",
            },
        ],
        "customer_research": {
            "customer_segments": [
                {
                    "segment_name": "Dessert and Sweets Enthusiasts",
                    "evidence": (
                        "One of the most popular spots for sweets in Bahria Town "
                        "Islamabad is Sweets and Bakers. Known for its quality and "
                        "variety, this shop offers a range of traditional Pakistani "
                        "sweets such as barfi, ras malai, and jalebi."
                    ),
                    "needs": [
                        "Fresh and authentic traditional Pakistani sweets like barfi, ras malai, and jalebi",
                        "Fusion and modern dessert options",
                        "Affordable pricing for daily indulgences",
                    ],
                    "source": (
                        "https://vocal.media/art/where-to-find-the-best-sweets-in-bahria-town-islamabad"
                    ),
                },
                {
                    "segment_name": "Event Planners and Gift Givers",
                    "evidence": (
                        "Their freshly prepared desserts are perfect for celebrations "
                        "or a quick indulgence... Ideal for gifting or events"
                    ),
                    "needs": [
                        "Custom cakes, freshly baked bundts, and signature cupcake bouquets for celebrations",
                        "On-spot purchase availability for good food items to take to gatherings",
                    ],
                    "source": (
                        "https://vocal.media/art/where-to-find-the-best-sweets-in-bahria-town-islamabad"
                    ),
                },
            ],
            "customer_pain_points": [
                {
                    "pain_point": (
                        "Difficulty finding reliable reasonable home bakers or on-spot "
                        "shops for specific party items like mini pizza, mini burgers, "
                        "and brownies in Bahria Town."
                    ),
                    "source": (
                        "https://www.facebook.com/groups/2650072818601204/posts/3189766181298529"
                    ),
                },
                {
                    "pain_point": (
                        "Limited selection of cakes when trying to send orders or gifts "
                        "to friends in areas like Bahria Enclave Islamabad."
                    ),
                    "source": (
                        "https://www.reddit.com/r/islamabad/comments/1gzpf55/"
                        "want_to_send_a_cake_to_my_friend_in_isb_bahria"
                    ),
                },
            ],
            "confidence": "strong",
        },
        "hiring_signal": {
            "postings_found": [
                {
                    "role": "Bakery Consultant",
                    "detail": (
                        "Sales Executive, Waitress, and Night Shift Inventory & "
                        "Stock Management positions in Bahria Town/Rawalpindi."
                    ),
                },
                {
                    "role": "Store Keeper",
                    "detail": "Position at Rahat Bakers in Bahria Town.",
                },
                {
                    "role": "Dessert Chef",
                    "detail": "Role requiring 5-6 years experience in Bahria Town.",
                },
                {
                    "role": "Food Technologist",
                    "detail": (
                        "Bakery team opening in Islamabad and Rawalpindi area "
                        "by Continental Bakers."
                    ),
                },
            ],
            "employer_count": 4,
            "confidence": "strong",
            "query_used": "bakery jobs hiring Bahria Town Islamabad",
        },
        "location_data": {
            "location_name": "Bahria Town Phase 4, Bahria Town, Islamabad Capital Territory, Pakistan",
            "coordinates": {"latitude": 33.5502173, "longitude": 73.1239422},
            "competitor_count_nearby": 14,
            "score": 45,
            "explanation": (
                "Within 1500m: 14 direct competitors and 44 commercial POIs nearby. "
                "Competition score: 0/55, commercial activity score: 45/45."
            ),
            "data_confidence": "high",
            "note": None,
        },
        "strategy": {
            "positioning": (
                "Position as a specialized artisanal bakery focusing on custom "
                "celebration items and quick-indulgence party snacks that bridge the "
                "gap between traditional sweet shops and modern custom bakeries in "
                "Bahria Town."
            ),
            "target_customers": (
                "Dessert and sweets enthusiasts seeking daily indulgences, alongside "
                "event planners and local residents looking for reliable custom cakes, "
                "cupcake bouquets, and party snacks like mini pizzas and brownies."
            ),
            "pricing_strategy": (
                "Competitive and value-driven mid-market pricing, mirroring successful "
                "local actors like AMU'S Bakeshop to appeal to daily purchase needs, "
                "while offering premium tiers for custom event orders."
            ),
            "key_differentiators": [
                "Specialized focus on hard-to-find party items like mini burgers and mini pizzas alongside baked goods",
                "Fresh daily artisanal offerings tailored specifically to local residential demand in Bahria Town phases",
                "Streamlined custom cake ordering and gifting service to address regional delivery limitations",
                "Strong emphasis on product hygiene and consistent quality control",
            ],
            "confidence": 65,
            "assumptions": [
                "Demand for specialty party items and custom celebration cakes remains underserved by existing traditional bakeries",
                "The 14 nearby competitors reported in location data leave specific gaps in niche artisanal offerings",
                "Hiring signal indicates healthy local labor availability for scaling bakery operations",
            ],
            "verification_needed": [
                "Reconcile the high density of 14 nearby competitors reported by location data with specific gaps identified in artisanal and party snack offerings",
                "Verify the exact market saturation and unique value propositions of named competitors such as Bundtly, AMU'S Bakeshop, and Cravings By Anza",
                "Confirm the willingness of local customers to pay premium prices for specialized custom cakes versus established traditional sweet shops",
            ],
            "opportunity_tier": "viable_but_contested",
            "first_call_question": (
                "What specific party items or custom cake styles are local residents "
                "currently struggling to find from existing competitors like AMU'S "
                "Bakeshop and local home bakers?"
            ),
        },
        "task_plan": [
            {
                "task_id": "T1",
                "phase": "validate",
                "title": (
                    "What specific party items or custom cake styles are local residents "
                    "currently struggling to find from existing competitors like AMU'S "
                    "Bakeshop and local home bakers?"
                ),
                "priority": "P0",
                "source": "first_call_question",
                "status": "pending",
            },
            {
                "task_id": "T2",
                "phase": "validate",
                "title": (
                    "Reconcile the high density of 14 nearby competitors reported by "
                    "location data with specific gaps identified in artisanal and party "
                    "snack offerings"
                ),
                "priority": "P0",
                "depends_on": ["T1"],
                "source": "verification_needed",
                "status": "pending",
            },
            {
                "task_id": "T3",
                "phase": "validate",
                "title": (
                    "Verify the exact market saturation and unique value propositions of "
                    "named competitors such as Bundtly, AMU'S Bakeshop, and Cravings By Anza"
                ),
                "priority": "P0",
                "depends_on": ["T1"],
                "source": "verification_needed",
                "status": "pending",
            },
            {
                "task_id": "T4",
                "phase": "validate",
                "title": (
                    "Confirm the willingness of local customers to pay premium prices for "
                    "specialized custom cakes versus established traditional sweet shops"
                ),
                "priority": "P0",
                "depends_on": ["T1"],
                "source": "verification_needed",
                "status": "pending",
            },
            {
                "task_id": "T8",
                "phase": "decide",
                "title": "Finalize positioning and pricing based on validation results",
                "priority": "P0",
                "depends_on": ["T1", "T2", "T3", "T4", "T5", "T6", "T7"],
                "source": "positioning",
                "status": "pending",
            },
            {
                "task_id": "T9",
                "phase": "build",
                "title": (
                    "Develop a production line for niche party items including mini "
                    "burgers, mini pizzas, and specialty baked goods"
                ),
                "priority": "P1",
                "depends_on": ["T1", "T2", "T5"],
                "source": "key_differentiators",
                "status": "pending",
            },
            {
                "task_id": "T13",
                "phase": "launch",
                "title": "Prepare go-to-market plan based on finalized strategy",
                "priority": "P1",
                "depends_on": ["T8"],
                "source": "target_customers",
                "status": "pending",
            },
        ],
    }

    report = generate_final_report(sample)
    printable = {k: v for k, v in report.items() if k != "full_data"}
    print(json.dumps(printable, indent=2, ensure_ascii=False))
