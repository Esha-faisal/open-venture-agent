import json
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from gemini_config import GEMINI_MODEL
from gemini_retry import invoke_with_retry

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


def _customer_bundle(venture_data: dict) -> dict:
    """Normalize customer fields whether nested or flat."""
    nested = venture_data.get("customer_research")
    if isinstance(nested, dict):
        return {
            "customer_segments": nested.get("customer_segments") or [],
            "customer_pain_points": nested.get("customer_pain_points") or [],
            "confidence": nested.get("confidence") or "none",
            "no_local_data_reason": nested.get("no_local_data_reason"),
        }
    return {
        "customer_segments": venture_data.get("customer_segments") or [],
        "customer_pain_points": venture_data.get("customer_pain_points") or [],
        "confidence": venture_data.get("customer_confidence") or "none",
        "no_local_data_reason": venture_data.get("no_local_data_reason"),
    }


def _extract_key_findings(venture_data: dict, *, limit: int = 5) -> list[str]:
    """Pick highlight findings in priority order (capped at limit).

    Priority: verification contradictions → real named competitors → customer
    pain points → location score → other supporting signals. Vendor/technology
    and pricing-intelligence entries never compete for these slots.
    """
    highlights: list[str] = []

    def add(text: str | None) -> None:
        cleaned = (text or "").strip()
        if cleaned and cleaned not in highlights and len(highlights) < limit:
            highlights.append(cleaned)

    strategy = venture_data.get("strategy") or {}
    location = venture_data.get("location_data") or {}

    # 1) Verification / contradiction signals
    for item in strategy.get("verification_needed") or []:
        text = str(item).strip()
        if not text:
            continue
        lowered = text.lower()
        if any(
            marker in lowered
            for marker in (
                "contradict",
                "conflict",
                "discrepan",
                "disagree",
                "mismatch",
                "inconsist",
            )
        ):
            add(text)
    if location.get("data_confidence") == "low" and location.get("note"):
        add(f"Location data conflict: {location['note']}")

    # 2) Real named competitors only (missing category treated as competitor)
    for item in venture_data.get("competitor_findings") or []:
        category = item.get("category")
        if category in {
            "vendor_technology",
            "pricing_intelligence",
            "directory_source",
        }:
            continue
        name = item.get("competitor_name")
        finding = item.get("finding")
        if name and finding:
            add(f"Competitor {name}: {finding}")
        elif finding:
            add(str(finding))

    # 3) Customer pain points (and explicit local-data gap)
    customer = _customer_bundle(venture_data)
    reason = customer.get("no_local_data_reason")
    if reason and not (customer.get("customer_segments") or customer.get("customer_pain_points")):
        add(f"Local customer data gap: {reason}")
    for pain in customer.get("customer_pain_points") or []:
        add(f"Customer pain: {pain.get('pain_point')}")

    # 4) Location score
    if location.get("score") is not None and location.get("location_name"):
        add(
            f"Location score {location['score']}/100 for "
            f"{location['location_name']} "
            f"({location.get('competitor_count_nearby', '?')} nearby competitors)."
        )

    # 5) Supporting fillers only if slots remain
    for segment in customer.get("customer_segments") or []:
        name = segment.get("segment_name")
        needs = segment.get("needs") or []
        if name and needs:
            add(f"Customer segment '{name}' needs: {', '.join(str(n) for n in needs[:3])}")
        elif name:
            add(f"Customer segment: {name}")

    hiring = venture_data.get("hiring_signal") or {}
    if hiring.get("confidence") in {"strong", "weak"}:
        employers = hiring.get("employer_count")
        add(
            f"Hiring signal is {hiring['confidence']} "
            f"({employers} employer(s) posting related roles)."
        )

    for item in strategy.get("verification_needed") or []:
        add(str(item))

    for item in venture_data.get("research_findings") or []:
        add(item.get("finding"))

    return highlights[:limit]


def _extract_top_risks(strategy: dict, *, limit: int = 3) -> list[str]:
    risks = [
        str(item).strip()
        for item in (strategy.get("verification_needed") or [])
        if str(item).strip()
    ]
    return risks[:limit]


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


def _generate_headline(venture_data: dict, strategy: dict, key_findings: list[str]) -> str:
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
    return headline


def generate_final_report(venture_data: dict) -> dict:
    """Build a consolidated summary report from full pipeline output."""
    strategy = venture_data.get("strategy") or {}
    task_plan = venture_data.get("task_plan") or []

    key_findings = _extract_key_findings(venture_data)
    top_risks = _extract_top_risks(strategy)
    immediate_next_steps = _extract_immediate_next_steps(task_plan)
    headline = _generate_headline(venture_data, strategy, key_findings)

    return {
        "headline": headline,
        "opportunity_tier": strategy.get("opportunity_tier"),
        "confidence": strategy.get("confidence"),
        "key_findings": key_findings,
        "top_risks": top_risks,
        "immediate_next_steps": immediate_next_steps,
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
                "Bakery Consultant, Sales Executive, Waitress, and Night Shift Inventory & Stock Management positions in Bahria Town/Rawalpindi",
                "Store Keeper position at Rahat Bakers in Bahria Town",
                "Dessert Chef position with 5-6 years experience in Bahria Town",
                "Food Technologist position for bakery team in Islamabad and Rawalpindi area by Continental Bakers",
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
