import asyncio
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_serializer
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import (
    AsyncSessionLocal,
    CustomerResearch,
    Location,
    ResearchFinding,
    Strategy,
    Task,
    Venture,
    create_tables,
)
from orchestrator import run_venture_pipeline
from text_sanitize import sanitize_display_text


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await create_tables()
    yield


app = FastAPI(lifespan=lifespan)
frontend_url = os.getenv("FRONTEND_URL", "").rstrip("/")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url] if frontend_url else [],
   allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?|https://ova-[a-z0-9]+-esha-797a\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)


class HealthResponse(BaseModel):
    status: str


class VentureRequest(BaseModel):
    business_idea: str


class InvalidInputResponse(BaseModel):
    status: Literal["invalid_input"] = "invalid_input"
    message: str


class MissingLocationResponse(BaseModel):
    status: Literal["missing_location"] = "missing_location"
    message: str


class ExistingBusinessResponse(BaseModel):
    status: Literal["existing_business_detected"] = "existing_business_detected"
    message: str


EarlyExitResponse = (
    InvalidInputResponse | MissingLocationResponse | ExistingBusinessResponse
)


class FindingResponse(BaseModel):
    finding: str
    source: str
    source_agent: str
    # Only present for Competitor Agent findings.
    competitor_name: str | None = None
    category: str | None = None
    possible_duplicate_of: str | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, serializer):
        data = serializer(self)
        if self.source_agent != "competitor":
            data.pop("competitor_name", None)
            data.pop("category", None)
            data.pop("possible_duplicate_of", None)
        else:
            if not data.get("competitor_name"):
                data.pop("competitor_name", None)
            if not data.get("category"):
                data.pop("category", None)
            if not data.get("possible_duplicate_of"):
                data.pop("possible_duplicate_of", None)
        return data


class CoordinatesResponse(BaseModel):
    latitude: float | None = None
    longitude: float | None = None


class LocationResponse(BaseModel):
    location_name: str | None = None
    coordinates: CoordinatesResponse | None = None
    competitor_count: int | None = None
    score: int | None = None
    explanation: str | None = None
    status: str | None = None
    data_status: str | None = None
    data_confidence: str | None = "high"
    note: str | None = None


class ProsConsItemResponse(BaseModel):
    point: str = ""
    reasoning: str = ""
    source: str = ""


class ProsAndConsResponse(BaseModel):
    pros: list[ProsConsItemResponse] = []
    cons: list[ProsConsItemResponse] = []


class StrategyResponse(BaseModel):
    positioning: str
    target_customers: str
    pricing_strategy: str
    estimated_budget: str | None = None
    key_differentiators: list[str]
    confidence: int
    idea_score: int | None = None
    idea_score_label: str | None = None
    assumptions: list[str]
    verification_needed: list[str]
    opportunity_tier: str | None = None
    first_call_question: str | None = None
    pros_and_cons: ProsAndConsResponse | None = None


class HiringPostingResponse(BaseModel):
    role: str = ""
    detail: str = ""


class HiringSignalResponse(BaseModel):
    postings_found: list[HiringPostingResponse] = []
    employer_count: int = 0
    confidence: str = "none"
    query_used: str | None = None


class CustomerSegmentResponse(BaseModel):
    segment_name: str
    evidence: str
    needs: list[str]
    source: str


class CustomerPainPointResponse(BaseModel):
    pain_point: str
    source: str


class CustomerResearchResponse(BaseModel):
    customer_segments: list[CustomerSegmentResponse] = []
    customer_pain_points: list[CustomerPainPointResponse] = []
    operational_pain_points: list[CustomerPainPointResponse] = []
    confidence: str = "none"
    no_local_data_reason: str | None = None


class TaskResponse(BaseModel):
    task_id: str
    phase: str
    title: str
    priority: str
    depends_on: list[str] = []
    source: str
    status: str

    @model_serializer(mode="wrap")
    def _serialize(self, serializer):
        data = serializer(self)
        if not data.get("depends_on"):
            data.pop("depends_on", None)
        return data


class FinalReportTaskResponse(BaseModel):
    task_id: str | None = None
    phase: str | None = None
    title: str | None = None
    priority: str | None = None
    depends_on: list[str] = []
    source: str | None = None
    status: str | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, serializer):
        data = serializer(self)
        if not data.get("depends_on"):
            data.pop("depends_on", None)
        return data


class RecommendedStrategyResponse(BaseModel):
    positioning: str = ""
    target_customers: str = ""
    pricing_strategy: str = ""
    estimated_budget: str = ""
    key_differentiators: list[str] = []


class KeyFindingsResponse(BaseModel):
    market_research: list[str] = []
    competitors: list[str] = []
    customer_pain_points: list[str] = []
    operational_pain_points: list[str] = []
    customer_segments: list[str] = []
    hiring_signal: list[str] = []


class FinalReportResponse(BaseModel):
    headline: str
    opportunity_tier: str | None = None
    confidence: int | None = None
    idea_score: int | None = None
    idea_score_label: str | None = None
    recommended_strategy: RecommendedStrategyResponse | None = None
    key_findings: KeyFindingsResponse = Field(default_factory=KeyFindingsResponse)
    top_risks: list[str] = []
    data_limitations: list[str] = []
    immediate_next_steps: list[FinalReportTaskResponse] = []
    full_analysis: str | None = None
    full_data: dict | None = None


class VentureResponse(BaseModel):
    id: int
    status: str
    delivery_mode: str | None = None
    findings: list[FindingResponse]
    hiring_signal: HiringSignalResponse | None = None
    customer_research: CustomerResearchResponse | None = None
    location: LocationResponse | None = None
    strategy: StrategyResponse | None = None
    task_plan: list[TaskResponse] = []
    final_report: FinalReportResponse | None = None


class VentureInfo(BaseModel):
    id: int
    business_idea: str
    status: str
    delivery_mode: str | None = None
    created_at: datetime


class StoredResearchFinding(BaseModel):
    id: int
    finding: str
    source: str


class StoredCompetitorFinding(BaseModel):
    id: int
    competitor_name: str | None
    finding: str
    source: str
    category: str | None = None


class VentureDetailResponse(BaseModel):
    venture: VentureInfo
    research_findings: list[StoredResearchFinding]
    competitor_findings: list[StoredCompetitorFinding]
    hiring_signal: HiringSignalResponse | None = None
    customer_research: CustomerResearchResponse | None = None
    location: LocationResponse | None = None
    strategy: StrategyResponse | None = None
    task_plan: list[TaskResponse] = []
    final_report: FinalReportResponse | None = None


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


def _coordinates_response(
    latitude: float | None,
    longitude: float | None,
) -> CoordinatesResponse | None:
    if latitude is None and longitude is None:
        return None
    return CoordinatesResponse(
        latitude=latitude,
        longitude=longitude,
    )


def _location_response_from_analysis(analysis: dict | None) -> LocationResponse | None:
    if analysis is None:
        return None
    data_status = analysis.get("data_status") or analysis.get("status")
    coordinates = analysis.get("coordinates") or {}
    coords = _coordinates_response(
        coordinates.get("latitude"),
        coordinates.get("longitude"),
    )
    if data_status in {"skipped", "location_not_found", "overpass_unavailable"}:
        return LocationResponse(
            location_name=analysis.get("location_name"),
            coordinates=coords,
            competitor_count=None,
            score=None,
            explanation=analysis.get("explanation") or analysis.get("reason"),
            status=analysis.get("status") or data_status,
            data_status=data_status,
            data_confidence=analysis.get("data_confidence") or "high",
            note=analysis.get("note"),
        )
    return LocationResponse(
        location_name=analysis.get("location_name"),
        coordinates=coords,
        competitor_count=analysis.get("competitor_count_nearby"),
        score=analysis.get("score"),
        explanation=analysis.get("explanation"),
        status=analysis.get("status", "ok"),
        data_status=analysis.get("data_status", "ok"),
        data_confidence=analysis.get("data_confidence") or "high",
        note=analysis.get("note"),
    )


def _location_response_from_model(location: Location | None) -> LocationResponse | None:
    if location is None:
        return None
    return LocationResponse(
        location_name=location.location_name,
        coordinates=_coordinates_response(
            location.latitude,
            location.longitude,
        ),
        competitor_count=location.competitor_count,
        score=location.score,
        explanation=location.explanation,
        status="ok",
        data_status="ok",
        data_confidence=location.data_confidence or "high",
        note=location.note,
    )


def _pros_and_cons_response(raw) -> ProsAndConsResponse | None:
    if not isinstance(raw, dict):
        return None

    def _items(key: str) -> list[ProsConsItemResponse]:
        items: list[ProsConsItemResponse] = []
        for item in raw.get(key) or []:
            if isinstance(item, dict):
                point = sanitize_display_text(str(item.get("point") or "")).strip()
                reasoning = sanitize_display_text(
                    str(item.get("reasoning") or "")
                ).strip()
                source = sanitize_display_text(str(item.get("source") or "")).strip()
                if point or reasoning:
                    items.append(
                        ProsConsItemResponse(
                            point=point,
                            reasoning=reasoning,
                            source=source,
                        )
                    )
            else:
                text = sanitize_display_text(str(item or "")).strip()
                if text:
                    items.append(ProsConsItemResponse(point=text, reasoning="", source=""))
        return items

    pros = _items("pros")
    cons = _items("cons")
    if not pros and not cons:
        return None
    return ProsAndConsResponse(pros=pros, cons=cons)


def _strategy_response_from_data(data: dict | None) -> StrategyResponse | None:
    if data is None:
        return None
    return StrategyResponse(
        positioning=data["positioning"],
        target_customers=data["target_customers"],
        pricing_strategy=data["pricing_strategy"],
        estimated_budget=data.get("estimated_budget") or None,
        key_differentiators=data["key_differentiators"],
        confidence=data["confidence"],
        idea_score=data.get("idea_score"),
        idea_score_label=data.get("idea_score_label"),
        assumptions=data["assumptions"],
        verification_needed=data["verification_needed"],
        opportunity_tier=data.get("opportunity_tier"),
        first_call_question=data.get("first_call_question"),
        pros_and_cons=_pros_and_cons_response(data.get("pros_and_cons")),
    )


def _strategy_response_from_model(strategy: Strategy | None) -> StrategyResponse | None:
    if strategy is None:
        return None
    return StrategyResponse(
        positioning=strategy.positioning,
        target_customers=strategy.target_customers,
        pricing_strategy=strategy.pricing_strategy,
        estimated_budget=strategy.estimated_budget,
        key_differentiators=strategy.key_differentiators,
        confidence=strategy.confidence,
        idea_score=strategy.idea_score,
        idea_score_label=strategy.idea_score_label,
        assumptions=strategy.assumptions,
        verification_needed=strategy.verification_needed,
        opportunity_tier=strategy.opportunity_tier,
        first_call_question=strategy.first_call_question,
        pros_and_cons=_pros_and_cons_response(strategy.pros_and_cons),
    )


def _normalize_hiring_posting(item) -> HiringPostingResponse | None:
    """Accept structured {role, detail} or legacy plain-string postings."""
    if isinstance(item, dict):
        role = sanitize_display_text(str(item.get("role") or "")).strip()
        detail = sanitize_display_text(str(item.get("detail") or "")).strip()
        if not role and not detail:
            return None
        return HiringPostingResponse(role=role, detail=detail)
    text = sanitize_display_text(str(item or "")).strip()
    if not text:
        return None
    # Legacy blended string — best-effort title split on first colon/dash.
    for sep in (": ", " — ", " - "):
        if sep in text:
            role, _, detail = text.partition(sep)
            role = role.strip()
            detail = detail.strip()
            if role and detail:
                return HiringPostingResponse(role=role, detail=detail)
    return HiringPostingResponse(role="", detail=text)


def _hiring_signal_response(
    data: dict | None,
) -> HiringSignalResponse | None:
    if not data:
        return None
    postings = [
        posting
        for posting in (
            _normalize_hiring_posting(item)
            for item in (data.get("postings_found") or [])
        )
        if posting is not None
    ]
    return HiringSignalResponse(
        postings_found=postings,
        employer_count=int(data.get("employer_count") or 0),
        confidence=str(data.get("confidence") or "none"),
        query_used=data.get("query_used"),
    )


def _customer_research_response(
    *,
    segments: list[dict] | None = None,
    pain_points: list[dict] | None = None,
    operational_pain_points: list[dict] | None = None,
    confidence: str | None = None,
    no_local_data_reason: str | None = None,
) -> CustomerResearchResponse | None:
    segments = segments or []
    pain_points = pain_points or []
    operational_pain_points = operational_pain_points or []
    conf = confidence or "none"
    return CustomerResearchResponse(
        customer_segments=[
            CustomerSegmentResponse(
                segment_name=item.get("segment_name") or "",
                evidence=item.get("evidence") or "",
                needs=list(item.get("needs") or []),
                source=item.get("source") or "",
            )
            for item in segments
        ],
        customer_pain_points=[
            CustomerPainPointResponse(
                pain_point=item.get("pain_point") or "",
                source=item.get("source") or "",
            )
            for item in pain_points
        ],
        operational_pain_points=[
            CustomerPainPointResponse(
                pain_point=item.get("pain_point") or "",
                source=item.get("source") or "",
            )
            for item in operational_pain_points
        ],
        confidence=conf,
        no_local_data_reason=no_local_data_reason,
    )


def _customer_research_response_from_model(
    row: CustomerResearch | None,
) -> CustomerResearchResponse | None:
    if row is None:
        return None
    return _customer_research_response(
        segments=row.customer_segments or [],
        pain_points=row.customer_pain_points or [],
        operational_pain_points=getattr(row, "operational_pain_points", None) or [],
        confidence=row.confidence,
        no_local_data_reason=row.no_local_data_reason,
    )


def _save_customer_research(
    venture_id: int,
    *,
    segments: list[dict],
    pain_points: list[dict],
    operational_pain_points: list[dict] | None = None,
    confidence: str,
    no_local_data_reason: str | None = None,
) -> CustomerResearch:
    return CustomerResearch(
        venture_id=venture_id,
        customer_segments=segments,
        customer_pain_points=pain_points,
        operational_pain_points=operational_pain_points or [],
        confidence=confidence or "none",
        no_local_data_reason=no_local_data_reason,
    )


def _save_location(venture_id: int, analysis: dict) -> Location:
    coordinates = analysis.get("coordinates") or {}
    return Location(
        venture_id=venture_id,
        location_name=analysis.get("location_name"),
        latitude=coordinates.get("latitude"),
        longitude=coordinates.get("longitude"),
        competitor_count=analysis.get("competitor_count_nearby"),
        score=analysis.get("score"),
        explanation=analysis.get("explanation"),
        data_confidence=analysis.get("data_confidence") or "high",
        note=analysis.get("note"),
    )


def _save_strategy(venture_id: int, data: dict) -> Strategy:
    return Strategy(
        venture_id=venture_id,
        positioning=data["positioning"],
        target_customers=data["target_customers"],
        pricing_strategy=data["pricing_strategy"],
        estimated_budget=data.get("estimated_budget") or None,
        key_differentiators=data["key_differentiators"],
        confidence=data["confidence"],
        idea_score=data.get("idea_score"),
        idea_score_label=data.get("idea_score_label"),
        assumptions=data["assumptions"],
        verification_needed=data["verification_needed"],
        opportunity_tier=data.get("opportunity_tier"),
        first_call_question=data.get("first_call_question"),
        pros_and_cons=data.get("pros_and_cons"),
    )


def _task_response_from_data(item: dict) -> TaskResponse:
    return TaskResponse(
        task_id=item["task_id"],
        phase=item["phase"],
        title=item["title"],
        priority=item["priority"],
        depends_on=list(item.get("depends_on") or []),
        source=item["source"],
        status=item.get("status") or "pending",
    )


def _sanitize_finding_list(items: list | None) -> list[str]:
    return [
        sanitize_display_text(str(item))
        for item in (items or [])
        if str(item).strip()
    ]


def _key_findings_response(raw) -> KeyFindingsResponse:
    """Normalize stored key_findings (categorized dict or legacy flat list)."""
    if isinstance(raw, dict):
        return KeyFindingsResponse(
            market_research=_sanitize_finding_list(raw.get("market_research")),
            competitors=_sanitize_finding_list(raw.get("competitors")),
            customer_pain_points=_sanitize_finding_list(
                raw.get("customer_pain_points")
            ),
            operational_pain_points=_sanitize_finding_list(
                raw.get("operational_pain_points")
            ),
            customer_segments=_sanitize_finding_list(raw.get("customer_segments")),
            hiring_signal=_sanitize_finding_list(raw.get("hiring_signal")),
        )
    if isinstance(raw, list):
        # Legacy flat list — keep visible under market_research.
        return KeyFindingsResponse(market_research=_sanitize_finding_list(raw))
    return KeyFindingsResponse()


def _final_report_response(data: dict | None) -> FinalReportResponse | None:
    if not data:
        return None
    recommended = data.get("recommended_strategy")
    recommended_response = None
    if isinstance(recommended, dict):
        recommended_response = RecommendedStrategyResponse(
            positioning=sanitize_display_text(recommended.get("positioning") or ""),
            target_customers=sanitize_display_text(
                recommended.get("target_customers") or ""
            ),
            pricing_strategy=sanitize_display_text(
                recommended.get("pricing_strategy") or ""
            ),
            estimated_budget=sanitize_display_text(
                recommended.get("estimated_budget") or ""
            ),
            key_differentiators=[
                sanitize_display_text(str(item))
                for item in (recommended.get("key_differentiators") or [])
            ],
        )
    full_analysis = data.get("full_analysis")
    return FinalReportResponse(
        headline=sanitize_display_text(data.get("headline") or ""),
        opportunity_tier=data.get("opportunity_tier"),
        confidence=data.get("confidence"),
        idea_score=data.get("idea_score"),
        idea_score_label=(
            sanitize_display_text(str(data.get("idea_score_label") or "").strip())
            or None
        ),
        recommended_strategy=recommended_response,
        key_findings=_key_findings_response(data.get("key_findings")),
        top_risks=[
            sanitize_display_text(str(item))
            for item in (data.get("top_risks") or [])
            if str(item).strip()
        ],
        data_limitations=[
            sanitize_display_text(str(item))
            for item in (data.get("data_limitations") or [])
            if str(item).strip()
        ],
        immediate_next_steps=[
            FinalReportTaskResponse(
                task_id=step.get("task_id"),
                phase=step.get("phase"),
                title=sanitize_display_text(step.get("title") or "") or step.get("title"),
                priority=step.get("priority"),
                depends_on=list(step.get("depends_on") or []),
                source=step.get("source"),
                status=step.get("status"),
            )
            for step in (data.get("immediate_next_steps") or [])
            if isinstance(step, dict)
        ],
        full_analysis=(
            sanitize_display_text(str(full_analysis).strip()) if full_analysis else None
        ),
        full_data=data.get("full_data"),
    )


def _task_response_from_model(task: Task) -> TaskResponse:
    return TaskResponse(
        task_id=task.task_id,
        phase=task.phase,
        title=task.title,
        priority=task.priority,
        depends_on=list(task.depends_on or []),
        source=task.source,
        status=task.status,
    )


def _save_task(venture_id: int, item: dict) -> Task:
    return Task(
        venture_id=venture_id,
        task_id=item["task_id"],
        phase=item["phase"],
        title=item["title"],
        priority=item["priority"],
        depends_on=list(item.get("depends_on") or []),
        source=item["source"],
        status=item.get("status") or "pending",
    )


@app.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/venture", response_model=VentureResponse | EarlyExitResponse)
async def create_venture(
    payload: VentureRequest,
) -> VentureResponse | EarlyExitResponse:
    business_idea = (payload.business_idea or "").strip()
    if not business_idea:
        return InvalidInputResponse(
            message=(
                "This doesn't look like a business idea — try describing what "
                "you'd like to sell or offer, and ideally where."
            )
        )

    # Run pipeline first so early exits never create a venture row or report.
    pipeline = await asyncio.to_thread(run_venture_pipeline, business_idea)
    if pipeline.get("status") == "invalid_input":
        return InvalidInputResponse(
            message=str(
                pipeline.get("message")
                or (
                    "This doesn't look like a business idea — try describing what "
                    "you'd like to sell or offer, and ideally where."
                )
            )
        )
    if pipeline.get("status") == "existing_business_detected":
        return ExistingBusinessResponse(
            message=str(
                pipeline.get("message")
                or (
                    "This looks like it might be naming an existing company rather "
                    "than describing a new business idea. Try describing your own "
                    "concept instead — e.g. 'a mid-range clothing brand in Lahore.'"
                )
            )
        )
    if pipeline.get("status") == "missing_location":
        return MissingLocationResponse(
            message=str(
                pipeline.get("message")
                or (
                    "This looks like a physical/onsite business — please include "
                    "a city or area so we can research the local market."
                )
            )
        )

    # Short-lived sessions only — never hold a Neon connection across the
    # multi-minute research pipeline.
    async with AsyncSessionLocal() as db:
        venture = Venture(business_idea=business_idea)
        db.add(venture)
        await db.commit()
        await db.refresh(venture)
        venture_id = venture.id
        venture_status = venture.status

    research_data = pipeline.get("research_findings") or []
    competitor_data = pipeline.get("competitor_findings") or []
    location_data = pipeline.get("location_data")  # None when remote
    strategy_data = pipeline.get("strategy")
    delivery_mode = pipeline.get("delivery_mode")
    hiring_signal = pipeline.get("hiring_signal")
    task_plan = pipeline.get("task_plan") or []
    customer_segments = pipeline.get("customer_segments") or []
    customer_pain_points = pipeline.get("customer_pain_points") or []
    operational_pain_points = pipeline.get("operational_pain_points") or []
    customer_confidence = pipeline.get("customer_confidence") or "none"
    no_local_data_reason = pipeline.get("no_local_data_reason")
    final_report = pipeline.get("final_report")

    async with AsyncSessionLocal() as db:
        venture = await db.get(Venture, venture_id)
        if venture is None:
            raise HTTPException(status_code=500, detail="Venture disappeared after create")

        venture.delivery_mode = delivery_mode
        venture.hiring_signal = hiring_signal
        venture.final_report = final_report
        await db.commit()
        await db.refresh(venture)

        saved_findings: list[ResearchFinding] = []
        for item in research_data:
            finding = ResearchFinding(
                venture_id=venture.id,
                finding=item["finding"],
                source=item["source"],
                source_agent="research",
            )
            db.add(finding)
            saved_findings.append(finding)

        for item in competitor_data:
            finding = ResearchFinding(
                venture_id=venture.id,
                finding=item["finding"],
                source=item["source"],
                source_agent="competitor",
                competitor_name=item.get("competitor_name"),
                category=item.get("category"),
            )
            db.add(finding)
            saved_findings.append(finding)

        # Only persist location when Location Agent actually ran.
        if location_data is not None and location_data.get("status") != "skipped":
            db.add(_save_location(venture.id, location_data))

        if strategy_data is not None:
            db.add(_save_strategy(venture.id, strategy_data))

        db.add(
            _save_customer_research(
                venture.id,
                segments=customer_segments,
                pain_points=customer_pain_points,
                operational_pain_points=operational_pain_points,
                confidence=customer_confidence,
                no_local_data_reason=no_local_data_reason,
            )
        )

        for item in task_plan:
            db.add(_save_task(venture.id, item))

        await db.commit()

        for finding in saved_findings:
            await db.refresh(finding)

        response_status = venture.status
        response_delivery_mode = venture.delivery_mode

    # Map competitor source+name -> possible_duplicate_of from pipeline (not in DB yet).
    duplicate_flags = {
        (item.get("source"), item.get("competitor_name")): item.get(
            "possible_duplicate_of"
        )
        for item in competitor_data
        if item.get("possible_duplicate_of")
    }

    return VentureResponse(
        id=venture_id,
        status=response_status or venture_status,
        delivery_mode=response_delivery_mode,
        findings=[
            FindingResponse(
                finding=sanitize_display_text(item.finding),
                source=item.source,
                source_agent=item.source_agent,
                **(
                    {
                        "competitor_name": (
                            sanitize_display_text(item.competitor_name)
                            if item.competitor_name
                            else None
                        ),
                        "category": item.category,
                        "possible_duplicate_of": duplicate_flags.get(
                            (item.source, item.competitor_name)
                        ),
                    }
                    if item.source_agent == "competitor"
                    else {}
                ),
            )
            for item in saved_findings
        ],
        hiring_signal=_hiring_signal_response(hiring_signal),
        customer_research=_customer_research_response(
            segments=customer_segments,
            pain_points=customer_pain_points,
            operational_pain_points=operational_pain_points,
            confidence=customer_confidence,
            no_local_data_reason=no_local_data_reason,
        ),
        location=_location_response_from_analysis(location_data),
        strategy=_strategy_response_from_data(strategy_data),
        task_plan=[_task_response_from_data(item) for item in task_plan],
        final_report=_final_report_response(final_report),
    )


@app.get("/venture/{venture_id}", response_model=VentureDetailResponse)
async def get_venture(
    venture_id: int,
    db: AsyncSession = Depends(get_db),
) -> VentureDetailResponse:
    venture = await db.get(Venture, venture_id)
    if venture is None:
        raise HTTPException(status_code=404, detail="Venture not found")

    findings_result = await db.execute(
        select(ResearchFinding).where(ResearchFinding.venture_id == venture_id)
    )
    findings = findings_result.scalars().all()

    location_result = await db.execute(
        select(Location).where(Location.venture_id == venture_id)
    )
    location = location_result.scalar_one_or_none()

    strategy_result = await db.execute(
        select(Strategy).where(Strategy.venture_id == venture_id)
    )
    strategy = strategy_result.scalar_one_or_none()

    tasks_result = await db.execute(
        select(Task).where(Task.venture_id == venture_id).order_by(Task.id)
    )
    tasks = tasks_result.scalars().all()

    customer_result = await db.execute(
        select(CustomerResearch).where(CustomerResearch.venture_id == venture_id)
    )
    customer_research = customer_result.scalar_one_or_none()

    research_findings = [
        StoredResearchFinding(
            id=f.id,
            finding=sanitize_display_text(f.finding),
            source=f.source,
        )
        for f in findings
        if f.source_agent == "research"
    ]
    competitor_findings = [
        StoredCompetitorFinding(
            id=f.id,
            competitor_name=(
                sanitize_display_text(f.competitor_name) if f.competitor_name else None
            ),
            finding=sanitize_display_text(f.finding),
            source=f.source,
            category=f.category,
        )
        for f in findings
        if f.source_agent == "competitor"
    ]

    return VentureDetailResponse(
        venture=VentureInfo(
            id=venture.id,
            business_idea=venture.business_idea,
            status=venture.status,
            delivery_mode=venture.delivery_mode,
            created_at=venture.created_at,
        ),
        research_findings=research_findings,
        competitor_findings=competitor_findings,
        hiring_signal=_hiring_signal_response(venture.hiring_signal),
        customer_research=_customer_research_response_from_model(customer_research),
        location=_location_response_from_model(location),
        strategy=_strategy_response_from_model(strategy),
        task_plan=[_task_response_from_model(t) for t in tasks],
        final_report=_final_report_response(venture.final_report),
    )


@app.delete("/venture/{venture_id}")
async def delete_venture(
    venture_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    venture = await db.get(Venture, venture_id)
    if venture is None:
        raise HTTPException(status_code=404, detail="Venture not found")

    for model in (ResearchFinding, Location, Strategy, Task, CustomerResearch):
        await db.execute(delete(model).where(model.venture_id == venture_id))
    await db.delete(venture)
    await db.commit()
    return {"status": "deleted", "id": venture_id}