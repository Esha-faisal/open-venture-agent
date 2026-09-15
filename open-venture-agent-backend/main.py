import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, model_serializer
from sqlalchemy import select
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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await create_tables()
    yield


app = FastAPI(lifespan=lifespan)


class HealthResponse(BaseModel):
    status: str


class VentureRequest(BaseModel):
    business_idea: str


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


class StrategyResponse(BaseModel):
    positioning: str
    target_customers: str
    pricing_strategy: str
    key_differentiators: list[str]
    confidence: int
    assumptions: list[str]
    verification_needed: list[str]
    opportunity_tier: str | None = None
    first_call_question: str | None = None


class HiringSignalResponse(BaseModel):
    postings_found: list[str]
    employer_count: int
    confidence: str
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


class FinalReportResponse(BaseModel):
    headline: str
    opportunity_tier: str | None = None
    confidence: int | None = None
    key_findings: list[str] = []
    top_risks: list[str] = []
    immediate_next_steps: list[FinalReportTaskResponse] = []
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


def _strategy_response_from_data(data: dict | None) -> StrategyResponse | None:
    if data is None:
        return None
    return StrategyResponse(
        positioning=data["positioning"],
        target_customers=data["target_customers"],
        pricing_strategy=data["pricing_strategy"],
        key_differentiators=data["key_differentiators"],
        confidence=data["confidence"],
        assumptions=data["assumptions"],
        verification_needed=data["verification_needed"],
        opportunity_tier=data.get("opportunity_tier"),
        first_call_question=data.get("first_call_question"),
    )


def _strategy_response_from_model(strategy: Strategy | None) -> StrategyResponse | None:
    if strategy is None:
        return None
    return StrategyResponse(
        positioning=strategy.positioning,
        target_customers=strategy.target_customers,
        pricing_strategy=strategy.pricing_strategy,
        key_differentiators=strategy.key_differentiators,
        confidence=strategy.confidence,
        assumptions=strategy.assumptions,
        verification_needed=strategy.verification_needed,
        opportunity_tier=strategy.opportunity_tier,
        first_call_question=strategy.first_call_question,
    )


def _hiring_signal_response(
    data: dict | None,
) -> HiringSignalResponse | None:
    if not data:
        return None
    return HiringSignalResponse(
        postings_found=data.get("postings_found") or [],
        employer_count=int(data.get("employer_count") or 0),
        confidence=str(data.get("confidence") or "none"),
        query_used=data.get("query_used"),
    )


def _customer_research_response(
    *,
    segments: list[dict] | None = None,
    pain_points: list[dict] | None = None,
    confidence: str | None = None,
    no_local_data_reason: str | None = None,
) -> CustomerResearchResponse | None:
    segments = segments or []
    pain_points = pain_points or []
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
        confidence=row.confidence,
        no_local_data_reason=row.no_local_data_reason,
    )


def _save_customer_research(
    venture_id: int,
    *,
    segments: list[dict],
    pain_points: list[dict],
    confidence: str,
    no_local_data_reason: str | None = None,
) -> CustomerResearch:
    return CustomerResearch(
        venture_id=venture_id,
        customer_segments=segments,
        customer_pain_points=pain_points,
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
        key_differentiators=data["key_differentiators"],
        confidence=data["confidence"],
        assumptions=data["assumptions"],
        verification_needed=data["verification_needed"],
        opportunity_tier=data.get("opportunity_tier"),
        first_call_question=data.get("first_call_question"),
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


def _final_report_response(data: dict | None) -> FinalReportResponse | None:
    if not data:
        return None
    return FinalReportResponse(
        headline=str(data.get("headline") or ""),
        opportunity_tier=data.get("opportunity_tier"),
        confidence=data.get("confidence"),
        key_findings=list(data.get("key_findings") or []),
        top_risks=list(data.get("top_risks") or []),
        immediate_next_steps=[
            FinalReportTaskResponse(
                task_id=step.get("task_id"),
                phase=step.get("phase"),
                title=step.get("title"),
                priority=step.get("priority"),
                depends_on=list(step.get("depends_on") or []),
                source=step.get("source"),
                status=step.get("status"),
            )
            for step in (data.get("immediate_next_steps") or [])
            if isinstance(step, dict)
        ],
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


@app.post("/venture", response_model=VentureResponse)
async def create_venture(
    payload: VentureRequest,
    db: AsyncSession = Depends(get_db),
) -> VentureResponse:
    venture = Venture(business_idea=payload.business_idea)
    db.add(venture)
    await db.commit()
    await db.refresh(venture)

    pipeline = await asyncio.to_thread(run_venture_pipeline, venture.business_idea)

    research_data = pipeline.get("research_findings") or []
    competitor_data = pipeline.get("competitor_findings") or []
    location_data = pipeline.get("location_data")  # None when remote
    strategy_data = pipeline.get("strategy")
    delivery_mode = pipeline.get("delivery_mode")
    hiring_signal = pipeline.get("hiring_signal")
    task_plan = pipeline.get("task_plan") or []
    customer_segments = pipeline.get("customer_segments") or []
    customer_pain_points = pipeline.get("customer_pain_points") or []
    customer_confidence = pipeline.get("customer_confidence") or "none"
    no_local_data_reason = pipeline.get("no_local_data_reason")
    final_report = pipeline.get("final_report")

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
            confidence=customer_confidence,
            no_local_data_reason=no_local_data_reason,
        )
    )

    for item in task_plan:
        db.add(_save_task(venture.id, item))

    await db.commit()

    # Map competitor source+name -> possible_duplicate_of from pipeline (not in DB yet).
    duplicate_flags = {
        (item.get("source"), item.get("competitor_name")): item.get(
            "possible_duplicate_of"
        )
        for item in competitor_data
        if item.get("possible_duplicate_of")
    }

    return VentureResponse(
        id=venture.id,
        status=venture.status,
        delivery_mode=venture.delivery_mode,
        findings=[
            FindingResponse(
                finding=item.finding,
                source=item.source,
                source_agent=item.source_agent,
                **(
                    {
                        "competitor_name": item.competitor_name,
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
        StoredResearchFinding(id=f.id, finding=f.finding, source=f.source)
        for f in findings
        if f.source_agent == "research"
    ]
    competitor_findings = [
        StoredCompetitorFinding(
            id=f.id,
            competitor_name=f.competitor_name,
            finding=f.finding,
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
