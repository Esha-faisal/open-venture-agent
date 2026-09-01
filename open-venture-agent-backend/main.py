import asyncio
from collections.abc import AsyncGenerator

from fastapi import Depends, FastAPI
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from competitor_agent import run_competitor_research
from database import AsyncSessionLocal, ResearchFinding, Venture
from research_agent import run_research

app = FastAPI()


class HealthResponse(BaseModel):
    status: str


class VentureRequest(BaseModel):
    business_idea: str


class FindingResponse(BaseModel):
    finding: str
    source: str
    source_agent: str
    competitor_name: str | None = None


class VentureResponse(BaseModel):
    id: int
    status: str
    findings: list[FindingResponse]


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


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

    research_data, competitor_data = await asyncio.gather(
        asyncio.to_thread(run_research, venture.business_idea),
        asyncio.to_thread(run_competitor_research, venture.business_idea),
    )

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
        )
        db.add(finding)
        saved_findings.append(finding)

    await db.commit()

    return VentureResponse(
        id=venture.id,
        status=venture.status,
        findings=[
            FindingResponse(
                finding=item.finding,
                source=item.source,
                source_agent=item.source_agent,
                competitor_name=item.competitor_name,
            )
            for item in saved_findings
        ],
    )
