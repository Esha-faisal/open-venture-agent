import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import DateTime, Float, ForeignKey, JSON, String, Text, func, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

load_dotenv()

_raw_url = os.getenv("DATABASE_URL")
if not _raw_url:
    raise RuntimeError("DATABASE_URL is not set")

# .env uses a standard postgresql:// URL; async SQLAlchemy needs the asyncpg driver.
# Drop libpq-only query params that asyncpg does not accept.
_url = make_url(_raw_url).set(drivername="postgresql+asyncpg")
_query = {key: value for key, value in _url.query.items() if key not in {"sslmode", "channel_binding"}}
_url = _url.set(query=_query)

engine = create_async_engine(
    _url,
    connect_args={"ssl": True, "statement_cache_size": 0},
)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Venture(Base):
    __tablename__ = "ventures"

    id: Mapped[int] = mapped_column(primary_key=True)
    business_idea: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="received", server_default="received")
    delivery_mode: Mapped[str | None] = mapped_column(String, nullable=True)
    hiring_signal: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    final_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class ResearchFinding(Base):
    __tablename__ = "research_findings"

    id: Mapped[int] = mapped_column(primary_key=True)
    venture_id: Mapped[int] = mapped_column(ForeignKey("ventures.id"), nullable=False)
    finding: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    source_agent: Mapped[str] = mapped_column(String, nullable=False, server_default="research")
    competitor_name: Mapped[str | None] = mapped_column(String, nullable=True)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class Location(Base):
    __tablename__ = "locations"

    id: Mapped[int] = mapped_column(primary_key=True)
    venture_id: Mapped[int] = mapped_column(ForeignKey("ventures.id"), nullable=False)
    location_name: Mapped[str | None] = mapped_column(String, nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    competitor_count: Mapped[int | None] = mapped_column(nullable=True)
    score: Mapped[int | None] = mapped_column(nullable=True)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    data_confidence: Mapped[str | None] = mapped_column(
        String, nullable=True, server_default="high"
    )
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class Strategy(Base):
    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(primary_key=True)
    venture_id: Mapped[int] = mapped_column(ForeignKey("ventures.id"), nullable=False)
    positioning: Mapped[str] = mapped_column(Text, nullable=False)
    target_customers: Mapped[str] = mapped_column(Text, nullable=False)
    pricing_strategy: Mapped[str] = mapped_column(Text, nullable=False)
    key_differentiators: Mapped[list] = mapped_column(JSON, nullable=False)
    confidence: Mapped[int] = mapped_column(nullable=False)
    assumptions: Mapped[list] = mapped_column(JSON, nullable=False)
    verification_needed: Mapped[list] = mapped_column(JSON, nullable=False)
    opportunity_tier: Mapped[str | None] = mapped_column(String, nullable=True)
    first_call_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    venture_id: Mapped[int] = mapped_column(ForeignKey("ventures.id"), nullable=False)
    task_id: Mapped[str] = mapped_column(String, nullable=False)
    phase: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[str] = mapped_column(String, nullable=False)
    depends_on: Mapped[list] = mapped_column(JSON, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class CustomerResearch(Base):
    __tablename__ = "customer_research"

    id: Mapped[int] = mapped_column(primary_key=True)
    venture_id: Mapped[int] = mapped_column(ForeignKey("ventures.id"), nullable=False)
    customer_segments: Mapped[list] = mapped_column(JSON, nullable=False)
    customer_pain_points: Mapped[list] = mapped_column(JSON, nullable=False)
    confidence: Mapped[str] = mapped_column(String, nullable=False, server_default="none")
    no_local_data_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


__all__ = [
    "AsyncSessionLocal",
    "Base",
    "CustomerResearch",
    "Location",
    "ResearchFinding",
    "Strategy",
    "Task",
    "Venture",
    "create_tables",
    "engine",
]


async def create_tables() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                "ALTER TABLE research_findings "
                "ADD COLUMN IF NOT EXISTS source_agent VARCHAR DEFAULT 'research'"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE research_findings "
                "ADD COLUMN IF NOT EXISTS competitor_name VARCHAR"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE research_findings "
                "ADD COLUMN IF NOT EXISTS category VARCHAR"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE ventures "
                "ADD COLUMN IF NOT EXISTS delivery_mode VARCHAR"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE ventures "
                "ADD COLUMN IF NOT EXISTS hiring_signal JSON"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE strategies "
                "ADD COLUMN IF NOT EXISTS opportunity_tier VARCHAR"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE strategies "
                "ADD COLUMN IF NOT EXISTS first_call_question TEXT"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE locations "
                "ADD COLUMN IF NOT EXISTS data_confidence VARCHAR DEFAULT 'high'"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE locations "
                "ADD COLUMN IF NOT EXISTS note TEXT"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE ventures "
                "ADD COLUMN IF NOT EXISTS final_report JSON"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE customer_research "
                "ADD COLUMN IF NOT EXISTS no_local_data_reason TEXT"
            )
        )
