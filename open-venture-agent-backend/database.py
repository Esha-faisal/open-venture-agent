import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import DateTime, ForeignKey, String, Text, func, text
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


__all__ = [
    "AsyncSessionLocal",
    "Base",
    "ResearchFinding",
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
