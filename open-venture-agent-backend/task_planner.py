import json
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from gemini_config import GEMINI_MODEL
from gemini_retry import invoke_with_retry

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY is not set")

Phase = Literal["validate", "decide", "build", "launch"]
Priority = Literal["P0", "P1", "P2"]

_PHASE_ORDER = {"validate": 0, "decide": 1, "build": 2, "launch": 3}
_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2}


class BuildTaskDraft(BaseModel):
    differentiator_index: int = Field(
        description="0-based index into the key_differentiators list"
    )
    title: str = Field(
        description=(
            "Clear, actionable build task title (verb-led). "
            "Do not merely repeat the differentiator text."
        )
    )
    depends_on: list[str] = Field(
        description=(
            "Validate-phase task_ids that must finish before this build task. "
            "Only use ids from the provided validate task list."
        )
    )


class BuildTasksList(BaseModel):
    tasks: list[BuildTaskDraft]


class AssumptionValidationDraft(BaseModel):
    assumption_index: int = Field(
        description="0-based index into the assumptions list"
    )
    title: str = Field(
        description=(
            "Concrete, specific validation ACTION (verb-led). "
            "Must be executable in the real world — not a restatement of the belief. "
            "Examples: 'Visit or call the 5 nearest competitor bakeries and document "
            "their product range', 'Survey 15-20 local residents about which celebration "
            "items they currently struggle to find locally'."
        )
    )


class AssumptionValidationList(BaseModel):
    tasks: list[AssumptionValidationDraft]


class VerificationTaskDraft(BaseModel):
    verification_index: int = Field(
        description="0-based index into the verification_needed list"
    )
    title: str = Field(
        description=(
            "Concrete, actionable VALIDATE task (verb-led). Must read differently "
            "from the risk statement — turn the concern into a real-world check, "
            "do not copy the risk text."
        )
    )


class VerificationTaskList(BaseModel):
    tasks: list[VerificationTaskDraft]


def _next_task_id(counter: list[int]) -> str:
    task_id = f"T{counter[0]}"
    counter[0] += 1
    return task_id


def _task(
    *,
    task_id: str,
    phase: Phase,
    title: str,
    priority: Priority,
    depends_on: list[str],
    source: str,
) -> dict:
    task = {
        "task_id": task_id,
        "phase": phase,
        "title": title,
        "priority": priority,
        "source": source,
        "status": "pending",
    }
    if depends_on:
        task["depends_on"] = depends_on
    return task


def _build_tasks_from_differentiators(
    differentiators: list[str],
    validate_tasks: list[dict],
) -> list[BuildTaskDraft]:
    if not differentiators:
        return []

    validate_summary = [
        {
            "task_id": t["task_id"],
            "title": t["title"],
            "source": t["source"],
        }
        for t in validate_tasks
    ]
    validate_ids = {t["task_id"] for t in validate_tasks}

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(BuildTasksList)

    prompt = (
        "Convert each key differentiator into exactly one concrete BUILD task.\n"
        "Rules:\n"
        "- Produce one task per differentiator (same count and order by index).\n"
        "- title must be actionable (start with a verb); do not just repeat the text.\n"
        "- depends_on must only include validate-phase task_ids from the list below "
        "that logically should finish before building this differentiator.\n"
        "- If unsure which validate tasks apply, depend on all of them.\n"
        "- differentiator_index is 0-based into the differentiators list.\n\n"
        f"Key differentiators:\n{json.dumps(differentiators, indent=2)}\n\n"
        f"Validate tasks:\n{json.dumps(validate_summary, indent=2)}"
    )

    result = invoke_with_retry(
        lambda: structured_llm.invoke(prompt),
        label="task-planner-build",
    )

    # Keep one draft per differentiator index; clamp depends_on to known validate ids.
    by_index: dict[int, BuildTaskDraft] = {}
    for draft in result.tasks:
        idx = draft.differentiator_index
        if idx < 0 or idx >= len(differentiators):
            continue
        filtered = [dep for dep in draft.depends_on if dep in validate_ids]
        if not filtered:
            filtered = sorted(validate_ids, key=lambda x: int(x[1:]))
        by_index[idx] = BuildTaskDraft(
            differentiator_index=idx,
            title=draft.title.strip() or f"Implement: {differentiators[idx]}",
            depends_on=filtered,
        )

    drafts: list[BuildTaskDraft] = []
    for idx, text in enumerate(differentiators):
        if idx in by_index:
            drafts.append(by_index[idx])
        else:
            drafts.append(
                BuildTaskDraft(
                    differentiator_index=idx,
                    title=f"Implement: {text}",
                    depends_on=sorted(validate_ids, key=lambda x: int(x[1:])),
                )
            )
    return drafts


def _validation_titles_from_verification_needed(
    verification_items: list[str],
) -> list[str]:
    """Turn risk statements into distinct actionable Validate task titles."""
    if not verification_items:
        return []

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(VerificationTaskList)

    prompt = (
        "Convert each VERIFICATION / RISK statement into exactly one concrete "
        "VALIDATE task title.\n"
        "The inputs are pure risk or discrepancy statements (what is uncertain). "
        "They are NOT tasks. Do NOT copy or lightly rephrase them as the title.\n"
        "Rules:\n"
        "- Produce one task per item (same count; use verification_index).\n"
        "- title must be an executable ACTION that starts with a verb "
        "(Walk, Visit, Call, Count, Survey, Interview, Compare, Map, Time, "
        "Document, Ask, Photograph, etc.).\n"
        "- Make it specific: who/what/where/how when possible.\n"
        "- The action must resolve or test the underlying concern — it should "
        "read clearly differently from the risk text even though it addresses "
        "the same issue.\n"
        "  Example risk: \"Location data shows many nearby competitors, but no "
        "specific rivals were named — the real level of direct competition is "
        "unclear.\"\n"
        "  Example task: \"Walk the target neighborhood and count how many "
        "direct competitors in this exact niche are actually operating nearby, "
        "since map density and named research disagree.\"\n"
        "- Do not invent unrelated work; stay tightly tied to the concern.\n"
        "- Do not start with \"Verify that…\" restating the risk as a belief.\n\n"
        f"Verification / risk statements:\n{json.dumps(verification_items, indent=2)}"
    )

    result = invoke_with_retry(
        lambda: structured_llm.invoke(prompt),
        label="task-planner-verification",
    )

    by_index: dict[int, str] = {}
    for draft in result.tasks:
        idx = draft.verification_index
        if idx < 0 or idx >= len(verification_items):
            continue
        title = draft.title.strip()
        if title:
            by_index[idx] = title

    titles: list[str] = []
    for idx, text in enumerate(verification_items):
        titles.append(
            by_index.get(idx)
            or f"Check this unresolved concern with a concrete local visit: {text}"
        )
    return titles


def _validation_titles_from_assumptions(assumptions: list[str]) -> list[str]:
    """Convert belief-style assumptions into concrete validation action titles."""
    if not assumptions:
        return []

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(AssumptionValidationList)

    prompt = (
        "Convert each strategy ASSUMPTION into exactly one concrete VALIDATE task.\n"
        "Assumptions are beliefs, not tasks. Do NOT copy or lightly rephrase the "
        "assumption text.\n"
        "Rules:\n"
        "- Produce one task per assumption (same count; use assumption_index).\n"
        "- title must be an executable validation ACTION that starts with a verb "
        "(Visit, Call, Survey, Count, Interview, Time, Compare, Document, etc.).\n"
        "- Make it specific: include who/what/how many/where when possible "
        "(e.g. 'Visit or call the 5 nearest competitor bakeries and document their "
        "current product range to identify actual gaps', "
        "'Survey 15-20 local residents about which party/celebration items they "
        "currently struggle to find locally').\n"
        "- The action must test whether the assumption is true — not assert it.\n"
        "- Do not invent unrelated work; stay tightly tied to what the assumption claims.\n\n"
        f"Assumptions:\n{json.dumps(assumptions, indent=2)}"
    )

    result = invoke_with_retry(
        lambda: structured_llm.invoke(prompt),
        label="task-planner-assumptions",
    )

    by_index: dict[int, str] = {}
    for draft in result.tasks:
        idx = draft.assumption_index
        if idx < 0 or idx >= len(assumptions):
            continue
        title = draft.title.strip()
        if title:
            by_index[idx] = title

    titles: list[str] = []
    for idx, text in enumerate(assumptions):
        titles.append(
            by_index.get(idx)
            or f"Test assumption with a concrete local check: {text}"
        )
    return titles


def generate_task_plan(strategy: dict) -> list[dict]:
    """Convert Strategy Agent output into a fixed-phase task list."""
    counter = [1]
    validate_tasks: list[dict] = []

    first_call = (strategy.get("first_call_question") or "").strip()
    t1 = _next_task_id(counter)
    validate_tasks.append(
        _task(
            task_id=t1,
            phase="validate",
            title=first_call or "Ask the primary first-call validation question",
            priority="P0",
            depends_on=[],
            source="first_call_question",
        )
    )

    verification_items: list[str] = []
    for item in strategy.get("verification_needed") or []:
        text = str(item).strip()
        if not text:
            continue
        verification_items.append(text)

    verification_titles = _validation_titles_from_verification_needed(
        verification_items
    )
    for title in verification_titles:
        validate_tasks.append(
            _task(
                task_id=_next_task_id(counter),
                phase="validate",
                title=title,
                priority="P0",
                depends_on=[t1],
                source="verification_needed",
            )
        )

    assumptions = [
        str(item).strip()
        for item in (strategy.get("assumptions") or [])
        if str(item).strip()
    ]
    assumption_titles = _validation_titles_from_assumptions(assumptions)
    for title in assumption_titles:
        validate_tasks.append(
            _task(
                task_id=_next_task_id(counter),
                phase="validate",
                title=title,
                priority="P1",
                depends_on=[t1],
                source="assumptions",
            )
        )

    validate_ids = [t["task_id"] for t in validate_tasks]

    decide_id = _next_task_id(counter)
    decide_task = _task(
        task_id=decide_id,
        phase="decide",
        title="Finalize positioning and pricing based on validation results",
        priority="P0",
        depends_on=list(validate_ids),
        source="positioning",
    )

    differentiators = [
        str(d).strip()
        for d in (strategy.get("key_differentiators") or [])
        if str(d).strip()
    ]
    build_drafts = _build_tasks_from_differentiators(differentiators, validate_tasks)
    build_tasks: list[dict] = []
    for draft in build_drafts:
        build_tasks.append(
            _task(
                task_id=_next_task_id(counter),
                phase="build",
                title=draft.title,
                priority="P1",
                depends_on=list(draft.depends_on),
                source="key_differentiators",
            )
        )

    launch_task = _task(
        task_id=_next_task_id(counter),
        phase="launch",
        title="Prepare go-to-market plan based on finalized strategy",
        priority="P1",
        depends_on=[decide_id],
        source="target_customers",
    )

    tasks = [*validate_tasks, decide_task, *build_tasks, launch_task]
    tasks.sort(
        key=lambda t: (
            _PHASE_ORDER[t["phase"]],
            _PRIORITY_ORDER[t["priority"]],
            int(t["task_id"][1:]),
        )
    )
    return tasks


if __name__ == "__main__":
    # Real strategy output from POST /venture "beauty parlour in lahore"
    strategy = {
        "positioning": (
            "Positioned as a modern, premium yet accessible neighborhood beauty "
            "studio focusing on hygiene, transparent product usage, and "
            "personalized styling in key high-demand areas like DHA or Johar Town."
        ),
        "target_customers": (
            "Urban professional women, university students, and brides-to-be in "
            "Lahore seeking reliable, high-quality grooming and bridal services."
        ),
        "pricing_strategy": (
            "Value-based tiered pricing with transparent service menus and bundled "
            "packages for bridal and regular maintenance routines to compete against "
            "established local salons."
        ),
        "key_differentiators": [
            "Strict hygiene and premium international product standards",
            "Personalized consultation-first approach for all hair and skin treatments",
            "Digital-first booking system and transparent loyalty rewards program",
            "Specialized modern bridal and contemporary styling packages",
        ],
        "confidence": 65,
        "assumptions": [
            "Consumer demand for professional grooming in Lahore continues to grow due to rising disposable incomes.",
            "Target customers value hygiene and transparent product usage over the lowest possible price point.",
        ],
        "verification_needed": [
            "Reconcile location data indicating 13 nearby competitors with specific competitor profiles like Seeme Beauty Parlour and Hira's Beauty Salon to map true market saturation in chosen neighborhoods.",
            "Verify local rental rates and staffing availability for skilled beauticians in DHA or Johar Town.",
        ],
        "opportunity_tier": "viable_but_contested",
        "first_call_question": (
            "What specific service gaps or pain points do you consistently "
            "experience at existing salons in your neighborhood that we could "
            "immediately solve?"
        ),
    }

    tasks = generate_task_plan(strategy)

    print(f"opportunity_tier: {strategy['opportunity_tier']}")
    print(f"total tasks: {len(tasks)}\n")

    current_phase = None
    for task in tasks:
        if task["phase"] != current_phase:
            current_phase = task["phase"]
            print(f"=== {current_phase.upper()} ===")
        print(
            f"  {task['task_id']} [{task['priority']}] {task['title']}\n"
            f"      depends_on={task.get('depends_on', [])}  source={task['source']}  "
            f"status={task['status']}"
        )
