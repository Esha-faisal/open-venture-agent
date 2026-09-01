import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

if not os.getenv("TAVILY_API_KEY"):
    raise RuntimeError("TAVILY_API_KEY is not set")
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY is not set")


class CompetitorFinding(BaseModel):
    competitor_name: str
    finding: str
    source: str


class CompetitorFindingsList(BaseModel):
    findings: list[CompetitorFinding] = Field(min_length=3, max_length=5)


class CompetitorState(TypedDict):
    business_idea: str
    search_results: str
    findings: list[dict]


def search_competitors(state: CompetitorState) -> dict:
    tool = TavilySearch(max_results=5)
    business_idea = state["business_idea"]
    query = f"{business_idea} Pakistan reviews pricing local named competitors"
    result = tool.invoke({"query": query})
    return {"search_results": str(result)}


def summarize_competitors(state: CompetitorState) -> dict:
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    structured_llm = llm.with_structured_output(CompetitorFindingsList)

    prompt = (
        "You are a competitive intelligence analyst. From the search results below, "
        "identify 3-5 specific named competitors (real business names, not generic "
        "categories). Prioritize local and regional businesses near the stated "
        "location over well-known international chains — only include global brands "
        "if they actually operate in that specific area.\n"
        "For each competitor, provide:\n"
        "- competitor_name: the actual business name\n"
        "- finding: a concise insight about them (e.g. pricing, strengths, weaknesses, "
        "customer reviews)\n"
        "- source: a URL from the search results supporting this finding\n\n"
        f"Business idea: {state['business_idea']}\n\n"
        f"Search results:\n{state['search_results']}"
    )
    result = structured_llm.invoke(prompt)
    return {"findings": [item.model_dump() for item in result.findings]}


def build_competitor_graph():
    graph = StateGraph(CompetitorState)
    graph.add_node("search", search_competitors)
    graph.add_node("summarize", summarize_competitors)
    graph.add_edge(START, "search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


competitor_agent = build_competitor_graph()


def run_competitor_research(business_idea: str) -> list[dict]:
    result = competitor_agent.invoke({"business_idea": business_idea})
    return result["findings"]


if __name__ == "__main__":
    findings = run_competitor_research("gym in the middle of Jhelum")
    for item in findings:
        print(item)
