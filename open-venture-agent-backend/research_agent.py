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


class Finding(BaseModel):
    finding: str
    source: str


class FindingsList(BaseModel):
    findings: list[Finding] = Field(min_length=3, max_length=5)


class ResearchState(TypedDict):
    business_idea: str
    search_results: str
    findings: list[dict]


def search_market(state: ResearchState) -> dict:
    tool = TavilySearch(max_results=5)
    business_idea = state["business_idea"]
    query = f"{business_idea} Pakistan local market trends demand opportunities"
    result = tool.invoke({"query": query})
    return {"search_results": str(result)}


def summarize_findings(state: ResearchState) -> dict:
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    structured_llm = llm.with_structured_output(FindingsList)

    prompt = (
        "You are a market research analyst. From the search results below, "
        "produce 3-5 concise findings. Prioritize local and regional market data "
        "for the stated location over global industry statistics — only use "
        "worldwide figures when no local data is available.\n"
        "Each finding must include a short 'finding' insight and a 'source' URL "
        "taken from the search results.\n\n"
        f"Business idea: {state['business_idea']}\n\n"
        f"Search results:\n{state['search_results']}"
    )
    result = structured_llm.invoke(prompt)
    return {"findings": [item.model_dump() for item in result.findings]}


def build_research_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("search", search_market)
    graph.add_node("summarize", summarize_findings)
    graph.add_edge(START, "search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


research_agent = build_research_graph()


def run_research(business_idea: str) -> list[dict]:
    result = research_agent.invoke({"business_idea": business_idea})
    return result["findings"]


if __name__ == "__main__":
    findings = run_research("coffee shop in Lahore")
    for item in findings:
        print(item)
