from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from rag import build_index, get_query_engine

# Build index once at startup
index = build_index()
query_engine = get_query_engine(index)

class AgentState(TypedDict):
    question: str
    retrieved_context: str
    answer: str
    needs_retry: bool
    attempts: int

def retrieve_node(state: AgentState) -> AgentState:
    """Hit the RAG engine"""
    response = query_engine.query(state["question"])
    return {
        **state,
        "retrieved_context": str(response),
        "attempts": state.get("attempts", 0) + 1
    }

def evaluate_node(state: AgentState) -> AgentState:
    """Decide if answer is good enough or retry"""
    answer = state["retrieved_context"]
    # Simple heuristic — if answer is too short or says "I don't know", retry
    needs_retry = (
        len(answer) < 50 or
        "i don't know" in answer.lower() or
        "no information" in answer.lower()
    ) and state["attempts"] < 2      # max 2 retries

    return {**state, "answer": answer, "needs_retry": needs_retry}

def should_retry(state: AgentState) -> str:
    return "retrieve" if state["needs_retry"] else "end"

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("evaluate", evaluate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "evaluate")
workflow.add_conditional_edges("evaluate", should_retry, {
    "retrieve": "retrieve",
    "end": END
})

agent = workflow.compile()

def run_query(question: str) -> dict:
    result = agent.invoke({
        "question": question,
        "retrieved_context": "",
        "answer": "",
        "needs_retry": False,
        "attempts": 0
    })
    return {"answer": result["answer"], "attempts": result["attempts"]}