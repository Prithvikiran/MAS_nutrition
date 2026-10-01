"""
Shared building blocks for both orchestration graphs (centralized and
decentralized). Both graphs are built from the exact same three specialist
agent nodes -- the only difference between them is *who decides* which node
runs next (see agents/centralized_graph.py vs agents/decentralized_graph.py).
"""
from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages

from app.config import get_llm
from app.retrieval import hybrid_rag, multihop_rag, naive_rag, query_transform_rag

RETRIEVAL_PIPELINES = {
    "naive": naive_rag.retrieve,
    "hybrid": hybrid_rag.retrieve,
    "query_transform": query_transform_rag.retrieve,
    "multihop": multihop_rag.retrieve,
}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    retrieval_pipeline: str          # which of the 4 pipelines to use
    retrieved_context: list[dict]    # filled in by the retrieval agent
    macros: dict                     # filled in by the calculator agent
    next_agent: str                  # who should run next (used by both graphs)
    visited: list[str]               # trace of node visits, for the eval script


def calculate_macros(calories: int, goal: Literal["lose", "maintain", "gain"]) -> dict:
    """Deterministic macro split -- not an LLM call, a plain calculation tool."""
    protein_pct = {"lose": 0.35, "maintain": 0.30, "gain": 0.30}[goal]
    fat_pct = 0.25
    carb_pct = 1 - protein_pct - fat_pct
    return {
        "calories": calories,
        "protein_g": round(calories * protein_pct / 4),
        "fat_g": round(calories * fat_pct / 9),
        "carb_g": round(calories * carb_pct / 4),
    }


def retrieval_agent(state: AgentState) -> AgentState:
    pipeline_fn = RETRIEVAL_PIPELINES[state.get("retrieval_pipeline", "hybrid")]
    last_user_message = state["messages"][-1].content
    results = pipeline_fn(last_user_message)

    state["retrieved_context"] = results
    state["visited"] = state.get("visited", []) + ["retrieval_agent"]
    return state


def calculator_agent(state: AgentState) -> AgentState:
    # In a full build this would parse calorie/goal from the conversation via
    # an LLM call; kept deterministic here to keep the tool's behavior
    # inspectable and testable without an API call.
    state["macros"] = calculate_macros(calories=2000, goal="lose")
    state["visited"] = state.get("visited", []) + ["calculator_agent"]
    return state


def meal_planning_agent(state: AgentState) -> AgentState:
    llm = get_llm()
    context_text = "\n".join(f"- {c['text']}" for c in state.get("retrieved_context", []))
    macros_text = (
        f"Target: {state['macros']}" if state.get("macros") else "No specific macro target set."
    )

    system_prompt = (
        "You are a nutrition assistant. Using the retrieved context below, "
        f"write a short, concrete meal suggestion.\n\nContext:\n{context_text}\n\n{macros_text}\n\n"
        "Cite which context bullet(s) informed your answer."
    )

    response = llm.invoke([("system", system_prompt), *state["messages"]])
    state["messages"] = state["messages"] + [response]
    state["visited"] = state.get("visited", []) + ["meal_planning_agent"]
    return state
