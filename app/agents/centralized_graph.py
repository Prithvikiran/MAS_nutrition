"""
Centralized orchestration: one supervisor node, on every turn, looks at the
conversation + which agents have already run, and decides which specialist
agent runs next (or whether to finish). Specialist agents never decide
routing themselves -- they just do their job and return control to the
supervisor. This is the classic LangGraph "supervisor" pattern.
"""
import json

from langgraph.graph import END, StateGraph

from app.agents_common import AgentState, calculator_agent, meal_planning_agent, retrieval_agent
from app.config import get_llm

_SUPERVISOR_SYSTEM = """You route a nutrition-assistant conversation between
specialist agents: "retrieval_agent" (looks up nutrition facts), "calculator_agent"
(computes calorie/macro targets), "meal_planning_agent" (writes the final answer
using whatever context/macros are available), or "FINISH" (conversation is done).

Rules: retrieval_agent should usually run before meal_planning_agent unless
context is already retrieved. Never call the same agent twice in one turn
(check "already_visited"). Once meal_planning_agent has run, respond FINISH.

Already visited this turn: {visited}

Respond with ONLY a JSON object: {{"next": "<agent_name_or_FINISH>"}}"""


def supervisor_node(state: AgentState) -> AgentState:
    llm = get_llm()
    visited = state.get("visited", [])
    system = _SUPERVISOR_SYSTEM.format(visited=visited)

    response = llm.invoke([("system", system), *state["messages"]])
    try:
        decision = json.loads(response.content.strip())
        next_agent = decision.get("next", "FINISH")
    except json.JSONDecodeError:
        next_agent = "FINISH"

    state["next_agent"] = next_agent
    return state


def _route(state: AgentState) -> str:
    return state["next_agent"] if state["next_agent"] != "FINISH" else END


def build_centralized_graph():
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("calculator_agent", calculator_agent)
    graph.add_node("meal_planning_agent", meal_planning_agent)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route,
        {
            "retrieval_agent": "retrieval_agent",
            "calculator_agent": "calculator_agent",
            "meal_planning_agent": "meal_planning_agent",
            END: END,
        },
    )
    # every specialist returns control to the supervisor -- it never routes itself
    graph.add_edge("retrieval_agent", "supervisor")
    graph.add_edge("calculator_agent", "supervisor")
    graph.add_edge("meal_planning_agent", "supervisor")

    return graph.compile()
