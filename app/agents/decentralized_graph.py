"""
Decentralized orchestration: there is no supervisor node. Each specialist
agent, after doing its own job, makes its own small routing decision about
which peer to hand off to next (or to finish) -- agent-to-agent delegation
rather than a hub-and-spoke router. Compare this file's shape to
centralized_graph.py: routing logic is now duplicated across each agent
instead of centralized in one place, which is exactly the tradeoff explored
in DESIGN.md.
"""
import json

from langgraph.graph import END, StateGraph

from app.agents_common import AgentState, calculate_macros, retrieval_agent
from app.config import get_llm

_HANDOFF_SYSTEM = """You just ran as "{current_agent}" in a nutrition-assistant
agent pipeline. Decide which peer agent should run next: "retrieval_agent",
"calculator_agent", "meal_planning_agent", or "FINISH" if the user's question
is fully answered. Don't hand off to yourself. Don't hand off to an agent
already visited this turn unless truly necessary.

Already visited this turn: {visited}

Respond with ONLY a JSON object: {{"next": "<agent_name_or_FINISH>"}}"""


def _decide_next(state: AgentState, current_agent: str) -> str:
    llm = get_llm()
    system = _HANDOFF_SYSTEM.format(current_agent=current_agent, visited=state.get("visited", []))
    response = llm.invoke([("system", system), *state["messages"]])
    try:
        decision = json.loads(response.content.strip())
        return decision.get("next", "FINISH")
    except json.JSONDecodeError:
        return "FINISH"


def retrieval_agent_decentralized(state: AgentState) -> AgentState:
    state = retrieval_agent(state)  # reuse the same retrieval logic
    state["next_agent"] = _decide_next(state, "retrieval_agent")
    return state


def calculator_agent_decentralized(state: AgentState) -> AgentState:
    state["macros"] = calculate_macros(calories=2000, goal="lose")
    state["visited"] = state.get("visited", []) + ["calculator_agent"]
    state["next_agent"] = _decide_next(state, "calculator_agent")
    return state


def meal_planning_agent_decentralized(state: AgentState) -> AgentState:
    llm = get_llm()
    context_text = "\n".join(f"- {c['text']}" for c in state.get("retrieved_context", []))
    macros_text = f"Target: {state['macros']}" if state.get("macros") else ""
    system_prompt = (
        f"You are a nutrition assistant. Context:\n{context_text}\n\n{macros_text}\n\n"
        "Write a short, concrete meal suggestion citing the context."
    )
    response = llm.invoke([("system", system_prompt), *state["messages"]])
    state["messages"] = state["messages"] + [response]
    state["visited"] = state.get("visited", []) + ["meal_planning_agent"]
    # meal_planning_agent is treated as the natural terminal node -- it always
    # finishes rather than asking itself where to go next, since by this point
    # the user's question has been answered.
    state["next_agent"] = "FINISH"
    return state


def _route(state: AgentState) -> str:
    return state["next_agent"] if state["next_agent"] != "FINISH" else END


def build_decentralized_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieval_agent", retrieval_agent_decentralized)
    graph.add_node("calculator_agent", calculator_agent_decentralized)
    graph.add_node("meal_planning_agent", meal_planning_agent_decentralized)

    # Entry point: a cheap heuristic instead of a supervisor call -- this is
    # the one place decentralized orchestration still needs *some* starting
    # decision, since a graph has to begin somewhere.
    graph.set_entry_point("retrieval_agent")

    routes = {
        "retrieval_agent": "retrieval_agent",
        "calculator_agent": "calculator_agent",
        "meal_planning_agent": "meal_planning_agent",
        END: END,
    }
    graph.add_conditional_edges("retrieval_agent", _route, routes)
    graph.add_conditional_edges("calculator_agent", _route, routes)
    graph.add_conditional_edges("meal_planning_agent", _route, routes)

    return graph.compile()
