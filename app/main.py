"""
CLI entrypoint.

Usage:
    python -m app.main --query "What should I eat before a marathon?" \\
        --pipeline hybrid --mode centralized
"""
import argparse

from langchain_core.messages import HumanMessage

from app.agents.centralized_graph import build_centralized_graph
from app.agents.decentralized_graph import build_decentralized_graph
from app.agents_common import AgentState


def run(query: str, pipeline: str, mode: str) -> None:
    graph = build_centralized_graph() if mode == "centralized" else build_decentralized_graph()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "retrieval_pipeline": pipeline,
        "retrieved_context": [],
        "macros": {},
        "next_agent": "",
        "visited": [],
    }

    final_state = graph.invoke(initial_state)

    print(f"\n--- {mode} orchestration, {pipeline} retrieval ---")
    print(f"Agent visit order: {final_state['visited']}")
    print(f"Final answer:\n{final_state['messages'][-1].content}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--pipeline", default="hybrid", choices=["naive", "hybrid", "query_transform", "multihop"]
    )
    parser.add_argument("--mode", default="centralized", choices=["centralized", "decentralized"])
    args = parser.parse_args()

    run(args.query, args.pipeline, args.mode)
