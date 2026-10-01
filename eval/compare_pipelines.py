"""
Comparison script -- not a formal benchmark, but exactly the kind of run
that produced the tradeoff observations in DESIGN.md. For each sample query:

1. Runs all 4 retrieval pipelines and shows which sources each one returns
   (so you can see naive vs hybrid vs query-transform vs multi-hop disagree).
2. Runs both orchestration graphs and logs the agent visit sequence + wall
   time (a cheap proxy for the "routing observability" and "scalability"
   tradeoffs called out in the resume bullet).
"""
import time

from langchain_core.messages import HumanMessage

from app.agents.centralized_graph import build_centralized_graph
from app.agents.decentralized_graph import build_decentralized_graph
from app.agents_common import AgentState
from app.retrieval import hybrid_rag, multihop_rag, naive_rag, query_transform_rag

SAMPLE_QUERIES = [
    "What should I eat before a marathon?",
    "I'm diabetic and want to lose weight, what should my meals look like?",
    "Why am I always tired, could it be my diet?",
]

PIPELINES = {
    "naive": naive_rag.retrieve,
    "hybrid": hybrid_rag.retrieve,
    "query_transform": query_transform_rag.retrieve,
    "multihop": multihop_rag.retrieve,
}


def compare_retrieval_pipelines() -> None:
    print("=" * 70)
    print("RETRIEVAL PIPELINE COMPARISON")
    print("=" * 70)
    for query in SAMPLE_QUERIES:
        print(f"\nQuery: {query}")
        for name, fn in PIPELINES.items():
            results = fn(query)
            sources = [r["source"] for r in results]
            print(f"  {name:16s} -> {sources}")


def compare_orchestration_modes() -> None:
    print("\n" + "=" * 70)
    print("ORCHESTRATION MODE COMPARISON")
    print("=" * 70)

    centralized = build_centralized_graph()
    decentralized = build_decentralized_graph()

    for query in SAMPLE_QUERIES:
        print(f"\nQuery: {query}")
        for name, graph in [("centralized", centralized), ("decentralized", decentralized)]:
            state: AgentState = {
                "messages": [HumanMessage(content=query)],
                "retrieval_pipeline": "hybrid",
                "retrieved_context": [],
                "macros": {},
                "next_agent": "",
                "visited": [],
            }
            start = time.perf_counter()
            final_state = graph.invoke(state)
            elapsed = time.perf_counter() - start
            print(
                f"  {name:14s} -> visited={final_state['visited']} "
                f"({elapsed:.2f}s, {len(final_state['visited'])} agent calls)"
            )


if __name__ == "__main__":
    compare_retrieval_pipelines()
    compare_orchestration_modes()
