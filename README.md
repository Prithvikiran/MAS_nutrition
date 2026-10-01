# Multi-Agent Nutrition RAG Assistant

A personalized nutrition assistant built with **LangGraph + ChromaDB**, implementing four retrieval pipelines and two agent-orchestration architectures, so they can be directly compared rather than just built once and left unexamined.

Recreated from a resume bullet describing:
> *Designed and developed a multi-agent RAG system for personalized nutrition assistance using LangGraph, ChromaDB, enabling graph-structured retrieval and contextual multi-turn reasoning. Implemented four retrieval pipelines: naive RAG, hybrid retrieval (BM25 + vector search), query transformation, and multi-hop decomposition. Built and evaluated both centralized orchestration with LLM-based routing and decentralized agent-to-agent delegation.*

See `DESIGN.md` for the reasoning behind every choice below.

---

## Structure

```
app/
├── config.py                  # LLM + path config, shared across everything
├── ingest.py                  # builds the Chroma + BM25 indices from sample docs
├── agents_common.py           # shared AgentState + the 3 specialist agent nodes
├── retrieval/
│   ├── naive_rag.py           # Pipeline 1: raw vector similarity
│   ├── hybrid_rag.py          # Pipeline 2: BM25 + vector, RRF-merged
│   ├── query_transform_rag.py # Pipeline 3: LLM rewrites query before retrieval
│   └── multihop_rag.py        # Pipeline 4: LLM decomposes into sub-questions
├── agents/
│   ├── centralized_graph.py   # supervisor node routes between agents
│   └── decentralized_graph.py # agents route to each other directly
└── main.py                    # CLI entrypoint
eval/
└── compare_pipelines.py       # runs all 4 pipelines + both graphs side by side
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

## Run a single query

```bash
python -m app.main --query "What should I eat before a marathon?" --pipeline hybrid --mode centralized
python -m app.main --query "I'm diabetic and want to lose weight" --pipeline multihop --mode decentralized
```

## Compare everything at once

```bash
python -m eval.compare_pipelines
```
Prints, for each sample query: which sources each of the 4 retrieval pipelines returned, and the full agent-visit sequence + wall time for both orchestration modes.

## The 3 specialist agents (shared by both orchestration graphs)

- **`retrieval_agent`** — runs whichever of the 4 pipelines was selected
- **`calculator_agent`** — deterministic macro-split calculation (not an LLM call — kept as a plain function so its output is exact and testable)
- **`meal_planning_agent`** — synthesizes retrieved context + macro targets into a final, cited answer

## The 2 orchestration modes

- **Centralized** (`agents/centralized_graph.py`) — a supervisor node makes every routing decision; specialists never route themselves.
- **Decentralized** (`agents/decentralized_graph.py`) — no supervisor; each agent decides its own handoff after finishing its job.

Both are built from the *same* three specialist agents — only who decides what runs next differs. That's deliberate: it's what makes the comparison in `DESIGN.md` fair.
