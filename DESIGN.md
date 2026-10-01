# Design Choices

## Why ChromaDB
Local, embedded, zero infrastructure to stand up — appropriate for a project whose point is comparing retrieval *strategies*, not benchmarking vector database throughput. The retrieval pipelines are written against a thin `get_chroma_collection()` interface (`app/ingest.py`), so swapping in Pinecone/Weaviate later is a one-function change, not a rewrite.

## Why these four retrieval pipelines specifically
Each one exists to fail differently, so running all four on the same query set is genuinely informative rather than four variations on the same idea:

- **Naive RAG** is the control group — no rewriting, no lexical signal, no decomposition. It's what "just call vector search" looks like, and it's necessary as a baseline even though it's the weakest pipeline, because without it you can't say the other three are actually improvements.
- **Hybrid (BM25 + vector)** fixes naive RAG's blind spot for exact terminology — a query mentioning "vitamin D" should retrieve the vitamin D document even if the surrounding phrasing doesn't paraphrase well.
- **Query transformation** fixes a different problem: users don't phrase questions the way a knowledge base is written. "I want to lose weight fast" doesn't share much vocabulary with a document about "caloric deficit strategies," even though it's exactly what the user needs. Rewriting the query into the corpus's own register closes that gap *before* retrieval happens, rather than hoping the embedding model bridges it.
- **Multi-hop decomposition** fixes a problem neither of the above can: a single query embedding can only point in one direction. "What should a diabetic marathon runner eat before and after a race?" touches three sub-topics (diabetes management, pre-race carb-loading, post-exercise recovery) — a single retrieval call tends to surface docs for whichever sub-topic dominates the embedding, silently dropping the others. Decomposing into sub-questions and retrieving independently for each is the only pipeline here that reliably covers compound questions.

**Tradeoff, stated plainly:** pipelines 3 and 4 both cost an extra LLM call before retrieval even starts, which roughly doubles latency and cost per query compared to pipelines 1-2. `eval/compare_pipelines.py` is written to make that tradeoff visible (source overlap per pipeline) rather than asserting one pipeline is "best" — in practice, query transformation and multi-hop are worth the cost on ambiguous/compound questions and wasted cost on simple, well-phrased ones. A production version would route between "hybrid" and "multi-hop" using something like the query classifier built in the companion RAG project's retrieval-service, rather than always paying for the expensive pipeline.

## Why compare centralized vs. decentralized orchestration at all
It would have been simpler to just pick one pattern. The reason both exist here is that they trade off along three axes that only become visible once you've built both and can point at the code:

**Scalability.** Centralized: adding a 4th specialist agent means updating exactly one place — the supervisor's routing prompt and the graph's conditional-edge map (`centralized_graph.py`). Decentralized: adding a 4th agent means every *existing* agent's handoff prompt now needs to know that agent exists, since routing logic is duplicated across every node (`decentralized_graph.py`). Centralized scales more cleanly as the agent roster grows.

**Routing observability.** Centralized: every routing decision passes through one function (`supervisor_node`), so logging/tracing/debugging "why did it go here" has exactly one place to instrument. Decentralized: routing decisions are scattered across every agent's `_decide_next` call, so understanding a single conversation's path means reconstructing it from N different decision points instead of one. This is the sharpest practical tradeoff of the two — it's also why `eval/compare_pipelines.py` explicitly logs `visited` order for both modes, to make this difference something you can point at rather than just assert.

**Compositional flexibility.** Decentralized wins here: an agent can hand off directly to whichever peer makes sense for the situation, without that path needing to have been anticipated by a central router's prompt in advance. If you later wanted `calculator_agent` to sometimes hand off straight back to itself with refined inputs (e.g., recompute macros after the user corrects their weight), decentralized supports that pattern natively; centralized would need the supervisor's prompt explicitly updated to allow it.

**Bottom line:** centralized orchestration is the better default for a system with a small, stable set of agents where debuggability matters (which is most production systems). Decentralized becomes worth its added complexity when the agent roster is large, evolving, or when agent-to-agent handoff patterns are genuinely hard to anticipate from a central point — which is a real but narrower set of cases.

## Why the calculator agent is not an LLM call
`calculate_macros()` in `agents_common.py` is a plain deterministic function, not a prompt. Arithmetic that has one correct answer should never be delegated to a non-deterministic model call — this is a small design choice, but it's the kind of thing worth stating explicitly: agentic systems shouldn't reach for an LLM call by default just because everything else in the pipeline does.

## Known simplifications (and what a fuller build would add)
- `calculator_agent` uses a fixed 2000-calorie/"lose" target rather than extracting it from the conversation — a full build would add a small extraction step (LLM or structured-output parse) to pull calorie target and goal from the user's message.
- No conversation memory across separate CLI invocations — `main.py` runs one query per process. A real multi-turn build would persist `AgentState` (LangGraph supports this via checkpointers) between turns.
- The eval script compares pipelines qualitatively (which sources got retrieved) rather than against labeled ground truth — see the companion `litmind-project-design.md`'s Phase 7 for what a rigorous precision/recall eval suite on top of this would look like.
