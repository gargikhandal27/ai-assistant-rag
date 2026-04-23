"""
agent.py  (LangGraph version)

HOW LANGGRAPH WORKS — Simple explanation:
==========================================
  Think of it like a flowchart:
    - Each box = a NODE (a Python function)
    - Each arrow = an EDGE (goes from one node to next)
    - Some arrows are CONDITIONAL (only go there IF something is true)

GRAPH STRUCTURE:
================
  START
    ↓
  [decide_tools]     → should I web search?
    ↓ (conditional edge — branches here!)
  [web_search]       → runs ONLY if needed
    ↓
  [search_pdfs]      → always runs
    ↓
  [pick_model]       → router picks best LLM
    ↓
  [generate_answer]  → LLM answers
    ↓
  END

KEY CONCEPT — State:
====================
  A shared dictionary ALL nodes can read and write to.
  Like a notebook passed between every step.
"""

import os
import re
from groq import Groq
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional

# ── LangGraph imports ──────────────────────────────────────────
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# ── Your existing files (no changes needed) ────────────────────
from agent.router import RouterAgent, AVAILABLE_MODELS
from agent.tools import search_web, format_web_results, TOOL_REGISTRY

load_dotenv()


# ══════════════════════════════════════════════════════════════
# STEP 1 — State
# The shared notebook passed between all nodes
# ══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    question:         str
    tool_decision:    Dict[str, Any]
    web_results:      List[Dict[str, Any]]
    web_context:      str
    pdf_results:      List[Dict[str, Any]]
    sources:          List[Dict[str, Any]]
    routing_decision: Dict[str, Any]
    answer:           str
    active_model:     str
    email_result:     Optional[Dict[str, Any]]
    reasoning_trail:  List[Dict[str, Any]]


# ══════════════════════════════════════════════════════════════
# STEP 2 — Nodes (each does ONE job)
# Returns only the keys it updated — LangGraph merges the rest
# ══════════════════════════════════════════════════════════════

def node_decide_tools(state: AgentState) -> dict:
    """Node 1: Should we web search? Email?"""
    question = state["question"]
    print(f"\n[Node 1] Deciding tools...")

    tool_descriptions = "\n".join([
        f'- "{name}": {info["description"]}'
        for name, info in TOOL_REGISTRY.items()
    ])

    prompt = f"""You are a tool selection agent. Be STRICT about web search.

Available tools:
{tool_descriptions}

User question: "{question}"

RULES for NEED_WEB_SEARCH:
- Say YES only if question is about: current news, today's weather, live prices,
  recent events, sports scores, or anything time-sensitive
- Say NO if question is about: concepts, definitions, explanations, document content,
  history, science, math, or anything a textbook would cover
- When in doubt → say NO

Examples:
  "What is supervised learning?" → NO (it's a concept)
  "What is the weather today?"   → YES (time-sensitive)
  "Explain neural networks"      → NO (concept, in PDFs)
  "Latest news about AI?"        → YES (current info needed)

Answer in EXACT format:
NEED_WEB_SEARCH: yes/no
NEED_EMAIL: yes/no
EMAIL_ADDRESS: <email if mentioned in question, else "none">
REASONING: <one sentence why>
"""
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=80,
        )
        raw = response.choices[0].message.content.strip()
        tool_decision = _parse_tool_decision(raw, question)
    except Exception as e:
        tool_decision = {"need_web_search": False, "need_email": False,
                         "email_address": None, "reasoning": f"Failed: {e}"}

    return {
        "tool_decision": tool_decision,
        "reasoning_trail": state.get("reasoning_trail", []) + [
            {"step": "Tool Selection", "decision": tool_decision}
        ],
    }


def node_web_search(state: AgentState) -> dict:
    """Node 2: Search the web"""
    print(f"\n[Node 2] Running web search...")

    # Make query more specific so Tavily returns better results
    query = state["question"] + " 2026"

    search_output = search_web(query)
    web_results   = search_output.get("results", [])

    return {
        "web_results": web_results,
        "web_context": format_web_results(search_output),
        "reasoning_trail": state.get("reasoning_trail", []) + [
            {"step": "Web Search", "results": f"Found {len(web_results)} result(s)"}
        ],
    }


def node_search_pdfs(state: AgentState, retriever) -> dict:
    """Node 3: Search your PDFs (always runs)"""
    print(f"\n[Node 3] Searching PDFs...")
    pdf_results = retriever.retrieve(state["question"], top_k=5, score_threshold=0.0)
    sources = [
        {
            "source":  doc["metadata"].get("source_file", "unknown"),
            "page":    doc["metadata"].get("page", "unknown"),
            "score":   doc["similarity_score"],
            "preview": doc["content"][:120] + "...",
        }
        for doc in pdf_results
    ]

    return {
        "pdf_results": pdf_results,
        "sources": sources,
        "reasoning_trail": state.get("reasoning_trail", []) + [
            {"step": "PDF Retrieval", "results": f"Found {len(pdf_results)} chunk(s)"}
        ],
    }


def node_pick_model(state: AgentState, router: RouterAgent) -> dict:
    """Node 4: Router AI picks best LLM"""
    print(f"\n[Node 4] Picking model...")
    routing_decision = router.decide_model(state["question"])

    return {
        "routing_decision": routing_decision,
        "reasoning_trail": state.get("reasoning_trail", []) + [
            {"step": "Model Routing", "decision": routing_decision}
        ],
    }


def node_generate_answer(state: AgentState, available_llms: dict) -> dict:
    """Node 5: Generate the final answer"""
    routing_decision = state.get("routing_decision", {})
    pdf_context = "\n\n".join([doc["content"] for doc in state.get("pdf_results", [])])

    full_context = ""
    if pdf_context:
        full_context += f"=== PDF Documents ===\n{pdf_context}\n\n"
    if state.get("web_context"):
        full_context += f"=== Web Results ===\n{state['web_context']}"
    if not full_context:
        full_context = "No context found."

    chosen_key = routing_decision.get("chosen_key", "groq")
    llm = available_llms.get(chosen_key) or next(iter(available_llms.values()))

    try:
        answer = llm.generate_response(state["question"], full_context)
        active_model = llm.model_name
    except Exception as e:
        answer = f"Model failed: {e}"
        active_model = "none"

    # Add PDF citations
    citations = [
        f"[{i+1}] {s['source']} (page {s['page']})"
        for i, s in enumerate(state.get("sources", []))
    ]
    if citations:
        answer += "\n\n**Sources:**\n" + "\n".join(citations)

    return {
        "answer": answer,
        "active_model": active_model,
        "reasoning_trail": state.get("reasoning_trail", []) + [
            {"step": "Answer Generation", "model": active_model}
        ],
    }


# ══════════════════════════════════════════════════════════════
# STEP 3 — Conditional Edge
# After Node 1, should we go to web_search or skip it?
# ══════════════════════════════════════════════════════════════

def should_web_search(state: AgentState) -> str:
    """Returns name of next node to go to."""
    if state["tool_decision"].get("need_web_search", False):
        return "web_search"   # → go to web search
    return "search_pdfs"      # → skip web search


# ══════════════════════════════════════════════════════════════
# STEP 4 — Helper
# ══════════════════════════════════════════════════════════════

def _parse_tool_decision(raw: str, question: str) -> Dict[str, Any]:
    result = {"need_web_search": False, "need_email": False,
              "email_address": None, "reasoning": ""}
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("NEED_WEB_SEARCH:"):
            result["need_web_search"] = "yes" in line.lower()
        elif line.upper().startswith("NEED_EMAIL:"):
            result["need_email"] = "yes" in line.lower()
        elif line.upper().startswith("EMAIL_ADDRESS:"):
            addr = line.split(":", 1)[1].strip()
            result["email_address"] = None if addr.lower() == "none" else addr
        elif line.upper().startswith("REASONING:"):
            result["reasoning"] = line.split(":", 1)[1].strip()
    if not result["email_address"]:
        m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", question)
        if m:
            result["email_address"] = m.group()
            result["need_email"] = True
    return result


# ══════════════════════════════════════════════════════════════
# STEP 5 — Build the Graph
# Wire all nodes and edges together
# ══════════════════════════════════════════════════════════════

def build_graph(retriever, router, available_llms):
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("decide_tools",    node_decide_tools)
    graph.add_node("web_search",      node_web_search)
    graph.add_node("search_pdfs",     lambda s: node_search_pdfs(s, retriever))
    graph.add_node("pick_model",      lambda s: node_pick_model(s, router))
    graph.add_node("generate_answer", lambda s: node_generate_answer(s, available_llms))

    # Entry point
    graph.set_entry_point("decide_tools")

    # Edges
    graph.add_conditional_edges(
        "decide_tools",
        should_web_search,
        {"web_search": "web_search", "search_pdfs": "search_pdfs"}
    )
    graph.add_edge("web_search",      "search_pdfs")
    graph.add_edge("search_pdfs",     "pick_model")
    graph.add_edge("pick_model",      "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════
# STEP 6 — Pipeline class (same interface, main.py unchanged)
# ══════════════════════════════════════════════════════════════

class AgenticRAGPipeline:
    def __init__(self, retriever, available_llms: dict):
        self.retriever      = retriever
        self.available_llms = available_llms
        self.router         = RouterAgent()
        self.history        = []
        self.graph          = build_graph(retriever, self.router, available_llms)
        print("AgenticRAGPipeline (LangGraph) initialized ✅")

    def query(self, question: str) -> Dict[str, Any]:
        """
        Run the full agent graph for a question.
        Now with conversation memory — agent remembers previous messages.
        """
        print(f"\n{'='*60}")
        print(f"[LangGraph Agent] Question: {question}")
        print(f"{'='*60}")

        # ── Add conversation memory ────────────────────────────────
        if self.history:
            last_2 = self.history[-2:]
            history_text = "\n".join([
                f"User: {h['question']}\nAssistant: {h['answer'][:200]}"
                for h in last_2
            ])
            question_with_history = f"""Previous conversation:
{history_text}

Current question: {question}"""
            print(f"[Agent] Added history context from {len(last_2)} previous exchange(s)")
        else:
            question_with_history = question

        # ── Initial state ──────────────────────────────────────────
        initial_state: AgentState = {
            "question":         question_with_history,
            "tool_decision":    {},
            "web_results":      [],
            "web_context":      "",
            "pdf_results":      [],
            "sources":          [],
            "routing_decision": {},
            "answer":           "",
            "active_model":     "",
            "email_result":     None,
            "reasoning_trail":  [],
        }

        # ── Run the graph ──────────────────────────────────────────
        final_state = self.graph.invoke(initial_state)

        # ── Save to history ────────────────────────────────────────
        self.history.append({
            "question": question,
            "answer":   final_state["answer"][:200],
            "model":    final_state["active_model"],
        })

        print(f"\n[LangGraph Agent] Done! Model: {final_state['active_model']}")

        return {
            "question":         question,
            "answer":           final_state["answer"],
            "sources":          final_state["sources"],
            "web_results":      final_state["web_results"],
            "active_model":     final_state["active_model"],
            "routing_decision": final_state["routing_decision"],
            "tool_decision":    final_state["tool_decision"],
            "reasoning_trail":  final_state["reasoning_trail"],
            "email_result":     final_state.get("email_result"),
            "history":          self.history,
        }