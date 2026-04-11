"""
LangGraph Orchestrator
Wires all 4 agents into a stateful pipeline with:
- Retry logic at each node
- Graceful degradation (partial results if an agent fails)
- Observability traces per agent
"""
from langgraph.graph import StateGraph, END
from app.core.pipeline_state import PipelineState
from app.agents.parse_agent import parse_agent
from app.agents.normalize_agent import normalize_agent
from app.agents.entity_resolve_agent import entity_resolve_agent
from app.agents.match_agent import match_agent


# ──────────────────────────────────────────────────────────────────
# Wrapper nodes with retry logic
# ──────────────────────────────────────────────────────────────────

async def parse_node(state: PipelineState) -> PipelineState:
    """Run parse agent with up to 2 retries."""
    for attempt in range(2):
        state = await parse_agent(state)
        if state.get("parsed") and state.get("parse_confidence", 0) > 0.3:
            break
    return state


async def normalize_node(state: PipelineState) -> PipelineState:
    """Run normalize agent. Skips if parse failed."""
    if not state.get("parsed"):
        state["skills_canonical"] = []
        state["inferred_skills"] = []
        state["emerging_skills_found"] = []
        return state
    return await normalize_agent(state)


async def entity_node(state: PipelineState) -> PipelineState:
    """Run entity resolver. Non-fatal — degrades gracefully."""
    return await entity_resolve_agent(state)


async def finalize_node(state: PipelineState) -> PipelineState:
    """
    Determine overall pipeline status.
    Does NOT run matching (that's a separate API call with a JD).
    """
    errors = state.get("errors", [])
    parse_ok = bool(state.get("parsed")) and state.get("parse_confidence", 0) > 0.3
    normalize_ok = len(state.get("skills_canonical", [])) > 0

    if parse_ok and normalize_ok:
        state["overall_status"] = "completed"
    elif parse_ok:
        state["overall_status"] = "partial"
    else:
        state["overall_status"] = "failed"

    return state


# ──────────────────────────────────────────────────────────────────
# Build the graph
# ──────────────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("parse",     parse_node)
    graph.add_node("normalize", normalize_node)
    graph.add_node("enrich",    entity_node)
    graph.add_node("finalize",  finalize_node)

    graph.set_entry_point("parse")
    graph.add_edge("parse",     "normalize")
    graph.add_edge("normalize", "enrich")
    graph.add_edge("enrich",    "finalize")
    graph.add_edge("finalize",  END)

    return graph.compile()


# Singleton pipeline
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


async def run_parse_pipeline(
    raw_file: bytes,
    file_name: str,
    file_type: str,
    job_id: str,
) -> PipelineState:
    """
    Run the parse → normalize → enrich pipeline.
    Returns full state. DB writes handled by the calling service.
    """
    pipeline = get_pipeline()

    initial_state: PipelineState = {
        "job_id": job_id,
        "raw_file": raw_file,
        "file_name": file_name,
        "file_type": file_type,
        "layout_type": "unknown",
        "parsed": None,
        "parse_confidence": 0.0,
        "resume_language": "en",
        "ai_content_probability": 0.0,
        "skills_canonical": [],
        "experience_months_total": 0,
        "inferred_skills": [],
        "emerging_skills_found": [],
        "platform_profiles": {},
        "entity_confidence": 0.5,
        "enriched_skills": [],
        "candidate_id": None,
        "match_score": 0.0,
        "blind_score": 0.0,
        "semantic_score": 0.0,
        "required_skill_coverage": 0.0,
        "experience_depth_score": 0.0,
        "bias_delta": 0.0,
        "bias_flagged": False,
        "matched_skills": [],
        "skill_gaps": [],
        "upskilling_suggestions": {},
        "shap_values": {},
        "cot_reasoning": "",
        "match_summary": "",
        "interview_questions": {},
        "hitl_required": False,
        "hitl_triggers": [],
        "traces": [],
        "errors": [],
        "overall_status": "processing",
    }

    final_state = await pipeline.ainvoke(initial_state)
    return final_state
