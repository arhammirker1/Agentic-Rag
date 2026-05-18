"""
agenticrag.agents -- Multi-agent retrieval system.

Agents:
  - KeywordAgent     : Expands a question into search keywords for pre-filtering.
  - PlannerAgent     : Selects which documents to search based on graph metadata.
  - HunterAgent      : Searches individual document trees (parallel).
  - SynthesizerAgent : Combines findings into a cohesive answer with citations.
  - EvaluatorAgent   : Decides if more evidence is needed (iterative loop).
  - CriticAgent      : Verifies zero hallucination against source text.
  - Orchestrator     : Runs the full agentic loop.
"""

from .keyword_agent import KeywordAgent
from .planner import PlannerAgent
from .hunter import HunterAgent, HuntResult
from .synthesizer import SynthesizerAgent
from .evaluator import EvaluatorAgent, EvalResult
from .critic import CriticAgent, VerificationResult
from .orchestrator import Orchestrator, ForestResult

__all__ = [
    "KeywordAgent",
    "PlannerAgent",
    "HunterAgent",
    "HuntResult",
    "SynthesizerAgent",
    "EvaluatorAgent",
    "EvalResult",
    "CriticAgent",
    "VerificationResult",
    "Orchestrator",
    "ForestResult",
]