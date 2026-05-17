"""
orchestrator.py — Agentic loop that ties all agents together.

The Orchestrator runs the full multi-agent retrieval pipeline:

  ┌──────────────────────────────────────────────────────────────┐
  │  User Question                                               │
  │       │                                                      │
  │       ▼                                                      │
  │  ┌─────────┐    Queries Graph DB                             │
  │  │ Planner │──────────────────▶ doc_ids                     │
  │  └─────────┘                                                 │
  │       │                                                      │
  │       ▼                                                      │
  │  ┌─────────┐    Parallel TreeSearcher                        │
  │  │ Hunters │──────────────────▶ text chunks                 │
  │  └─────────┘                                                 │
  │       │                                                      │
  │       ▼                                                      │
  │  ┌──────────────┐                                            │
  │  │ Synthesizer  │──────────────▶ draft answer               │
  │  └──────────────┘                                            │
  │       │                                                      │
  │       ▼                                                      │
  │  ┌────────┐                                                  │
  │  │ Critic │──────────────────▶ verified answer              │
  │  └────────┘                                                  │
  │       │                                                      │
  │       ▼                                                      │
  │  ForestResult                                                │
  └──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import PageIndexConfig
from ..graph.base import DocumentGraph
from ..storage.base import TreeStore
from .planner import PlannerAgent
from .hunter import HunterAgent, HuntResult
from .synthesizer import SynthesizerAgent
from .critic import CriticAgent, VerificationResult
from .evaluator import EvaluatorAgent, EvalResult

log = logging.getLogger(__name__)


# ── Trail logging helpers ─────────────────────────────────────────────────

def _trail_header(title: str) -> None:
    """Print a boxed trail header."""
    print(f"\n{'='*70}")
    print(f"  >> TRAIL: {title}")
    print(f"{'='*70}")

def _trail_step(agent: str, action: str, detail: str = "") -> None:
    """Print a trail step with agent name."""
    prefix = f"  [{agent}]"
    print(f"{prefix} {action}")
    if detail:
        # Indent detail lines
        for line in detail.split('\n')[:20]:  # cap at 20 lines
            print(f"  │  {line}")
        if detail.count('\n') > 20:
            print(f"  │  ... ({detail.count(chr(10)) - 20} more lines)")

def _trail_data(label: str, data: str, max_chars: int = 500) -> None:
    """Print a labeled data block (truncated)."""
    preview = data[:max_chars]
    if len(data) > max_chars:
        preview += f"\n  ... [truncated, {len(data)} total chars]"
    print(f"  ├─ {label}:")
    for line in preview.split('\n'):
        print(f"  │  {line}")

def _trail_separator() -> None:
    print(f"  {'─'*60}")


@dataclass
class ForestResult:
    """
    The output of a Forest.ask() call.

    Attributes
    ----------
    text              : The final verified answer.
    sources           : List of source chunks used (with doc/page metadata).
    confidence        : 0.0 to 1.0 — how confident the system is.
    documents_searched: List of doc_ids that were searched.
    reasoning_trace   : Step-by-step trace of the full pipeline.
    was_rewritten     : Whether the Critic modified the answer.
    hallucinations    : Any hallucinations detected by the Critic.
    elapsed_seconds   : Total time taken.
    """
    text:               str
    sources:            List[Dict[str, Any]]  = field(default_factory=list)
    confidence:         float                 = 1.0
    documents_searched: List[str]             = field(default_factory=list)
    reasoning_trace:    List[str]             = field(default_factory=list)
    was_rewritten:      bool                  = False
    hallucinations:     List[Dict]            = field(default_factory=list)
    elapsed_seconds:    float                 = 0.0

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        n_sources = len(self.sources)
        n_docs = len(self.documents_searched)
        return (
            f"ForestResult(confidence={self.confidence:.2f}, "
            f"sources={n_sources}, docs={n_docs}, "
            f"time={self.elapsed_seconds:.1f}s)"
        )


class Orchestrator:
    """
    Runs the full multi-agent agentic loop.

    Parameters
    ----------
    config  : PageIndexConfig.
    store   : TreeStore backend.
    graph   : DocumentGraph backend.
    """

    def __init__(
        self,
        config: PageIndexConfig,
        store: TreeStore,
        graph: DocumentGraph,
    ):
        self.config = config
        self.store  = store
        self.graph  = graph

        # Initialise agents
        self.planner = PlannerAgent(
            graph,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            max_docs=config.max_docs_per_query,
            quiet=config.quiet,
            enable_thinking=config.enable_thinking,
            num_ctx=config.num_ctx,
        )
        self.hunter = HunterAgent(store, config)
        self.synthesizer = SynthesizerAgent(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            max_output_tokens=config.max_output_tokens,
            quiet=config.quiet,
            enable_thinking=config.enable_thinking,
            num_ctx=config.num_ctx,
        )
        self.critic = CriticAgent(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            quiet=config.quiet,
            enable_thinking=config.enable_thinking,
            num_ctx=config.num_ctx,
        )
        self.evaluator = EvaluatorAgent(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            quiet=config.quiet,
            enable_thinking=config.enable_thinking,
            num_ctx=config.num_ctx,
        )

    def ask(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        skip_critic: bool = False,
    ) -> ForestResult:
        """
        Run the full agentic retrieval pipeline.

        Parameters
        ----------
        question    : The user's question.
        history     : Optional conversation history.
        skip_critic : If True, skip the hallucination check (faster).

        Returns
        -------
        ForestResult with the verified answer and metadata.
        """
        start_time = time.time()
        history = history or []
        trace: List[str] = []
        verbose = self.config.verbose and not self.config.quiet

        # ── Step 1: Plan ─────────────────────────────────────────────────
        if verbose:
            _trail_header("STEP 1 — PLANNER")
            _trail_step("Planner", "Selecting relevant documents from graph ...")
        self._log("Planning: selecting relevant documents ...")
        trace.append("[Planner] Selecting documents from graph …")

        plan = self.planner.plan(question, history=history)
        doc_ids = plan.doc_ids

        if verbose:
            _trail_step("Planner", f"Reasoning: {plan.reasoning}")
            _trail_step("Planner", f"Selected {len(doc_ids)} doc(s): {doc_ids}")
            _trail_separator()

        trace.append(f"[Planner] {plan.reasoning}")
        trace.append(f"[Planner] Selected {len(doc_ids)} documents: {doc_ids}")

        if not doc_ids:
            return ForestResult(
                text="No documents in the index match this question.",
                reasoning_trace=trace,
                elapsed_seconds=time.time() - start_time,
            )

        # ── Step 2+3: Iterative Hunt → Synthesize → Evaluate loop ────────
        max_rounds = getattr(self.config, 'max_retrieval_rounds', 2)
        all_chunks: List[Dict] = []
        visited_nodes: set = set()
        effective_question = question
        draft = ""

        # All doc_ids from the graph for expand_docs check
        all_available_docs = [d.doc_id for d in self.graph.list_documents()]

        for rnd in range(1, max_rounds + 1):
            # ── Hunt ─────────────────────────────────────────────────────
            if verbose:
                _trail_header(f"ROUND {rnd}/{max_rounds} — HUNTER")
                _trail_step("Hunter", f"Searching {len(doc_ids)} document(s) in parallel ...")
                if rnd > 1:
                    _trail_step("Hunter", f"Refined query: '{effective_question}'")
                    _trail_step("Hunter", f"Excluding {len(visited_nodes)} previously-visited node(s)")
            self._log(f"[Round {rnd}] Hunting across {len(doc_ids)} documents ...")
            trace.append(f"[Hunter:R{rnd}] Searching {len(doc_ids)} documents ...")

            hunt_results = self.hunter.hunt_parallel(
                doc_ids,
                effective_question,
                history=history,
                max_workers=min(5, len(doc_ids)),
                exclude_nodes=visited_nodes if visited_nodes else None,
            )

            # Collect new chunks
            new_chunks: List[Dict] = []
            for hr in hunt_results:
                if hr.success:
                    for chunk in hr.chunks:
                        nid = chunk.get('node_id', '')
                        if nid not in visited_nodes:
                            new_chunks.append(chunk)
                            visited_nodes.add(nid)
                    if verbose:
                        _trail_step("Hunter", f"Doc '{hr.doc_id[:30]}' -> {len(hr.chunks)} chunk(s), {hr.iterations} iteration(s)")
                        for chunk in hr.chunks:
                            text_len = len(chunk.get('text', ''))
                            _trail_step("Hunter", f"  Node '{chunk.get('node_title', '?')}' (pages {chunk.get('start_page','?')}-{chunk.get('end_page','?')}, {text_len} chars)")
                        for step in hr.reasoning_steps:
                            _trail_step("Hunter", f"  {step}")
                    for step in hr.reasoning_steps:
                        trace.append(f"[Hunter:R{rnd}:{hr.doc_id[:20]}] {step}")
                else:
                    if verbose:
                        _trail_step("Hunter", f"Doc '{hr.doc_id[:30]}' FAILED: {hr.error}")
                    trace.append(f"[Hunter:R{rnd}:{hr.doc_id[:20]}] FAILED: {hr.error}")

            all_chunks.extend(new_chunks)

            if verbose:
                _trail_separator()
                _trail_step("Hunter", f"New chunks this round: {len(new_chunks)}")
                _trail_step("Hunter", f"Total chunks accumulated: {len(all_chunks)}")
                total_text = sum(len(c.get('text', '')) for c in all_chunks)
                _trail_step("Hunter", f"Total text volume: {total_text:,} chars")
                _trail_separator()

            trace.append(f"[Hunter:R{rnd}] +{len(new_chunks)} new chunks, {len(all_chunks)} total.")

            if not all_chunks:
                if rnd == max_rounds:
                    return ForestResult(
                        text="I searched the relevant documents but could not find "
                             "specific information to answer this question.",
                        documents_searched=doc_ids,
                        reasoning_trace=trace,
                        elapsed_seconds=time.time() - start_time,
                    )
                continue  # try another round

            # If this is a subsequent round and no new chunks were found,
            # the corpus is exhausted — stop looping to save time.
            if rnd > 1 and len(new_chunks) == 0:
                if verbose:
                    _trail_step("Hunter", "No new chunks found — corpus exhausted, stopping loop.")
                self._log(f"[Round {rnd}] No new chunks found. Corpus exhausted.")
                trace.append(f"[Hunter:R{rnd}] Corpus exhausted — skipping re-synthesis.")
                break

            # ── Synthesize ───────────────────────────────────────────────
            if verbose:
                _trail_header(f"ROUND {rnd}/{max_rounds} — SYNTHESIZER")
                _trail_step("Synthesizer", f"Input: {len(all_chunks)} chunks, question='{question}'")
                for ci, chunk in enumerate(all_chunks, 1):
                    text = chunk.get('text', '')
                    _trail_step("Synthesizer", f"  Chunk {ci}: '{chunk.get('node_title', '?')}' -> {len(text)} chars")
            self._log(f"[Round {rnd}] Synthesizing answer ...")
            trace.append(f"[Synthesizer:R{rnd}] Combining {len(all_chunks)} chunks ...")

            draft = self.synthesizer.synthesize(question, all_chunks, history=history)

            if verbose:
                _trail_separator()
                _trail_step("Synthesizer", f"Draft answer: {len(draft)} chars")
                _trail_data("Draft answer", draft, max_chars=1000)
                _trail_separator()

            trace.append(f"[Synthesizer:R{rnd}] Draft: {len(draft)} chars.")

            # ── Evaluate (skip on last round) ────────────────────────────
            if rnd < max_rounds:
                if verbose:
                    _trail_header(f"ROUND {rnd}/{max_rounds} — EVALUATOR")
                    _trail_step("Evaluator", "Checking if evidence is sufficient ...")
                self._log(f"[Round {rnd}] Evaluating evidence sufficiency ...")

                evaluation = self.evaluator.evaluate(
                    question=question,
                    draft=draft,
                    chunks=all_chunks,
                    searched_doc_ids=doc_ids,
                    all_doc_ids=all_available_docs,
                )

                if verbose:
                    _trail_step("Evaluator", f"Sufficient: {evaluation.sufficient}")
                    _trail_step("Evaluator", f"Confidence: {evaluation.confidence:.0%}")
                    if evaluation.gaps:
                        _trail_step("Evaluator", f"Gaps: {evaluation.gaps}")
                    if evaluation.refined_query:
                        _trail_step("Evaluator", f"Refined query: '{evaluation.refined_query}'")
                    if evaluation.expand_docs:
                        _trail_step("Evaluator", "Recommends searching additional documents")
                    _trail_separator()

                trace.append(
                    f"[Evaluator:R{rnd}] sufficient={evaluation.sufficient}, "
                    f"confidence={evaluation.confidence:.2f}, gaps={evaluation.gaps}"
                )

                if evaluation.sufficient:
                    self._log(f"[Round {rnd}] Evidence sufficient, proceeding to Critic.")
                    break

                # Prepare next round
                effective_question = evaluation.refined_query or question
                self._log(
                    f"[Round {rnd}] Need more context. "
                    f"Gaps: {evaluation.gaps}. Refining query ..."
                )

                # Optionally expand document set
                if evaluation.expand_docs:
                    new_plan = self.planner.plan(effective_question, history=history)
                    for did in new_plan.doc_ids:
                        if did not in doc_ids:
                            doc_ids.append(did)
                    if verbose:
                        _trail_step("Evaluator", f"Expanded doc set to {len(doc_ids)} doc(s): {doc_ids}")
                    trace.append(f"[Evaluator:R{rnd}] Expanded to {len(doc_ids)} docs.")

        # ── Step 4: Verify (Critic) ──────────────────────────────────────
        if skip_critic:
            trace.append("[Critic] SKIPPED (skip_critic=True)")
            final_answer = draft
            confidence = 0.8  # default when unchecked
            was_rewritten = False
            hallucinations = []
        else:
            if verbose:
                _trail_header("STEP 4 — CRITIC")
                _trail_step("Critic", "Verifying answer against source evidence ...")
                _trail_step("Critic", f"Answer length: {len(draft)} chars")
                _trail_step("Critic", f"Evidence chunks: {len(all_chunks)}")
            self._log("Verifying answer (zero-hallucination check) ...")
            trace.append("[Critic] Verifying answer against source evidence …")

            verification = self.critic.verify(draft, all_chunks)
            final_answer = verification.answer
            confidence = verification.confidence
            was_rewritten = verification.was_rewritten
            hallucinations = verification.hallucinations

            if verbose:
                _trail_step("Critic", f"Verdict: {verification.verdict}")
                _trail_step("Critic", f"Confidence: {confidence:.0%}")
                if hallucinations:
                    _trail_step("Critic", f"Hallucinations found: {len(hallucinations)}")
                    for h in hallucinations:
                        _trail_step("Critic", f"  ❌ Claim: \"{h.get('claim', '?')}\"")
                        _trail_step("Critic", f"     Reason: {h.get('reason', '?')}")
                if was_rewritten:
                    _trail_step("Critic", f"Answer was REWRITTEN ({len(final_answer)} chars)")
                    _trail_data("Rewritten answer", final_answer, max_chars=1000)
                else:
                    _trail_step("Critic", "[OK] Answer PASSED -- no hallucinations")
                _trail_separator()

            if was_rewritten:
                trace.append(
                    f"[Critic] REWRITTEN — {len(hallucinations)} hallucination(s) removed."
                )
            else:
                trace.append(f"[Critic] PASSED — confidence {confidence:.2f}")

        elapsed = time.time() - start_time
        self._log(f"Done in {elapsed:.1f}s")

        if verbose:
            _trail_header("PIPELINE COMPLETE")
            _trail_step("Result", f"Total time: {elapsed:.1f}s")
            _trail_step("Result", f"Final answer length: {len(final_answer)} chars")
            _trail_step("Result", f"Confidence: {confidence:.0%}")
            _trail_step("Result", f"Sources: {len(all_chunks)}")
            _trail_step("Result", f"Was rewritten: {was_rewritten}")
            print(f"{'='*70}\n")

        # Build source citations
        sources = [
            {
                "doc_id": c["doc_id"],
                "doc_title": c.get("doc_title", ""),
                "section": c.get("node_title", ""),
                "pages": f"{c.get('start_page', '?')}-{c.get('end_page', '?')}",
            }
            for c in all_chunks
        ]

        return ForestResult(
            text=final_answer,
            sources=sources,
            confidence=confidence,
            documents_searched=doc_ids,
            reasoning_trace=trace,
            was_rewritten=was_rewritten,
            hallucinations=hallucinations,
            elapsed_seconds=elapsed,
        )

    def _log(self, msg: str) -> None:
        if self.config.verbose and not self.config.quiet:
            print(f"[agenticrag] {msg}")
        log.info(msg)
