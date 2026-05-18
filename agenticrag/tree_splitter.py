"""
tree_splitter.py — Split oversized trees into topic-based sub-trees.

When a document is very large (e.g. a 160-page SEC 10-K filing), the tree
index can contain 80-100+ nodes.  The compact tree JSON alone can be
10,000-40,000 tokens — far exceeding typical model context/rate limits.

This module detects oversized trees and splits them into multiple smaller
sub-trees, each covering a coherent topic section of the document.  The
split follows the Markdown heading hierarchy, so each sub-tree contains
a logical group of related sections.

Usage (called automatically by the ingestion pipeline):
    from agenticrag.tree_splitter import should_split, split_tree

    tree = build_tree("big_report.pdf", config)
    if should_split(tree):
        sub_trees = split_tree(tree)
        for st in sub_trees:
            store.save(st["doc_id"], st["tree"])
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────

# Maximum nodes before we consider splitting.
# A compact tree with 30 nodes ≈ 3-4K tokens — safe for 8K TPM.
MAX_NODES_PER_TREE = 30

# Maximum compact JSON characters before we force a split.
# ~4 chars per token, 5000 tokens ≈ 20,000 chars.
MAX_COMPACT_CHARS = 20_000

# Minimum nodes to bother creating a sub-tree.
# Very small fragments aren't worth the overhead.
MIN_NODES_PER_SUBTREE = 2


# ── Public API ────────────────────────────────────────────────────────────

def should_split(tree: Dict[str, Any]) -> bool:
    """
    Check if a tree is too large and should be split.

    Parameters
    ----------
    tree : The full tree dict from build_tree().

    Returns
    -------
    True if the tree should be split into sub-trees.
    """
    nodes = tree.get("nodes", [])
    total = _count_nodes(nodes)

    if total <= MAX_NODES_PER_TREE:
        return False

    # Also check the compact JSON size
    compact = _compact_json_size(nodes)
    if compact <= MAX_COMPACT_CHARS:
        return False

    log.info(
        f"Tree has {total} nodes, compact JSON is {compact:,} chars — "
        f"splitting required (thresholds: {MAX_NODES_PER_TREE} nodes, "
        f"{MAX_COMPACT_CHARS:,} chars)."
    )
    return True


def split_tree(
    tree: Dict[str, Any],
    parent_doc_id: str,
) -> List[Dict[str, Any]]:
    """
    Split a large tree into multiple topic-based sub-trees.

    Each sub-tree is a dict with:
      - "sub_doc_id"   : Unique ID (parent_doc_id + "__partNN")
      - "tree"         : The sub-tree dict (same schema as full tree)
      - "title"        : Title describing this sub-tree's content
      - "summary"      : Multi-sentence summary of the sub-tree's sections
      - "page_range"   : (start_page, end_page) tuple
      - "part_index"   : 0-based part number

    Parameters
    ----------
    tree           : The full tree dict from build_tree().
    parent_doc_id  : The parent document's doc_id.

    Returns
    -------
    List of sub-tree descriptors.
    """
    nodes = tree.get("nodes", [])
    source_file = tree.get("source_file", "")
    total_pages = tree.get("total_pages", 0)
    doc_description = tree.get("document_description", "")

    if not nodes:
        return []

    # Group top-level nodes into chunks that fit under the threshold
    groups = _group_nodes(nodes)

    sub_trees: List[Dict[str, Any]] = []
    for i, group in enumerate(groups):
        part_num = i + 1
        sub_doc_id = f"{parent_doc_id}__part{part_num:02d}"

        # Re-number node IDs within this sub-tree
        _renumber_nodes(group)

        # Compute page range
        start_page = _min_page(group)
        end_page = _max_page(group)

        # Build title from section titles
        titles = [n.get("title", "") for n in group if n.get("title")]
        if len(titles) <= 3:
            combined_title = " | ".join(titles)
        else:
            combined_title = f"{titles[0]} ... {titles[-1]} ({len(titles)} sections)"

        # Build summary from section content
        summary = _build_summary(group, source_file, start_page, end_page)

        sub_tree_dict = {
            "source_file": source_file,
            "total_pages": total_pages,
            "parent_doc_id": parent_doc_id,
            "part_index": i,
            "part_count": len(groups),
            "page_range": [start_page, end_page],
            "document_description": summary,
            "nodes": group,
        }

        sub_trees.append({
            "sub_doc_id": sub_doc_id,
            "tree": sub_tree_dict,
            "title": combined_title,
            "summary": summary,
            "page_range": (start_page, end_page),
            "part_index": i,
        })

    log.info(
        f"Split tree into {len(sub_trees)} sub-trees "
        f"(from {_count_nodes(nodes)} total nodes)."
    )
    return sub_trees


# ── Grouping Algorithm ────────────────────────────────────────────────────

def _group_nodes(nodes: List[Dict]) -> List[List[Dict]]:
    """
    Group top-level nodes into chunks that each fit under the threshold.

    Strategy:
    1. Walk top-level nodes in order.
    2. Add each node to the current group.
    3. When the group exceeds MAX_NODES_PER_TREE, start a new group.
    4. If a single top-level node has too many children, promote its
       children to top-level and group them instead.
    """
    # Pre-process: if there's a single dominant node with many children
    # (common in 10-K filings: one "3M COMPANY" node with 200+ children),
    # promote its children to the top level for better splitting.
    effective_nodes = _flatten_dominant_node(nodes)

    groups: List[List[Dict]] = []
    current_group: List[Dict] = []
    current_count = 0

    for node in effective_nodes:
        node_count = _count_nodes([node])

        # If this single node is still too big, recurse on its children
        if node_count > MAX_NODES_PER_TREE and node.get("nodes"):
            # Flush current group first
            if current_group:
                groups.append(current_group)
                current_group = []
                current_count = 0

            # Recursively group this node's children
            child_groups = _group_nodes(node["nodes"])
            groups.extend(child_groups)
        else:
            # Check if adding this node would exceed the threshold
            if current_count + node_count > MAX_NODES_PER_TREE and current_group:
                groups.append(current_group)
                current_group = []
                current_count = 0

            current_group.append(node)
            current_count += node_count

    if current_group:
        groups.append(current_group)

    # Merge tiny trailing groups with the previous group
    merged: List[List[Dict]] = []
    for g in groups:
        count = _count_nodes(g)
        if merged and count < MIN_NODES_PER_SUBTREE:
            merged[-1].extend(g)
        else:
            merged.append(g)

    return merged if merged else [nodes]


def _flatten_dominant_node(nodes: List[Dict]) -> List[Dict]:
    """
    If there's one dominant node containing most of the tree's children,
    promote its children to the top level for better splitting.

    Example: A 10-K with structure:
        UNITED STATES (leaf)
        SECURITIES... (leaf)
        FORM 10-K (leaf)
        3M COMPANY (236 children)  <-- dominant node

    Becomes:
        UNITED STATES (leaf)
        SECURITIES... (leaf)
        FORM 10-K (leaf)
        child_1, child_2, ..., child_236  <-- promoted
    """
    if len(nodes) <= 1:
        # Only one node — just return it, _group_nodes will recurse
        return nodes

    # Find if there's a dominant node (>80% of total nodes)
    total = _count_nodes(nodes)
    for i, node in enumerate(nodes):
        node_count = _count_nodes([node])
        if node_count > total * 0.8 and node.get("nodes"):
            # This node dominates — promote its children
            result = []
            # Keep the small nodes before it
            result.extend(nodes[:i])
            # Add a leaf version of this node (no children) as a header
            header = {k: v for k, v in node.items() if k != "nodes"}
            header["text"] = header.get("text", "")[:200]  # trim text
            header["nodes"] = []
            result.append(header)
            # Promote children
            result.extend(node["nodes"])
            # Keep nodes after it
            result.extend(nodes[i + 1:])
            return result

    return nodes


# ── Summary Builder ───────────────────────────────────────────────────────

def _build_summary(
    nodes: List[Dict],
    source_file: str,
    start_page: int,
    end_page: int,
) -> str:
    """
    Build a rich text summary of a sub-tree from its node titles and text.

    This summary is used by the Planner to select the right sub-tree.
    It includes section titles, key content snippets, and page range.
    """
    parts = []
    parts.append(
        f"This section of '{source_file}' covers pages {start_page}-{end_page}."
    )

    # Collect all section titles with their content previews
    section_details = []
    _collect_section_info(nodes, section_details, depth=0)

    if section_details:
        parts.append("Sections covered:")
        for title, preview, depth in section_details[:15]:  # cap at 15
            indent = "  " * depth
            if preview:
                parts.append(f"{indent}- {title}: {preview}")
            else:
                parts.append(f"{indent}- {title}")

    return "\n".join(parts)


def _collect_section_info(
    nodes: List[Dict],
    result: List,
    depth: int = 0,
    max_preview: int = 150,
) -> None:
    """Recursively collect (title, preview, depth) from nodes."""
    for node in nodes:
        title = node.get("title", "Untitled")
        text = node.get("text", "")

        # Extract a meaningful preview from the text
        preview = ""
        if text:
            # Take first ~150 chars, try to break at sentence
            snippet = text[:max_preview].strip()
            dot_pos = snippet.rfind(".")
            if dot_pos > 50:
                snippet = snippet[:dot_pos + 1]
            preview = snippet.replace("\n", " ")

        result.append((title, preview, depth))

        # Recurse into children (max 2 levels deep for summaries)
        if depth < 2 and node.get("nodes"):
            _collect_section_info(node["nodes"], result, depth + 1, max_preview)


# ── Helpers ───────────────────────────────────────────────────────────────

def _count_nodes(nodes: List[Dict]) -> int:
    """Count total nodes (including nested children)."""
    return sum(1 + _count_nodes(n.get("nodes", [])) for n in nodes)


def _compact_json_size(nodes: List[Dict]) -> int:
    """Estimate the compact JSON size (without text fields)."""
    def _strip(nodes):
        out = []
        for n in nodes:
            item = {k: v for k, v in n.items() if k not in ("text", "nodes")}
            kids = _strip(n.get("nodes", []))
            if kids:
                item["nodes"] = kids
            out.append(item)
        return out

    compact = _strip(nodes)
    return len(json.dumps(compact, indent=2))


def _renumber_nodes(nodes: List[Dict], counter: Optional[List[int]] = None) -> None:
    """Re-number node_ids starting from 0001."""
    if counter is None:
        counter = [1]
    for n in nodes:
        n["node_id"] = f"{counter[0]:04d}"
        counter[0] += 1
        if n.get("nodes"):
            _renumber_nodes(n["nodes"], counter)


def _min_page(nodes: List[Dict]) -> int:
    """Find the minimum start_index across all nodes."""
    pages = []
    for n in nodes:
        pages.append(n.get("start_index", 0))
        if n.get("nodes"):
            pages.append(_min_page(n["nodes"]))
    return min(pages) if pages else 0


def _max_page(nodes: List[Dict]) -> int:
    """Find the maximum end_index across all nodes."""
    pages = []
    for n in nodes:
        pages.append(n.get("end_index", 0))
        if n.get("nodes"):
            pages.append(_max_page(n["nodes"]))
    return max(pages) if pages else 0
