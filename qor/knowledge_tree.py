"""
QOR Knowledge Tree — Unified Knowledge Index
===============================================
Makes the RocksDB KnowledgeGraph the CENTRAL INDEX connecting all knowledge.
User feedback (corrections, preferences) creates graph nodes that filter future answers.
Trade outcomes generate lesson nodes that improve future analysis.

All existing stores keep working — the graph becomes a layer ON TOP that indexes,
filters, and connects them.

Node types: user, correction, blocked_fact, preference, trade_pattern, lesson, knowledge, historical_event
Edge predicates: corrected, blocked, prefers, dislikes, resulted_in_lesson, interests_in, learned
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Stop words for keyword matching in tree_search
_TREE_STOP_WORDS = frozenset({
    "the", "what", "how", "does", "who", "when", "where", "why",
    "is", "are", "was", "were", "can", "will", "would", "should",
    "about", "from", "with", "this", "that", "for", "and", "but",
    "tell", "me", "please", "show", "give", "do", "did", "has",
    "have", "been", "being", "its", "it", "a", "an", "of", "in",
    "to", "on", "at", "by", "or", "not", "no", "so", "if", "up",
    "my", "your", "our", "their", "any", "some", "most", "more",
})


# ==============================================================================
# FEEDBACK DETECTION — Keyword matching for user feedback signals
# ==============================================================================

CORRECTION_SIGNALS = [
    "that's wrong", "thats wrong", "that is wrong",
    "incorrect", "not correct", "not right", "not true",
    "bad analysis", "wrong analysis", "bad answer",
    "you're wrong", "youre wrong", "you are wrong",
    "actually no", "no that's", "no thats",
    "that was wrong", "wrong answer", "that's incorrect",
    "that's not right", "thats not right",
    "you made a mistake", "mistake in your",
    "inaccurate", "not accurate",
]

PREFERENCE_LIKE = [
    "i like", "i love", "i prefer", "i enjoy",
    "more of this", "more like this", "great answer",
    "good answer", "perfect answer", "exactly what i wanted",
    "keep doing this", "this is great", "this is good",
    "i appreciate", "well done",
]

PREFERENCE_DISLIKE = [
    "i don't like", "i dont like", "i dislike",
    "stop talking about", "not interested in",
    "too much about", "less of this", "i hate",
    "stop with the", "enough about", "don't want to hear",
    "dont want to hear", "boring",
]

BLOCK_SIGNALS = [
    "don't use that source", "dont use that source",
    "that tool is broken", "bad source", "unreliable source",
    "never use", "stop using", "don't trust", "dont trust",
    "that data is wrong", "bad data",
]

# Phase D: 4 new feedback signal types
STALE_SIGNALS = [
    "that's outdated", "thats outdated", "that is outdated",
    "out of date", "not current", "stale data", "old data",
    "update that", "that's old", "thats old", "that is old",
    "no longer accurate", "needs updating", "not up to date",
]

SOFT_CORRECTION_SIGNALS = [
    "i don't think that's right", "i dont think thats right",
    "not sure that's correct", "not sure thats correct",
    "that doesn't sound right", "that doesnt sound right",
    "are you sure about that", "i doubt that", "that seems off",
    "that might be wrong", "that could be wrong",
    "hmm that doesn't seem right", "hmm that doesnt seem right",
]

USER_PROVIDES_SIGNALS = [
    "actually it's", "actually its", "actually, it's", "actually, its",
    "the correct answer is", "the real answer is", "the right answer is",
    "it should be", "the real number is", "the actual number is",
    "no, it's", "no, its", "no it's", "no its",
    "let me correct that", "the truth is", "in reality",
]

SOURCE_BLOCK_SIGNALS = [
    "that source is bad", "unreliable source",
    "that api is broken", "that tool gives wrong data",
    "stop using that tool", "don't use that api", "dont use that api",
    "that feed is unreliable", "block that source",
    "that source lies", "untrustworthy source",
]


@dataclass
class FeedbackEvent:
    """Detected user feedback."""
    event_type: str  # correction, stale, soft_correction, block, user_provides,
                     # preference_like, preference_dislike, source_block
    confidence: float
    user_quote: str  # The exact user text
    target_content: str  # What the feedback is about (last answer or topic)
    topic: str = ""  # Extracted topic if preference
    provided_value: str = ""  # User-provided correct value (for user_provides)


class FeedbackDetector:
    """Detects user feedback from question text."""

    @staticmethod
    def detect(question: str, last_answer: Optional[dict] = None,
               chat_context: Optional[list] = None) -> Optional[FeedbackEvent]:
        """Detect feedback from user question.

        Args:
            question: Current user input.
            last_answer: Previous answer dict (from gate.answer()).
            chat_context: Recent chat messages.

        Returns:
            FeedbackEvent or None if no feedback detected.
        """
        q_lower = question.lower().strip()

        # Only detect feedback if there was a previous answer
        # (user is responding to AI, not asking a fresh question)
        target = ""
        if last_answer:
            target = last_answer.get("answer", "")[:200]

        # Check correction signals (hard correction)
        for signal in CORRECTION_SIGNALS:
            if signal in q_lower:
                if not target and not last_answer:
                    continue
                return FeedbackEvent(
                    event_type="correction",
                    confidence=0.85,
                    user_quote=question,
                    target_content=target,
                )

        # Check soft correction signals (less severe)
        for signal in SOFT_CORRECTION_SIGNALS:
            if signal in q_lower:
                if not target and not last_answer:
                    continue
                return FeedbackEvent(
                    event_type="soft_correction",
                    confidence=0.6,
                    user_quote=question,
                    target_content=target,
                )

        # Check stale signals
        for signal in STALE_SIGNALS:
            if signal in q_lower:
                return FeedbackEvent(
                    event_type="stale",
                    confidence=0.75,
                    user_quote=question,
                    target_content=target,
                )

        # Check user-provides signals (user gives correct answer)
        for signal in USER_PROVIDES_SIGNALS:
            if signal in q_lower:
                provided = _extract_feedback_topic(q_lower, signal)
                return FeedbackEvent(
                    event_type="user_provides",
                    confidence=0.85,
                    user_quote=question,
                    target_content=target,
                    provided_value=provided,
                )

        # Check source block signals
        for signal in SOURCE_BLOCK_SIGNALS:
            if signal in q_lower:
                source_name = _extract_feedback_topic(q_lower, signal)
                return FeedbackEvent(
                    event_type="source_block",
                    confidence=0.8,
                    user_quote=question,
                    target_content=target,
                    topic=source_name,
                )

        # Check block signals
        for signal in BLOCK_SIGNALS:
            if signal in q_lower:
                return FeedbackEvent(
                    event_type="block",
                    confidence=0.8,
                    user_quote=question,
                    target_content=target,
                )

        # Check preference like
        for signal in PREFERENCE_LIKE:
            if signal in q_lower:
                topic = _extract_feedback_topic(q_lower, signal)
                return FeedbackEvent(
                    event_type="preference_like",
                    confidence=0.7,
                    user_quote=question,
                    target_content=target,
                    topic=topic,
                )

        # Check preference dislike
        for signal in PREFERENCE_DISLIKE:
            if signal in q_lower:
                topic = _extract_feedback_topic(q_lower, signal)
                return FeedbackEvent(
                    event_type="preference_dislike",
                    confidence=0.7,
                    user_quote=question,
                    target_content=target,
                    topic=topic,
                )

        return None


def _extract_feedback_topic(text: str, signal: str) -> str:
    """Extract topic from feedback text after the signal phrase."""
    idx = text.find(signal)
    if idx < 0:
        return ""
    after = text[idx + len(signal):].strip().strip(".,!?")
    # Take first few meaningful words
    words = after.split()[:5]
    topic = " ".join(words).strip()
    # Remove common filler
    for filler in ["the ", "about ", "your ", "this "]:
        if topic.startswith(filler):
            topic = topic[len(filler):]
    return topic


# ==============================================================================
# FEEDBACK PROCESSOR — Writes feedback to graph
# ==============================================================================

class FeedbackProcessor:
    """Writes user feedback into the knowledge graph."""

    @staticmethod
    def process(event: FeedbackEvent, graph, user_id: str,
                memory_store=None, cache_store=None) -> str:
        """Process a feedback event and write to graph.

        Args:
            event: The detected feedback event.
            graph: QORGraph instance (must be open).
            user_id: User node ID (e.g. "user:ravi").
            memory_store: Optional MemoryStore for reducing confidence.
            cache_store: Optional CacheStore for invalidating stale entries.

        Returns:
            Description of what was recorded.
        """
        if graph is None or not hasattr(graph, 'is_open') or not graph.is_open:
            return ""

        now = datetime.now().isoformat()
        content_hash = hashlib.sha256(
            (event.user_quote + event.target_content).encode()
        ).hexdigest()[:8]

        # Ensure user node exists
        graph.add_node(user_id, node_type="user", properties={
            "created_at": now,
        })

        if event.event_type == "correction":
            return _process_correction(event, graph, user_id, content_hash,
                                       now, memory_store, cache_store)
        elif event.event_type == "soft_correction":
            return _process_soft_correction(event, graph, user_id,
                                            content_hash, now, memory_store)
        elif event.event_type == "stale":
            return _process_stale(event, graph, user_id, content_hash, now,
                                  cache_store)
        elif event.event_type == "user_provides":
            return _process_user_provides(event, graph, user_id,
                                          content_hash, now)
        elif event.event_type == "source_block":
            return _process_source_block(event, graph, user_id,
                                         content_hash, now)
        elif event.event_type == "block":
            return _process_block(event, graph, user_id, content_hash, now)
        elif event.event_type == "preference_like":
            return _process_preference(event, graph, user_id, "like", now)
        elif event.event_type == "preference_dislike":
            return _process_preference(event, graph, user_id, "dislike", now)

        return ""


def _process_correction(event, graph, user_id, content_hash, now,
                        memory_store, cache_store=None):
    """Create correction + blocked_fact nodes and invalidate stale data."""
    corr_id = f"corr:{content_hash}"
    blocked_id = f"blocked:{content_hash}"

    # Create correction node
    graph.add_node(corr_id, node_type="correction", properties={
        "original_claim": event.target_content[:200],
        "reason": event.user_quote[:200],
        "user_quote": event.user_quote[:200],
        "timestamp": now,
    })

    # Create blocked_fact node
    graph.add_node(blocked_id, node_type="blocked_fact", properties={
        "original_content": event.target_content[:200],
        "blocked_by": corr_id,
        "reason": event.user_quote[:200],
        "timestamp": now,
    })

    # Edges: user → corrected → correction → blocked → blocked_fact
    graph.add_edge(user_id, "corrected", corr_id,
                   confidence=1.0, source="user_feedback")
    graph.add_edge(corr_id, "blocked", blocked_id,
                   confidence=1.0, source="user_feedback")

    # Reduce confidence of matching memory entries
    if memory_store is not None and event.target_content:
        try:
            matches = memory_store.search(event.target_content, top_k=3)
            for key, entry, score in matches:
                if score >= 2:  # Only reduce high-match entries
                    entry.confidence = min(entry.confidence, 0.1)
                    memory_store._maybe_flush()
        except Exception:
            pass

    # Invalidate matching cache entries so next question re-fetches
    if cache_store is not None and event.target_content:
        try:
            removed = cache_store.invalidate_matching(event.target_content)
            if removed:
                logger.info("Correction: invalidated %d cache entries", removed)
        except Exception:
            pass

    # Invalidate matching knowledge nodes in the graph (mark stale)
    _invalidate_tree_knowledge(graph, event.target_content)

    return "correction recorded"


def _process_block(event, graph, user_id, content_hash, now):
    """Create blocked_fact node from explicit block request."""
    blocked_id = f"blocked:{content_hash}"

    graph.add_node(blocked_id, node_type="blocked_fact", properties={
        "original_content": event.target_content[:200],
        "blocked_by": user_id,
        "reason": event.user_quote[:200],
        "timestamp": now,
    })

    graph.add_edge(user_id, "blocked", blocked_id,
                   confidence=1.0, source="user_feedback")

    return "source blocked"


def _process_preference(event, graph, user_id, polarity, now):
    """Create/update preference node."""
    topic = event.topic or "general"
    pref_id = f"pref:{topic}:{polarity}"

    # Get existing or create new
    existing = graph.get_node(pref_id)
    strength = 0.5
    if existing:
        props = existing.get("properties", {})
        strength = min(1.0, props.get("strength", 0.5) + 0.1)

    graph.add_node(pref_id, node_type="preference", properties={
        "topic": topic,
        "polarity": polarity,
        "strength": strength,
        "last_updated": now,
    })

    predicate = "prefers" if polarity == "like" else "dislikes"
    graph.add_edge(user_id, predicate, pref_id,
                   confidence=strength, source="user_feedback")

    return f"preference updated ({polarity}: {topic})"


def _process_soft_correction(event, graph, user_id, content_hash, now,
                             memory_store):
    """Soft correction: reduce confidence but don't create blocked_fact."""
    corr_id = f"softcorr:{content_hash}"

    graph.add_node(corr_id, node_type="correction", properties={
        "original_claim": event.target_content[:200],
        "reason": event.user_quote[:200],
        "severity": "soft",
        "timestamp": now,
    })

    graph.add_edge(user_id, "corrected", corr_id,
                   confidence=0.6, source="user_feedback")

    # Soft: reduce confidence by 0.8x (not full block)
    if memory_store is not None and event.target_content:
        try:
            matches = memory_store.search(event.target_content, top_k=3)
            for key, entry, score in matches:
                if score >= 2:
                    entry.confidence = entry.confidence * 0.8
                    memory_store._maybe_flush()
        except Exception:
            pass

    return "soft correction recorded (confidence reduced)"


def _process_stale(event, graph, user_id, content_hash, now,
                   cache_store=None):
    """Stale: mark for re-fetch without penalty (data was correct at time)."""
    stale_id = f"stale:{content_hash}"

    graph.add_node(stale_id, node_type="correction", properties={
        "original_claim": event.target_content[:200],
        "reason": "Reported stale by user",
        "severity": "stale",
        "timestamp": now,
    })

    graph.add_edge(user_id, "reported_stale", stale_id,
                   confidence=0.75, source="user_feedback")

    # Invalidate matching cache entries so next question re-fetches
    if cache_store is not None and event.target_content:
        try:
            removed = cache_store.invalidate_matching(event.target_content)
            if removed:
                logger.info("Stale report: invalidated %d cache entries", removed)
        except Exception:
            pass

    # Invalidate matching knowledge nodes in the graph (mark stale)
    _invalidate_tree_knowledge(graph, event.target_content)

    return "marked stale (will re-fetch)"


def _process_user_provides(event, graph, user_id, content_hash, now):
    """User provides correct value: create knowledge node with user's data."""
    know_id = f"userprov:{content_hash}"

    graph.add_node(know_id, node_type="knowledge", properties={
        "content": event.provided_value[:500] or event.user_quote[:500],
        "source": "user_provided",
        "original_claim": event.target_content[:200],
        "user_quote": event.user_quote[:200],
        "timestamp": now,
        "confidence": 0.85,
    }, confidence=0.85, source="user_provided")

    graph.add_edge(user_id, "provided", know_id,
                   confidence=0.85, source="user_feedback")

    # If there was a previous answer, create correction link
    if event.target_content:
        corr_id = f"corr:{content_hash}"
        graph.add_node(corr_id, node_type="correction", properties={
            "original_claim": event.target_content[:200],
            "corrected_to": event.provided_value[:200],
            "reason": event.user_quote[:200],
            "severity": "user_corrected",
            "timestamp": now,
        })
        graph.add_edge(corr_id, "corrected_by", know_id,
                       confidence=0.85, source="user_feedback")

    return f"user-provided value recorded: {event.provided_value[:50]}"


def _process_source_block(event, graph, user_id, content_hash, now):
    """Block a specific source/tool from being used."""
    source_name = event.topic or "unknown_source"
    blocked_id = f"srcblock:{hashlib.sha256(source_name.encode()).hexdigest()[:8]}"

    graph.add_node(blocked_id, node_type="blocked_fact", properties={
        "original_content": f"Source blocked: {source_name}",
        "source_name": source_name,
        "blocked_by": user_id,
        "reason": event.user_quote[:200],
        "timestamp": now,
        "block_type": "source",
    })

    graph.add_edge(user_id, "blocked_source", blocked_id,
                   confidence=1.0, source="user_feedback")

    return f"source blocked: {source_name}"


# ==============================================================================
# TREE KNOWLEDGE INVALIDATION — Mark stale knowledge nodes after corrections
# ==============================================================================

def _invalidate_tree_knowledge(graph, target_content: str):
    """Mark matching knowledge/snapshot nodes as stale after user correction.

    Finds knowledge nodes whose content matches the corrected claim and
    sets their timestamp far in the past so tree_search returns low scores
    and the tool-needed check triggers a fresh fetch.

    Args:
        graph: QORGraph instance.
        target_content: The corrected content to match against.
    """
    if not graph or not getattr(graph, 'is_open', False) or not target_content:
        return

    # Extract meaningful keywords from the corrected content
    keywords = [w.lower() for w in target_content.split()
                if len(w) > 2 and w.lower() not in _TREE_STOP_WORDS]
    if not keywords:
        return

    invalidated = 0
    for node_type in ("knowledge", "snapshot"):
        try:
            nodes = graph.list_nodes(node_type=node_type)
            for nid, data in nodes:
                props = data.get("properties", {})
                content = (props.get("content", "") + " "
                           + props.get("question", "")).lower()
                if not content.strip():
                    continue
                matches = sum(1 for kw in keywords if kw in content)
                if matches >= max(1, len(keywords) // 2):
                    # Mark as stale by setting timestamp to epoch
                    props["timestamp"] = "2000-01-01T00:00:00+00:00"
                    props["invalidated_by"] = "user_correction"
                    try:
                        graph.add_node(nid, node_type=node_type,
                                       properties=props)
                        invalidated += 1
                    except Exception:
                        pass
        except Exception:
            pass

    if invalidated:
        logger.info("Invalidated %d tree knowledge nodes after correction",
                    invalidated)


# ==============================================================================
# CORRECTION CASCADE — Propagate corrections to dependent nodes (Phase D.15)
# ==============================================================================

def cascade_correction(graph, corrected_node_id: str,
                       max_depth: int = 2,
                       decay_factor: float = 0.9) -> Dict[str, Any]:
    """
    Propagate confidence reduction to nodes dependent on a corrected node.

    PRD Section 20: When a correction happens:
    - Summary nodes: confidence *= 0.7, flagged NEEDS_RECOMPUTE
    - Event nodes: edge weight *= 0.5
    - General dependents: confidence *= 0.9
    - Limit to max_depth hops to prevent over-damping

    Args:
        graph: QORGraph instance
        corrected_node_id: ID of the corrected node
        max_depth: maximum cascade hops (default 2)
        decay_factor: per-hop confidence multiplier (default 0.9)

    Returns:
        {"affected_count": int, "affected_nodes": list, "depth_reached": int}
    """
    if graph is None or not hasattr(graph, 'is_open') or not graph.is_open:
        return {"affected_count": 0, "affected_nodes": [], "depth_reached": 0}

    affected = []
    visited = {corrected_node_id}
    current_layer = [corrected_node_id]
    depth_reached = 0

    for depth in range(max_depth):
        next_layer = []
        for node_id in current_layer:
            # Find outgoing edges
            try:
                edges = graph.get_edges(node_id)
            except Exception:
                edges = []

            for edge in edges:
                target_id = edge.get("object", "")
                if not target_id or target_id in visited:
                    continue
                visited.add(target_id)

                target = graph.get_node(target_id)
                if not target:
                    continue

                props = target.get("properties", {})
                ntype = props.get("type", "").lower()

                # Apply type-specific confidence reduction
                current_conf = props.get("confidence", 0.5)
                if "summary" in ntype:
                    new_conf = current_conf * 0.7
                    props["flags"] = props.get("flags", 0) | 0x10  # STALE
                elif "event" in ntype:
                    # Weaken edges, not node confidence
                    try:
                        graph.add_edge(
                            node_id, edge.get("predicate", "related_to"),
                            target_id,
                            confidence=edge.get("confidence", 0.5) * 0.5,
                            source="cascade_correction",
                        )
                    except Exception:
                        pass
                    new_conf = current_conf * 0.95
                else:
                    new_conf = current_conf * decay_factor

                props["confidence"] = new_conf
                props["cascade_from"] = corrected_node_id
                props["cascade_timestamp"] = datetime.now().isoformat()

                try:
                    graph.add_node(target_id, node_type=ntype or "knowledge",
                                   properties=props,
                                   confidence=new_conf)
                except Exception:
                    pass

                affected.append(target_id)
                next_layer.append(target_id)

        if next_layer:
            depth_reached = depth + 1
        current_layer = next_layer

    # Negative Hebbian on edges leading to corrected node
    try:
        if hasattr(graph, 'hebbian_update'):
            # Find incoming edges to corrected node
            data = graph.get_node(corrected_node_id)
            if data:
                # Apply negative reinforcement
                for affected_id in affected[:10]:  # Limit to prevent explosion
                    graph.hebbian_update(affected_id, corrected_node_id,
                                        reward=-1.0)
    except Exception:
        pass

    logger.info("Cascade from %s: %d nodes affected, depth %d",
                corrected_node_id, len(affected), depth_reached)

    return {
        "affected_count": len(affected),
        "affected_nodes": affected,
        "depth_reached": depth_reached,
    }


def _compute_content_diff(old_content: str, new_content: str) -> Dict[str, Any]:
    """
    Compare old vs new content and compute divergence metrics (Step 4).

    Returns:
        Dict with "divergence" (0.0-1.0), "diff_summary" (human-readable),
        "old_numbers", "new_numbers", "numeric_pct_change" (if applicable).
    """
    import re

    result = {
        "divergence": 0.0,
        "diff_summary": "",
        "old_numbers": [],
        "new_numbers": [],
        "numeric_pct_change": None,
    }

    if not old_content and not new_content:
        return result

    if not old_content or not new_content:
        result["divergence"] = 1.0
        result["diff_summary"] = "One side empty"
        return result

    # Numeric comparison — extract numbers
    old_nums = re.findall(r'[\d,]+\.?\d*', old_content)
    new_nums = re.findall(r'[\d,]+\.?\d*', new_content)
    result["old_numbers"] = old_nums[:5]
    result["new_numbers"] = new_nums[:5]

    if old_nums and new_nums:
        try:
            old_val = float(old_nums[0].replace(",", ""))
            new_val = float(new_nums[0].replace(",", ""))
            if abs(old_val) > 0.001:
                pct_change = abs(new_val - old_val) / abs(old_val) * 100
                result["numeric_pct_change"] = round(pct_change, 2)
                # Map pct_change to divergence: 0% -> 0.0, 100%+ -> 1.0
                result["divergence"] = min(pct_change / 100.0, 1.0)
                result["diff_summary"] = (
                    f"Numeric change: {old_val} -> {new_val} "
                    f"({pct_change:+.1f}%)"
                )
                return result
        except (ValueError, IndexError):
            pass

    # Text similarity — word overlap (Jaccard)
    old_words = set(old_content.lower().split())
    new_words = set(new_content.lower().split())
    union = old_words | new_words
    intersection = old_words & new_words
    if union:
        jaccard = len(intersection) / len(union)
        result["divergence"] = round(1.0 - jaccard, 3)
    else:
        result["divergence"] = 1.0

    # Build human-readable diff summary
    added = new_words - old_words
    removed = old_words - new_words
    parts = []
    if removed:
        parts.append(f"removed: {', '.join(list(removed)[:5])}")
    if added:
        parts.append(f"added: {', '.join(list(added)[:5])}")
    result["diff_summary"] = "; ".join(parts) if parts else "minor text changes"

    return result


def cascade_correction_full(graph, corrected_node_id: str,
                            old_content: str, new_content: str,
                            max_depth: int = 2,
                            decay_factor: float = 0.9) -> Dict[str, Any]:
    """
    Complete correction cascade: Steps 4 and 5 of the correction engine.

    Step 4: Compare old vs new content — calculate diff/divergence between
    the correction's new content and the original node's content. Store the
    diff as a property on the corrected node.

    Step 5: Cascade to dependents — find all INCOMING edges (nodes that
    depend on the corrected node), and for each dependent:
      a. Reduce the dependent edge's confidence by 20%
      b. Add a NEEDS_RECHECK flag (0x100) to the dependent node
      c. Log which nodes were cascade-affected

    Args:
        graph: QORGraph instance.
        corrected_node_id: ID of the node that was corrected.
        old_content: Previous content of the node.
        new_content: New/corrected content of the node.
        max_depth: Maximum cascade hops (default 2).
        decay_factor: Per-hop confidence multiplier for general dependents
                      (default 0.9).

    Returns:
        {
            "diff": dict (from _compute_content_diff),
            "affected_count": int,
            "affected_nodes": list of node IDs,
            "depth_reached": int,
            "cascade_log": list of dicts describing each affected node,
        }
    """
    result = {
        "diff": {},
        "affected_count": 0,
        "affected_nodes": [],
        "depth_reached": 0,
        "cascade_log": [],
    }

    if graph is None or not hasattr(graph, 'is_open') or not graph.is_open:
        return result

    # --- Step 4: Compare old vs new content ---
    diff = _compute_content_diff(old_content, new_content)
    result["diff"] = diff

    # Store the diff as a property on the corrected node
    try:
        node_data = graph.get_node(corrected_node_id)
        if node_data:
            props = node_data.get("properties", {})
            props["correction_diff"] = diff.get("diff_summary", "")
            props["correction_divergence"] = diff.get("divergence", 0.0)
            if diff.get("numeric_pct_change") is not None:
                props["correction_pct_change"] = diff["numeric_pct_change"]
            props["correction_old_content"] = old_content[:200]
            props["correction_new_content"] = new_content[:200]
            props["correction_timestamp"] = datetime.now().isoformat()
            ntype = props.get("type", "knowledge")
            graph.add_node(corrected_node_id, node_type=ntype,
                           properties=props)
    except Exception as e:
        logger.debug("Failed to store diff on corrected node: %s", e)

    # --- Step 5: Cascade to dependents (incoming edges) ---
    # NEEDS_RECHECK flag = 0x100 (from graph.py NodeFlags)
    NEEDS_RECHECK = 0x100

    affected = []
    cascade_log = []
    visited = {corrected_node_id}
    current_layer = [corrected_node_id]
    depth_reached = 0

    for depth in range(max_depth):
        next_layer = []
        for node_id in current_layer:
            # Find INCOMING edges — nodes that point TO this node
            try:
                incoming_edges = graph.get_edges(node_id, direction="in")
            except Exception:
                incoming_edges = []

            for edge in incoming_edges:
                # Incoming edge: subject -> predicate -> node_id
                dependent_id = edge.get("subject", "")
                if not dependent_id or dependent_id in visited:
                    continue
                visited.add(dependent_id)

                dependent_node = graph.get_node(dependent_id)
                if not dependent_node:
                    continue

                # (a) Reduce the dependent edge's confidence by 20%
                old_edge_conf = edge.get("confidence", 1.0)
                new_edge_conf = old_edge_conf * 0.8  # 20% reduction
                predicate = edge.get("predicate", "related_to")
                try:
                    graph.add_edge(
                        dependent_id, predicate, node_id,
                        confidence=new_edge_conf,
                        source="cascade_correction",
                    )
                except Exception:
                    pass

                # (b) Add NEEDS_RECHECK flag (0x100) to the dependent node
                dep_props = dependent_node.get("properties", {})
                old_flags = dep_props.get("flags", 0)
                dep_props["flags"] = old_flags | NEEDS_RECHECK
                dep_props["recheck_reason"] = (
                    f"Dependency {corrected_node_id} was corrected"
                )
                dep_props["recheck_timestamp"] = datetime.now().isoformat()

                # Also apply per-hop confidence decay on the node itself
                dep_conf = dep_props.get("confidence", 0.5)
                dep_props["confidence"] = dep_conf * decay_factor
                dep_props["cascade_from"] = corrected_node_id

                dep_ntype = dep_props.get("type", "knowledge")
                try:
                    graph.add_node(dependent_id, node_type=dep_ntype,
                                   properties=dep_props)
                except Exception:
                    pass

                # (c) Log this cascade effect
                log_entry = {
                    "node_id": dependent_id,
                    "depth": depth + 1,
                    "edge_confidence_old": round(old_edge_conf, 3),
                    "edge_confidence_new": round(new_edge_conf, 3),
                    "flags_before": old_flags,
                    "flags_after": dep_props["flags"],
                    "node_confidence_old": round(dep_conf, 3),
                    "node_confidence_new": round(dep_conf * decay_factor, 3),
                }
                cascade_log.append(log_entry)
                affected.append(dependent_id)
                next_layer.append(dependent_id)

        if next_layer:
            depth_reached = depth + 1
        current_layer = next_layer

    result["affected_count"] = len(affected)
    result["affected_nodes"] = affected
    result["depth_reached"] = depth_reached
    result["cascade_log"] = cascade_log

    logger.info(
        "Correction cascade (full) from %s: divergence=%.2f, "
        "%d dependents affected, depth %d",
        corrected_node_id, diff.get("divergence", 0.0),
        len(affected), depth_reached,
    )

    return result


# ==============================================================================
# TRADE LEARNER — Analyzes closed trades -> creates lesson nodes
# ==============================================================================

class TradeLearner:
    """Analyzes closed trades and creates pattern/lesson nodes in the graph."""

    @staticmethod
    def analyze_trades(trade_store, graph, user_id: str,
                       min_trades: int = 5) -> List[str]:
        """Analyze closed trades and create lesson nodes.

        Args:
            trade_store: TradeStore or FuturesTradeStore instance.
            graph: QORGraph instance.
            user_id: User node ID.
            min_trades: Minimum total closed trades before analysis.

        Returns:
            List of new lesson descriptions created.
        """
        if graph is None or not hasattr(graph, 'is_open') or not graph.is_open:
            return []
        if trade_store is None or not hasattr(trade_store, 'trades'):
            return []

        # Get all closed trades
        closed = [t for t in trade_store.trades.values()
                  if t.get("status") != "open"]
        if len(closed) < min_trades:
            return []

        # Group by symbol
        by_symbol = {}
        for t in closed:
            sym = t.get("symbol", "?")
            by_symbol.setdefault(sym, []).append(t)

        now = datetime.now().isoformat()
        new_lessons = []

        for symbol, trades in by_symbol.items():
            if len(trades) < 3:
                continue

            # Analyze SHORT trades
            shorts = [t for t in trades if t.get("direction", t.get("side", "")) == "SHORT"]
            if len(shorts) >= 3:
                short_wins = sum(1 for t in shorts if t.get("pnl", 0) > 0)
                short_wr = short_wins / len(shorts) * 100
                if short_wr < 40:
                    lesson = _create_trade_lesson(
                        graph, symbol, "losing_short",
                        f"Avoid SHORT on {symbol} - historically {short_wr:.0f}% win rate "
                        f"({short_wins}/{len(shorts)} wins)",
                        trades=shorts, now=now,
                    )
                    if lesson:
                        new_lessons.append(lesson)

            # Analyze winning strategies
            by_strategy = {}
            for t in trades:
                strat = t.get("strategy", "unknown")
                by_strategy.setdefault(strat, []).append(t)

            for strat, strat_trades in by_strategy.items():
                if len(strat_trades) < 3 or strat == "unknown":
                    continue
                wins = sum(1 for t in strat_trades if t.get("pnl", 0) > 0)
                wr = wins / len(strat_trades) * 100
                if wr > 60:
                    lesson = _create_trade_lesson(
                        graph, symbol, f"winning_{strat}",
                        f"Strategy '{strat}' works well on {symbol} - "
                        f"{wr:.0f}% win rate ({wins}/{len(strat_trades)})",
                        trades=strat_trades, now=now,
                    )
                    if lesson:
                        new_lessons.append(lesson)

            # Detect tight stop losses
            sl_trades = [t for t in trades
                         if t.get("exit_reason", "") == "stop_loss"
                         and t.get("pnl_pct", 0) < 0]
            if len(sl_trades) >= 3:
                avg_loss = sum(t.get("pnl_pct", 0) for t in sl_trades) / len(sl_trades)
                if avg_loss > -1.0:  # Losses less than 1% = too tight
                    lesson = _create_trade_lesson(
                        graph, symbol, "sl_too_tight",
                        f"Stop loss may be too tight on {symbol} - "
                        f"avg SL loss {avg_loss:.2f}%, consider wider stops",
                        trades=sl_trades, now=now,
                    )
                    if lesson:
                        new_lessons.append(lesson)

            # DCA effectiveness
            dca_trades = [t for t in trades if t.get("dca_count", 0) > 0]
            non_dca = [t for t in trades if t.get("dca_count", 0) == 0]
            if len(dca_trades) >= 3 and len(non_dca) >= 3:
                dca_wr = sum(1 for t in dca_trades if t.get("pnl", 0) > 0) / len(dca_trades)
                non_dca_wr = sum(1 for t in non_dca if t.get("pnl", 0) > 0) / len(non_dca)
                if dca_wr > non_dca_wr + 0.1:
                    lesson = _create_trade_lesson(
                        graph, symbol, "dca_effective",
                        f"DCA improves results on {symbol} - "
                        f"DCA win rate {dca_wr:.0%} vs non-DCA {non_dca_wr:.0%}",
                        trades=dca_trades, now=now,
                    )
                    if lesson:
                        new_lessons.append(lesson)

        return new_lessons

    @staticmethod
    def get_active_lessons(graph) -> List[str]:
        """Get all lesson content strings with confidence > 0.5.

        Args:
            graph: QORGraph instance.

        Returns:
            List of lesson content strings.
        """
        if graph is None or not hasattr(graph, 'is_open') or not graph.is_open:
            return []

        lessons = []
        try:
            lesson_nodes = graph.list_nodes(node_type="lesson")
            for nid, data in lesson_nodes:
                props = data.get("properties", {})
                content = props.get("content", "")
                if content:
                    # Check edge confidence
                    edges = graph.get_edges(nid, direction="in")
                    max_conf = max((e.get("confidence", 0) for e in edges), default=0.7)
                    if max_conf > 0.5:
                        lessons.append(content)
        except Exception:
            pass
        return lessons


def _create_trade_lesson(graph, symbol, pattern_name, lesson_content,
                         trades, now):
    """Create trade_pattern + lesson nodes in graph."""
    tp_id = f"tp:{symbol.lower()}:{pattern_name}"
    lesson_hash = hashlib.sha256(lesson_content.encode()).hexdigest()[:8]
    lesson_id = f"lesson:{lesson_hash}"

    # Check if this exact lesson already exists
    existing = graph.get_node(lesson_id)
    if existing:
        # Update evidence count
        props = existing.get("properties", {})
        count = props.get("evidence_count", 1) + 1
        graph.add_node(lesson_id, node_type="lesson", properties={
            "evidence_count": count,
            "last_confirmed": now,
        })
        return None  # Not a new lesson

    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    losses = len(trades) - wins
    wr = wins / len(trades) * 100 if trades else 0
    avg_pnl = sum(t.get("pnl_pct", 0) for t in trades) / len(trades) if trades else 0

    # Create trade_pattern node
    graph.add_node(tp_id, node_type="trade_pattern", properties={
        "symbol": symbol,
        "pattern_name": pattern_name,
        "win_count": wins,
        "loss_count": losses,
        "win_rate": round(wr, 1),
        "avg_pnl": round(avg_pnl, 2),
        "sample_size": len(trades),
        "last_updated": now,
    })

    # Create lesson node
    graph.add_node(lesson_id, node_type="lesson", properties={
        "lesson_type": "trade",
        "content": lesson_content,
        "evidence_count": 1,
        "first_seen": now,
        "last_confirmed": now,
    })

    # Edge: trade_pattern → resulted_in_lesson → lesson
    graph.add_edge(tp_id, "resulted_in_lesson", lesson_id,
                   confidence=min(0.7 + len(trades) * 0.03, 1.0),
                   source="trade_analysis")

    return lesson_content


# ==============================================================================
# ANSWER FILTER — Checks graph before generating answer
# ==============================================================================

@dataclass
class ContextAdjustments:
    """Adjustments to apply before generating an answer."""
    blocked_patterns: List[str] = field(default_factory=list)
    trade_lessons: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
    boost_topics: List[str] = field(default_factory=list)
    suppress_topics: List[str] = field(default_factory=list)


class AnswerFilter:
    """Checks the knowledge graph for adjustments before answer generation."""

    @staticmethod
    def get_adjustments(question: str, graph, user_id: str) -> ContextAdjustments:
        """Query graph for blocked content, lessons, corrections, preferences.

        Args:
            question: The user's question.
            graph: QORGraph instance.
            user_id: User node ID.

        Returns:
            ContextAdjustments with all applicable adjustments.
        """
        adj = ContextAdjustments()

        if graph is None or not hasattr(graph, 'is_open') or not graph.is_open:
            return adj

        try:
            # Blocked facts
            blocked_nodes = graph.list_nodes(node_type="blocked_fact")
            for nid, data in blocked_nodes:
                props = data.get("properties", {})
                content = props.get("original_content", "")
                if content:
                    adj.blocked_patterns.append(content)

            # Recent corrections
            corr_nodes = graph.list_nodes(node_type="correction")
            for nid, data in corr_nodes:
                props = data.get("properties", {})
                claim = props.get("original_claim", "")
                reason = props.get("reason", "")
                if claim:
                    adj.corrections.append(f"Previously corrected: {claim}")

            # Trade lessons
            adj.trade_lessons = TradeLearner.get_active_lessons(graph)

            # Preferences from user
            pref_nodes = graph.list_nodes(node_type="preference")
            for nid, data in pref_nodes:
                props = data.get("properties", {})
                topic = props.get("topic", "")
                polarity = props.get("polarity", "")
                if topic:
                    if polarity == "like":
                        adj.boost_topics.append(topic)
                    elif polarity == "dislike":
                        adj.suppress_topics.append(topic)

        except Exception as e:
            logger.debug(f"AnswerFilter error: {e}")

        return adj


# ==============================================================================
# SYSTEM ID — Machine-level unique identity (generated once, stored forever)
# ==============================================================================

def _get_machine_fingerprint() -> str:
    """Get a deterministic fingerprint from THIS machine's hardware.

    Combines:
      1. Motherboard UUID (BIOS — unique per physical machine)
      2. MAC address (network card — unique per NIC)
      3. Hostname (OS-level machine name)

    These don't change unless you swap hardware. Same machine = same
    fingerprint every time. Different machine = different fingerprint.

    Returns:
        SHA-256 hash of combined hardware identifiers (first 16 chars).
    """
    import platform
    import subprocess

    parts = []

    # 1. Motherboard UUID (most reliable — burned into BIOS)
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["wmic", "csproduct", "get", "UUID"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if line and line != "UUID":
                    parts.append(f"bios:{line}")
                    break
        else:
            # Linux/Mac
            for path in [
                "/sys/class/dmi/id/product_uuid",
                "/etc/machine-id",
            ]:
                try:
                    with open(path) as f:
                        val = f.read().strip()
                    if val:
                        parts.append(f"bios:{val}")
                        break
                except Exception:
                    continue
    except Exception:
        pass

    # 2. MAC address (network card hardware)
    try:
        import uuid as _uuid
        mac = _uuid.getnode()
        # getnode returns a random if no MAC found — check bit 0
        if not (mac & (1 << 40)):  # Bit 40 = 0 means real MAC
            parts.append(f"mac:{mac:012x}")
    except Exception:
        pass

    # 3. Hostname
    try:
        parts.append(f"host:{platform.node()}")
    except Exception:
        pass

    if not parts:
        # Absolute last resort — should never happen
        import uuid as _uuid
        parts.append(f"rand:{_uuid.uuid4().hex}")

    raw = "|".join(sorted(parts))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_or_create_system_id(data_dir: str) -> str:
    """Get the system ID for this machine.

    Built from REAL hardware identifiers (motherboard UUID, MAC address,
    hostname). Same machine always produces the same ID. Different machine
    produces a different ID. No randomness.

    The ID is cached in qor-data/.system_id so we don't shell out to
    wmic on every startup. If the file is deleted, it regenerates the
    SAME ID from hardware (deterministic).

    Args:
        data_dir: Path to qor-data directory.

    Returns:
        System ID string like "sys-a1b2c3d4e5f678".
    """
    import os

    id_path = os.path.join(data_dir, ".system_id")

    # Read cached (avoids wmic call on every start)
    if os.path.exists(id_path):
        try:
            with open(id_path, "r") as f:
                sid = f.read().strip()
            if sid:
                return sid
        except Exception:
            pass

    # Generate from hardware — deterministic, same machine = same ID
    fingerprint = _get_machine_fingerprint()
    sid = f"sys-{fingerprint}"

    # Cache it
    try:
        os.makedirs(data_dir, exist_ok=True)
        with open(id_path, "w") as f:
            f.write(sid)
    except Exception:
        pass
    return sid


# ==============================================================================
# PROFILE — User profile stored in the tree (user node properties)
# ==============================================================================

def save_profile_to_tree(graph, user_id: str, profile: dict):
    """Save profile into the user node in the tree.

    ONLY stores: name, interests, cautions, preferences.
    NEVER stores: API keys, trading credentials, secrets.
    Credentials stay in profile.json (encrypted by .keyfile).

    Args:
        graph: QORGraph instance.
        user_id: User node ID.
        profile: Profile dict.
    """
    if graph is None or not getattr(graph, 'is_open', False):
        return

    import json

    props = {
        "name": profile.get("user_name", "unknown"),
        "interests_json": json.dumps(profile.get("interests", {})),
        "cautions_json": json.dumps(profile.get("cautions", [])),
        "preferred_detail_level": profile.get("preferred_detail_level", "detailed"),
        "updated_at": datetime.now().isoformat(),
    }
    # NO credentials — they stay in profile.json, encrypted by .keyfile

    try:
        existing = graph.get_node(user_id)
        if existing:
            graph.update_node(user_id, props)
        else:
            props["created_at"] = datetime.now().isoformat()
            graph.add_node(user_id, node_type="user", properties=props)
    except Exception as e:
        logger.debug(f"save_profile_to_tree error: {e}")


def get_profile_from_tree(graph, user_id: str) -> Optional[dict]:
    """Read profile from the user node in the tree.

    Returns name, interests, cautions, preferences.
    NEVER returns credentials — those come from profile.json.

    Args:
        graph: QORGraph instance.
        user_id: User node ID.

    Returns:
        Profile dict or None.
    """
    if graph is None or not getattr(graph, 'is_open', False):
        return None

    import json

    try:
        node = graph.get_node(user_id)
        if node is None:
            return None

        props = node.get("properties", {})
        profile = {
            "user_name": props.get("name", ""),
            "preferred_detail_level": props.get("preferred_detail_level", "detailed"),
        }
        try:
            profile["interests"] = json.loads(props.get("interests_json", "{}"))
        except Exception:
            profile["interests"] = {}
        try:
            profile["cautions"] = json.loads(props.get("cautions_json", "[]"))
        except Exception:
            profile["cautions"] = []
        # NO credentials here — they come from profile.json
        return profile
    except Exception as e:
        logger.debug(f"get_profile_from_tree error: {e}")
        return None


def update_profile_field(graph, user_id: str, field: str, value):
    """Update a single profile field in the tree.

    Args:
        graph: QORGraph instance.
        user_id: User node ID.
        field: Profile field name (e.g. "interests", "user_name").
        value: New value.
    """
    if graph is None or not getattr(graph, 'is_open', False):
        return

    import json

    try:
        # Map profile field names to node property names
        # NEVER store credentials in the tree
        if field == "trading_credentials":
            return  # Credentials stay in profile.json only
        elif field == "user_name":
            graph.update_node(user_id, {"name": value})
        elif field == "interests":
            graph.update_node(user_id, {"interests_json": json.dumps(value)})
        elif field == "cautions":
            graph.update_node(user_id, {"cautions_json": json.dumps(value)})
        elif field == "preferred_detail_level":
            graph.update_node(user_id, {"preferred_detail_level": value})
        else:
            graph.update_node(user_id, {field: json.dumps(value)
                                         if isinstance(value, (dict, list))
                                         else value})
        graph.update_node(user_id, {"updated_at": datetime.now().isoformat()})
    except Exception as e:
        logger.debug(f"update_profile_field error: {e}")


# ==============================================================================
# TREE SEARCH — Single unified search (AI calls ONLY this, not individual stores)
# ==============================================================================

def tree_search(query: str, graph, user_id: str,
                verbose: bool = False,
                ngre_brain=None) -> List[Dict[str, Any]]:
    """Search the Knowledge Tree — THE single database.

    The AI calls ONLY this function. No memory, no cache, no RAG, no
    historical parquet. Everything lives in the tree (graph.rocksdb).
    If the tree has no data → returns empty → tool gets called → tool
    saves to tree → next time tree has data.

    Search strategy:
      1. NGRE embedding search — Mamba 768-dim vector similarity (best)
      2. Semantic query — structured edge facts (graph traversal)
      3. Knowledge nodes — stored tool results, keyword matched
      4. Lesson/pattern nodes — trade experience
      5. Historical events — archived significant events

    Args:
        query: User's question.
        graph: QORGraph instance.
        user_id: User node ID.
        verbose: Print debug output.
        ngre_brain: NGREBrain instance for embedding search (optional).

    Returns:
        List of {"content": str, "source": str, "score": float,
                 "timestamp": str}
        sorted by relevance score, up to 8 results.
    """
    results = []

    if graph is None or not getattr(graph, 'is_open', False):
        return []  # No tree → empty → tool will be called

    # --- 0. NGRE embedding search — vector similarity via Mamba ---
    # This is the primary search: compute query embedding, find nearest
    # nodes by cosine similarity in the graph's embedding index.
    if ngre_brain is not None:
        try:
            query_emb = ngre_brain.compute_embedding(query)
            if query_emb is not None:
                emb_results = graph.search_by_embedding(
                    query_emb, k=8)
                for nid, distance in emb_results:
                    data = graph.get_node(nid)
                    if data:
                        props = data.get("properties", {})
                        content = props.get("content", "")
                        if content:
                            # Convert distance to similarity score (0-1)
                            sim = max(0.0, 1.0 - float(distance))
                            results.append({
                                "content": content,
                                "source": f"graph:embedding:{data.get('type', '')}",
                                "score": sim,
                                "timestamp": props.get("timestamp", ""),
                            })
                if verbose and results:
                    print(f"  ├─ NGRE embedding search: "
                          f"{len(results)} nodes (cosine similarity)")
        except Exception:
            pass

    # --- 1. Semantic query — structured edge facts (BFS traversal) ---
    try:
        from .confidence import _format_graph_facts
        sem = graph.semantic_query(query)
        if sem.get("edge_count", 0) > 0:
            facts = _format_graph_facts(sem.get("path", []))
            if facts:
                results.append({
                    "content": facts,
                    "source": "graph",
                    "score": max(sem.get("confidence", 0.7), 0.7),
                    "timestamp": "",
                })
    except Exception:
        pass

    # --- 2. Knowledge nodes — keyword search on stored content ---
    keywords = [w.lower() for w in query.split()
                if len(w) > 2 and w.lower() not in _TREE_STOP_WORDS]

    now_ts = datetime.now()

    # Detect temporal queries: "yesterday", "last week", specific dates
    # If temporal → we WANT old data, so don't penalize staleness
    _temporal_words = {
        "yesterday", "last", "ago", "previous", "earlier", "before",
        "week", "month", "history", "historical", "past",
    }
    query_lower = query.lower()
    is_temporal = any(tw in query_lower for tw in _temporal_words)

    # Resolve temporal date references for matching
    # "yesterday" → "2026-02-13", "last week" → date range
    target_date = None
    if "yesterday" in query_lower:
        from datetime import timedelta
        target_date = (now_ts - timedelta(days=1)).strftime("%Y-%m-%d")

    if keywords:
        # Search these node types — all have "content" in properties
        _searchable = [
            ("knowledge", 0.85),       # Tool results stored as nodes
            ("snapshot", 0.9),         # Ingested time-series (prices, weather, TA)
            ("event", 0.8),            # Ingested news/articles
            ("historical_event", 0.8), # Session open/close + significant events
            ("lesson", 0.9),           # Trade lessons (high value)
            ("trade_pattern", 0.8),    # Trade patterns
            ("topic", 0.6),            # Topic nodes (name-based)
        ]
        for node_type, type_weight in _searchable:
            try:
                nodes = graph.list_nodes(node_type=node_type)
                for nid, data in nodes:
                    props = data.get("properties", {})
                    content = props.get("content", "")
                    name = props.get("name", "")
                    question_stored = props.get("question", "")
                    query_stored = props.get("query", "")
                    entity_stored = props.get("entity", "")
                    date_field = props.get("date", "")
                    searchable = (f"{content} {name} {question_stored} "
                                  f"{query_stored} {entity_stored}").lower()
                    if not searchable.strip():
                        continue
                    matches = sum(1 for kw in keywords if kw in searchable)

                    # Require at least one SPECIFIC keyword match (not just
                    # generic words like "price", "today", "analysis").
                    # Without this, a BTC node matching "price" gets returned
                    # for a gold query, causing wrong-asset answers.
                    _generic_words = {"price", "today", "analysis", "current",
                                      "now", "latest", "update", "market",
                                      "trading", "value", "worth", "cost",
                                      "rate", "quote", "data", "info",
                                      "chart", "report", "summary", "news"}
                    specific_matches = sum(
                        1 for kw in keywords
                        if kw in searchable and kw not in _generic_words)
                    if specific_matches == 0 and len(keywords) > 1:
                        continue  # Only generic matches — skip this node

                    # For temporal queries, also match by date field
                    if target_date and date_field == target_date:
                        matches += 2  # Strong boost for exact date match

                    if matches > 0:
                        score = (matches / max(len(keywords), 1)) * type_weight

                        ts_str = props.get("timestamp", "")

                        # Freshness scoring — depends on query intent
                        if node_type == "historical_event":
                            # Historical events: NEVER penalize for age.
                            # These are permanent records (session open/close).
                            # Boost if the date matches the temporal query.
                            if target_date and date_field == target_date:
                                score = max(score, 0.95)
                            elif is_temporal:
                                score = max(score, 0.8)
                        elif is_temporal:
                            # Temporal query but non-historical node: no
                            # freshness penalty (user wants old data)
                            pass
                        elif ts_str:
                            # Non-temporal query: fresh data ranks higher
                            try:
                                entry_time = datetime.fromisoformat(ts_str)
                                age_min = (now_ts - entry_time).total_seconds() / 60
                                if node_type == "snapshot":
                                    # Snapshots: heavily reward freshness
                                    if age_min <= 10:
                                        score = max(score, 0.95)
                                    elif age_min <= 30:
                                        score = max(score, 0.85)
                                    elif age_min <= 60:
                                        score = max(score, 0.7)
                                    elif age_min > 360:
                                        score *= 0.5
                                else:
                                    if age_min <= 10:
                                        score += 0.1
                                    elif age_min > 360:
                                        score -= 0.2
                            except Exception:
                                pass

                        results.append({
                            "content": content or name or nid,
                            "source": f"graph:{node_type}",
                            "score": score,
                            "timestamp": ts_str,
                        })
            except Exception:
                pass

    # --- 3. Sort by score, deduplicate ---
    results.sort(key=lambda x: x["score"], reverse=True)
    seen = set()
    deduped = []
    for r in results:
        ck = r["content"][:200]
        if ck not in seen:
            seen.add(ck)
            deduped.append(r)
    results = deduped

    if verbose and results:
        src_set = set(r["source"] for r in results[:8])
        print(f"  ├─ Tree: {len(results[:8])} results from {src_set}")

    return results[:8]


# ==============================================================================
# TREE STATUS — Display function for "tree" command
# ==============================================================================

def tree_status(graph, user_id: str):
    """Display knowledge tree statistics.

    Args:
        graph: QORGraph instance.
        user_id: User node ID.
    """
    if graph is None or not hasattr(graph, 'is_open') or not graph.is_open:
        print("  Knowledge graph not available")
        return

    print("\n  Knowledge Tree:")

    try:
        corrections = len(graph.list_nodes(node_type="correction"))
        blocked = len(graph.list_nodes(node_type="blocked_fact"))
        preferences = len(graph.list_nodes(node_type="preference"))
        patterns = len(graph.list_nodes(node_type="trade_pattern"))
        lessons = len(graph.list_nodes(node_type="lesson"))
        users = len(graph.list_nodes(node_type="user"))
        historical = len(graph.list_nodes(node_type="historical_event"))
        topics = len(graph.list_nodes(node_type="topic"))
        knowledge = len(graph.list_nodes(node_type="knowledge"))

        total_special = (corrections + blocked + preferences + patterns
                         + lessons + historical + topics + knowledge)

        print(f"    User nodes:        {users}")
        print(f"    Knowledge nodes:   {knowledge}")
        print(f"    Corrections:       {corrections}")
        print(f"    Blocked facts:     {blocked}")
        print(f"    Preferences:       {preferences}")
        print(f"    Trade patterns:    {patterns}")
        print(f"    Lessons learned:   {lessons}")
        print(f"    Historical events: {historical}")
        print(f"    Topics:            {topics}")
        print(f"    Total tree nodes:  {total_special}")

        # Show active lessons
        active_lessons = TradeLearner.get_active_lessons(graph)
        if active_lessons:
            print(f"\n    Active lessons:")
            for lesson in active_lessons[:5]:
                print(f"      - {lesson[:80]}")

        # Show preferences
        pref_nodes = graph.list_nodes(node_type="preference")
        if pref_nodes:
            likes = []
            dislikes = []
            for nid, data in pref_nodes:
                props = data.get("properties", {})
                topic = props.get("topic", "?")
                if props.get("polarity") == "like":
                    likes.append(topic)
                else:
                    dislikes.append(topic)
            if likes:
                print(f"\n    Likes: {', '.join(likes)}")
            if dislikes:
                print(f"    Dislikes: {', '.join(dislikes)}")

    except Exception as e:
        print(f"    Error reading tree: {e}")


# ==============================================================================
# BOOTSTRAP — Seed the tree from existing qor-data stores
# ==============================================================================

def bootstrap_tree(graph, user_id: str, profile: dict = None,
                   chat_store=None, trade_store=None,
                   futures_store=None, memory_store=None):
    """One-time bootstrap: seed Knowledge Tree from existing qor-data.

    Reads existing stores and creates tree nodes/edges:
    1. User node from profile
    2. Interest edges from profile.interests
    3. Trade lessons from futures/spot trade history
    4. Entity extraction from recent chat sessions

    Safe to run multiple times — existing nodes are merged, not duplicated.

    Args:
        graph: QORGraph instance (must be open).
        user_id: User node ID (e.g. "user:ravi").
        profile: Profile dict from profile.json.
        chat_store: ChatStore instance.
        trade_store: Spot TradeStore instance.
        futures_store: Futures FuturesTradeStore instance.
        memory_store: MemoryStore instance.

    Returns:
        Dict with counts of what was created.
    """
    if graph is None or not hasattr(graph, 'is_open') or not graph.is_open:
        print("  Graph not available — cannot bootstrap")
        return {}

    now = datetime.now().isoformat()
    result = {"user": 0, "interests": 0, "trade_lessons": 0,
              "chat_entities": 0, "chat_topics": 0}

    print(f"\n  Bootstrapping Knowledge Tree for {user_id}...")

    # 1. Create user node
    user_name = "unknown"
    if profile:
        user_name = profile.get("user_name", "unknown")
    graph.add_node(user_id, node_type="user", properties={
        "name": user_name,
        "created_at": now,
    })
    result["user"] = 1
    print(f"    User node: {user_id} ({user_name})")

    # 2. Mirror profile interests as graph edges
    if profile and "interests" in profile:
        interests = profile["interests"]
        for topic, data in interests.items():
            score = data.get("score", 0.1)
            count = data.get("count", 0)
            # Create topic entity node
            topic_id = topic.lower().replace(" ", "_")
            graph.add_node(topic_id, node_type="topic", properties={
                "name": topic,
                "interest_score": score,
                "mention_count": count,
            })
            # Edge: user → interests_in → topic
            graph.add_edge(user_id, "interests_in", topic_id,
                           confidence=min(score + 0.2, 1.0),
                           source="profile_bootstrap")
            result["interests"] += 1
        if result["interests"] > 0:
            print(f"    Interests: {result['interests']} topics linked")

    # 3. Analyze trade history for lessons
    for store_name, store in [("spot", trade_store), ("futures", futures_store)]:
        if store is None:
            continue
        try:
            lessons = TradeLearner.analyze_trades(store, graph, user_id,
                                                   min_trades=3)
            result["trade_lessons"] += len(lessons)
            if lessons:
                print(f"    Trade lessons ({store_name}): {len(lessons)} created")
                for lesson in lessons[:3]:
                    print(f"      - {lesson[:70]}")
        except Exception as e:
            print(f"    Trade analysis ({store_name}) error: {e}")

    # 4. Extract entities from chat history
    if chat_store is not None:
        try:
            from .confidence import _extract_entities_and_edges
            sessions = chat_store.list_sessions()
            # Process last 10 sessions
            session_ids = [s.get("session_id", "") for s in sessions[-10:]]
            for sid in session_ids:
                if not sid:
                    continue
                msgs = chat_store.get_history(sid, last_n=20)
                text = " ".join(getattr(m, 'content', '') for m in msgs
                                if getattr(m, 'role', '') == "assistant")
                if not text:
                    continue
                triples = _extract_entities_and_edges(text)
                for subj, pred, obj in triples[:15]:
                    graph.add_edge(subj, pred, obj,
                                   confidence=0.6, source="chat_bootstrap")
                    result["chat_entities"] += 1

                # Also extract topics using simple keyword detection
                try:
                    # Import TOPIC_MAP from __main__ — fallback if unavailable
                    from .__main__ import _detect_topics
                    user_text = " ".join(getattr(m, 'content', '') for m in msgs
                                         if getattr(m, 'role', '') == "user")
                    topics = _detect_topics(user_text)
                    for topic in topics:
                        topic_id = topic.lower().replace(" ", "_")
                        graph.add_node(topic_id, node_type="topic", properties={
                            "name": topic,
                        })
                        graph.add_edge(user_id, "interests_in", topic_id,
                                       confidence=0.4, source="chat_bootstrap")
                        result["chat_topics"] += 1
                except Exception:
                    pass

            if result["chat_entities"] > 0:
                print(f"    Chat entities: {result['chat_entities']} triples extracted")
            if result["chat_topics"] > 0:
                print(f"    Chat topics: {result['chat_topics']} interest edges")

        except Exception as e:
            print(f"    Chat extraction error: {e}")

    # 5. Import historical events into graph
    if memory_store is not None:
        try:
            # Search memory for historical entries (category="historical")
            hist_count = 0
            # Also check for a HistoricalStore if passed via memory_store's parent
            # The historical store is separate — check if it was wired
            historical_store = None
            # Try to find historical.parquet in the data dir
            import os
            data_dir = getattr(memory_store, '_data_dir', None)
            if data_dir is None:
                # Guess common data dir
                for candidate in ["qor-data", "."]:
                    hp = os.path.join(candidate, "historical", "historical.parquet")
                    if os.path.exists(hp):
                        data_dir = candidate
                        break

            if data_dir:
                hist_path = os.path.join(data_dir, "historical", "historical.parquet")
                if os.path.exists(hist_path):
                    try:
                        import pyarrow.parquet as pq
                        table = pq.read_table(hist_path)
                        for i in range(len(table)):
                            row = {col: table.column(col)[i].as_py()
                                   for col in table.column_names}
                            content = row.get("content", "")
                            source = row.get("source", "historical")
                            key = row.get("key", f"hist:{i}")
                            if not content:
                                continue
                            evt_hash = hashlib.sha256(
                                content[:200].encode()).hexdigest()[:8]
                            evt_id = f"hist:{evt_hash}"
                            graph.add_node(evt_id, node_type="historical_event",
                                           properties={
                                               "content": content[:200],
                                               "source": source,
                                               "timestamp": row.get("timestamp", now),
                                           })
                            hist_count += 1
                        if hist_count > 0:
                            print(f"    Historical events: {hist_count} archived to graph")
                    except Exception as e:
                        print(f"    Historical import error: {e}")
            result["historical_events"] = hist_count
        except Exception as e:
            print(f"    Historical bootstrap error: {e}")

    # 6. Import memory entries as knowledge nodes (static + tool results)
    if memory_store is not None:
        try:
            mem_count = 0
            # Access the internal table for bulk import
            table = getattr(memory_store, '_table', None)
            if table is not None and len(table) > 0:
                import pyarrow as pa
                for i in range(len(table)):
                    try:
                        key = table.column("key")[i].as_py()
                        content = table.column("content")[i].as_py()
                        source = table.column("source")[i].as_py()
                        category = table.column("category")[i].as_py()
                        # Skip internal/transient entries
                        if not content or not key:
                            continue
                        if key.startswith(("context:", "psych:", "kb:", "answer:")):
                            continue
                        if category == "live":
                            continue  # Live entries expire — don't import
                        # Create knowledge node
                        know_hash = hashlib.sha256(
                            content[:200].encode()).hexdigest()[:8]
                        know_id = f"know:{know_hash}"
                        graph.add_node(know_id, node_type="knowledge",
                                       properties={
                                           "content": content[:500],
                                           "source": source or "memory",
                                           "question": key[:100],
                                           "timestamp": now,
                                       })
                        mem_count += 1
                    except Exception:
                        continue
                if mem_count > 0:
                    print(f"    Memory entries: {mem_count} imported to tree")
            result["memory_entries"] = mem_count
        except Exception as e:
            print(f"    Memory bootstrap error: {e}")

    # 7. Import RAG documents (knowledge/*.txt, *.md) into tree
    try:
        import os
        data_dir = getattr(memory_store, '_data_dir', None) if memory_store else None
        if data_dir is None:
            for candidate in ["qor-data", "."]:
                kd = os.path.join(candidate, "knowledge")
                if os.path.isdir(kd):
                    data_dir = candidate
                    break
        if data_dir:
            kb_dir = os.path.join(data_dir, "knowledge")
            doc_count = 0
            if os.path.isdir(kb_dir):
                for fname in os.listdir(kb_dir):
                    if not fname.endswith((".txt", ".md")):
                        continue
                    fpath = os.path.join(kb_dir, fname)
                    if os.path.isdir(fpath):
                        continue
                    try:
                        with open(fpath, "r", encoding="utf-8",
                                  errors="ignore") as f:
                            text = f.read()
                        if not text.strip():
                            continue
                        # Split into chunks (max 500 chars each)
                        chunks = [text[i:i+500]
                                  for i in range(0, min(len(text), 3000), 500)]
                        for ci, chunk in enumerate(chunks):
                            doc_hash = hashlib.sha256(
                                chunk[:200].encode()).hexdigest()[:8]
                            doc_id = f"doc:{doc_hash}"
                            graph.add_node(doc_id, node_type="knowledge",
                                           properties={
                                               "content": chunk,
                                               "source": f"rag:{fname}",
                                               "question": fname,
                                               "timestamp": now,
                                           })
                            doc_count += 1
                    except Exception:
                        continue
                if doc_count > 0:
                    print(f"    RAG documents: {doc_count} chunks imported to tree")
            result["rag_docs"] = doc_count
    except Exception as e:
        print(f"    RAG bootstrap error: {e}")

    total = sum(result.values())
    print(f"    Total: {total} items bootstrapped")
    return result


# ==============================================================================
# TOPIC CLUSTERING — Group related TOPIC nodes under shared CATEGORY nodes
# ==============================================================================

def cluster_topics(graph, min_cluster_size: int = 3) -> int:
    """Group related TOPIC nodes under shared CATEGORY nodes.

    Finds TOPIC nodes that share edges with the same entities,
    groups them into clusters, and creates CATEGORY summary nodes.
    Returns number of clusters created.
    """
    if not graph or not getattr(graph, 'is_open', False):
        return 0

    try:
        topic_nodes = graph.list_nodes(node_type="topic")
    except Exception:
        return 0

    if len(topic_nodes) < min_cluster_size:
        return 0

    # Build adjacency: topic -> set of connected entities
    topic_neighbors = {}
    for nid, data in topic_nodes:
        try:
            edges = graph.get_edges(nid, direction="both")
        except Exception:
            edges = []
        neighbors = set()
        for e in edges:
            if e.get("subject") == nid:
                neighbors.add(e.get("object", ""))
            else:
                neighbors.add(e.get("subject", ""))
        neighbors.discard("")
        topic_neighbors[nid] = neighbors

    # Simple clustering: topics sharing > 50% neighbors go in same cluster
    clusters = []
    assigned = set()
    for nid_a, neighbors_a in topic_neighbors.items():
        if nid_a in assigned or not neighbors_a:
            continue
        cluster = [nid_a]
        assigned.add(nid_a)
        for nid_b, neighbors_b in topic_neighbors.items():
            if nid_b in assigned or not neighbors_b:
                continue
            overlap = len(neighbors_a & neighbors_b)
            union = len(neighbors_a | neighbors_b)
            if union > 0 and overlap / union > 0.5:
                cluster.append(nid_b)
                assigned.add(nid_b)
        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    # Create category nodes for clusters
    created = 0
    for cluster in clusters:
        cluster_key = ":".join(sorted(cluster))
        cat_id = f"cat:{hashlib.sha256(cluster_key.encode()).hexdigest()[:8]}"

        # Skip if already exists
        existing = graph.get_node(cat_id, track_access=False)
        if existing:
            continue

        graph.add_node(cat_id, node_type="category", properties={
            "members": cluster,
            "size": len(cluster),
        })
        for member in cluster:
            graph.add_edge(member, "belongs_to", cat_id,
                           confidence=0.8, source="clustering")
        created += 1

    if created > 0:
        logger.info("Topic clustering: created %d category nodes", created)

    return created
