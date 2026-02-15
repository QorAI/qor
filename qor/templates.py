"""
QOR Query Templates — Bridge Between Natural Language and Knowledge Graph
==========================================================================
Templates define patterns for decomposing questions into graph traversal queries.
Each template specifies: what the question looks like, how many hops it needs,
and how to traverse the RocksDB knowledge graph.

Based on the Hereditary Tree-LSTM approach to Complex KBQA (C-KBQA),
adapted for QOR's self-modifying architecture.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FilledQuery:
    """A template filled with extracted entities/relations/constraints."""
    template_name: str
    hops: int
    entity: Optional[str] = None
    entity_type: Optional[str] = None
    relation: Optional[str] = None
    relation1: Optional[str] = None
    relation2: Optional[str] = None
    relation_chain: list = field(default_factory=list)
    constraint: Optional[str] = None
    constraint_field: Optional[str] = None
    constraint_op: Optional[str] = None
    constraint_value: Optional[str] = None
    target_property: Optional[str] = None
    entities: list = field(default_factory=list)


# Keywords that signal each template type (used for matching)
TEMPLATES = {
    "entity_lookup": {
        "description": "Simple: one entity lookup (who/what is X?)",
        "signals": ["who is", "what is", "tell me about", "describe"],
        "hops": 1,
        "needs_relation": False,
        "needs_constraint": False,
    },

    "single_relation": {
        "description": "One-hop relation (X relation ?)",
        "signals": [],  # fallback when entity + relation found
        "hops": 1,
        "needs_relation": True,
        "needs_constraint": False,
    },

    "two_hop": {
        "description": "Two-hop: entity -> relation1 -> intermediate -> relation2 -> answer",
        "signals": ["of the", "of a", "'s"],
        "hops": 2,
        "needs_relation": True,
        "needs_constraint": False,
    },

    "constrained": {
        "description": "Entity + relation + filter condition",
        "signals": ["where", "when", "before", "after", "more than", "less than",
                     "greater than", "above", "below", "between"],
        "hops": 1,
        "needs_relation": True,
        "needs_constraint": True,
    },

    "multi_hop_constrained": {
        "description": "Multi-hop + constraint (the hard ones)",
        "signals": [],  # matched when 2+ relations AND constraint detected
        "hops": 2,
        "needs_relation": True,
        "needs_constraint": True,
    },

    "comparison": {
        "description": "Comparing two entities on a property",
        "signals": ["compare", "difference between", "versus", "vs"],
        "hops": 1,
        "needs_relation": False,
        "needs_constraint": False,
    },

    "aggregation": {
        "description": "Count, sum, average operations",
        "signals": ["how many", "how much", "count", "total", "average", "sum"],
        "hops": 1,
        "needs_relation": True,
        "needs_constraint": False,
    },

    "transaction_chain": {
        "description": "Trace/track a chain of transactions or relationships",
        "signals": ["trace", "track", "chain", "path", "route", "follow"],
        "hops": 3,
        "needs_relation": True,
        "needs_constraint": False,
    },
}

# Constraint operators mapped from natural language
CONSTRAINT_OPS = {
    "before": "<",
    "after": ">",
    "more than": ">",
    "greater than": ">",
    "above": ">",
    "less than": "<",
    "below": "<",
    "at least": ">=",
    "at most": "<=",
    "equal to": "==",
    "equals": "==",
    "is": "==",
    "between": "between",
}
