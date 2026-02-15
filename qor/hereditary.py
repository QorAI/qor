"""
QOR Hereditary Question Decomposition
========================================
Breaks complex multi-hop questions into sub-questions using bottom-up
hereditary attention, matches them to query templates, and traverses
the RocksDB knowledge graph.

Based on the Hereditary Tree-LSTM (HTL) approach to Complex KBQA,
adapted for QOR: instead of a separate Tree-LSTM, QOR's self-mod
layers learn decomposition over time through CMS.

Flow:
  User question
    -> Confidence gate (simple vs complex?)
    -> Decompose into question tree
    -> Hereditary attention (bottom-up)
    -> Match to template
    -> Fill template, traverse graph
    -> Combine results, generate answer
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from .templates import TEMPLATES, CONSTRAINT_OPS, FilledQuery


# =========================================================================
# Question Tree
# =========================================================================

@dataclass
class QuestionNode:
    """A node in the question decomposition tree."""
    text: str
    node_type: str = "unknown"  # entity, relation, constraint, root
    attention: float = 0.0
    children: list = field(default_factory=list)
    value: Optional[str] = None  # resolved entity/relation/constraint value

    def is_leaf(self):
        return len(self.children) == 0

    def depth(self):
        if self.is_leaf():
            return 0
        return 1 + max(c.depth() for c in self.children)

    def __repr__(self):
        kids = f" [{len(self.children)} children]" if self.children else ""
        return f"QNode({self.node_type}: '{self.text}' attn={self.attention:.2f}{kids})"


# =========================================================================
# Hereditary Question Decomposer
# =========================================================================

class HereditaryQuestionDecomposer:
    """
    Breaks complex questions into sub-questions using bottom-up attention.

    Instead of Tree-LSTM, uses QOR's knowledge graph to resolve entities
    and relations, then applies hereditary attention scoring.
    """

    def __init__(self, graph=None):
        """
        Args:
            graph: QORGraph instance for entity/relation resolution
        """
        self.graph = graph
        self._known_entities = set()
        self._known_predicates = set()
        self._refresh_knowledge()

    def _refresh_knowledge(self):
        """Cache known entities and predicates from the graph."""
        if self.graph is None:
            return
        try:
            nodes = self.graph.list_nodes()
            self._known_entities = {nid.lower() for nid, _ in nodes}
            # Get all predicate names from synonym entries
            preds = set()
            for syn_key, syn_data in self.graph._prefix_scan("syn:"):
                pred = syn_key.replace("syn:", "")
                preds.add(pred)
                if isinstance(syn_data, dict):
                    canonical = syn_data.get("canonical", "")
                    if canonical:
                        preds.add(canonical.lower())
                    for alias in syn_data.get("aliases", []):
                        preds.add(alias.lower())
            # Also add predicates from actual edges
            for edge_key, _ in self.graph._prefix_scan("edge:"):
                parts = edge_key.split(":")
                if len(parts) >= 3:
                    preds.add(parts[2].lower())
            self._known_predicates = preds
        except Exception:
            pass

    def decompose(self, question: str) -> dict:
        """
        Full decomposition pipeline.

        Returns:
            dict with keys: tree, template, query, entities, relations, constraints
        """
        question_lower = question.lower().strip().rstrip("?").strip()

        # Step 1: Extract components
        entities = self.extract_entities(question_lower)
        relations = self.extract_relations(question_lower)
        constraints = self.extract_constraints(question_lower)

        # Step 2: Build question tree
        tree = self.build_tree(question_lower, entities, relations, constraints)

        # Step 3: Apply hereditary attention (bottom-up)
        self.apply_hereditary_attention(tree)

        # Step 4: Match to template
        template_name = self.match_template(question_lower, entities, relations, constraints)

        # Step 5: Fill template into a query
        query = self.fill_template(template_name, entities, relations, constraints)

        return {
            "tree": tree,
            "template": template_name,
            "query": query,
            "entities": entities,
            "relations": relations,
            "constraints": constraints,
        }

    def extract_entities(self, question: str) -> list:
        """Find entities mentioned in the question that exist in the knowledge graph."""
        found = []
        if not self._known_entities:
            return found

        words = re.findall(r'[a-z_]+', question)

        # Check single words
        for w in words:
            if w in self._known_entities:
                found.append(w)

        # Check multi-word phrases (2-3 word combinations)
        for i in range(len(words)):
            for j in range(i + 2, min(i + 4, len(words) + 1)):
                phrase = "_".join(words[i:j])
                if phrase in self._known_entities:
                    found.append(phrase)

        # Deduplicate preserving order
        seen = set()
        unique = []
        for e in found:
            if e not in seen:
                seen.add(e)
                unique.append(e)

        return unique

    def extract_relations(self, question: str) -> list:
        """Find relations/predicates mentioned in the question."""
        found = []
        if self.graph is None:
            return found

        # Stop words that should never be treated as relations
        stop_words = {
            "who", "what", "where", "when", "why", "how", "is", "are", "was",
            "were", "do", "does", "did", "the", "a", "an", "of", "in", "on",
            "at", "to", "for", "with", "from", "by", "that", "which", "this",
            "it", "and", "or", "not", "no", "but", "if", "than", "more", "less",
            "many", "much", "some", "all", "any", "have", "get", "give", "tell",
            "me", "about", "can", "could", "would", "should", "will",
        }

        words = re.findall(r'[a-z_]+', question)

        # Track (position, resolved_predicate) to preserve question order
        found_with_pos = []

        # Check each word against predicate synonyms
        for i, w in enumerate(words):
            if w in stop_words:
                continue
            resolved = self.graph.resolve_predicate(w)
            if resolved != w or w in self._known_predicates:
                pos = question.find(w)
                found_with_pos.append((pos, resolved))

        # Check multi-word phrases (only if they match known predicates exactly)
        for i in range(len(words)):
            for j in range(i + 2, min(i + 4, len(words) + 1)):
                phrase_space = " ".join(words[i:j])
                phrase_under = "_".join(words[i:j])
                if phrase_space in self._known_predicates or phrase_under in self._known_predicates:
                    resolved = self.graph.resolve_predicate(phrase_space)
                    pos = question.find(phrase_space)
                    found_with_pos.append((pos, resolved))

        # Sort by position in question, then deduplicate
        found_with_pos.sort(key=lambda x: x[0])
        found = [r for _, r in found_with_pos]

        # Deduplicate
        seen = set()
        unique = []
        for r in found:
            if r not in seen:
                seen.add(r)
                unique.append(r)

        return unique

    def extract_constraints(self, question: str) -> list:
        """Extract constraint conditions (before X, more than Y, etc.)."""
        constraints = []

        for phrase, op in CONSTRAINT_OPS.items():
            idx = question.find(phrase)
            if idx == -1:
                continue

            # Extract the value after the constraint phrase
            after = question[idx + len(phrase):].strip()
            # Take the next word/number as the value
            match = re.match(r'[\w.]+', after)
            if match:
                value = match.group()
                constraints.append({
                    "phrase": phrase,
                    "op": op,
                    "value": value,
                    "position": idx,
                })

        return constraints

    def build_tree(self, question: str, entities: list,
                   relations: list, constraints: list) -> QuestionNode:
        """Build a question decomposition tree."""
        root = QuestionNode(text=question, node_type="root")

        # Add entity nodes as children
        for entity in entities:
            root.children.append(QuestionNode(
                text=entity, node_type="entity", value=entity
            ))

        # Add relation nodes
        for relation in relations:
            root.children.append(QuestionNode(
                text=relation, node_type="relation", value=relation
            ))

        # Add constraint nodes
        for constraint in constraints:
            node = QuestionNode(
                text=f"{constraint['phrase']} {constraint['value']}",
                node_type="constraint",
                value=constraint['value'],
            )
            root.children.append(node)

        # If we have both relations and entities, create hierarchy:
        # entity -> relation -> result, with constraints on the side
        if len(entities) >= 1 and len(relations) >= 2:
            # Multi-hop: restructure into chain
            root.children = []
            current = root
            for i, relation in enumerate(relations):
                rel_node = QuestionNode(text=relation, node_type="relation", value=relation)
                if i < len(entities):
                    ent_node = QuestionNode(
                        text=entities[i], node_type="entity", value=entities[i]
                    )
                    rel_node.children.append(ent_node)
                current.children.append(rel_node)
                current = rel_node

            # Attach constraints to deepest node
            for constraint in constraints:
                current.children.append(QuestionNode(
                    text=f"{constraint['phrase']} {constraint['value']}",
                    node_type="constraint",
                    value=constraint['value'],
                ))

        return root

    def apply_hereditary_attention(self, node: QuestionNode):
        """
        Bottom-up hereditary attention.

        Leaf nodes get base attention scores.
        Parent attention = own_score + 0.5 * sum(children_attention).
        This ensures important fragments at the bottom propagate up.
        """
        if node.children:
            # Process all children first (bottom-up)
            for child in node.children:
                self.apply_hereditary_attention(child)

            # Parent inherits children's attention
            child_attention = sum(c.attention for c in node.children)

            # Own score based on node type importance
            own_attention = self._score_node(node)

            # Hereditary combination
            node.attention = own_attention + 0.5 * child_attention
        else:
            # Leaf node: just own attention
            node.attention = self._score_node(node)

    def _score_node(self, node: QuestionNode) -> float:
        """Score a node's importance based on type and content."""
        base_scores = {
            "entity": 0.9,
            "relation": 0.8,
            "constraint": 0.7,
            "root": 0.3,
            "unknown": 0.1,
        }
        score = base_scores.get(node.node_type, 0.1)

        # Boost if entity exists in knowledge graph
        if node.node_type == "entity" and node.value:
            if node.value.lower() in self._known_entities:
                score += 0.1

        return min(score, 1.0)

    def match_template(self, question: str, entities: list,
                       relations: list, constraints: list) -> str:
        """Find the best matching template for the question structure."""
        scores = {}

        for name, template in TEMPLATES.items():
            score = 0.0

            # Signal word matching
            for signal in template["signals"]:
                if signal in question:
                    score += 1.0

            # Structural matching
            has_relation = len(relations) > 0
            has_constraint = len(constraints) > 0
            n_entities = len(entities)
            n_relations = len(relations)

            if template["needs_relation"] and not has_relation:
                score -= 2.0  # Heavy penalty
            if template["needs_constraint"] and not has_constraint:
                score -= 2.0

            # Boost for structural fit
            if name == "entity_lookup" and n_entities >= 1 and n_relations == 0:
                score += 1.5
            elif name == "single_relation" and n_entities >= 1 and n_relations == 1:
                score += 1.5
            elif name == "two_hop" and n_relations >= 2:
                score += 2.0
            elif name == "constrained" and has_constraint and n_relations >= 1:
                score += 1.5
            elif name == "multi_hop_constrained" and has_constraint and n_relations >= 2:
                score += 2.5
            elif name == "comparison" and n_entities >= 2:
                score += 1.5
            elif name == "aggregation" and any(s in question for s in ["how many", "count", "total"]):
                score += 2.0
            elif name == "transaction_chain" and any(s in question for s in ["trace", "track", "chain"]):
                score += 2.0

            scores[name] = score

        # Return the highest scoring template
        best = max(scores, key=scores.get)
        return best

    def fill_template(self, template_name: str, entities: list,
                      relations: list, constraints: list) -> FilledQuery:
        """Fill a template with extracted components to create an executable query."""
        template = TEMPLATES[template_name]
        query = FilledQuery(
            template_name=template_name,
            hops=template["hops"],
            entities=entities[:],
        )

        if entities:
            query.entity = entities[0]

        # For multi-hop templates, reorder relations based on graph connectivity
        # instead of sentence position. The relation that connects to the known
        # entity should be relation1 (first hop).
        ordered_relations = self._order_relations_by_graph(
            entities, relations) if template["hops"] >= 2 else relations

        if ordered_relations:
            query.relation = ordered_relations[0]
            query.relation1 = ordered_relations[0]
        if len(ordered_relations) >= 2:
            query.relation2 = ordered_relations[1]
        if len(ordered_relations) >= 2:
            query.relation_chain = ordered_relations

        if constraints:
            c = constraints[0]
            query.constraint = f"{c['op']} {c['value']}"
            query.constraint_op = c['op']
            query.constraint_value = c['value']

        return query

    def _order_relations_by_graph(self, entities: list, relations: list) -> list:
        """
        Reorder relations so that the first relation is the one that connects
        to the known entity. This fixes multi-hop chain direction.

        For "What features does the project that ravi created have?":
          entity=ravi, relations=['has','created'] (sentence order)
          Graph has: ravi->created->qora, qora->has->staking
          So 'created' connects to ravi -> relation1='created', relation2='has'
        """
        if not entities or not relations or len(relations) < 2 or self.graph is None:
            return relations

        entity = entities[0]

        # Check which relation has edges from/to the known entity
        for i, rel in enumerate(relations):
            edges = self.graph.query_pattern(subject=entity, predicate=rel, obj=None)
            if not edges:
                edges = self.graph.query_pattern(subject=None, predicate=rel, obj=entity)
            if edges:
                # This relation connects to the entity — it should be first
                if i == 0:
                    return relations  # already correct order
                # Move this relation to front, keep rest in order
                reordered = [rel] + [r for j, r in enumerate(relations) if j != i]
                return reordered

        # No relation matched the entity in the graph — keep original order
        return relations


# =========================================================================
# Graph Query Executor
# =========================================================================

class GraphQueryExecutor:
    """
    Takes a filled template and traverses the RocksDB knowledge graph.
    Handles single-hop, multi-hop, and chain traversals.
    """

    def __init__(self, graph):
        self.graph = graph

    def execute(self, query: FilledQuery) -> dict:
        """Execute a filled query against the knowledge graph."""
        if query.template_name == "entity_lookup":
            return self.entity_lookup(query)
        elif query.template_name == "comparison":
            return self.comparison(query)
        elif query.template_name == "aggregation":
            return self.aggregation(query)
        elif query.hops == 1:
            return self.single_hop(query)
        elif query.hops == 2:
            return self.multi_hop(query)
        elif query.hops >= 3:
            return self.chain_hop(query)
        else:
            return self.single_hop(query)

    def entity_lookup(self, query: FilledQuery) -> dict:
        """Direct entity lookup — return all properties and edges."""
        entity = query.entity
        if not entity:
            return {"answer": "No entity found in question.", "results": [], "confidence": 0.0}

        node = self.graph.get_node(entity)
        edges = self.graph.get_edges(entity, direction="both")

        if node is None and not edges:
            return {"answer": f"Entity '{entity}' not found.", "results": [], "confidence": 0.0}

        results = []
        if node:
            results.append({"type": "node", "data": node})
        for edge in edges:
            results.append({"type": "edge", "data": edge})

        # Build readable answer
        parts = []
        if node:
            props = node.get("properties", {})
            if props:
                for k, v in props.items():
                    parts.append(f"{k}: {v}")
        for edge in edges:
            parts.append(f"{edge['subject']} {edge['predicate']} {edge['object']} "
                         f"(confidence: {edge.get('confidence', 0):.2f})")

        answer = "; ".join(parts) if parts else f"Found entity '{entity}' but no details."
        confidence = sum(e.get("confidence", 0.5) for e in edges) / max(len(edges), 1)

        return {"answer": answer, "results": results, "confidence": confidence,
                "entity": entity, "edge_count": len(edges)}

    def single_hop(self, query: FilledQuery) -> dict:
        """Single-hop: entity + relation -> results, with optional constraint."""
        entity = query.entity
        relation = query.relation

        if not entity:
            return {"answer": "No entity found.", "results": [], "confidence": 0.0}

        # Get edges from entity
        if relation:
            # Specific relation
            edges = self.graph.query_pattern(subject=entity, predicate=relation, obj=None)
            # Also check reverse
            edges += self.graph.query_pattern(subject=None, predicate=relation, obj=entity)
        else:
            edges = self.graph.get_edges(entity, direction="both")

        # Apply constraint filter
        if query.constraint_op and query.constraint_value:
            edges = self._filter_edges(edges, query.constraint_op, query.constraint_value)

        if not edges:
            return {"answer": f"No results for '{entity}' with relation '{relation}'.",
                    "results": [], "confidence": 0.0}

        results = [{"type": "edge", "data": e} for e in edges]
        parts = [f"{e['subject']} {e['predicate']} {e['object']}" for e in edges]
        answer = "; ".join(parts)
        confidence = sum(e.get("confidence", 0.5) for e in edges) / len(edges)

        return {"answer": answer, "results": results, "confidence": confidence,
                "edge_count": len(edges)}

    def multi_hop(self, query: FilledQuery) -> dict:
        """Two-hop: entity -> relation1 -> intermediate -> relation2 -> results."""
        entity = query.entity
        relation1 = query.relation1
        relation2 = query.relation2

        if not entity or not relation1:
            return {"answer": "Insufficient information for multi-hop query.",
                    "results": [], "confidence": 0.0}

        # First hop
        hop1_edges = self.graph.query_pattern(subject=entity, predicate=relation1, obj=None)
        if not hop1_edges:
            hop1_edges = self.graph.query_pattern(subject=None, predicate=relation1, obj=entity)

        # Apply constraint to first hop if present
        if query.constraint_op and query.constraint_value:
            hop1_edges = self._filter_edges(hop1_edges, query.constraint_op, query.constraint_value)

        if not hop1_edges:
            return {"answer": f"No results for first hop: '{entity}' -> '{relation1}'.",
                    "results": [], "confidence": 0.0}

        # Second hop on each intermediate result
        final_results = []
        for edge in hop1_edges:
            # The intermediate entity is the object of hop1
            intermediate = edge["object"] if edge["subject"] == entity else edge["subject"]

            if relation2:
                hop2_edges = self.graph.query_pattern(
                    subject=intermediate, predicate=relation2, obj=None)
                if not hop2_edges:
                    hop2_edges = self.graph.query_pattern(
                        subject=None, predicate=relation2, obj=intermediate)
            else:
                hop2_edges = self.graph.get_edges(intermediate, direction="both")

            for h2 in hop2_edges:
                final_results.append({
                    "type": "multi_hop",
                    "hop1": edge,
                    "hop2": h2,
                    "intermediate": intermediate,
                })

        if not final_results:
            intermediates = [e["object"] for e in hop1_edges]
            return {"answer": f"Found intermediates {intermediates} but no second hop results.",
                    "results": [], "confidence": 0.3}

        # Build answer
        parts = []
        for r in final_results:
            h2 = r["hop2"]
            parts.append(f"{h2['subject']} {h2['predicate']} {h2['object']} "
                         f"(via {r['intermediate']})")

        answer = "; ".join(parts)
        confidences = [r["hop2"].get("confidence", 0.5) for r in final_results]
        confidence = sum(confidences) / len(confidences)

        return {"answer": answer, "results": final_results, "confidence": confidence,
                "edge_count": len(final_results), "hops_completed": 2}

    def chain_hop(self, query: FilledQuery) -> dict:
        """Follow a chain of relations (3+ hops)."""
        entity = query.entity
        chain = query.relation_chain

        if not entity or not chain:
            return {"answer": "Insufficient information for chain query.",
                    "results": [], "confidence": 0.0}

        current_entities = [entity]
        all_paths = []

        for i, relation in enumerate(chain):
            next_entities = []
            for ent in current_entities:
                edges = self.graph.query_pattern(subject=ent, predicate=relation, obj=None)
                if not edges:
                    edges = self.graph.query_pattern(subject=None, predicate=relation, obj=ent)

                for edge in edges:
                    target = edge["object"] if edge["subject"] == ent else edge["subject"]
                    next_entities.append(target)
                    all_paths.append({
                        "hop": i + 1,
                        "relation": relation,
                        "from": ent,
                        "to": target,
                        "edge": edge,
                    })

            current_entities = list(set(next_entities))

            if not current_entities:
                return {"answer": f"Chain broke at hop {i+1}: '{relation}' from {ent}.",
                        "results": all_paths, "confidence": 0.2,
                        "hops_completed": i + 1}

        parts = [f"{p['from']} -[{p['relation']}]-> {p['to']}" for p in all_paths]
        answer = " | ".join(parts)
        final = ", ".join(current_entities)

        return {"answer": f"Final: {final}. Path: {answer}",
                "results": all_paths, "final_entities": current_entities,
                "confidence": 0.7, "edge_count": len(all_paths),
                "hops_completed": len(chain)}

    def comparison(self, query: FilledQuery) -> dict:
        """Compare two entities on their properties/edges."""
        entities = query.entities
        if len(entities) < 2:
            return {"answer": "Need at least 2 entities to compare.",
                    "results": [], "confidence": 0.0}

        results = {}
        for entity in entities[:2]:
            node = self.graph.get_node(entity)
            edges = self.graph.get_edges(entity, direction="both")
            results[entity] = {"node": node, "edges": edges}

        parts = []
        for entity, data in results.items():
            edge_count = len(data["edges"])
            props = data["node"].get("properties", {}) if data["node"] else {}
            parts.append(f"{entity}: {edge_count} connections, properties={props}")

        answer = " vs ".join(parts)
        return {"answer": answer, "results": results, "confidence": 0.6,
                "entities_compared": list(results.keys())}

    def aggregation(self, query: FilledQuery) -> dict:
        """Count/aggregate results."""
        entity = query.entity
        relation = query.relation

        if relation:
            edges = self.graph.query_pattern(subject=entity, predicate=relation, obj=None)
            if not edges and entity:
                edges = self.graph.query_pattern(subject=None, predicate=relation, obj=entity)
            if not edges:
                # Just search by predicate
                edges = self.graph.query_pattern(subject=None, predicate=relation, obj=None)
        elif entity:
            edges = self.graph.get_edges(entity, direction="both")
        else:
            return {"answer": "Not enough information for aggregation.",
                    "results": [], "confidence": 0.0}

        count = len(edges)
        answer = f"Count: {count}"
        if entity:
            answer = f"{entity} has {count} matching connections"
        if relation:
            answer += f" for '{relation}'"

        return {"answer": answer, "results": edges, "count": count,
                "confidence": 0.8, "edge_count": count}

    def _filter_edges(self, edges: list, op: str, value: str) -> list:
        """Filter edges based on constraint operator and value."""
        filtered = []
        for edge in edges:
            # Check object field (often contains the value to filter)
            obj = edge.get("object", "")
            # Try numeric comparison
            try:
                obj_num = float(obj)
                val_num = float(value)
                if op == ">" and obj_num > val_num:
                    filtered.append(edge)
                elif op == "<" and obj_num < val_num:
                    filtered.append(edge)
                elif op == ">=" and obj_num >= val_num:
                    filtered.append(edge)
                elif op == "<=" and obj_num <= val_num:
                    filtered.append(edge)
                elif op == "==" and obj_num == val_num:
                    filtered.append(edge)
                continue
            except (ValueError, TypeError):
                pass

            # String comparison
            if op == "==" and obj.lower() == value.lower():
                filtered.append(edge)
            elif op == ">" and obj.lower() > value.lower():
                filtered.append(edge)
            elif op == "<" and obj.lower() < value.lower():
                filtered.append(edge)
            else:
                # If we can't apply the filter, include the edge
                filtered.append(edge)

        return filtered


# =========================================================================
# High-level API
# =========================================================================

def hereditary_query(graph, question: str) -> dict:
    """
    One-call API: decompose question, traverse graph, return answer.

    Args:
        graph: QORGraph instance
        question: natural language question

    Returns:
        dict with: answer, confidence, template, tree, results, decomposition
    """
    decomposer = HereditaryQuestionDecomposer(graph)
    decomposition = decomposer.decompose(question)

    executor = GraphQueryExecutor(graph)
    result = executor.execute(decomposition["query"])

    return {
        "answer": result.get("answer", "No answer found."),
        "confidence": result.get("confidence", 0.0),
        "template": decomposition["template"],
        "entities_found": decomposition["entities"],
        "relations_found": decomposition["relations"],
        "constraints": decomposition["constraints"],
        "edge_count": result.get("edge_count", 0),
        "hops_completed": result.get("hops_completed", decomposition["query"].hops),
        "tree_depth": decomposition["tree"].depth(),
        "results": result.get("results", []),
    }
