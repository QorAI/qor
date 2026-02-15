"""
QOR Knowledge Base — Psychology + RAPTOR Tree + Document Store
===============================================================
Ported from Go infrastructure into native Python.

Three knowledge layers:
  1. Built-in knowledge (psychology, persuasion, harm reduction)
  2. Document store (load .txt/.md files, search by keyword)
  3. RAPTOR tree (hierarchical retrieval with summarization)

All searchable by QOR's confidence gate.

Usage:
    from qor.knowledge import KnowledgeBase

    kb = KnowledgeBase("knowledge/")
    kb.load()

    # Search
    results = kb.search("how to redirect anger")

    # Get full context for a query
    context = kb.get_context("user is angry about money")
"""

import os
import re
import math
import json
import glob
from typing import List, Dict, Tuple, Optional
from collections import Counter
from dataclasses import dataclass, field


# ==============================================================================
# BUILT-IN PSYCHOLOGY KNOWLEDGE
# ==============================================================================

PSYCHOLOGY_KNOWLEDGE = {

    # ---- Core Principles ----
    "core_principles": """
Meeting People Where They Are:
- Don't judge the question, understand the person
- Everyone has reasons for what they want — find the underlying need
- Resistance means you're pushing too hard — step back and listen
- People change when THEY decide to, not when told to

The "Yes, And" Approach:
- Never start with "no" or "I can't"
- Acknowledge their interest/need first
- Add information that expands their perspective
- Let them reach conclusions themselves

Practical Over Moral:
- "This is dangerous because..." beats "This is wrong because..."
- Focus on what works/doesn't work, not good/bad
- People respond to self-interest, not lectures
- Show the better path, don't just block the current one
""",

    # ---- Motivational Interviewing ----
    "motivational_interviewing": """
OARS Technique:
- Open questions: "What interests you about this?"
- Affirmations: "It makes sense you'd want to understand this"
- Reflections: "So you're looking for..."
- Summaries: "Let me make sure I understand..."

Developing Discrepancy:
- Help them see gap between current path and their actual goals
- "What do you really want to achieve here?"
- "How does this fit with your other goals?"
- Let THEM identify the problems, don't point them out

Rolling with Resistance:
- Arguing creates more resistance
- Agree with part of what they say
- Redirect rather than oppose
- "You're right that... and also..."

Supporting Self-Efficacy:
- Express confidence they can make good choices
- Highlight their strengths and past successes
- "You seem like someone who thinks things through"

The Change Talk Types (DARN-CAT):
- Desire: "I want to..." "I wish..."
- Ability: "I could..." "I can..."
- Reasons: "I would feel better if..."
- Need: "I have to..." "I must..."
- Commitment: "I will..." "I'm going to..."
- Activation: "I'm ready to..."
- Taking Steps: "I've already started..."

The Spirit of MI:
- Partnership (not expert/patient)
- Acceptance (not judgment)
- Compassion (genuine care)
- Evocation (draw out their wisdom, don't pour in yours)
""",

    # ---- Influence Principles (Cialdini) ----
    "influence_principles": """
Reciprocity:
- Give useful information freely first
- People feel obligated to reciprocate openness
- Being genuinely helpful builds trust

Social Proof:
- "Many people have this question..."
- "It's common to be curious about..."
- Normalizing reduces shame and opens conversation

Authority:
- Share expertise without being condescending
- Explain the "why" behind recommendations
- Admit uncertainty when appropriate — builds credibility

Liking:
- Find common ground
- Be warm and non-judgmental
- Use "we" language when appropriate

Commitment/Consistency:
- Get small agreements first
- "Would you agree that safety is important?"
- Build on their stated values

Scarcity:
- Highlight what they might miss with current path
- "The opportunity for X decreases if..."
""",

    # ---- Harm Reduction ----
    "harm_reduction": """
Core Philosophy:
- Accept that people will do risky things
- Reducing harm is better than demanding abstinence
- Non-judgmental support is more effective than moralizing
- Meet people where they are, not where you want them

Why It Works:
- Maintains trust and relationship (enables future influence)
- Provides accurate information (prevents worse outcomes)
- Acknowledges autonomy (people change when they choose to)
- Reduces shame (shame drives worse behaviors)

Practical Application:
- If they're going to do X, help them do it more safely
- Provide accurate information, not scare tactics
- Acknowledge autonomy — it's their choice
- Keep the door open for future conversations

The Goal:
- Keep people safe enough to live another day
- Another day means another chance to make different choices
- Relationship and trust enable influence over time
""",

    # ---- Criminal Psychology ----
    "criminal_psychology": """
Why People Do "Bad" Things:
- Unmet needs (money, status, connection, excitement)
- Limited perceived options
- Short-term thinking under stress
- Normalization within their environment
- Rationalization ("everyone does it", "they deserve it")

What Actually Changes Behavior:
- Seeing viable alternatives that meet the same needs
- Positive relationships with non-criminal others
- Having something to lose (stake in conventional life)
- Skills and opportunities for legitimate success
- Changed self-identity ("I'm not that kind of person")

What Doesn't Work:
- Moral lectures
- Threats of punishment (already discounted)
- Shaming
- Cutting off all contact
- Demanding immediate change

Effective Approaches:
- Find the underlying need and show legal ways to meet it
- Highlight strengths they already have
- Connect to future goals they care about
- Small steps, not complete transformation
- Maintain relationship even when disagreeing

Financial Motivation:
- Show legitimate paths to same money (or more)
- Calculate real risk-adjusted returns
- Highlight skills they have that are marketable

Thrill/Excitement:
- Acknowledge the appeal of risk and excitement
- Suggest legal high-adrenaline alternatives
- Channel competitive drive productively

Power/Control:
- Recognize the need for agency and control
- Show paths to legitimate influence
- Address underlying feelings of powerlessness

Revenge/Anger:
- Validate the feelings without endorsing actions
- Explore what justice would actually look like
- Discuss long-term satisfaction vs short-term release

Belonging/Identity:
- Understand the community aspect
- Don't dismiss their social connections
- Identity shifts happen gradually
""",

    # ---- Persuasion Techniques ----
    "persuasion_techniques": """
Principle 1: Understand Before Persuading
- You cannot change someone's mind until you understand how it works
- Ask questions: Why do you want this? What would change if you had it?
- Listen more than you talk
- People are more open to influence from those who understand them

Principle 2: Find Common Ground
- Start with areas of agreement, no matter how small
- "You're right that..." before "and also..."
- Disagreement is easier to handle after agreement is established

Principle 3: Use Questions Instead of Statements
- "What would happen if...?" instead of "You should..."
- Questions engage the mind; statements trigger defense
- Self-generated insights stick better than external advice

Principle 4: Appeal to Self-Interest
- What's in it for them? Make that clear
- Frame recommendations in terms of their goals, not yours
- Altruistic appeals work less than you'd think

Principle 5: Reduce Friction
- Make the desired action easy
- Remove obstacles proactively
- Break big changes into small steps
- Quick wins build momentum

Principle 6: Know When to Let Go
- Sometimes people aren't ready to change
- Pushing harder doesn't help
- Plant seeds and let them grow
- Your goal is helping them, not winning the argument
""",
}


# ==============================================================================
# DOCUMENT NODE (for RAPTOR tree)
# ==============================================================================

@dataclass
class KnowledgeNode:
    """A node in the knowledge tree."""
    id: str
    title: str
    content: str
    category: str = ""
    keywords: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)  # child node IDs
    metadata: Dict = field(default_factory=dict)
    score: float = 0.0  # relevance score (set during search)


# ==============================================================================
# KNOWLEDGE BASE
# ==============================================================================

class KnowledgeBase:
    """
    Full knowledge base with built-in psychology + document loading + search.

    Ported from Go: knowledge.go + knowledge_loader.go
    """

    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = knowledge_dir
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.index: Dict[str, List[str]] = {}  # word → node IDs
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self):
        """Load built-in knowledge + document files."""
        # 1. Load built-in psychology knowledge
        self._load_builtin_knowledge()

        # 2. Load documents from knowledge directory
        self._load_directory()

        # 3. Build search index
        self._build_index()

        self._loaded = True
        print(f"[Knowledge] Loaded {len(self.nodes)} nodes, "
              f"{len(self.index)} indexed terms")

    def _load_builtin_knowledge(self):
        """Load all built-in psychology/communication knowledge."""
        for topic, content in PSYCHOLOGY_KNOWLEDGE.items():
            node = KnowledgeNode(
                id=f"builtin:{topic}",
                title=topic.replace("_", " ").title(),
                content=content.strip(),
                category="psychology",
                keywords=_extract_keywords(content),
                metadata={"source": "built-in", "type": "psychology"},
            )
            self.nodes[node.id] = node

    def _load_directory(self):
        """Load .txt and .md files from knowledge directory."""
        if not os.path.isdir(self.knowledge_dir):
            os.makedirs(self.knowledge_dir, exist_ok=True)
            self._create_defaults()
            return

        patterns = ["**/*.txt", "**/*.md"]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(
                os.path.join(self.knowledge_dir, pattern), recursive=True
            ))

        if not files:
            self._create_defaults()
            return

        for fpath in sorted(set(files)):
            try:
                with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                rel = os.path.relpath(fpath, self.knowledge_dir)
                name = os.path.splitext(os.path.basename(fpath))[0]
                category = os.path.dirname(rel) or ""

                node = KnowledgeNode(
                    id=rel,
                    title=name,
                    content=content,
                    category=category,
                    keywords=_extract_keywords(content),
                    metadata={"source": fpath, "type": "document"},
                )
                self.nodes[node.id] = node
            except Exception as e:
                print(f"[Knowledge] Warning: failed to load {fpath}: {e}")

        print(f"[Knowledge] Loaded {len(files)} files from {self.knowledge_dir}/")

    def _create_defaults(self):
        """Create default knowledge documents if directory is empty."""
        defaults = {
            "qora.md": """# Qora Blockchain

Qora is a high-performance Layer 1 blockchain with integrated AI capabilities.

## Key Features
- Fast finality (sub-second)
- AI inference on-chain
- Smart contracts via CosmWasm
- EVM compatibility

## Token
- Symbol: QOR
- Smallest unit: uqor (1 QOR = 1,000,000 uqor)

## Staking
- Minimum stake: 1 QOR
- Unbonding period: 21 days
- Rewards distributed each block
""",
            "api.md": """# Qora API Reference

## Balance Query
GET /cosmos/bank/v1beta1/balances/{address}

## Transaction Query
GET /cosmos/tx/v1beta1/txs/{hash}

## Block Query
GET /cosmos/base/tendermint/v1beta1/blocks/{height}

## Validators
GET /cosmos/staking/v1beta1/validators

## AI Models
GET /qora/ai/v1/models

## AI Inference
POST /qora/ai/v1/infer/agent
""",
            "commands.md": """# Common Commands

## Check Balance
qorad query bank balances <address>

## Send Tokens
qorad tx bank send <from> <to> <amount>uqor

## Stake Tokens
qorad tx staking delegate <validator> <amount>uqor

## Query Validators
qorad query staking validators
""",
        }

        os.makedirs(self.knowledge_dir, exist_ok=True)
        for name, content in defaults.items():
            path = os.path.join(self.knowledge_dir, name)
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    f.write(content)

            node = KnowledgeNode(
                id=name,
                title=os.path.splitext(name)[0],
                content=content,
                category="",
                keywords=_extract_keywords(content),
                metadata={"source": "default", "type": "document"},
            )
            self.nodes[node.id] = node

        print(f"[Knowledge] Created {len(defaults)} default documents")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _build_index(self):
        """Build keyword search index over all nodes."""
        self.index = {}
        for node_id, node in self.nodes.items():
            # Index keywords
            for kw in node.keywords:
                self.index.setdefault(kw, []).append(node_id)
            # Index title words
            for word in _tokenize(node.title):
                self.index.setdefault(word, []).append(node_id)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> List[KnowledgeNode]:
        """Search for relevant knowledge nodes."""
        if not self._loaded:
            self.load()

        query_words = _tokenize(query)
        scores: Dict[str, int] = {}

        for word in query_words:
            for node_id in self.index.get(word, []):
                scores[node_id] = scores.get(node_id, 0) + 1

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for node_id, score in ranked[:limit]:
            node = self.nodes[node_id]
            node.score = score
            results.append(node)

        return results

    def get_context(self, query: str, max_chars: int = 3000) -> str:
        """Get relevant context string for a query (used by confidence gate)."""
        nodes = self.search(query, limit=5)
        if not nodes:
            return ""

        parts = []
        total = 0
        for node in nodes:
            # Extract the most relevant section
            section = _extract_relevant_section(node.content, query, 600)
            if not section:
                continue
            if total + len(section) > max_chars:
                break
            parts.append(f"## {node.title}\n{section}")
            total += len(section)

        return "\n\n".join(parts)

    def get_psychology_context(self, query: str) -> str:
        """Get psychology-specific knowledge relevant to a query."""
        query_lower = query.lower()
        relevant = []

        # Always include core principles
        relevant.append(PSYCHOLOGY_KNOWLEDGE["core_principles"])

        # Money/theft
        if any(w in query_lower for w in
               ["money", "steal", "rich", "cash", "financial"]):
            relevant.append(PSYCHOLOGY_KNOWLEDGE["criminal_psychology"])

        # Hacking/thrill
        if any(w in query_lower for w in
               ["hack", "thrill", "exciting", "adrenaline"]):
            relevant.append(PSYCHOLOGY_KNOWLEDGE["criminal_psychology"])

        # Drugs/addiction
        if any(w in query_lower for w in
               ["drug", "addict", "substance", "using"]):
            relevant.append(PSYCHOLOGY_KNOWLEDGE["harm_reduction"])

        # Violence/anger
        if any(w in query_lower for w in
               ["kill", "hurt", "attack", "revenge", "angry", "hate"]):
            relevant.append(PSYCHOLOGY_KNOWLEDGE["criminal_psychology"])
            relevant.append(PSYCHOLOGY_KNOWLEDGE["motivational_interviewing"])

        # Persuasion/communication
        if any(w in query_lower for w in
               ["convince", "persuade", "negotiate", "influence"]):
            relevant.append(PSYCHOLOGY_KNOWLEDGE["persuasion_techniques"])
            relevant.append(PSYCHOLOGY_KNOWLEDGE["influence_principles"])

        return "\n\n".join(relevant)

    # ------------------------------------------------------------------
    # Adding knowledge
    # ------------------------------------------------------------------

    def add_text(self, text: str, title: str = "untitled",
                 category: str = "custom"):
        """Add a text document to the knowledge base."""
        node_id = f"custom:{title}"
        node = KnowledgeNode(
            id=node_id,
            title=title,
            content=text,
            category=category,
            keywords=_extract_keywords(text),
            metadata={"source": "runtime", "type": "custom"},
        )
        self.nodes[node_id] = node

        # Update index
        for kw in node.keywords:
            self.index.setdefault(kw, []).append(node_id)
        for word in _tokenize(node.title):
            self.index.setdefault(word, []).append(node_id)

    def add_file(self, path: str):
        """Add a file to the knowledge base."""
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        name = os.path.splitext(os.path.basename(path))[0]
        self.add_text(content, title=name, category="file")

    def add_folder(self, folder: str):
        """Add all .txt/.md files from a folder."""
        for ext in ["*.txt", "*.md"]:
            for fpath in glob.glob(os.path.join(folder, "**", ext),
                                   recursive=True):
                self.add_file(fpath)
        self._build_index()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self):
        """Print knowledge base statistics."""
        categories = {}
        for node in self.nodes.values():
            cat = node.metadata.get("type", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        total_chars = sum(len(n.content) for n in self.nodes.values())
        print(f"\n  Knowledge Base Stats:")
        print(f"    Total nodes:    {len(self.nodes)}")
        print(f"    Total chars:    {total_chars:,}")
        print(f"    Index terms:    {len(self.index)}")
        for cat, count in sorted(categories.items()):
            print(f"    {cat}: {count} nodes")


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "is", "it", "this", "that", "be",
    "are", "was", "were", "been", "have", "has", "can", "will",
    "do", "does", "did", "not", "so", "if", "from", "as", "they",
    "we", "you", "he", "she", "its", "my", "your", "our", "their",
    "what", "which", "who", "when", "where", "how", "why",
})


def _tokenize(text: str) -> List[str]:
    """Split text into lowercase words, remove stop words."""
    text = text.lower()
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = cleaned.split()
    return [w for w in words if w not in _STOP_WORDS and len(w) >= 2]


def _extract_keywords(content: str, max_keywords: int = 20) -> List[str]:
    """Extract top keywords from content by frequency."""
    words = _tokenize(content)
    freq = Counter(w for w in words if len(w) >= 3)
    return [w for w, _ in freq.most_common(max_keywords)]


def _extract_relevant_section(content: str, query: str,
                               max_len: int = 500) -> str:
    """Extract the most relevant section of content for a query."""
    query_words = set(_tokenize(query))
    lines = content.split('\n')

    # Score each line
    scored = []
    for i, line in enumerate(lines):
        line_words = set(_tokenize(line))
        overlap = len(query_words & line_words)
        if overlap > 0:
            scored.append((i, overlap))

    if not scored:
        # No match — return beginning
        return content[:max_len] + ("..." if len(content) > max_len else "")

    # Get context around best match
    scored.sort(key=lambda x: x[1], reverse=True)
    best_idx = scored[0][0]
    start = max(0, best_idx - 2)
    end = min(len(lines), best_idx + 6)

    section = '\n'.join(lines[start:end])
    if len(section) > max_len:
        section = section[:max_len] + "..."
    return section
