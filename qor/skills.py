"""
QOR Skill System — SKILL.md Text Files (OpenClaw Pattern)
==========================================================
Skills are just markdown files. The AI reads them and follows the instructions.
Drop a SKILL.md in qor-data/skills/, QOR immediately knows it. No code, no rebuild.

How it works:
  1. SkillLoader scans qor-data/skills/ recursively
  2. Reads every SKILL.md → parses YAML frontmatter + markdown body
  3. Builds index: {name → SkillInfo(name, description, keywords, instructions)}
  4. At query time: match_skill(question) → inject instructions into system prompt
  5. Model follows the SKILL.md steps using existing tools

SKILL.md format:
  ---
  name: polymarket
  description: Fetch prediction market odds from Polymarket
  keywords: [polymarket, prediction, odds, probability]
  tools: [polymarket, news_search]
  ---

  # Polymarket Skill
  ## When to use
  User asks about prediction odds...
  ## How to execute
  1. Call polymarket tool...
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SkillInfo:
    """A loaded skill from a SKILL.md file."""
    name: str                  # "polymarket"
    description: str           # "Fetch prediction market odds"
    keywords: list             # ["polymarket", "prediction", "odds"]
    tools: list                # ["polymarket", "news_search"]
    instructions: str          # full markdown body (the AI reads this)
    path: str                  # file path for hot-reload


def _parse_skill_md(text: str, path: str) -> Optional[SkillInfo]:
    """Parse a SKILL.md file into a SkillInfo.

    Expected format: YAML frontmatter between --- markers, then markdown body.
    """
    # Split frontmatter from body
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', text, re.DOTALL)
    if not match:
        return None

    frontmatter_text = match.group(1)
    body = match.group(2).strip()

    # Parse YAML frontmatter (simple key: value parsing, no PyYAML dependency)
    fm = {}
    for line in frontmatter_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        colon_pos = line.find(':')
        if colon_pos < 0:
            continue
        key = line[:colon_pos].strip()
        value = line[colon_pos + 1:].strip()

        # Parse YAML lists: [item1, item2, item3]
        if value.startswith('[') and value.endswith(']'):
            items = value[1:-1].split(',')
            fm[key] = [item.strip().strip('"').strip("'") for item in items if item.strip()]
        else:
            # Strip quotes
            fm[key] = value.strip('"').strip("'")

    name = fm.get('name')
    if not name:
        return None

    return SkillInfo(
        name=name,
        description=fm.get('description', ''),
        keywords=fm.get('keywords', []),
        tools=fm.get('tools', []),
        instructions=body,
        path=path,
    )


class SkillLoader:
    """Loads and matches SKILL.md files from a directory."""

    def __init__(self, skills_dir: str):
        self.skills_dir = skills_dir
        self.skills: dict = {}          # name → SkillInfo
        self._timestamps: dict = {}     # path → mtime

    def load_all(self):
        """Scan skills_dir recursively for SKILL.md files."""
        self.skills.clear()
        self._timestamps.clear()

        if not os.path.isdir(self.skills_dir):
            return

        for root, dirs, files in os.walk(self.skills_dir):
            for fname in files:
                if fname.upper() == 'SKILL.MD':
                    path = os.path.join(root, fname)
                    self._load_file(path)

    def _load_file(self, path: str):
        """Load a single SKILL.md file."""
        try:
            with open(path, encoding='utf-8') as f:
                text = f.read()
            skill = _parse_skill_md(text, path)
            if skill:
                self.skills[skill.name] = skill
                self._timestamps[path] = os.path.getmtime(path)
        except Exception:
            pass

    def match(self, question: str) -> Optional[SkillInfo]:
        """Find best matching skill for a question.

        Scoring: keyword hits (+2 each), description word overlap (+1 each).
        Returns best match above threshold (>= 2), or None.
        """
        q = question.lower()
        best_skill = None
        best_score = 0

        for skill in self.skills.values():
            score = 0
            # Keyword hits (strongest signal)
            for kw in skill.keywords:
                if kw.lower() in q:
                    score += 2
            # Description word overlap (weaker signal)
            for word in skill.description.lower().split():
                if len(word) > 3 and word in q:
                    score += 1

            if score > best_score:
                best_score = score
                best_skill = skill

        return best_skill if best_score >= 2 else None

    def check_for_updates(self) -> list:
        """Hot-reload: detect changed/new SKILL.md files. Returns list of reloaded names."""
        reloaded = []

        if not os.path.isdir(self.skills_dir):
            return reloaded

        # Check for new or modified files
        for root, dirs, files in os.walk(self.skills_dir):
            for fname in files:
                if fname.upper() == 'SKILL.MD':
                    path = os.path.join(root, fname)
                    mtime = os.path.getmtime(path)
                    if path not in self._timestamps or self._timestamps[path] < mtime:
                        self._load_file(path)
                        # Find the skill name for this path
                        for name, skill in self.skills.items():
                            if skill.path == path:
                                reloaded.append(name)
                                break

        return reloaded

    def get_descriptions(self) -> str:
        """Format all skill names + descriptions for display."""
        if not self.skills:
            return "No skills loaded."
        lines = []
        for name, skill in sorted(self.skills.items()):
            lines.append(f"  {name}: {skill.description}")
            lines.append(f"    keywords: {', '.join(skill.keywords)}")
        return '\n'.join(lines)
