"""Skill system for the Ollama harness.

Skills are markdown documents stored in ./skill/ that define reusable
agent behaviors. The registry loads them lazily and exposes them as
callable tools under the "skill:" namespace.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class Skill(BaseModel):
    """A loaded skill from a markdown document."""

    name: str = Field(..., description="The skill's folder name (e.g. 'fix-bug')")
    description: str = Field(..., description="Parsed from the markdown header")
    tools_used: list[str] = Field(default_factory=list, description="Tools this skill uses")
    steps: list[str] = Field(default_factory=list, description="Step descriptions")
    _invoke: Callable[[dict], str] | None = Field(default=None)

    @classmethod
    def from_markdown(cls, path: Path) -> "Skill":
        """Parse a skill.md file into a Skill object."""
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()

        # Extract header (title)
        title_match = re.match(r"^#\s*(.+)$", lines[0], re.IGNORECASE)
        name = title_match.group(1).strip().lower().replace(" ", "-") if title_match else path.stem

        # Extract description
        desc_match = re.search(r"^##\s*description\s*\n\n(.*?)(?=\n##|\Z)", text, re.IGNORECASE | re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else ""

        # Extract tools used
        tools_match = re.search(r"^##\s*tools used\s*\n\n((?:- .*\n?)+)", text, re.IGNORECASE | re.DOTALL)
        tools_used = []
        if tools_match:
            for line in tools_match.group(1).splitlines():
                m = re.match(r"^\s*-?\s*(.+)$", line.strip())
                if m:
                    tools_used.append(m.group(1).strip())

        # Extract steps
        steps_match = re.search(r"^##\s*steps\s*\n\n((?:\d+\.\s*.+?)(?:\n(?!\d+\.\s).*?\n)+)", text, re.IGNORECASE | re.DOTALL)
        steps = []
        if steps_match:
            block = steps_match.group(1)
            # Split by numbered lines
            step_blocks = re.split(r"(?=\d+\.\s+)", block)
            for block in step_blocks:
                if block.strip():
                    steps.append(block.strip())

        return cls(name=name, description=description, tools_used=tools_used, steps=steps)

    def invoke(self, args: dict[str, Any]) -> str:
        """Execute the skill with the given arguments.

        This is a placeholder that returns a formatted summary of the skill.
        Subclasses or custom skills can override this to perform actual work.
        """
        if self._invoke:
            return self._invoke(args)
        # Default behavior: summarize what the skill would do
        lines = [f"[SkillCallEvent] {self.name}: {self.description}"]
        for step in self.steps:
            lines.append(f"  → {step}")
        if args:
            lines.append(f"  args: {json.dumps(args, indent=2)[:500]}")
        return "\n".join(lines)


class SkillRegistry:
    """Singleton registry that scans ./skill/ and loads skills lazily."""

    _instance: "SkillRegistry | None" = None
    _skills: dict[str, Skill] = {}
    _loaded: set[str] = set()

    def __new__(cls, skill_dir: Path | None = None) -> "SkillRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._skill_dir = Path(skill_dir or "./skill")
            cls._instance._lock = False
        return cls._instance

    @property
    def skill_dir(self) -> Path:
        return self._skill_dir

    def register(self, path: str | Path) -> Skill:
        """Register a skill from a markdown file or directory.

        Args:
            path: Either a path to a skill.md file or a directory containing one.

        Returns:
            The loaded Skill object.
        """
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"skill not found: {p}")

        # Handle directory case
        if p.is_dir():
            md_path = p / "skill.md"
            if not md_path.exists():
                raise ValueError(f"directory has no skill.md: {p}")
            p = md_path

        # Load the skill
        skill = Skill.from_markdown(p)

        # Register it
        self._skills[skill.name] = skill
        self._loaded.add(skill.name)

        return skill

    def unregister(self, name: str) -> None:
        """Unregister a skill by name."""
        if name in self._skills:
            del self._skills[name]
            self._loaded.discard(name)

    def get(self, name: str) -> Skill | None:
        """Get a skill by name. Loads it lazily if not yet loaded."""
        if name not in self._skills:
            # Try to load from the skill directory
            candidate = self.skill_dir / name / "skill.md"
            if candidate.exists():
                skill = Skill.from_markdown(candidate)
                self._skills[name] = skill
                self._loaded.add(name)
            else:
                return None
        return self._skills.get(name)

    def list_skills(self) -> list[Skill]:
        """Return all registered skills."""
        return list(self._skills.values())

    def load_all(self) -> None:
        """Pre-load all skills from the skill directory."""
        if not self.skill_dir.exists():
            return
        for entry in self.skill_dir.iterdir():
            if entry.is_dir() and (entry / "skill.md").exists():
                self.register(entry)

    def __contains__(self, name: str) -> bool:
        skill = self.get(name)
        return skill is not None

    def __len__(self) -> int:
        return len(self._skills)
