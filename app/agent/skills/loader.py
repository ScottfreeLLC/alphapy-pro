"""Parse markdown skill files with YAML frontmatter."""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import frontmatter

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """A parsed trading strategy skill."""

    name: str
    file_path: str
    body: str  # The markdown strategy text (sent to Claude)

    # Frontmatter fields
    timeframes: List[str] = field(default_factory=lambda: ["1d"])
    risk_per_trade: float = 0.02
    max_positions: int = 3
    enabled: bool = True
    tags: List[str] = field(default_factory=list)

    # Runtime metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches_timeframe(self, timeframe: str) -> bool:
        return timeframe in self.timeframes


class SkillLoader:
    """Loads and parses markdown skill files from a directory."""

    def __init__(self, skills_dir: str):
        self.skills_dir = skills_dir

    def load_all(self) -> List[Skill]:
        """Load all .md files from the skills directory."""
        if not os.path.isdir(self.skills_dir):
            logger.warning(f"Skills directory not found: {self.skills_dir}")
            return []

        skills = []
        for filename in sorted(os.listdir(self.skills_dir)):
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(self.skills_dir, filename)
            skill = self.load_file(filepath)
            if skill:
                skills.append(skill)

        logger.info(f"Loaded {len(skills)} skills from {self.skills_dir}")
        return skills

    def load_file(self, filepath: str) -> Optional[Skill]:
        """Parse a single skill file."""
        try:
            post = frontmatter.load(filepath)
            meta = post.metadata

            skill = Skill(
                name=meta.get("name", os.path.splitext(os.path.basename(filepath))[0]),
                file_path=filepath,
                body=post.content,
                timeframes=meta.get("timeframes", ["1d"]),
                risk_per_trade=meta.get("risk_per_trade", 0.02),
                max_positions=meta.get("max_positions", 3),
                enabled=meta.get("enabled", True),
                tags=meta.get("tags", []),
            )

            logger.debug(f"Loaded skill: {skill.name} from {filepath}")
            return skill

        except Exception as e:
            logger.error(f"Failed to load skill from {filepath}: {e}")
            return None
