"""Skill registry with hot-reload support."""

import logging
import os
import time
from typing import Dict, List, Optional

from .loader import Skill, SkillLoader

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Central registry for trading skills. Supports filtering and hot-reload."""

    def __init__(self, skills_dir: str):
        self.skills_dir = skills_dir
        self.loader = SkillLoader(skills_dir)
        self._skills: Dict[str, Skill] = {}
        self._last_load_time: float = 0
        self._file_mtimes: Dict[str, float] = {}

    def load(self):
        """Load all skills from disk."""
        skills = self.loader.load_all()
        self._skills = {s.name: s for s in skills}
        self._last_load_time = time.time()
        self._update_mtimes()
        logger.info(f"Registry loaded {len(self._skills)} skills")

    def reload_if_changed(self) -> bool:
        """Reload skills if any files have been modified. Returns True if reloaded."""
        if not os.path.isdir(self.skills_dir):
            return False

        changed = False
        for filename in os.listdir(self.skills_dir):
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(self.skills_dir, filename)
            mtime = os.path.getmtime(filepath)
            if filepath not in self._file_mtimes or mtime > self._file_mtimes[filepath]:
                changed = True
                break

        if changed:
            logger.info("Skill files changed, reloading...")
            self.load()
            return True
        return False

    def get_all(self) -> List[Skill]:
        """Get all loaded skills."""
        return list(self._skills.values())

    def get_enabled(self) -> List[Skill]:
        """Get only enabled skills."""
        return [s for s in self._skills.values() if s.enabled]

    def get_by_name(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def get_for_timeframe(self, timeframe: str) -> List[Skill]:
        """Get enabled skills matching a timeframe."""
        return [s for s in self.get_enabled() if s.matches_timeframe(timeframe)]

    def toggle_skill(self, name: str, enabled: bool) -> bool:
        """Enable or disable a skill by name. Returns True if found."""
        skill = self._skills.get(name)
        if skill:
            skill.enabled = enabled
            logger.info(f"Skill '{name}' {'enabled' if enabled else 'disabled'}")
            return True
        return False

    def list_skills(self) -> List[Dict]:
        """Return a summary list of all skills for the API."""
        return [
            {
                "name": s.name,
                "enabled": s.enabled,
                "timeframes": s.timeframes,
                "tags": s.tags,
                "risk_per_trade": s.risk_per_trade,
                "max_positions": s.max_positions,
            }
            for s in self._skills.values()
        ]

    def _update_mtimes(self):
        """Cache file modification times."""
        self._file_mtimes = {}
        if not os.path.isdir(self.skills_dir):
            return
        for filename in os.listdir(self.skills_dir):
            if filename.endswith(".md"):
                filepath = os.path.join(self.skills_dir, filename)
                self._file_mtimes[filepath] = os.path.getmtime(filepath)
