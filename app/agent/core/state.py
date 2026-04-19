"""Agent state management."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from ..core.signal import TradeSignal


class AgentType(str, Enum):
    SWING = "swing"
    DAY = "day"


class AgentStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class AutonomyMode(str, Enum):
    APPROVAL = "approval"
    SEMI_AUTONOMOUS = "semi_autonomous"
    AUTONOMOUS = "autonomous"


@dataclass
class CycleResult:
    """Result of a single engine cycle."""
    cycle_number: int
    timestamp: datetime
    signals_generated: int
    signals_executed: int
    signals_pending_approval: int
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class AgentState:
    """Tracks the current state of the Alfi agent."""

    agent_type: AgentType = AgentType.SWING
    status: AgentStatus = AgentStatus.STOPPED
    autonomy_mode: AutonomyMode = AutonomyMode.APPROVAL
    cycle_count: int = 0
    started_at: Optional[datetime] = None
    last_cycle_at: Optional[datetime] = None

    # Signals
    pending_signals: List[TradeSignal] = field(default_factory=list)
    recent_signals: List[TradeSignal] = field(default_factory=list)

    # Activity log
    activity_log: List[Dict] = field(default_factory=list)

    # Error tracking
    consecutive_errors: int = 0
    last_error: Optional[str] = None

    def log_activity(self, action: str, detail: str = "", level: str = "info"):
        """Add an entry to the activity log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "detail": detail,
            "level": level,
        }
        self.activity_log.append(entry)
        # Keep only last 200 entries
        if len(self.activity_log) > 200:
            self.activity_log = self.activity_log[-200:]

    def to_dict(self) -> Dict:
        """Serialize state for API/WebSocket."""
        return {
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "autonomy_mode": self.autonomy_mode.value,
            "cycle_count": self.cycle_count,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_cycle_at": self.last_cycle_at.isoformat() if self.last_cycle_at else None,
            "pending_signals_count": len(self.pending_signals),
            "recent_signals_count": len(self.recent_signals),
            "consecutive_errors": self.consecutive_errors,
            "last_error": self.last_error,
            "recent_activity": self.activity_log[-10:],
        }
