"""Memory tool for persistent state tracking."""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .base import Tool

logger = logging.getLogger(__name__)


@dataclass
class MemoryTool(Tool):
    """Tool for persistent state tracking across agent sessions.

    Stores signals, orders, P&L, and other state in JSON files.
    """

    name: str = "agent_memory"
    description: str = """
Stores and retrieves persistent state:
- Today's signals generated
- Orders submitted and filled
- Cumulative P&L tracking
- Error logs and recoveries
Use this to maintain context across trading sessions.
"""
    input_schema: dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "write", "append", "list", "clear"],
                "description": "Operation to perform",
            },
            "key": {
                "type": "string",
                "description": "State key (e.g., 'signals', 'orders', 'pnl')",
            },
            "value": {
                "type": "string",
                "description": "Value to store (JSON string for write/append)",
            },
        },
        "required": ["action", "key"],
    })

    memory_dir: Path = field(default_factory=lambda: Path("agent/memory"))

    def __post_init__(self):
        """Ensure memory directory exists."""
        self.memory_dir = Path(self.memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, key: str, dated: bool = True) -> Path:
        """Get file path for a memory key.

        Args:
            key: Memory key
            dated: If True, include today's date in filename

        Returns:
            Path to the memory file.
        """
        if dated:
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"{key}_{date_str}.json"
        else:
            filename = f"{key}.json"

        return self.memory_dir / filename

    async def execute(
        self,
        action: str,
        key: str,
        value: Optional[str] = None,
    ) -> str:
        """Perform memory operation.

        Returns:
            JSON string with operation result.
        """
        try:
            if action == "read":
                return await self._read(key)
            elif action == "write":
                return await self._write(key, value)
            elif action == "append":
                return await self._append(key, value)
            elif action == "list":
                return await self._list_keys()
            elif action == "clear":
                return await self._clear(key)
            else:
                return json.dumps({
                    "error": f"Unknown action: {action}",
                    "valid_actions": ["read", "write", "append", "list", "clear"],
                })

        except Exception as e:
            logger.error(f"Memory operation error: {e}")
            return json.dumps({
                "error": str(e),
                "action": action,
                "key": key,
            })

    async def _read(self, key: str) -> str:
        """Read a memory key."""
        file_path = self._get_file_path(key)

        if not file_path.exists():
            # Try undated version
            file_path = self._get_file_path(key, dated=False)

        if not file_path.exists():
            return json.dumps({
                "key": key,
                "value": None,
                "exists": False,
            })

        with open(file_path) as f:
            data = json.load(f)

        return json.dumps({
            "key": key,
            "value": data,
            "exists": True,
            "file": str(file_path),
        })

    async def _write(self, key: str, value: Optional[str]) -> str:
        """Write a memory key (overwrites existing)."""
        if value is None:
            return json.dumps({
                "error": "Value required for write operation",
                "key": key,
            })

        file_path = self._get_file_path(key)

        # Parse value as JSON if possible
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            data = {"value": value}

        # Add metadata
        data["_updated"] = datetime.now().isoformat()

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Memory written: {key} -> {file_path}")

        return json.dumps({
            "success": True,
            "key": key,
            "file": str(file_path),
        })

    async def _append(self, key: str, value: Optional[str]) -> str:
        """Append to a memory key (for lists)."""
        if value is None:
            return json.dumps({
                "error": "Value required for append operation",
                "key": key,
            })

        file_path = self._get_file_path(key)

        # Load existing data
        if file_path.exists():
            with open(file_path) as f:
                data = json.load(f)
        else:
            data = {"items": []}

        # Parse new value
        try:
            new_item = json.loads(value)
        except json.JSONDecodeError:
            new_item = {"value": value}

        # Add timestamp
        new_item["_timestamp"] = datetime.now().isoformat()

        # Append to items list
        if "items" not in data:
            data["items"] = []
        data["items"].append(new_item)
        data["_updated"] = datetime.now().isoformat()

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Memory appended: {key} (now {len(data['items'])} items)")

        return json.dumps({
            "success": True,
            "key": key,
            "item_count": len(data["items"]),
            "file": str(file_path),
        })

    async def _list_keys(self) -> str:
        """List all memory keys."""
        files = list(self.memory_dir.glob("*.json"))

        keys = []
        for f in files:
            name = f.stem
            # Remove date suffix if present
            parts = name.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) == 10:  # YYYY-MM-DD
                key = parts[0]
                date = parts[1]
            else:
                key = name
                date = None

            keys.append({
                "key": key,
                "date": date,
                "file": f.name,
                "size": f.stat().st_size,
            })

        return json.dumps({
            "keys": keys,
            "total": len(keys),
            "memory_dir": str(self.memory_dir),
        })

    async def _clear(self, key: str) -> str:
        """Clear a memory key (delete file)."""
        file_path = self._get_file_path(key)

        if file_path.exists():
            file_path.unlink()
            logger.info(f"Memory cleared: {key}")
            return json.dumps({
                "success": True,
                "key": key,
                "deleted": str(file_path),
            })
        else:
            return json.dumps({
                "success": False,
                "key": key,
                "message": "Key not found",
            })

    # Convenience methods for common operations

    async def log_signal(
        self,
        symbol: str,
        signal: str,
        probability: float,
        price: float,
    ) -> str:
        """Log a generated signal."""
        value = json.dumps({
            "symbol": symbol,
            "signal": signal,
            "probability": probability,
            "price": price,
        })
        return await self._append("signals", value)

    async def log_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_id: str,
        status: str,
    ) -> str:
        """Log an executed order."""
        value = json.dumps({
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "order_id": order_id,
            "status": status,
        })
        return await self._append("orders", value)

    async def log_pnl(
        self,
        equity: float,
        daily_pnl: float,
        daily_pnl_pct: float,
    ) -> str:
        """Log daily P&L snapshot."""
        value = json.dumps({
            "equity": equity,
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": daily_pnl_pct,
        })
        return await self._append("pnl", value)
