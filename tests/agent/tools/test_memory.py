"""Tests for agent.tools.memory module."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from agent.tools.memory import MemoryTool


class TestMemoryToolInit:
    """Tests for MemoryTool initialization."""

    def test_tool_metadata(self, memory_temp_dir):
        """Test tool name and description."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        assert tool.name == "agent_memory"
        assert "persistent state" in tool.description.lower()

    def test_creates_memory_directory(self, tmp_path):
        """Test that memory directory is created."""
        memory_dir = tmp_path / "new_memory"
        assert not memory_dir.exists()

        tool = MemoryTool(memory_dir=memory_dir)

        assert memory_dir.exists()

    def test_input_schema(self, memory_temp_dir):
        """Test input schema has required fields."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        assert "action" in tool.input_schema["properties"]
        assert "key" in tool.input_schema["properties"]
        assert "value" in tool.input_schema["properties"]


class TestMemoryToolWrite:
    """Tests for write operation."""

    @pytest.mark.asyncio
    async def test_write_json_value(self, memory_temp_dir):
        """Test writing a JSON value."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        result = await tool.execute(
            action="write",
            key="test_key",
            value='{"data": "test_value"}',
        )

        data = json.loads(result)
        assert data["success"] is True
        assert data["key"] == "test_key"

    @pytest.mark.asyncio
    async def test_write_string_value(self, memory_temp_dir):
        """Test writing a plain string value."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        result = await tool.execute(
            action="write",
            key="test_key",
            value="plain_string",
        )

        data = json.loads(result)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_write_requires_value(self, memory_temp_dir):
        """Test write requires value parameter."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        result = await tool.execute(
            action="write",
            key="test_key",
            value=None,
        )

        data = json.loads(result)
        assert "error" in data
        assert "Value required" in data["error"]

    @pytest.mark.asyncio
    async def test_write_adds_timestamp(self, memory_temp_dir):
        """Test write adds _updated timestamp."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        await tool.execute(
            action="write",
            key="test_key",
            value='{"data": "value"}',
        )

        # Read the file directly
        file_path = tool._get_file_path("test_key")
        with open(file_path) as f:
            data = json.load(f)

        assert "_updated" in data


class TestMemoryToolRead:
    """Tests for read operation."""

    @pytest.mark.asyncio
    async def test_read_existing_key(self, memory_temp_dir):
        """Test reading an existing key."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        # Write first
        await tool.execute(
            action="write",
            key="test_key",
            value='{"data": "value"}',
        )

        # Then read
        result = await tool.execute(
            action="read",
            key="test_key",
        )

        data = json.loads(result)
        assert data["exists"] is True
        assert data["value"]["data"] == "value"

    @pytest.mark.asyncio
    async def test_read_nonexistent_key(self, memory_temp_dir):
        """Test reading a nonexistent key."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        result = await tool.execute(
            action="read",
            key="nonexistent",
        )

        data = json.loads(result)
        assert data["exists"] is False
        assert data["value"] is None


class TestMemoryToolAppend:
    """Tests for append operation."""

    @pytest.mark.asyncio
    async def test_append_to_new_key(self, memory_temp_dir):
        """Test appending to a new key."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        result = await tool.execute(
            action="append",
            key="signals",
            value='{"symbol": "AAPL", "signal": "long"}',
        )

        data = json.loads(result)
        assert data["success"] is True
        assert data["item_count"] == 1

    @pytest.mark.asyncio
    async def test_append_multiple_items(self, memory_temp_dir):
        """Test appending multiple items."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        await tool.execute(
            action="append",
            key="signals",
            value='{"symbol": "AAPL"}',
        )

        await tool.execute(
            action="append",
            key="signals",
            value='{"symbol": "TSLA"}',
        )

        result = await tool.execute(
            action="append",
            key="signals",
            value='{"symbol": "MSFT"}',
        )

        data = json.loads(result)
        assert data["item_count"] == 3

    @pytest.mark.asyncio
    async def test_append_adds_timestamp(self, memory_temp_dir):
        """Test append adds _timestamp to each item."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        await tool.execute(
            action="append",
            key="signals",
            value='{"symbol": "AAPL"}',
        )

        # Read file directly
        file_path = tool._get_file_path("signals")
        with open(file_path) as f:
            data = json.load(f)

        assert "_timestamp" in data["items"][0]

    @pytest.mark.asyncio
    async def test_append_requires_value(self, memory_temp_dir):
        """Test append requires value parameter."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        result = await tool.execute(
            action="append",
            key="signals",
            value=None,
        )

        data = json.loads(result)
        assert "error" in data


class TestMemoryToolList:
    """Tests for list operation."""

    @pytest.mark.asyncio
    async def test_list_empty(self, memory_temp_dir):
        """Test listing with no keys."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        result = await tool.execute(
            action="list",
            key="",
        )

        data = json.loads(result)
        assert data["total"] == 0
        assert data["keys"] == []

    @pytest.mark.asyncio
    async def test_list_with_keys(self, memory_temp_dir):
        """Test listing with existing keys."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        # Create some keys
        await tool.execute(action="write", key="signals", value='{"data": 1}')
        await tool.execute(action="write", key="orders", value='{"data": 2}')

        result = await tool.execute(
            action="list",
            key="",
        )

        data = json.loads(result)
        assert data["total"] >= 2
        key_names = [k["key"] for k in data["keys"]]
        assert "signals" in key_names
        assert "orders" in key_names


class TestMemoryToolClear:
    """Tests for clear operation."""

    @pytest.mark.asyncio
    async def test_clear_existing_key(self, memory_temp_dir):
        """Test clearing an existing key."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        # Write first
        await tool.execute(action="write", key="test_key", value='{"data": 1}')

        # Then clear
        result = await tool.execute(
            action="clear",
            key="test_key",
        )

        data = json.loads(result)
        assert data["success"] is True

        # Verify file is deleted
        file_path = tool._get_file_path("test_key")
        assert not file_path.exists()

    @pytest.mark.asyncio
    async def test_clear_nonexistent_key(self, memory_temp_dir):
        """Test clearing a nonexistent key."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        result = await tool.execute(
            action="clear",
            key="nonexistent",
        )

        data = json.loads(result)
        assert data["success"] is False
        assert "not found" in data["message"].lower()


class TestMemoryToolUnknownAction:
    """Tests for unknown action handling."""

    @pytest.mark.asyncio
    async def test_unknown_action(self, memory_temp_dir):
        """Test handling of unknown action."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        result = await tool.execute(
            action="invalid",
            key="test",
        )

        data = json.loads(result)
        assert "error" in data
        assert "Unknown action" in data["error"]
        assert "valid_actions" in data


class TestMemoryToolConvenienceMethods:
    """Tests for convenience methods."""

    @pytest.mark.asyncio
    async def test_log_signal(self, memory_temp_dir):
        """Test log_signal convenience method."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        result = await tool.log_signal(
            symbol="AAPL",
            signal="long",
            probability=0.72,
            price=150.0,
        )

        data = json.loads(result)
        assert data["success"] is True
        assert data["key"] == "signals"

    @pytest.mark.asyncio
    async def test_log_order(self, memory_temp_dir):
        """Test log_order convenience method."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        result = await tool.log_order(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=150.0,
            order_id="test-123",
            status="filled",
        )

        data = json.loads(result)
        assert data["success"] is True
        assert data["key"] == "orders"

    @pytest.mark.asyncio
    async def test_log_pnl(self, memory_temp_dir):
        """Test log_pnl convenience method."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        result = await tool.log_pnl(
            equity=100500.0,
            daily_pnl=500.0,
            daily_pnl_pct=0.5,
        )

        data = json.loads(result)
        assert data["success"] is True
        assert data["key"] == "pnl"


class TestMemoryToolFilePath:
    """Tests for file path generation."""

    def test_get_file_path_dated(self, memory_temp_dir):
        """Test dated file path generation."""
        tool = MemoryTool(memory_dir=memory_temp_dir)
        date_str = datetime.now().strftime("%Y-%m-%d")

        path = tool._get_file_path("signals", dated=True)

        assert f"signals_{date_str}.json" in str(path)

    def test_get_file_path_undated(self, memory_temp_dir):
        """Test undated file path generation."""
        tool = MemoryTool(memory_dir=memory_temp_dir)

        path = tool._get_file_path("config", dated=False)

        assert path.name == "config.json"
