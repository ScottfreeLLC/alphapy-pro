"""Tests for agent.tools.base module."""

import pytest

from agent.tools.base import Tool


class TestTool:
    """Tests for Tool base class."""

    def test_cannot_instantiate_directly(self):
        """Test that Tool cannot be instantiated without required fields."""
        with pytest.raises(TypeError):
            Tool()

    def test_create_with_fields(self):
        """Test creating a tool with required fields."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.input_schema == {"type": "object", "properties": {}}

    def test_to_dict(self):
        """Test to_dict method returns Claude API format."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "param": {"type": "string"},
                },
            },
        )

        result = tool.to_dict()

        assert result["name"] == "test_tool"
        assert result["description"] == "A test tool"
        assert "input_schema" in result
        assert result["input_schema"]["properties"]["param"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_execute_not_implemented(self):
        """Test execute raises NotImplementedError."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            input_schema={},
        )

        with pytest.raises(NotImplementedError, match="must implement execute"):
            await tool.execute()
