"""Tests for agent.tools.signal_generator module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.signal_generator import SignalGeneratorTool


class TestSignalGeneratorToolInit:
    """Tests for SignalGeneratorTool initialization."""

    def test_tool_metadata(self):
        """Test tool name and description."""
        with patch("agent.tools.signal_generator.FeatureCalculator"):
            tool = SignalGeneratorTool()

            assert tool.name == "generate_signals"
            assert "trading signals" in tool.description.lower()

    def test_input_schema(self):
        """Test input schema has required fields."""
        with patch("agent.tools.signal_generator.FeatureCalculator"):
            tool = SignalGeneratorTool()

            assert "bar_data" in tool.input_schema["properties"]
            assert "prob_min" in tool.input_schema["properties"]
            assert "signal_type" in tool.input_schema["properties"]


class TestSignalGeneratorToolLoadModel:
    """Tests for load_model method."""

    def test_load_model_success(self, mock_xgb_model):
        """Test loading a model successfully."""
        with patch("agent.tools.signal_generator.FeatureCalculator"):
            with patch("agent.tools.signal_generator.ModelLoader") as mock_loader_cls:
                mock_loader = MagicMock()
                mock_loader.is_loaded = True
                mock_loader_cls.return_value = mock_loader

                tool = SignalGeneratorTool()
                tool.load_model(str(mock_xgb_model["run_dir"]), "xgb")

                mock_loader_cls.assert_called_once()
                mock_loader.load.assert_called_once()


class TestSignalGeneratorToolExecute:
    """Tests for execute method."""

    @pytest.fixture
    def mock_tool(self):
        """Create a mock signal generator tool."""
        with patch("agent.tools.signal_generator.FeatureCalculator") as mock_fc:
            mock_feature_calc = MagicMock()
            mock_fc.return_value = mock_feature_calc

            tool = SignalGeneratorTool()

            # Mock model loader
            mock_loader = MagicMock()
            mock_loader.is_loaded = True
            mock_loader.predict_latest.return_value = {
                "prediction": 1,
                "probability": 0.72,
            }
            tool._model_loader = mock_loader
            tool._feature_calculator = mock_feature_calc

            yield tool

    @pytest.mark.asyncio
    async def test_execute_without_model_loaded(self):
        """Test execute returns error when model not loaded."""
        with patch("agent.tools.signal_generator.FeatureCalculator"):
            tool = SignalGeneratorTool()
            # Model not loaded

            result = await tool.execute(
                bar_data='{"AAPL": {"bars": []}}',
                prob_min=0.55,
            )

            data = json.loads(result)
            assert "error" in data
            assert "Model not loaded" in data["error"]

    @pytest.mark.asyncio
    async def test_execute_generates_long_signal(self, mock_tool):
        """Test generating a long signal."""
        bar_data = json.dumps({
            "AAPL": {
                "bars": [
                    {"datetime": "2024-01-15T10:00:00", "open": 150, "high": 152,
                     "low": 149, "close": 151, "volume": 100000}
                ],
                "latest_close": 151.0,
                "latest_time": "2024-01-15T10:00:00",
            }
        })

        result = await mock_tool.execute(
            bar_data=bar_data,
            prob_min=0.55,
            signal_type="long_only",
        )

        data = json.loads(result)
        assert "signals" in data
        assert "AAPL" in data["signals"]
        assert data["signals"]["AAPL"]["signal"] == "long"

    @pytest.mark.asyncio
    async def test_execute_no_signal_below_threshold(self, mock_tool):
        """Test no signal when probability below threshold."""
        mock_tool._model_loader.predict_latest.return_value = {
            "prediction": 1,
            "probability": 0.52,  # Below 0.55 threshold
        }

        bar_data = json.dumps({
            "AAPL": {
                "bars": [{"datetime": "2024-01-15T10:00:00", "open": 150, "high": 152,
                         "low": 149, "close": 151, "volume": 100000}],
                "latest_close": 151.0,
            }
        })

        result = await mock_tool.execute(
            bar_data=bar_data,
            prob_min=0.55,
            signal_type="long_only",
        )

        data = json.loads(result)
        assert data["signals"]["AAPL"]["signal"] == "none"

    @pytest.mark.asyncio
    async def test_execute_short_signal(self, mock_tool):
        """Test generating a short signal."""
        mock_tool._model_loader.predict_latest.return_value = {
            "prediction": 0,
            "probability": 0.30,  # 70% probability of being wrong = 70% short
        }

        bar_data = json.dumps({
            "AAPL": {
                "bars": [{"datetime": "2024-01-15T10:00:00", "open": 150, "high": 152,
                         "low": 149, "close": 151, "volume": 100000}],
                "latest_close": 151.0,
            }
        })

        result = await mock_tool.execute(
            bar_data=bar_data,
            prob_min=0.55,
            signal_type="both",
        )

        data = json.loads(result)
        assert data["signals"]["AAPL"]["signal"] == "short"

    @pytest.mark.asyncio
    async def test_execute_long_only_no_short(self, mock_tool):
        """Test long_only signal type ignores short signals."""
        mock_tool._model_loader.predict_latest.return_value = {
            "prediction": 0,
            "probability": 0.30,
        }

        bar_data = json.dumps({
            "AAPL": {
                "bars": [{"datetime": "2024-01-15T10:00:00", "open": 150, "high": 152,
                         "low": 149, "close": 151, "volume": 100000}],
            }
        })

        result = await mock_tool.execute(
            bar_data=bar_data,
            prob_min=0.55,
            signal_type="long_only",
        )

        data = json.loads(result)
        assert data["signals"]["AAPL"]["signal"] == "none"

    @pytest.mark.asyncio
    async def test_execute_multiple_symbols(self, mock_tool):
        """Test generating signals for multiple symbols."""
        bar_data = json.dumps({
            "AAPL": {
                "bars": [{"datetime": "2024-01-15T10:00:00", "open": 150, "high": 152,
                         "low": 149, "close": 151, "volume": 100000}],
                "latest_close": 151.0,
            },
            "TSLA": {
                "bars": [{"datetime": "2024-01-15T10:00:00", "open": 250, "high": 255,
                         "low": 248, "close": 252, "volume": 200000}],
                "latest_close": 252.0,
            },
        })

        result = await mock_tool.execute(
            bar_data=bar_data,
            prob_min=0.55,
        )

        data = json.loads(result)
        assert "AAPL" in data["signals"]
        assert "TSLA" in data["signals"]
        assert data["summary"]["total_symbols"] == 2

    @pytest.mark.asyncio
    async def test_execute_skips_metadata_keys(self, mock_tool):
        """Test that metadata keys starting with _ are skipped."""
        bar_data = json.dumps({
            "AAPL": {
                "bars": [{"datetime": "2024-01-15T10:00:00", "open": 150, "high": 152,
                         "low": 149, "close": 151, "volume": 100000}],
            },
            "_summary": {
                "total": 1,
            },
        })

        result = await mock_tool.execute(
            bar_data=bar_data,
            prob_min=0.55,
        )

        data = json.loads(result)
        assert "_summary" not in data["signals"]
        assert "AAPL" in data["signals"]

    @pytest.mark.asyncio
    async def test_execute_returns_actionable_signals(self, mock_tool):
        """Test that actionable signals are filtered."""
        bar_data = json.dumps({
            "AAPL": {
                "bars": [{"datetime": "2024-01-15T10:00:00", "open": 150, "high": 152,
                         "low": 149, "close": 151, "volume": 100000}],
            },
        })

        result = await mock_tool.execute(
            bar_data=bar_data,
            prob_min=0.55,
        )

        data = json.loads(result)
        assert "actionable" in data
        # AAPL should be in actionable since it has a long signal
        assert "AAPL" in data["actionable"]

    @pytest.mark.asyncio
    async def test_execute_includes_summary(self, mock_tool):
        """Test that response includes summary."""
        bar_data = json.dumps({
            "AAPL": {
                "bars": [{"datetime": "2024-01-15T10:00:00", "open": 150, "high": 152,
                         "low": 149, "close": 151, "volume": 100000}],
            },
        })

        result = await mock_tool.execute(
            bar_data=bar_data,
            prob_min=0.55,
        )

        data = json.loads(result)
        assert "summary" in data
        assert "total_symbols" in data["summary"]
        assert "long_signals" in data["summary"]
        assert "short_signals" in data["summary"]
        assert "prob_threshold" in data["summary"]

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, mock_tool):
        """Test error handling when processing fails."""
        result = await mock_tool.execute(
            bar_data="invalid json",
            prob_min=0.55,
        )

        data = json.loads(result)
        assert "error" in data


class TestDetermineSignal:
    """Tests for _determine_signal method."""

    @pytest.fixture
    def tool(self):
        """Create a signal generator tool."""
        with patch("agent.tools.signal_generator.FeatureCalculator"):
            return SignalGeneratorTool()

    def test_none_prediction(self, tool):
        """Test with None prediction."""
        result = tool._determine_signal(
            prediction=None,
            probability=0.72,
            prob_min=0.55,
            signal_type="both",
        )
        assert result == "none"

    def test_none_probability(self, tool):
        """Test with None probability."""
        result = tool._determine_signal(
            prediction=1,
            probability=None,
            prob_min=0.55,
            signal_type="both",
        )
        assert result == "none"

    def test_long_signal(self, tool):
        """Test long signal generation."""
        result = tool._determine_signal(
            prediction=1,
            probability=0.72,
            prob_min=0.55,
            signal_type="long_only",
        )
        assert result == "long"

    def test_short_signal(self, tool):
        """Test short signal generation."""
        result = tool._determine_signal(
            prediction=0,
            probability=0.30,  # 1 - 0.30 = 0.70 > 0.55
            prob_min=0.55,
            signal_type="short_only",
        )
        assert result == "short"

    def test_below_threshold(self, tool):
        """Test signal below threshold."""
        result = tool._determine_signal(
            prediction=1,
            probability=0.50,  # Below threshold
            prob_min=0.55,
            signal_type="both",
        )
        assert result == "none"
