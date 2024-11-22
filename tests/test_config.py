"""Tests for configuration module."""

from nxtint.utils.config import Config, GenConfig


def test_config_override():
    """Test that configuration overrides work."""
    # Override a value
    original_value = Config.gen.max_int
    with GenConfig.override(max_int=10):
        assert Config.gen.max_int == 10

    # Check that the original value is restored
    assert Config.gen.max_int == original_value

    return
