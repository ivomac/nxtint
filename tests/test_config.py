"""Tests for configuration module."""

from nxtint.utils.config import Config, GenConfig


def test_config_override():
    """Test that configuration overrides work."""
    # Save original value
    original_value = Config.gen.max_int

    # Override a value
    with GenConfig.override(max_int=10):
        assert Config.gen.max_int == 10

    # Check that the original value is restored
    assert Config.gen.max_int == original_value


def test_config_set():
    """Test that Config set method works."""
    # Save original value
    original_value = Config.model.x_len

    # Override a value
    overrides = {
        "model": {
            "x_len": 1000,
        },
        "gen": {
            "max_int": 10,
        },
    }
    with Config.override(**overrides):
        assert Config.model.x_len == 1000
        assert Config.gen.max_int == 10

    # Check that the original value is restored
    assert Config.model.x_len == original_value
