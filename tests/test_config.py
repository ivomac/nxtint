"""Tests for configuration module."""

from pathlib import Path
from tempfile import TemporaryDirectory

from nxtint.utils.config import Config, GenConfig


def test_config_instance():
    """Test that configuration instances cannot be created."""
    try:
        Config()
    except TypeError:
        pass
    else:
        raise AssertionError("Config instance created")


def test_config_setattr():
    """Test that configuration attributes cannot be set."""
    try:
        Config.model.x_len = 10
    except AttributeError:
        pass
    else:
        raise AssertionError("Config attribute set")


def test_config_override():
    """Test that configuration overrides work."""
    # Save original value
    original_value = Config.gen.max_int

    # Override a value
    with GenConfig.override(max_int=10):
        assert Config.gen.max_int == 10

    # Check that the original value is restored
    assert Config.gen.max_int == original_value


def test_config_nonexistent():
    """Test that non-existent attributes cannot be overridden."""
    try:
        with GenConfig.override(none12=12):
            assert True
    except AttributeError:
        pass
    else:
        raise AssertionError("Non-existent attribute overridden")


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


def test_config_init_dirs():
    """Test that Config initdirs method works."""
    # Save original value
    original_value = Config.save.dir

    log_dir = TemporaryDirectory()
    save_dir = TemporaryDirectory()

    # Override a value
    with Config.override(
        log={"dir": log_dir.name},
        save={"dir": save_dir.name},
    ):
        Config.init_dirs()
        assert Path(Config.save.dir).exists()
        assert Path(Config.log.dir).exists()

    # Check that the original value is restored
    assert Config.save.dir == original_value
