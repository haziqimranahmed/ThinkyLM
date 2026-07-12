"""
ThinkyLM — Configuration Tests
"""

import pytest
from thinkylm.config import ModelConfig, ThinkyLMConfig, TrainingConfig


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.vocab_size == 4_000
    assert cfg.hidden_size == 128
    assert cfg.num_layers == 3
    assert cfg.num_heads == 4
    assert cfg.head_dim == 32  # 128 // 4


def test_model_config_validation_heads():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(hidden_size=100, num_heads=3)  # 100 % 3 != 0


def test_model_config_validation_layers():
    with pytest.raises(ValueError):
        ModelConfig(num_layers=0)


def test_model_config_validation_vocab():
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=5)


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.batch_size == 2
    assert cfg.num_workers == 0
    assert cfg.pin_memory is False
    assert cfg.persistent_workers is False


def test_training_config_validation():
    with pytest.raises(ValueError):
        TrainingConfig(batch_size=0)
    with pytest.raises(ValueError):
        TrainingConfig(learning_rate=-0.001)


def test_config_estimate_params():
    cfg = ThinkyLMConfig()
    params = cfg.estimate_params()
    # Should be at least 100K for our tiny debug config
    assert params > 100_000
    assert params < 10_000_000  # Must be under local safety limit


def test_config_from_yaml(tmp_path):
    import yaml

    config = {
        "name": "test",
        "model": {
            "vocab_size": 100,
            "context_length": 32,
            "hidden_size": 64,
            "num_layers": 2,
            "num_heads": 4,
            "intermediate_size": 128,
            "dropout": 0.0,
            "tie_embeddings": True,
        },
        "training": {"batch_size": 1, "max_steps": 5},
    }
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(yaml.dump(config))

    loaded = ThinkyLMConfig.from_yaml(cfg_file)
    assert loaded.model.vocab_size == 100
    assert loaded.model.hidden_size == 64
    assert loaded.training.batch_size == 1


def test_config_from_yaml_missing_file():
    with pytest.raises(FileNotFoundError):
        ThinkyLMConfig.from_yaml("nonexistent_config.yaml")
