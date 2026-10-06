"""Exercise schema validation, safe diagnostics and legacy resolution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from agent.config import Settings
from agent.model_config import (
    ModelConfigError,
    ModelsConfig,
    ProviderSpec,
    ReasoningSpec,
    check_main,
    load_models,
)

_EXAMPLE = Path(__file__).resolve().parents[2] / "config" / "models.example.yaml"
_SECRET = "super-secret-api-key-that-must-not-leak"


def _data() -> dict[str, Any]:
    return {
        "providers": {
            "gateway": {
                "kind": "openai",
                "base_url": "https://example.test/v1",
                "api_key_env": "MODEL_KEY",
            },
        },
        "models": {
            "fast": {
                "provider": "gateway",
                "model": "upstream-model",
                "context_window": 32768,
                "max_tokens": 8192,
            },
        },
        "routing": {"default": "fast"},
    }


def _write(tmp_path: Path, data: dict[str, Any]) -> Path:
    filename = tmp_path / "models.yaml"
    filename.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return filename


def test_example_loads_and_resolves_secrets() -> None:
    registry = load_models(_EXAMPLE, {"AGENT_LLM_API_KEY": _SECRET}, None)
    assert registry.source == "yaml"
    assert registry.default_id == "main"
    assert registry.ids() == ["main", "light"]
    assert registry.fallback == ["light"]
    assert registry.roles == {}
    model = registry.get("main")
    assert model.provider_id == "gateway"
    assert model.kind == "openai"
    assert model.api_key == _SECRET
    assert model.configured
    assert model.context_window == 131070
    assert model.max_tokens == 16000
    assert model.timeout_seconds == 120
    assert model.max_retries == 3
    assert model.reasoning.allowed_efforts == ["none", "high"]
    assert model.reasoning.default_effort == "high"
    assert model.reasoning.wire_parameter == "reasoning_effort"
    light = registry.get("light")
    assert light.model == "gemma-4-26b"
    assert not light.reasoning.supported
    assert registry.warnings == []
    assert _SECRET not in repr(model) + repr(registry)
    with pytest.raises(FrozenInstanceError):
        model.max_tokens = 10
    with pytest.raises(KeyError, match="Unknown model ID.*missing"):
        registry.get("missing")


def test_unknown_field_has_location_and_no_secret(tmp_path: Path) -> None:
    data = _data()
    data["providers"]["gateway"]["api_key"] = _SECRET
    filename = _write(tmp_path, data)
    with pytest.raises(ModelConfigError) as caught:
        load_models(filename, {"MODEL_KEY": _SECRET}, None)
    assert f"{filename}:" in str(caught.value)
    assert "providers.gateway.api_key" in str(caught.value)
    assert _SECRET not in str(caught.value) + repr(caught.value)


def test_duplicate_key_has_exact_line_and_field(tmp_path: Path) -> None:
    filename = tmp_path / "duplicate.yaml"
    filename.write_text("routing:\n  default: fast\n  default: private-secret\n")
    with pytest.raises(ModelConfigError, match=r":3:3: routing.default: duplicate mapping key"):
        load_models(filename, {}, None)


@pytest.mark.parametrize(
    ("change", "field_path"),
    [
        (lambda data: data["models"]["fast"].update(provider="missing"), "models.fast.provider"),
        (lambda data: data["routing"].update(default="missing"), "routing.default"),
        (lambda data: data["routing"].update(fallback=["gateway"]), "routing.fallback[0]"),
        (lambda data: data["routing"].update(roles={"review": "missing"}), "routing.roles.review"),
        (lambda data: data.update(decisions={"provider": "missing"}), "decisions.provider"),
    ],
)
def test_broken_reference_has_field_path(
    tmp_path: Path, change: Callable[[dict[str, Any]], None], field_path: str
) -> None:
    data = _data()
    change(data)
    with pytest.raises(ModelConfigError) as caught:
        load_models(_write(tmp_path, data), {"MODEL_KEY": _SECRET}, None)
    assert field_path in str(caught.value)
    assert _SECRET not in str(caught.value)


@pytest.mark.parametrize("env", [{}, {"MODEL_KEY": ""}])
def test_missing_secret_leaves_provider_unconfigured(tmp_path: Path, env: dict[str, str]) -> None:
    registry = load_models(_write(tmp_path, _data()), env, None)
    assert not registry.get("fast").configured
    assert any("providers.gateway.api_key_env" in warning for warning in registry.warnings)


def test_keyless_provider_is_configured(tmp_path: Path) -> None:
    data = _data()
    data["providers"]["gateway"]["api_key_env"] = None
    registry = load_models(_write(tmp_path, data), {}, None)
    assert registry.get("fast").api_key == ""
    assert registry.get("fast").configured


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "127.0.0.1",
        "[::1]",
        "host.docker.internal",
        "LOCALHOST",
    ],
)
def test_local_http_has_no_warning(tmp_path: Path, host: str) -> None:
    data = _data()
    data["providers"]["gateway"]["base_url"] = f"http://{host}:11434/v1"
    assert not load_models(_write(tmp_path, data), {"MODEL_KEY": _SECRET}, None).warnings


def test_http_warning_and_strict_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = _data()
    data["providers"]["gateway"]["base_url"] = "http://remote.example/v1"
    filename = _write(tmp_path, data)
    registry = load_models(filename, {"MODEL_KEY": _SECRET}, None)
    assert len(registry.warnings) == 1
    assert "providers.gateway.base_url" in registry.warnings[0]
    monkeypatch.setenv("MODEL_KEY", _SECRET)
    assert check_main(["--file", str(filename)]) == 0
    assert check_main(["--file", str(filename), "--strict"]) == 2
    output = capsys.readouterr()
    assert "OK:" in output.out
    assert "WARNING:" in output.err
    assert _SECRET not in output.out + output.err


@pytest.mark.parametrize("api_key", ["", _SECRET])
def test_legacy_single_preserves_fields_and_configured(tmp_path: Path, api_key: str) -> None:
    settings = Settings(
        llm_base_url="https://legacy.example/v1",
        llm_api_key=api_key,
        llm_model="legacy-model",
        llm_timeout_seconds=45,
        llm_max_retries=4,
        llm_max_tokens=6000,
        context_window=65536,
    )
    registry = load_models(tmp_path / "absent.yaml", {}, settings)
    model = registry.get("default")
    assert registry.source == "legacy"
    assert registry.default_id == "default"
    assert registry.ids() == ["default"]
    assert registry.fallback == [] and registry.roles == {}
    assert not registry.warnings
    assert model.base_url == settings.resolve_llm_profile().base_url
    assert model.api_key == api_key
    assert model.configured == bool(api_key)
    assert model.model == "legacy-model"
    assert model.timeout_seconds == 45 and model.max_retries == 4
    assert model.max_tokens == 6000 and model.context_window == 65536
    assert not model.reasoning.supported and model.pricing is None
    assert _SECRET not in repr(registry)


def test_legacy_profiles_override_globals_and_warn() -> None:
    settings = Settings(
        llm_model="ignored-global",
        default_llm_profile="deep",
        context_window=65536,
        llm_profiles={
            "fast": {
                "base_url": "https://fast.example/v1",
                "api_key": "",
                "model": "fast-model",
                "max_tokens": 4000,
            },
            "deep": {
                "base_url": "https://deep.example/v1",
                "api_key": _SECRET,
                "model": "deep-model",
                "timeout_seconds": 90,
                "max_retries": 5,
                "max_tokens": 16000,
            },
        },
    )
    registry = load_models(None, {}, settings)
    assert registry.default_id == "deep"
    assert registry.ids() == ["fast", "deep"]
    assert not registry.get("fast").configured
    assert registry.get("deep").configured
    for name in registry.ids():
        model = registry.get(name)
        profile = settings.resolve_llm_profile(name)
        assert (
            model.base_url,
            model.api_key,
            model.model,
            model.timeout_seconds,
            model.max_retries,
            model.max_tokens,
        ) == (
            profile.base_url,
            profile.api_key,
            profile.model,
            profile.timeout_seconds,
            profile.max_retries,
            profile.max_tokens,
        )
        assert model.context_window == settings.context_window
    assert "deprecated" in registry.warnings[0]
    assert _SECRET not in repr(registry)


def test_legacy_unknown_default_is_rejected() -> None:
    with pytest.raises(ModelConfigError, match="routing.default.*unknown model"):
        load_models(None, {}, Settings(default_llm_profile="missing"))


def test_decisions_enabled_is_rejected(tmp_path: Path) -> None:
    data = _data()
    data["decisions"] = {"enabled": True}
    with pytest.raises(ModelConfigError, match="decisions.enabled.*unsupported in this phase"):
        load_models(_write(tmp_path, data), {"MODEL_KEY": _SECRET}, None)


@pytest.mark.parametrize("kind", ["jev", "anthropic"])
@pytest.mark.parametrize("routing", ["default", "fallback", "roles"])
def test_unsupported_transport_selection_is_rejected(
    tmp_path: Path, kind: str, routing: str
) -> None:
    data = _data()
    data["providers"]["future"] = {"kind": kind, "base_url": "https://future.example/v1"}
    data["models"]["future"] = {
        "provider": "future",
        "model": "future-model",
        "context_window": 4096,
        "max_tokens": 1000,
    }
    declared = load_models(_write(tmp_path, data), {"MODEL_KEY": _SECRET}, None)
    assert declared.get("future").kind == kind
    data["routing"][routing] = {
        "default": "future",
        "fallback": ["future"],
        "roles": {"review": "future"},
    }[routing]
    with pytest.raises(ModelConfigError, match=f"routing.{routing}.*unsupported transport"):
        load_models(_write(tmp_path, data), {"MODEL_KEY": _SECRET}, None)


def test_checksum_tracks_file_bytes_not_credentials(tmp_path: Path) -> None:
    filename = _write(tmp_path, _data())
    first = load_models(filename, {"MODEL_KEY": _SECRET}, None)
    assert first.checksum == load_models(filename, {"MODEL_KEY": "different"}, None).checksum
    filename.write_text(filename.read_text() + "\n# изменён файл\n")
    assert first.checksum != load_models(filename, {"MODEL_KEY": _SECRET}, None).checksum
    assert len(first.checksum) == 64


@pytest.mark.parametrize(
    "url",
    [
        f"https://user:{_SECRET}@example.test/v1",
        f"https://example.test/v1?token={_SECRET}",
        f"https://example.test/v1#{_SECRET}",
        "https://example.test/v1?",
        "https://example.test/v1#",
        "file:///tmp/model",
        "https://example.test:bad/v1",
        "https://example.test\n/v1",
    ],
)
def test_url_errors_never_leak_values(tmp_path: Path, url: str) -> None:
    data = _data()
    data["providers"]["gateway"]["base_url"] = url
    with pytest.raises(ModelConfigError) as caught:
        load_models(_write(tmp_path, data), {"MODEL_KEY": _SECRET}, None)
    assert "providers.gateway.base_url" in str(caught.value)
    assert _SECRET not in str(caught.value) + repr(caught.value)
    with pytest.raises(ValidationError) as direct:
        ProviderSpec(kind="openai", base_url=url)
    assert _SECRET not in str(direct.value) + repr(direct.value)


@pytest.mark.parametrize(
    "reasoning",
    [
        {"allowed_efforts": ["low"]},
        {"supported": True, "allowed_efforts": ["auto"]},
        {"supported": True, "allowed_efforts": ["low"], "default_effort": "high"},
        {"default_effort": "none"},
        {"supported": True, "wire_parameter": "thinking"},
    ],
)
def test_invalid_reasoning_is_rejected(tmp_path: Path, reasoning: dict[str, Any]) -> None:
    data = _data()
    data["models"]["fast"]["reasoning"] = reasoning
    with pytest.raises(ModelConfigError, match="models.fast.reasoning"):
        load_models(_write(tmp_path, data), {"MODEL_KEY": _SECRET}, None)


def test_reasoning_is_frozen_and_all_efforts_are_supported() -> None:
    reasoning = ReasoningSpec(
        supported=True,
        allowed_efforts=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default_effort="max",
    )
    with pytest.raises(ValidationError, match="frozen"):
        reasoning.supported = False
    assert ModelsConfig.model_config["extra"] == "forbid"


@pytest.mark.parametrize("model_id", ["", "bad id", "tab\tid", "a" * 101, "newline\n"])
def test_invalid_model_id_is_rejected(tmp_path: Path, model_id: str) -> None:
    data = _data()
    data["models"][model_id] = data["models"].pop("fast")
    data["routing"]["default"] = model_id
    with pytest.raises(ModelConfigError, match="models"):
        load_models(_write(tmp_path, data), {"MODEL_KEY": _SECRET}, None)


@pytest.mark.parametrize(
    ("change", "field_path"),
    [
        (lambda data: data["providers"]["gateway"].update(timeout_seconds=0), "timeout_seconds"),
        (
            lambda data: data["providers"]["gateway"].update(timeout_seconds=float("inf")),
            "timeout_seconds",
        ),
        (lambda data: data["providers"]["gateway"].update(max_retries=11), "max_retries"),
        (lambda data: data["providers"]["gateway"].update(max_retries=-1), "max_retries"),
        (lambda data: data["models"]["fast"].update(context_window=4095), "context_window"),
        (lambda data: data["models"]["fast"].update(max_tokens=0), "max_tokens"),
        (lambda data: data["models"]["fast"].update(max_tokens=32768), "max_tokens"),
        (
            lambda data: data["models"]["fast"].update(
                pricing={"input_per_mtok": -1, "output_per_mtok": 0}
            ),
            "input_per_mtok",
        ),
    ],
)
def test_numeric_bounds(
    tmp_path: Path, change: Callable[[dict[str, Any]], None], field_path: str
) -> None:
    data = _data()
    change(data)
    with pytest.raises(ModelConfigError, match=field_path):
        load_models(_write(tmp_path, data), {"MODEL_KEY": _SECRET}, None)


@pytest.mark.parametrize(
    "content",
    [
        "!!python/object/apply:os.system ['secret']",
        "routing: [secret",
        "value: &cycle [*cycle]",
        "value: !!unknown secret",
        "true: secret",
        "",
    ],
)
def test_unsafe_or_invalid_yaml_is_rejected(tmp_path: Path, content: str) -> None:
    filename = tmp_path / "invalid.yaml"
    filename.write_text(content)
    with pytest.raises(ModelConfigError) as caught:
        load_models(filename, {}, None)
    assert f"{filename}:" in str(caught.value)
    assert "secret" not in str(caught.value)


def test_yaml_size_limit(tmp_path: Path) -> None:
    filename = tmp_path / "large.yaml"
    filename.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ModelConfigError, match="1 MiB"):
        load_models(filename, {}, None)


def test_yaml_precedence_override_and_legacy_warning(tmp_path: Path) -> None:
    data = _data()
    data["models"]["deep"] = {**data["models"]["fast"], "model": "deep-model"}
    filename = _write(tmp_path, data)
    env = {
        "MODEL_KEY": _SECRET,
        "AGENT_DEFAULT_LLM_PROFILE": "deep",
        "AGENT_LLM_MODEL": "ignored-model",
    }
    registry = load_models(filename, env, Settings())
    assert registry.source == "yaml" and registry.default_id == "deep"
    assert "ignored" in registry.warnings[0]
    env["AGENT_DEFAULT_LLM_PROFILE"] = "missing"
    registry = load_models(filename, env, Settings())
    assert registry.default_id != "missing"
    assert any("AGENT_DEFAULT_LLM_PROFILE" in warning for warning in registry.warnings)


def test_invalid_file_never_falls_back_to_legacy(tmp_path: Path) -> None:
    filename = tmp_path / "invalid.yaml"
    filename.write_text("models: []")
    with pytest.raises(ModelConfigError):
        load_models(filename, {}, Settings())
    with pytest.raises(ModelConfigError, match="no YAML file or legacy settings"):
        load_models(None, {}, None)


def test_cli_invalid_file_exit_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    filename = tmp_path / "missing.yaml"
    assert check_main(["--file", str(filename)]) == 1
    assert "configuration file is missing" in capsys.readouterr().err
    filename.write_text(f"routing: [{_SECRET}")
    assert check_main(["--file", str(filename)]) == 1
    output = capsys.readouterr()
    assert _SECRET not in output.out + output.err


def test_cli_dotenv_and_process_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    filename = _write(tmp_path, _data())
    dotenv = tmp_path / "selected.env"
    dotenv.write_text(f"MODEL_KEY={_SECRET}\nAGENT_MODELS_FILE={filename}\n")
    monkeypatch.setenv("AGENT_ENV_FILE", str(dotenv))
    monkeypatch.delenv("MODEL_KEY", raising=False)
    monkeypatch.delenv("AGENT_MODELS_FILE", raising=False)  # let the selected .env choose it
    assert check_main(["--strict"]) == 0
    # An empty process variable wins over .env, leaving the provider unconfigured (a warning).
    monkeypatch.setenv("MODEL_KEY", "")
    assert check_main(["--strict"]) == 2
    output = capsys.readouterr()
    assert _SECRET not in output.out + output.err


def test_cli_has_no_runtime_imports_or_directory_side_effects(tmp_path: Path) -> None:
    data = _data()
    data["providers"]["gateway"]["api_key_env"] = None
    filename = _write(tmp_path, data)
    data_dir = tmp_path / "never-created"
    code = (
        "import json, sys; from agent.model_config import check_main; "
        f"assert check_main(['--file', {str(filename)!r}]) == 0; "
        "print(json.dumps(sorted(set(sys.modules) & {'agent.database', 'agent.api', "
        "'agent.runtime'})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "AGENT_DATA_DIR": str(data_dir)},
        timeout=15,
    )
    assert json.loads(result.stdout.splitlines()[-1]) == []
    assert not data_dir.exists()


@pytest.mark.parametrize("model_id", ["A.z_0-9", "a" * 100])
def test_valid_model_id_boundaries(tmp_path: Path, model_id: str) -> None:
    data = _data()
    data["models"][model_id] = data["models"].pop("fast")
    data["routing"]["default"] = model_id
    assert load_models(_write(tmp_path, data), {"MODEL_KEY": _SECRET}, None).ids() == [model_id]


def test_alias_expansion_is_bounded(tmp_path: Path) -> None:
    filename = tmp_path / "aliases.yaml"
    content = "level0: &level0 [value]\n"
    for level in range(1, 17):
        content += f"level{level}: &level{level} [*level{level - 1}, *level{level - 1}]\n"
    filename.write_text(content)
    with pytest.raises(ModelConfigError, match="alias expansion limit"):
        load_models(filename, {}, None)


def test_cli_legacy_profiles_has_no_runtime_side_effects(tmp_path: Path) -> None:
    data_dir = tmp_path / "legacy-never-created"
    profiles = {
        "fast": {
            "base_url": "https://legacy.example/v1",
            "api_key": _SECRET,
            "model": "legacy-model",
        }
    }
    code = (
        "import json, sys; from pathlib import Path; import agent.config; "
        f"agent.config.PROJECT_ROOT = Path({str(tmp_path)!r}); "
        "from agent.model_config import check_main; "
        "assert check_main(['--strict']) == 2; "
        "print(json.dumps(sorted(set(sys.modules) & {'agent.database', 'agent.api', "
        "'agent.runtime'})))"
    )
    env = {name: value for name, value in os.environ.items() if not name.startswith("AGENT_")}
    env.update(
        {
            "AGENT_ENV_FILE": "",
            "AGENT_DATA_DIR": str(data_dir),
            "AGENT_DEFAULT_LLM_PROFILE": "fast",
            "AGENT_LLM_PROFILES": json.dumps(profiles),
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
    )
    assert json.loads(result.stdout.splitlines()[-1]) == []
    assert "deprecated" in result.stderr
    assert _SECRET not in result.stdout + result.stderr
    assert not data_dir.exists()


def test_unknown_default_override_is_ignored_with_warning(tmp_path: Path) -> None:
    registry = load_models(
        _write(tmp_path, _data()),
        {"MODEL_KEY": _SECRET, "AGENT_DEFAULT_LLM_PROFILE": "default"},
        None,
    )
    assert registry.default_id == "fast"
    assert any("AGENT_DEFAULT_LLM_PROFILE" in warning for warning in registry.warnings)


def test_legacy_style_ids_are_accepted(tmp_path: Path) -> None:
    data = _data()
    data["models"]["local:qwen"] = data["models"].pop("fast")
    data["routing"]["default"] = "local:qwen"
    registry = load_models(_write(tmp_path, data), {"MODEL_KEY": _SECRET}, None)
    assert registry.default_id == "local:qwen"
