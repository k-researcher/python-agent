"""Engine port and opt-in loading contract; no installed adapter is needed."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from agent.config import Settings


def profile(**changes):
    from agent.config import SUPPORTED_ADAPTER_MODULES, EngineProfileSettings

    return EngineProfileSettings(
        **dict(
            adapter_module=sorted(SUPPORTED_ADAPTER_MODULES)[0],
            binary="test-engine",
            model="test-model",
            operator_user_id="operator",
        )
        | changes
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"adapter_module": "os"},
        {"enabled": True, "binary": ""},
        {"enabled": True, "model": ""},
        {"enabled": True, "operator_user_id": ""},
        {"process_limit": 0},
        {"timeouts": {"turn": -1}},
        {"env": {"TOKEN": "x"}},
    ],
)
def test_profile_rejects_invalid_settings(changes):
    with pytest.raises(ValidationError):
        profile(**changes)


def test_default_off_has_no_import_process_probe_or_timer(monkeypatch):
    from agent.engine_adapters import EngineRegistry

    forbidden = Mock(side_effect=AssertionError("off must be inert"))
    monkeypatch.setattr(importlib, "import_module", forbidden)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    monkeypatch.setattr(asyncio, "create_task", forbidden)
    assert Settings().engine_profiles == {}
    registry = EngineRegistry({"personal": profile()})
    registry.register("personal", forbidden)
    assert registry.get("personal") is None
    assert registry.get("removed") is None
    assert EngineRegistry({}).get("personal") is None
    forbidden.assert_not_called()


def test_enabled_import_is_lazy_and_missing_module_is_unavailable(monkeypatch):
    from agent.engine_adapters import EngineRegistry

    importer = Mock(side_effect=ModuleNotFoundError("removed adapter"))
    monkeypatch.setattr(importlib, "import_module", importer)
    registry = EngineRegistry({"personal": profile(enabled=True)})
    importer.assert_not_called()
    assert registry.get("personal") is None
    importer.assert_called_once_with(profile().adapter_module)


async def test_port_with_registered_fake_engine(tmp_path):
    from agent.engine import DoneEvent, Engine, EngineSpec, Health, Limits, TextEvent, ThreadRef
    from agent.engine_adapters import EngineRegistry

    class FakeEngine:
        async def health(self):
            return Health(status="ready")

        async def open(self, spec, thread_id=None):
            assert spec.project_root == tmp_path.resolve()
            assert spec.mode == "read"
            return ThreadRef(thread_id or "native")

        async def turn(self, thread, input, submission_id):
            yield TextEvent(
                submission_id=submission_id,
                thread_id=thread.native_id,
                turn_id="turn",
                item_id="reply",
                seq=1,
                text=input,
                phase="final",
            )
            yield DoneEvent(
                submission_id=submission_id,
                thread_id=thread.native_id,
                turn_id="turn",
                seq=2,
                status="completed",
                reason=None,
            )

        async def answer(self, request_id, decision):
            pass

        async def answer_user_input(self, request_id):
            pass

        async def interrupt(self, turn_id):
            pass

        async def read_limits(self):
            return Limits(windows=[], status="unknown", observed_at=None)

        async def close(self):
            pass

    async with asyncio.timeout(3):
        registry = EngineRegistry({"personal": profile(enabled=True)})
        registry.register("personal", lambda _: FakeEngine())
        engine = registry.get("personal")
        assert isinstance(engine, Engine)
        assert (await engine.health()).status == "ready"
        assert dataclasses.is_dataclass(ThreadRef)
        thread = await engine.open(EngineSpec(tmp_path.resolve(), "read", "instructions", {}))
        assert (
            await engine.open(EngineSpec(tmp_path.resolve(), "read", "instructions", {}), "saved")
        ).native_id == "saved"
        events = [event async for event in engine.turn(thread, "new input", "submission")]
        assert [event.kind for event in events] == ["text", "done"]
        assert [event.seq for event in events] == [1, 2]
        assert all(event.thread_id == "native" for event in events)
        assert events[0].text == "new input"
        assert (await engine.read_limits()).status == "unknown"
        await engine.close()


def test_profiles_load_from_yaml_without_loading_adapters(tmp_path: Path, monkeypatch):
    from agent.engine_adapters import configured_profiles

    example = Path("config/models.example.yaml").read_text()
    path = tmp_path / "models.yaml"
    # The shipped example has no active profiles.
    path.write_text(example)
    monkeypatch.setattr(importlib, "import_module", Mock(side_effect=AssertionError("import")))
    assert configured_profiles(Settings(models_file=str(path))) == {}
    import yaml

    content = yaml.safe_load(example)
    content["engine_profiles"] = {"personal": profile(enabled=True).model_dump()}
    path.write_text(yaml.safe_dump(content))
    assert configured_profiles(Settings(models_file=str(path)))["personal"].enabled is True
    # Removing the declaration and the absent adapter needs no fallback or installation.
    del content["engine_profiles"]
    path.write_text(yaml.safe_dump(content))
    assert configured_profiles(Settings(models_file=str(path))) == {}
    assert (
        configured_profiles(Settings(models_file="", engine_profiles={"p": profile()}))["p"].enabled
        is False
    )


@pytest.mark.parametrize("profiles,parallel", [({"": {}}, 4), ({"p": {"process_limit": 2}}, 1)])
def test_invalid_profile_names_and_process_ceiling(profiles, parallel):
    from agent.engine_adapters import configured_profiles

    with pytest.raises(ValueError):
        configured_profiles(
            Settings(
                models_file="",
                max_parallel_sessions=parallel,
                engine_profiles={name: profile(**value) for name, value in profiles.items()},
            )
        )
