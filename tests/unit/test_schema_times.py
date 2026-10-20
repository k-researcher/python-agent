from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from agent.schemas import ProjectRead


def _project(created_at: datetime) -> ProjectRead:
    return ProjectRead(id="p1", name="demo", root_path="/tmp/demo", created_at=created_at)


def test_time_without_a_zone_is_sent_as_utc() -> None:
    body = _project(datetime(2026, 10, 20, 15, 31)).model_dump(mode="json")

    assert body["created_at"] == "2026-10-20T15:31:00Z"


def test_time_with_a_zone_keeps_it() -> None:
    moscow = timezone(timedelta(hours=3))
    project = _project(datetime(2026, 10, 20, 18, 31, tzinfo=moscow))

    assert project.created_at == datetime(2026, 10, 20, 15, 31, tzinfo=UTC)
    assert project.model_dump(mode="json")["created_at"] == "2026-10-20T18:31:00+03:00"
