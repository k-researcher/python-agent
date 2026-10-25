from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agent.config import Settings
from agent.database import init_database, session_factory
from agent.models import KnowledgeDocument
from agent.tools.base import ToolContext
from agent.tools.knowledge import KbListTool


def _sha(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


@pytest.fixture
async def kb_documents() -> None:
    await init_database(recover_running=False)
    async with session_factory() as db:
        db.add_all(
            [
                KnowledgeDocument(
                    content="Alpha document body",
                    content_sha256=_sha("Alpha document body"),
                    keywords="alpha",
                    document_metadata={"env": "prod"},
                ),
                KnowledgeDocument(
                    content="Beta document body",
                    content_sha256=_sha("Beta document body"),
                    keywords="beta",
                    document_metadata={"env": "dev"},
                ),
                KnowledgeDocument(
                    content="Gamma document body",
                    content_sha256=_sha("Gamma document body"),
                    keywords="gamma",
                    document_metadata={"env": "prod", "team": "data"},
                ),
            ]
        )
        await db.commit()


def context(tmp_path: Path) -> ToolContext:
    return ToolContext(session_id="test", project_root=tmp_path)


async def test_kb_list_pagination(tmp_path: Path, kb_documents: None) -> None:
    tool = KbListTool(Settings(knowledge_base_enabled=True))
    ctx = context(tmp_path)

    first = await tool.execute(ctx, {"offset": 0, "count": 2})
    assert first["total"] == 3
    assert first["count"] == 2

    second = await tool.execute(ctx, {"offset": 2, "count": 2})
    assert second["count"] == 1
    assert first["documents"][0]["id"] != second["documents"][0]["id"]

    beyond = await tool.execute(ctx, {"offset": 10, "count": 5})
    assert beyond["documents"] == []
    assert beyond["total"] == 3


async def test_kb_list_metadata_filter(tmp_path: Path, kb_documents: None) -> None:
    tool = KbListTool(Settings(knowledge_base_enabled=True))
    ctx = context(tmp_path)

    filtered = await tool.execute(ctx, {"metadata": {"env": "prod"}})
    assert filtered["total"] == 2
    assert all(doc["metadata"]["env"] == "prod" for doc in filtered["documents"])

    exact = await tool.execute(ctx, {"metadata": {"env": "prod", "team": "data"}})
    assert exact["total"] == 1
    assert exact["documents"][0]["metadata"]["team"] == "data"


async def test_kb_list_entry_shape(tmp_path: Path, kb_documents: None) -> None:
    tool = KbListTool(Settings(knowledge_base_enabled=True))
    ctx = context(tmp_path)

    result = await tool.execute(ctx, {"metadata": {"env": "dev"}})
    doc = result["documents"][0]
    assert set(doc) == {"id", "title", "metadata", "created_at"}
    assert doc["id"]
    assert doc["title"]
    assert doc["metadata"] == {"env": "dev"}
    assert doc["created_at"]
