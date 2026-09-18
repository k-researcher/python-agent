import sqlite3
import zipfile
from pathlib import Path

import pytest
from openpyxl import Workbook

from agent.config import Settings
from agent.tools import ToolRegistry
from agent.tools.advanced import ExcelSheetsTool
from agent.tools.base import ToolContext, ToolError


async def test_excel_reader_is_bounded_to_project(tmp_path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Data"
    sheet.append(["name", "value"])
    sheet.append(["alpha", 42])
    workbook.save(tmp_path / "sample.xlsx")
    workbook.close()

    registry = ToolRegistry(settings=Settings())
    result = await registry.execute(
        "excel_read",
        "ask",
        ToolContext(session_id="test", project_root=tmp_path),
        {"path": "sample.xlsx", "sheet": "Data", "max_row": 2, "max_col": 2},
    )

    assert result["rows"] == [["name", "value"], ["alpha", 42]]


async def test_database_select_uses_named_read_only_connection(tmp_path: Path) -> None:
    database_path = tmp_path / "external.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("create table items (name text)")
        connection.execute("insert into items values ('alpha')")

    settings = Settings(
        allow_network_tools=True,
        database_connections={"reporting": f"sqlite+aiosqlite:///{database_path}"},
    )
    registry = ToolRegistry(settings=settings)
    result = await registry.execute(
        "db_select",
        "ask",
        ToolContext(session_id="test", project_root=tmp_path),
        {"connection": "reporting", "query": "select name from items"},
    )

    assert result["rows"] == [{"name": "alpha"}]


def test_excel_rejects_excessive_uncompressed_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "bomb.xlsx"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/large.xml", "x" * 1000)

    monkeypatch.setattr("agent.tools.advanced.MAX_WORKBOOK_UNCOMPRESSED_BYTES", 100)
    with pytest.raises(ToolError, match="uncompressed"):
        ExcelSheetsTool.validate_workbook(path)
