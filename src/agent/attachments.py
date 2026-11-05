"""Text files attached to a user message.

The file texts go into the message content as tagged blocks after the typed text, so every
part that reads the content as a string (context, summaries, history) sees them unchanged.
``Message.attachments`` keeps the file names, sizes and the end of the typed text for the UI.
"""

from __future__ import annotations

from typing import Any

from agent.schemas import AttachmentIn

MAX_FILES = 10
MAX_FILE_BYTES = 1024 * 1024
_UNSAFE_NAME = str.maketrans({'"': "'", "<": "(", ">": ")", "\n": " ", "\r": " ", "\t": " "})


def clean_name(name: str) -> str:
    """Return the base name of a file, safe inside the attachment tag."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1].translate(_UNSAFE_NAME).strip()
    return base or "file"


def check_files(files: list[AttachmentIn], context_window: int | None) -> None:
    """Raise ValueError when the files break a limit.

    All files together may take at most a quarter of the model context: about four
    characters make one token, so the limit in characters equals the window in tokens.
    """
    if len(files) > MAX_FILES:
        raise ValueError(f"Можно приложить не больше {MAX_FILES} файлов.")
    for item in files:
        if "\x00" in item.text:
            raise ValueError(f"Файл {clean_name(item.name)} не текстовый.")
        if len(item.text.encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError(f"Файл {clean_name(item.name)} больше 1 МБ.")
    total = sum(len(item.text) for item in files)
    if context_window is not None and total > context_window:
        raise ValueError("Файлы слишком большие для этой модели: всего больше четверти контекста.")


def compose(text: str, files: list[AttachmentIn]) -> tuple[str, dict[str, Any] | None]:
    """Return the message content and the attachment record for ``text`` and ``files``."""
    if not files:
        return text, None
    typed = text.rstrip()
    blocks = [
        f'<attachment name="{clean_name(item.name)}">\n{item.text}\n</attachment>' for item in files
    ]
    content = "\n\n".join(([typed] if typed else []) + blocks)
    record = {
        "text_end": len(typed),
        "files": [
            {
                "name": clean_name(item.name),
                "size": len(item.text.encode("utf-8")),
                "chars": len(item.text),
            }
            for item in files
        ],
    }
    return content, record


def replace_text(
    content: str, record: dict[str, Any] | None, text: str
) -> tuple[str, dict[str, Any] | None]:
    """Keep the attached files of ``content`` and put ``text`` in place of the typed text."""
    if not record:
        return text, None
    end = int(record.get("text_end", 0))
    blocks = content[end:].lstrip("\n")
    typed = text.rstrip()
    return "\n\n".join(([typed] if typed else []) + [blocks]), record | {"text_end": len(typed)}


def typed_text(content: str | None, record: dict[str, Any] | None) -> str:
    """Return only the text that the user typed."""
    if not record:
        return content or ""
    return (content or "")[: int(record.get("text_end", 0))]
