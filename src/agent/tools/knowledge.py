from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import httpx
from sqlalchemy import select

from agent.config import Settings
from agent.database import session_factory
from agent.models import KnowledgeDocument, utcnow
from agent.tools.advanced import audit
from agent.tools.base import RiskLevel, Tool, ToolContext, ToolError
from agent.tools.network import validate_outbound_url

WORD_PATTERN = re.compile(r"[\w-]{3,}", re.UNICODE)
MAX_KNOWLEDGE_CONTENT_CHARS = 1_000_000


def keywords_for(text: str, limit: int = 20) -> str:
    counts: dict[str, int] = {}
    for word in WORD_PATTERN.findall(text.lower()):
        counts[word] = counts.get(word, 0) + 1
    ordered = sorted(counts, key=lambda word: (-counts[word], word))
    return " ".join(ordered[:limit])


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def embed(self, context: ToolContext, text_value: str) -> list[float] | None:
        url = self.settings.knowledge_base_embedding_url
        if not url:
            return None
        await validate_outbound_url(url, self.settings.network_allowlist)
        payload: dict[str, Any] = {"input": text_value}
        if self.settings.knowledge_base_embedding_model:
            payload["model"] = self.settings.knowledge_base_embedding_model
        headers = {"Content-Type": "application/json"}
        if self.settings.knowledge_base_embedding_api_key:
            headers["Authorization"] = f"Bearer {self.settings.knowledge_base_embedding_api_key}"
        encoded = str(payload)
        await audit(context, "embedding", url, "embed", encoded, "started")
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                vector = response.json()["data"][0]["embedding"]
            if not isinstance(vector, list) or not vector:
                raise ToolError("Embedding service returned an invalid vector")
            result = [float(value) for value in vector]
            await audit(context, "embedding", url, "embed", encoded, "completed")
            return result
        except Exception as exc:
            await audit(context, "embedding", url, "embed", encoded, "error", str(exc))
            raise


class KnowledgeTool(Tool):
    knowledge_capability = True

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.network_capability = bool(settings.knowledge_base_embedding_url)
        if not self.network_capability and self.risk_level is RiskLevel.network_access:
            self.risk_level = RiskLevel.read_only
        self.embedding_service = EmbeddingService(settings)


class KbSearchTool(KnowledgeTool):
    name = "kb_search"
    description = "Search the explicitly enabled knowledge base. Query egress requires approval."
    risk_level = RiskLevel.network_access
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments["query"])
        limit = min(50, int(arguments.get("limit", 5)))
        query_embedding = await self.embedding_service.embed(context, query)
        async with session_factory() as db:
            documents = list((await db.execute(select(KnowledgeDocument))).scalars())

        query_words = set(WORD_PATTERN.findall(query.lower()))
        scored: list[tuple[float, KnowledgeDocument]] = []
        for document in documents:
            if query_embedding is not None and document.embedding:
                score = cosine(query_embedding, document.embedding)
            else:
                searchable = (document.keywords + " " + document.content).lower()
                document_words = set(WORD_PATTERN.findall(searchable))
                score = len(query_words & document_words) / max(1, len(query_words))
            if score > 0:
                scored.append((score, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return {
            "results": [
                {
                    "id": document.id,
                    "content": document.content,
                    "metadata": document.document_metadata,
                    "similarity": round(score, 6),
                }
                for score, document in scored[:limit]
            ]
        }


class KbReadTool(KnowledgeTool):
    name = "kb_read"
    description = "Read knowledge documents by UUID."
    risk_level = RiskLevel.network_access
    input_schema = {
        "type": "object",
        "properties": {"ids": {"type": "array", "items": {"type": "string"}, "maxItems": 50}},
        "required": ["ids"],
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        ids = [str(item) for item in arguments["ids"]][:50]
        async with session_factory() as db:
            query = select(KnowledgeDocument).where(KnowledgeDocument.id.in_(ids))
            documents = list((await db.execute(query)).scalars())
        by_id = {document.id: document for document in documents}
        return {
            "documents": [
                {
                    "id": item,
                    "content": by_id[item].content,
                    "keywords": by_id[item].keywords,
                    "metadata": by_id[item].document_metadata,
                }
                for item in ids
                if item in by_id
            ],
            "missing": [item for item in ids if item not in by_id],
        }


class KbSaveTool(KnowledgeTool):
    name = "kb_save"
    description = "Save an approved exact text fragment to the knowledge base."
    risk_level = RiskLevel.remote_write
    input_schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "maxLength": MAX_KNOWLEDGE_CONTENT_CHARS},
            "metadata": {"type": "object"},
        },
        "required": ["content"],
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        content = str(arguments["content"]).strip()
        if not content:
            raise ToolError("Knowledge content is empty")
        if len(content) > MAX_KNOWLEDGE_CONTENT_CHARS:
            raise ToolError("Knowledge content exceeds the configured safety limit")
        digest = hashlib.sha256(content.encode()).hexdigest()
        embedding = await self.embedding_service.embed(context, content)
        await audit(context, "knowledge_base", "primary-database", "save", content, "started")
        async with session_factory() as db:
            existing = (
                await db.execute(
                    select(KnowledgeDocument).where(KnowledgeDocument.content_sha256 == digest)
                )
            ).scalar_one_or_none()
            if existing is not None:
                raise ToolError(f"Identical knowledge document already exists: {existing.id}")
            document = KnowledgeDocument(
                content=content,
                content_sha256=digest,
                keywords=keywords_for(content),
                embedding=embedding,
                document_metadata=dict(arguments.get("metadata") or {}),
                source_session_id=context.session_id,
            )
            db.add(document)
            await db.commit()
            await db.refresh(document)
        await audit(context, "knowledge_base", "primary-database", "save", content, "completed")
        return {"id": document.id, "content_sha256": digest}


class KbEditTool(KbSaveTool):
    name = "kb_edit"
    description = "Replace an approved knowledge document and recompute its index."
    input_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "content": {"type": "string", "maxLength": MAX_KNOWLEDGE_CONTENT_CHARS},
            "metadata": {"type": "object"},
        },
        "required": ["id", "content"],
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        document_id = str(arguments["id"])
        content = str(arguments["content"]).strip()
        if not content:
            raise ToolError("Knowledge content is empty")
        if len(content) > MAX_KNOWLEDGE_CONTENT_CHARS:
            raise ToolError("Knowledge content exceeds the configured safety limit")
        embedding = await self.embedding_service.embed(context, content)
        digest = hashlib.sha256(content.encode()).hexdigest()
        await audit(context, "knowledge_base", "primary-database", "edit", content, "started")
        async with session_factory() as db:
            document = await db.get(KnowledgeDocument, document_id)
            if document is None:
                raise ToolError("Knowledge document not found")
            document.content = content
            document.content_sha256 = digest
            document.keywords = keywords_for(content)
            document.embedding = embedding
            document.document_metadata = dict(arguments.get("metadata") or {})
            document.updated_at = utcnow()
            await db.commit()
        await audit(context, "knowledge_base", "primary-database", "edit", content, "completed")
        return {"id": document_id, "content_sha256": digest}


class KbDeleteTool(KnowledgeTool):
    name = "kb_delete"
    description = "Delete one approved knowledge document by UUID."
    risk_level = RiskLevel.destructive
    input_schema = {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        document_id = str(arguments["id"])
        await audit(context, "knowledge_base", "primary-database", "delete", document_id, "started")
        async with session_factory() as db:
            document = await db.get(KnowledgeDocument, document_id)
            if document is None:
                raise ToolError("Knowledge document not found")
            await db.delete(document)
            await db.commit()
        await audit(
            context,
            "knowledge_base",
            "primary-database",
            "delete",
            document_id,
            "completed",
        )
        return {"deleted": document_id}


def knowledge_tools(settings: Settings) -> tuple[Tool, ...]:
    return (
        KbSearchTool(settings),
        KbReadTool(settings),
        KbSaveTool(settings),
        KbEditTool(settings),
        KbDeleteTool(settings),
    )
