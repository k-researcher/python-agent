from __future__ import annotations

from pathlib import Path


class PathSecurityError(ValueError):
    pass


def is_sensitive_path(path: Path) -> bool:
    for part in path.parts:
        lowered = part.lower()
        if lowered == ".env" or (lowered.startswith(".env.") and lowered != ".env.example"):
            return True
        if lowered in {
            ".npmrc",
            ".pypirc",
            ".netrc",
            "credentials",
            "credentials.json",
            "id_rsa",
            "id_dsa",
            "id_ecdsa",
            "id_ed25519",
        }:
            return True
    return path.suffix.lower() in {".key", ".p12", ".pfx"}


class PathGuard:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise PathSecurityError(f"Project root is not a directory: {self.root}")

    def resolve(
        self, value: str, *, must_exist: bool = True, allow_sensitive: bool = False
    ) -> Path:
        raw = Path(value).expanduser()
        candidate = raw if raw.is_absolute() else self.root / raw

        if must_exist:
            resolved = candidate.resolve(strict=True)
        else:
            parent = candidate.parent.resolve(strict=True)
            resolved = parent / candidate.name

        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise PathSecurityError("Path is outside the selected project") from exc
        if not allow_sensitive and is_sensitive_path(resolved):
            raise PathSecurityError("Access to secret-bearing files is blocked")
        return resolved
