"""受 trace 和病例约束的临时 Artifact Store。

大型特征向量仅保存在服务端；MCP 工具之间只传 128 bit 随机句柄。每个 artifact
不可变，并绑定类型、trace、病例、父对象、模型 schema、内容哈希、大小和 TTL。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from wei_multimodal.mcp_server.errors import ContractError, ErrorCode
from wei_multimodal.mcp_server.security import SHA256_PATTERN

ARTIFACT_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(qc|ctf|pathf|pred|rpt)_[a-f0-9]{32}$"
)
ARTIFACT_PREFIXES: Final[dict[str, str]] = {
    "case_qc": "qc",
    "ct_features": "ctf",
    "pathology_features": "pathf",
    "prediction": "pred",
    "report": "rpt",
}
DEFAULT_MEDIA_TYPES: Final[dict[str, str]] = {
    "case_qc": "application/json",
    "ct_features": "application/json",
    "pathology_features": "application/json",
    "prediction": "application/json",
    "report": "text/html; charset=utf-8",
}


def _utc_now() -> datetime:
    """返回带时区 UTC 时间，便于测试时替换时钟。"""

    return datetime.now(UTC)


def _format_utc(value: datetime) -> str:
    """序列化为 RFC 3339 ``Z`` 时间。"""

    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    """解析内部元数据时间；任何损坏都由调用方统一隐藏。"""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("artifact timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """不可变 artifact 元数据；敏感绑定信息不会出现在公共引用中。"""

    artifact_id: str
    artifact_type: str
    media_type: str
    trace_id: str
    case_binding_sha256: str
    parent_artifact_ids: tuple[str, ...]
    model_schema_version: str | None
    profile_snapshot: dict[str, Any] | None
    created_at: datetime
    expires_at: datetime
    content_sha256: str
    payload_size_bytes: int
    payload_encoding: str

    def to_public_dict(self) -> dict[str, str]:
        """只返回 MCP ``ArtifactRef`` 所需字段，不泄露内部目录或病例绑定。"""

        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "media_type": self.media_type,
            "content_sha256": self.content_sha256,
            "created_at": _format_utc(self.created_at),
            "expires_at": _format_utc(self.expires_at),
        }


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """仅供服务端内部消费的 artifact 与 payload。"""

    metadata: ArtifactMetadata
    payload: Any

    @property
    def artifact_id(self) -> str:
        return self.metadata.artifact_id

    @property
    def artifact_type(self) -> str:
        return self.metadata.artifact_type

    @property
    def trace_id(self) -> str:
        return self.metadata.trace_id

    @property
    def case_binding_sha256(self) -> str:
        return self.metadata.case_binding_sha256


class ArtifactStore:
    """文件系统实现的进程内并发安全临时对象仓库。

    写入通过“同目录临时文件 + fsync + os.replace”发布。容量判断和发布由重入锁串行化，
    保证同一进程内的并发请求不会突破单 trace 或全局限制。
    """

    def __init__(
        self,
        root: Path,
        *,
        ttl_seconds: int = 1800,
        ttl: int | timedelta | None = None,
        max_artifact_bytes: int = 16 * 1024 * 1024,
        max_artifacts_per_trace: int = 10,
        max_total_bytes: int = 256 * 1024 * 1024,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        # ``ttl`` 是运行时组装层的便捷别名；若给出则优先，避免各层自行换算。
        if ttl is not None:
            ttl_seconds = int(ttl.total_seconds()) if isinstance(ttl, timedelta) else ttl
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if min(max_artifact_bytes, max_artifacts_per_trace, max_total_bytes) <= 0:
            raise ValueError("artifact capacity limits must be positive")
        self._root = root.resolve()
        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_artifact_bytes = max_artifact_bytes
        self._max_artifacts_per_trace = max_artifacts_per_trace
        self._max_total_bytes = max_total_bytes
        self._clock = clock
        self._lock = threading.RLock()
        self._root.mkdir(parents=True, exist_ok=True)
        if not self._root.is_dir():
            raise ValueError("artifact root must be a directory")

    @property
    def root(self) -> Path:
        """仅供运行时健康检查使用；不得写入 MCP 响应。"""

        return self._root

    def _paths(self, artifact_id: str) -> tuple[Path, Path]:
        """由已校验的句柄生成固定内部路径。"""

        if not ARTIFACT_ID_PATTERN.fullmatch(artifact_id):
            raise ContractError(
                ErrorCode.ARTIFACT_UNAVAILABLE,
                message="Artifact is unavailable.",
            )
        return self._root / f"{artifact_id}.meta.json", self._root / f"{artifact_id}.payload"

    @staticmethod
    def _serialize_payload(payload: Any) -> tuple[bytes, str]:
        """确定性序列化 payload；JSON 明确拒绝 NaN/Infinity。"""

        if isinstance(payload, bytes):
            return payload, "bytes"
        if isinstance(payload, str):
            return payload.encode("utf-8"), "utf-8"
        try:
            serialized = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ContractError(
                ErrorCode.INVALID_FILE_FORMAT,
                message="Artifact payload is not valid finite JSON.",
            ) from exc
        return serialized, "json"

    @staticmethod
    def _deserialize_payload(raw: bytes, encoding: str) -> Any:
        """按创建时记录的编码恢复服务端 payload。"""

        if encoding == "bytes":
            return raw
        if encoding == "utf-8":
            return raw.decode("utf-8")
        if encoding == "json":
            return json.loads(raw.decode("utf-8"))
        raise ValueError("unknown artifact payload encoding")

    def _metadata_to_bytes(self, metadata: ArtifactMetadata) -> bytes:
        """把内部元数据编码成不含路径的确定性 JSON。"""

        payload = asdict(metadata)
        payload["created_at"] = _format_utc(metadata.created_at)
        payload["expires_at"] = _format_utc(metadata.expires_at)
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _metadata_from_bytes(raw: bytes) -> ArtifactMetadata:
        """从内部索引恢复强类型元数据。"""

        payload = json.loads(raw.decode("utf-8"))
        return ArtifactMetadata(
            artifact_id=str(payload["artifact_id"]),
            artifact_type=str(payload["artifact_type"]),
            media_type=str(payload["media_type"]),
            trace_id=str(payload["trace_id"]),
            case_binding_sha256=str(payload["case_binding_sha256"]),
            parent_artifact_ids=tuple(payload["parent_artifact_ids"]),
            model_schema_version=payload["model_schema_version"],
            profile_snapshot=payload["profile_snapshot"],
            created_at=_parse_utc(str(payload["created_at"])),
            expires_at=_parse_utc(str(payload["expires_at"])),
            content_sha256=str(payload["content_sha256"]),
            payload_size_bytes=int(payload["payload_size_bytes"]),
            payload_encoding=str(payload["payload_encoding"]),
        )

    def _atomic_write(self, destination: Path, content: bytes) -> None:
        """把完整内容原子发布到目标路径，并尽量清理失败的临时文件。"""

        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self._root,
                prefix=".tmp-",
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_name = temporary.name
            os.replace(temporary_name, destination)
            temporary_name = None
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)

    def _read_metadata_file(self, path: Path) -> ArtifactMetadata:
        """读取内部索引；损坏或消失统一表现为 unavailable。"""

        try:
            metadata = self._metadata_from_bytes(path.read_bytes())
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise ContractError(
                ErrorCode.ARTIFACT_UNAVAILABLE,
                message="Artifact is unavailable.",
            ) from exc
        if not ARTIFACT_ID_PATTERN.fullmatch(metadata.artifact_id):
            raise ContractError(ErrorCode.ARTIFACT_UNAVAILABLE, message="Artifact is unavailable.")
        return metadata

    def _iter_metadata(self) -> list[ArtifactMetadata]:
        """列出可解析索引；临时文件和孤立 payload 不参与容量计数。"""

        records: list[ArtifactMetadata] = []
        for path in self._root.glob("*.meta.json"):
            try:
                records.append(self._read_metadata_file(path))
            except ContractError:
                # 损坏索引不会被外部读到；健康检查可据此告警。
                continue
        return records

    def _cleanup_expired_locked(self, now: datetime) -> int:
        """锁内删除已过期对象；绝不为容量删除仍有效的其他 trace。"""

        removed = 0
        for metadata in self._iter_metadata():
            if metadata.expires_at <= now:
                meta_path, payload_path = self._paths(metadata.artifact_id)
                meta_path.unlink(missing_ok=True)
                payload_path.unlink(missing_ok=True)
                removed += 1
        return removed

    def cleanup_expired(self) -> int:
        """显式清理过期对象并返回数量。"""

        with self._lock:
            return self._cleanup_expired_locked(self._clock().astimezone(UTC))

    def put(
        self,
        *,
        artifact_type: str,
        trace_id: str,
        case_binding_sha256: str,
        payload: Any,
        parent_artifact_ids: Sequence[str] = (),
        model_schema_version: str | None = None,
        profile_snapshot: Mapping[str, Any] | None = None,
        media_type: str | None = None,
        ttl_seconds: int | None = None,
    ) -> ArtifactMetadata:
        """创建不可变 artifact 并返回安全元数据。

        父对象若存在，必须仍有效且与新对象属于同 trace、同病例，防止调用层忘记执行
        编排绑定检查。
        """

        if artifact_type not in ARTIFACT_PREFIXES:
            raise ValueError("unsupported core artifact type")
        if not trace_id or len(trace_id) > 128:
            raise ValueError("trace_id is required and must be at most 128 characters")
        if not SHA256_PATTERN.fullmatch(case_binding_sha256):
            raise ValueError("case_binding_sha256 must be a lowercase SHA-256 digest")
        effective_ttl = self._ttl if ttl_seconds is None else timedelta(seconds=ttl_seconds)
        if effective_ttl.total_seconds() <= 0:
            raise ValueError("ttl_seconds must be positive")
        payload_bytes, payload_encoding = self._serialize_payload(payload)
        payload_size = len(payload_bytes)
        if payload_size > self._max_artifact_bytes:
            raise ContractError(
                ErrorCode.CAPACITY_EXCEEDED,
                message="Artifact exceeds the configured capacity limit.",
            )

        with self._lock:
            now = self._clock().astimezone(UTC)
            self._cleanup_expired_locked(now)
            for parent_id in parent_artifact_ids:
                self._get_locked(
                    parent_id,
                    trace_id=trace_id,
                    case_binding_sha256=case_binding_sha256,
                    now=now,
                )
            records = self._iter_metadata()
            trace_count = sum(record.trace_id == trace_id for record in records)
            total_bytes = sum(record.payload_size_bytes for record in records)
            if trace_count >= self._max_artifacts_per_trace or (
                total_bytes + payload_size > self._max_total_bytes
            ):
                raise ContractError(
                    ErrorCode.CAPACITY_EXCEEDED,
                    message="Artifact store capacity is exhausted.",
                )

            # 128 bit CSPRNG 句柄；极低概率冲突时重新生成，不覆盖已有对象。
            artifact_id = ""
            meta_path = payload_path = self._root
            for _ in range(10):
                artifact_id = f"{ARTIFACT_PREFIXES[artifact_type]}_{secrets.token_hex(16)}"
                meta_path, payload_path = self._paths(artifact_id)
                if not meta_path.exists() and not payload_path.exists():
                    break
            else:
                raise ContractError(
                    ErrorCode.SERVICE_UNAVAILABLE,
                    message="A secure artifact identifier could not be allocated.",
                )

            metadata = ArtifactMetadata(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                media_type=media_type or DEFAULT_MEDIA_TYPES[artifact_type],
                trace_id=trace_id,
                case_binding_sha256=case_binding_sha256,
                parent_artifact_ids=tuple(parent_artifact_ids),
                model_schema_version=model_schema_version,
                profile_snapshot=(dict(profile_snapshot) if profile_snapshot is not None else None),
                created_at=now,
                expires_at=now + effective_ttl,
                content_sha256=hashlib.sha256(payload_bytes).hexdigest(),
                payload_size_bytes=payload_size,
                payload_encoding=payload_encoding,
            )
            try:
                # payload 先发布，metadata 最后发布；读者永远不会看到半成品索引。
                self._atomic_write(payload_path, payload_bytes)
                self._atomic_write(meta_path, self._metadata_to_bytes(metadata))
            except OSError as exc:
                meta_path.unlink(missing_ok=True)
                payload_path.unlink(missing_ok=True)
                raise ContractError(
                    ErrorCode.SERVICE_UNAVAILABLE,
                    message="Artifact storage is temporarily unavailable.",
                ) from exc
            return metadata

    def _get_locked(
        self,
        artifact_id: str,
        *,
        trace_id: str,
        expected_type: str | None = None,
        case_binding_sha256: str | None = None,
        now: datetime | None = None,
    ) -> StoredArtifact:
        """锁内读取并按“可用性→TTL→类型→病例”顺序执行硬校验。"""

        meta_path, payload_path = self._paths(artifact_id)
        if not meta_path.is_file() or not payload_path.is_file():
            raise ContractError(ErrorCode.ARTIFACT_UNAVAILABLE, message="Artifact is unavailable.")
        metadata = self._read_metadata_file(meta_path)
        if metadata.artifact_id != artifact_id or metadata.trace_id != trace_id:
            # 不存在和跨 trace 使用同一错误，避免泄露对象是否存在。
            raise ContractError(ErrorCode.ARTIFACT_UNAVAILABLE, message="Artifact is unavailable.")
        current_time = (now or self._clock()).astimezone(UTC)
        if metadata.expires_at <= current_time:
            raise ContractError(ErrorCode.ARTIFACT_EXPIRED, message="Artifact has expired.")
        if expected_type is not None and metadata.artifact_type != expected_type:
            raise ContractError(
                ErrorCode.ARTIFACT_TYPE_MISMATCH,
                message="Artifact type does not match this workflow stage.",
            )
        if (
            case_binding_sha256 is not None
            and metadata.case_binding_sha256 != case_binding_sha256
        ):
            raise ContractError(
                ErrorCode.CASE_BINDING_MISMATCH,
                message="Artifact does not belong to the requested case.",
            )
        try:
            raw = payload_path.read_bytes()
            if len(raw) != metadata.payload_size_bytes:
                raise ValueError("payload size mismatch")
            if hashlib.sha256(raw).hexdigest() != metadata.content_sha256:
                raise ValueError("payload digest mismatch")
            payload = self._deserialize_payload(raw, metadata.payload_encoding)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ContractError(
                ErrorCode.ARTIFACT_UNAVAILABLE,
                message="Artifact is unavailable.",
            ) from exc
        return StoredArtifact(metadata=metadata, payload=payload)

    def get(
        self,
        artifact_id: str,
        *,
        trace_id: str,
        expected_type: str | None = None,
        case_binding_sha256: str | None = None,
    ) -> StoredArtifact:
        """读取同 trace 的 artifact，并可追加类型和病例绑定检查。"""

        with self._lock:
            return self._get_locked(
                artifact_id,
                trace_id=trace_id,
                expected_type=expected_type,
                case_binding_sha256=case_binding_sha256,
            )

    def require(
        self,
        artifact_id: str,
        *,
        trace_id: str,
        expected_type: str | None = None,
        case_binding_sha256: str | None = None,
    ) -> StoredArtifact:
        """``get`` 的语义化别名，供服务层表达“前序 artifact 必须存在”。"""

        return self.get(
            artifact_id,
            trace_id=trace_id,
            expected_type=expected_type,
            case_binding_sha256=case_binding_sha256,
        )

    def require_case_bound(
        self,
        artifact_id: str,
        *,
        trace_id: str,
        expected_type: str,
        case_binding_sha256: str,
    ) -> StoredArtifact:
        """显式要求同 trace、同类型、同病例，供预测和报告阶段使用。"""

        return self.require(
            artifact_id,
            trace_id=trace_id,
            expected_type=expected_type,
            case_binding_sha256=case_binding_sha256,
        )
