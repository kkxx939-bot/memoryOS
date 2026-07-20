from __future__ import annotations

import json

import pytest

from infrastructure.store.filesystem import BundleIntegrityError, FileSystemSourceStore
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_type import ContextType


class _ProtectedDomain:
    """测试用领域分类器：受保护 URI 必须通过完整 bundle 发布。"""

    @staticmethod
    def owns_uri(uri: str) -> bool:
        return "/protected/" in uri

    @staticmethod
    def owns_object(obj: ContextObject) -> bool:
        return _ProtectedDomain.owns_uri(obj.uri)


def _object() -> ContextObject:
    return ContextObject(
        uri="memoryos://user/u1/protected/note",
        context_type=ContextType.RESOURCE,
        title="受保护对象",
        tenant_id="default",
        owner_user_id="u1",
    )


def test_source_bundle_preserves_content_during_metadata_only_republish(tmp_path) -> None:  # noqa: ANN001
    store = FileSystemSourceStore(tmp_path, domain_classifier=_ProtectedDomain())
    obj = _object()
    store.write_object(obj, "正文-v1")

    obj.title = "更新后的标题"
    store.write_object(obj)

    assert store.read_object(obj.uri).title == "更新后的标题"
    assert store.read_content(obj.uri) == "正文-v1"


def test_source_bundle_keeps_old_generation_visible_when_publish_is_interrupted(tmp_path) -> None:  # noqa: ANN001
    store = FileSystemSourceStore(tmp_path, domain_classifier=_ProtectedDomain())
    obj = _object()
    store.write_object(obj, "正文-v1")

    def interrupt(stage: str, _uri: str, _generation_id: str) -> None:
        if stage == "before_current_pointer":
            raise RuntimeError("模拟 current 指针发布前崩溃")

    store.test_hook = interrupt
    with pytest.raises(RuntimeError, match="发布前崩溃"):
        store.write_object(obj, "正文-v2")

    assert store.read_content(obj.uri) == "正文-v1"


def test_source_bundle_rejects_tampered_generation_content(tmp_path) -> None:  # noqa: ANN001
    store = FileSystemSourceStore(tmp_path, domain_classifier=_ProtectedDomain())
    obj = _object()
    store.write_object(obj, "正文-v1")

    object_dir = store._object_dir(obj.uri)
    pointer = json.loads((object_dir / ".bundle-current.json").read_text(encoding="utf-8"))
    content_path = object_dir / ".bundle-generations" / pointer["generation_id"] / "content.md"
    content_path.write_text("被篡改", encoding="utf-8")

    with pytest.raises(BundleIntegrityError, match="digest mismatch"):
        store.read_content(obj.uri)
