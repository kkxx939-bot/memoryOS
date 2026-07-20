from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest

from infrastructure.store.filesystem.memory_document_store import FileSystemMemoryDocumentStore
from infrastructure.store.memory.layout import tenant_control_root, user_memory_root
from memory.core.model import (
    ABSENT,
    ManagedDocument,
    PresentPath,
    QuarantinedDocument,
    UnmanagedDocument,
    UnsafePath,
)
from memory.core.structure.frontmatter import adopt_raw_document, render_new_document
from memory.ports.document_store import (
    DocumentConflictError,
    DocumentNotFoundError,
    DocumentUnsafeError,
)

TENANT = "tenant-a"
OWNER = "alice"
DOCUMENT_ID = "memdoc_0123456789ABCDEF"
OTHER_DOCUMENT_ID = "memdoc_FEDCBA9876543210"


def _memory_root(root: Path) -> Path:
    path = user_memory_root(root, TENANT, OWNER)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def _external_write(root: Path, relative_path: str, raw: bytes) -> Path:
    path = _memory_root(root) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(raw)
    return path


def _registration_by_path(scan) -> dict[str, object]:  # noqa: ANN001
    return {item.relative_path: item for item in scan.registrations}


@pytest.mark.parametrize(
    ("tenant_id", "owner_user_id"),
    [("default", None), (TENANT, None), (TENANT, OWNER)],
)
def test_filesystem_capability_probe_is_fail_closed_and_ephemeral(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tenant_id: str,
    owner_user_id: str | None,
) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    store.probe_write_capabilities(tenant_id, owner_user_id)
    probe_parent = (
        user_memory_root(tmp_path, tenant_id, owner_user_id)
        if owner_user_id is not None
        else tenant_control_root(tmp_path, tenant_id) / "system" / "memory-documents"
    )
    assert probe_parent.is_dir()
    assert not list(probe_parent.glob(".filesystem-probe-*"))

    failing_store = FileSystemMemoryDocumentStore(tmp_path)
    fsync_calls = 0

    def reject_temp_fsync(_descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("file fsync unsupported")
        return None

    monkeypatch.setattr(os, "fsync", reject_temp_fsync)
    with pytest.raises(DocumentUnsafeError, match="capability probe failed"):
        failing_store.probe_write_capabilities(tenant_id, owner_user_id)
    assert not list(probe_parent.glob(".filesystem-probe-*"))


def test_filesystem_capability_probe_cleanup_failure_is_not_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    probe_parent = tenant_control_root(tmp_path, TENANT) / "system" / "memory-documents"
    original_fsync = os.fsync
    saw_probe_directory = False

    def reject_cleanup_fsync(descriptor: int) -> None:
        nonlocal saw_probe_directory
        probe_exists = bool(list(probe_parent.glob(".filesystem-probe-*")))
        if probe_exists:
            saw_probe_directory = True
        elif saw_probe_directory:
            raise OSError("probe cleanup fsync unsupported")
        original_fsync(descriptor)

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "fsync", reject_cleanup_fsync)
        with pytest.raises(DocumentUnsafeError, match="capability probe cleanup failed"):
            store.probe_write_capabilities(TENANT)

    assert saw_probe_directory
    assert not list(probe_parent.glob(".filesystem-probe-*"))
    assert (TENANT, "__control__") not in store._files._probed_scopes

    store.probe_write_capabilities(TENANT)
    assert (TENANT, "__control__") in store._files._probed_scopes
    assert not list(probe_parent.glob(".filesystem-probe-*"))


def test_read_state_distinguishes_absent_empty_and_exact_lf_crlf_bytes(tmp_path: Path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    assert store.read_state(TENANT, OWNER, "profile.md") == ABSENT

    path = _external_write(tmp_path, "profile.md", b"")
    empty = store.read_state(TENANT, OWNER, "profile.md")
    assert empty == PresentPath("profile.md", hashlib.sha256(b"").hexdigest(), 0)

    lf = render_new_document(DOCUMENT_ID, "line one\nline two\n")
    crlf = lf.replace(b"\n", b"\r\n")
    path.write_bytes(lf)
    lf_state = store.read_state(TENANT, OWNER, "profile.md")
    path.write_bytes(crlf)
    crlf_state = store.read_state(TENANT, OWNER, "profile.md")
    path.write_bytes(crlf + b"\r\n")
    trailing_state = store.read_state(TENANT, OWNER, "profile.md")

    assert isinstance(lf_state, PresentPath)
    assert isinstance(crlf_state, PresentPath)
    assert isinstance(trailing_state, PresentPath)
    assert lf_state.raw_sha256 == hashlib.sha256(lf).hexdigest()
    assert crlf_state.raw_sha256 == hashlib.sha256(crlf).hexdigest()
    assert len({lf_state.raw_sha256, crlf_state.raw_sha256, trailing_state.raw_sha256}) == 3


def test_read_state_marks_symlink_hardlink_directory_and_permission_failure_unsafe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    root = _memory_root(tmp_path)
    topics = root / "knowledge" / "topics"
    topics.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_bytes(b"outside")
    (topics / "symlink.md").symlink_to(outside)
    source = topics / "hardlink-source.md"
    source.write_bytes(b"hard link")
    os.link(source, topics / "hardlink-copy.md")
    (topics / "directory.md").mkdir()

    assert isinstance(store.read_state(TENANT, OWNER, "knowledge/topics/symlink.md"), UnsafePath)
    assert isinstance(store.read_state(TENANT, OWNER, "knowledge/topics/hardlink-source.md"), UnsafePath)
    assert isinstance(store.read_state(TENANT, OWNER, "knowledge/topics/hardlink-copy.md"), UnsafePath)
    assert isinstance(store.read_state(TENANT, OWNER, "knowledge/topics/directory.md"), UnsafePath)
    unsafe_scan_paths = {item.relative_path for item in store.full_scan(TENANT, OWNER).unsafe_paths}
    assert {
        "knowledge/topics/symlink.md",
        "knowledge/topics/hardlink-source.md",
        "knowledge/topics/hardlink-copy.md",
        "knowledge/topics/directory.md",
    } <= unsafe_scan_paths

    permission_path = _external_write(tmp_path, "profile.md", b"content")

    def deny_read(*args: object, **kwargs: object) -> bytes:
        raise PermissionError("secret path")

    monkeypatch.setattr(store._files, "read_regular", deny_read)
    denied = store.read_state(TENANT, OWNER, "profile.md")
    assert denied == UnsafePath("profile.md", "permission denied while reading memory tree")
    assert permission_path.read_bytes() == b"content"


def test_read_state_marks_oversized_file_unsafe(tmp_path: Path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path, max_file_bytes=128, max_front_matter_bytes=64)
    _external_write(tmp_path, "profile.md", b"x" * 129)

    state = store.read_state(TENANT, OWNER, "profile.md")

    assert isinstance(state, UnsafePath)
    assert "byte limit" in state.reason


def test_full_scan_classifies_empty_missing_id_invalid_encoding_and_bad_yaml(tmp_path: Path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    _external_write(tmp_path, "profile.md", b"")
    _external_write(tmp_path, "preferences.md", b"---\nmemoryos_schema: 1\n---\n")
    _external_write(
        tmp_path,
        "knowledge/topics/bom.md",
        b"\xef\xbb\xbf---\nmemoryos_schema: 1\ndocument_id: memdoc_AAAAAAAAAAAAAAAA\n---\n",
    )
    _external_write(
        tmp_path,
        "knowledge/topics/utf8.md",
        b"---\nmemoryos_schema: 1\ndocument_id: memdoc_BBBBBBBBBBBBBBBB\n---\n\xff",
    )
    _external_write(
        tmp_path,
        "knowledge/topics/duplicate-key.md",
        b"---\nmemoryos_schema: 1\nmemoryos_schema: 1\ndocument_id: memdoc_CCCCCCCCCCCCCCCC\n---\n",
    )
    _external_write(
        tmp_path,
        "knowledge/topics/invalid-id.md",
        b"---\nmemoryos_schema: 1\ndocument_id: invalid\n---\n",
    )

    scan = store.full_scan(TENANT, OWNER)
    states = _registration_by_path(scan)

    assert scan.complete
    assert isinstance(states["profile.md"], UnmanagedDocument)
    assert isinstance(states["preferences.md"], UnmanagedDocument)
    for relative in (
        "knowledge/topics/bom.md",
        "knowledge/topics/utf8.md",
        "knowledge/topics/duplicate-key.md",
        "knowledge/topics/invalid-id.md",
    ):
        assert isinstance(states[relative], QuarantinedDocument)


def test_full_scan_quarantines_oversized_and_overdeep_front_matter(tmp_path: Path) -> None:
    store = FileSystemMemoryDocumentStore(
        tmp_path,
        max_file_bytes=2048,
        max_front_matter_bytes=128,
        max_front_matter_depth=2,
    )
    _external_write(
        tmp_path,
        "knowledge/topics/header.md",
        b"---\nmemoryos_schema: 1\ndocument_id: memdoc_AAAAAAAAAAAAAAAA\nnote: " + (b"x" * 256) + b"\n---\n",
    )
    _external_write(
        tmp_path,
        "knowledge/topics/deep.md",
        b"---\nmemoryos_schema: 1\ndocument_id: memdoc_BBBBBBBBBBBBBBBB\nvalue: [[[[1]]]]\n---\n",
    )

    states = _registration_by_path(store.full_scan(TENANT, OWNER))

    assert isinstance(states["knowledge/topics/header.md"], QuarantinedDocument)
    assert isinstance(states["knowledge/topics/deep.md"], QuarantinedDocument)


def test_full_scan_quarantines_duplicate_copied_and_changed_document_ids(tmp_path: Path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    first = render_new_document(DOCUMENT_ID, "first")
    _external_write(tmp_path, "knowledge/topics/first.md", first)
    initial = store.full_scan(TENANT, OWNER)
    assert isinstance(initial.registrations[0], ManagedDocument)

    _external_write(
        tmp_path,
        "knowledge/topics/first.md",
        render_new_document(OTHER_DOCUMENT_ID, "changed identity"),
    )
    changed = _registration_by_path(store.full_scan(TENANT, OWNER))
    assert isinstance(changed["knowledge/topics/first.md"], QuarantinedDocument)
    assert "changed" in changed["knowledge/topics/first.md"].reason
    with pytest.raises(DocumentNotFoundError):
        store.read_raw(TENANT, OWNER, document_id=DOCUMENT_ID)

    copied_store = FileSystemMemoryDocumentStore(tmp_path / "copied")
    duplicate_raw = render_new_document(DOCUMENT_ID, "copy")
    _external_write(tmp_path / "copied", "knowledge/topics/one.md", duplicate_raw)
    _external_write(tmp_path / "copied", "knowledge/entities/two.md", duplicate_raw)
    duplicate = copied_store.full_scan(TENANT, OWNER)

    assert len(duplicate.registrations) == 2
    assert all(isinstance(item, QuarantinedDocument) for item in duplicate.registrations)
    duplicate_reasons = [item.reason for item in duplicate.registrations if isinstance(item, QuarantinedDocument)]
    assert all("duplicate document_id" in reason for reason in duplicate_reasons)
    with pytest.raises(DocumentNotFoundError):
        copied_store.read_raw(TENANT, OWNER, document_id=DOCUMENT_ID)


def test_full_scan_quarantines_unicode_casefold_path_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    monkeypatch.setattr(
        "infrastructure.store.filesystem.memory_document_store.MemoryDocumentPathPolicy.collision_key",
        lambda _relative: "same-logical-path",
    )
    _external_write(
        tmp_path,
        "knowledge/topics/one.md",
        render_new_document(DOCUMENT_ID, "one"),
    )
    _external_write(
        tmp_path,
        "knowledge/topics/two.md",
        render_new_document(OTHER_DOCUMENT_ID, "two"),
    )

    scan = store.full_scan(TENANT, OWNER)

    assert len(scan.registrations) == 2
    assert all(isinstance(item, QuarantinedDocument) for item in scan.registrations)
    collision_reasons = [item.reason for item in scan.registrations if isinstance(item, QuarantinedDocument)]
    assert all("casefold" in reason for reason in collision_reasons)


def test_create_replace_delete_use_exact_digest_cas_and_private_modes(tmp_path: Path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    created_raw = render_new_document(DOCUMENT_ID, "# Initial\n")
    created = store.create(TENANT, OWNER, "knowledge/topics/file-memory.md", created_raw, expected=ABSENT)
    created_state = store.read_state(TENANT, OWNER, created.relative_path)

    assert isinstance(created_state, PresentPath)
    assert created_state.raw_sha256 == hashlib.sha256(created_raw).hexdigest()
    assert store.read_raw(TENANT, OWNER, document_id=DOCUMENT_ID) == created_raw
    path = _memory_root(tmp_path) / created.relative_path
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    replacement_raw = render_new_document(DOCUMENT_ID, "# Updated\n")
    replacement = store.replace(
        TENANT,
        OWNER,
        DOCUMENT_ID,
        replacement_raw,
        expected_state=created_state,
    )
    replacement_state = store.read_state(TENANT, OWNER, replacement.relative_path)
    assert isinstance(replacement_state, PresentPath)
    assert replacement.raw_sha256 == hashlib.sha256(replacement_raw).hexdigest()

    with pytest.raises(DocumentConflictError):
        store.replace(TENANT, OWNER, DOCUMENT_ID, created_raw, expected_state=created_state)
    assert store.read_raw(TENANT, OWNER, document_id=DOCUMENT_ID) == replacement_raw

    with pytest.raises(DocumentConflictError, match="change document_id"):
        store.replace(
            TENANT,
            OWNER,
            DOCUMENT_ID,
            render_new_document(OTHER_DOCUMENT_ID, "identity swap"),
            expected_state=replacement_state,
        )

    assert store.delete(TENANT, OWNER, DOCUMENT_ID, expected_state=replacement_state) == ABSENT
    assert store.read_state(TENANT, OWNER, replacement.relative_path) == ABSENT
    with pytest.raises(DocumentNotFoundError):
        store.read_raw(TENANT, OWNER, document_id=DOCUMENT_ID)


def test_delete_rejects_stale_digest_without_removing_user_edit(tmp_path: Path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    original = render_new_document(DOCUMENT_ID, "original")
    document = store.create(TENANT, OWNER, "profile.md", original, expected=ABSENT)
    expected = store.read_state(TENANT, OWNER, document.relative_path)
    user_edit = render_new_document(DOCUMENT_ID, "external edit")
    (_memory_root(tmp_path) / document.relative_path).write_bytes(user_edit)

    with pytest.raises(DocumentConflictError):
        store.delete(TENANT, OWNER, DOCUMENT_ID, expected_state=expected)
    assert store.read_raw(TENANT, OWNER, relative_path=document.relative_path) == user_edit


def test_replace_revalidates_after_temp_fsync_and_preserves_concurrent_edit(tmp_path: Path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    relative = "knowledge/topics/concurrent-replace.md"
    original = render_new_document(DOCUMENT_ID, "original")
    store.create(TENANT, OWNER, relative, original, expected=ABSENT)
    expected = store.read_state(TENANT, OWNER, relative)
    system_after = render_new_document(DOCUMENT_ID, "system update")
    external_after = render_new_document(DOCUMENT_ID, "external editor wins")
    path = _memory_root(tmp_path) / relative

    def edit_after_temp_fsync(stage: str) -> None:
        if stage == "temp_file_fsynced":
            path.write_bytes(external_after)

    with pytest.raises(DocumentConflictError, match="expected state"):
        store.replace(
            TENANT,
            OWNER,
            DOCUMENT_ID,
            system_after,
            expected_state=expected,
            operation_id="mdintent_concurrent_replace",
            fault_hook=edit_after_temp_fsync,
        )

    assert path.read_bytes() == external_after
    assert tuple(path.parent.glob("*.tmp")) == ()


def test_rename_preserves_id_bytes_and_uri_and_never_overwrites_target(tmp_path: Path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    raw = render_new_document(DOCUMENT_ID, "rename me")
    document = store.create(TENANT, OWNER, "knowledge/topics/old.md", raw, expected=ABSENT)
    expected = store.read_state(TENANT, OWNER, document.relative_path)
    renamed = store.rename(
        TENANT,
        OWNER,
        DOCUMENT_ID,
        "knowledge/entities/new.md",
        expected_old=expected,
        expected_new=ABSENT,
    )

    assert renamed.document_id == document.document_id
    assert renamed.uri == document.uri
    assert store.read_state(TENANT, OWNER, "knowledge/topics/old.md") == ABSENT
    assert store.read_raw(TENANT, OWNER, document_id=DOCUMENT_ID) == raw

    occupied_raw = render_new_document(OTHER_DOCUMENT_ID, "occupied")
    store.create(TENANT, OWNER, "knowledge/topics/occupied.md", occupied_raw, expected=ABSENT)
    renamed_state = store.read_state(TENANT, OWNER, renamed.relative_path)
    with pytest.raises(DocumentConflictError, match="not ABSENT"):
        store.rename(
            TENANT,
            OWNER,
            DOCUMENT_ID,
            "knowledge/topics/occupied.md",
            expected_old=renamed_state,
        )
    assert store.read_raw(TENANT, OWNER, document_id=OTHER_DOCUMENT_ID) == occupied_raw


def test_rename_can_install_exact_new_bytes_without_overwriting_target(tmp_path: Path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    old_path = "knowledge/topics/old-edit.md"
    new_path = "knowledge/entities/new-edit.md"
    before = render_new_document(DOCUMENT_ID, "before rename edit")
    after = render_new_document(DOCUMENT_ID, "after rename edit")
    store.create(TENANT, OWNER, old_path, before, expected=ABSENT)
    expected = store.read_state(TENANT, OWNER, old_path)

    renamed = store.rename(
        TENANT,
        OWNER,
        DOCUMENT_ID,
        new_path,
        expected_old=expected,
        expected_new=ABSENT,
        after_bytes=after,
        operation_id="mdintent_rename_edit",
    )

    assert renamed.raw_bytes == after
    assert store.read_state(TENANT, OWNER, old_path) == ABSENT
    assert store.read_raw(TENANT, OWNER, relative_path=new_path) == after


def test_adopt_is_explicit_digest_cas_and_preserves_user_body(tmp_path: Path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    original = b"# User-created note\r\n\r\nDo not reformat.\r\n"
    _external_write(tmp_path, "knowledge/topics/user-note.md", original)
    state = store.read_state(TENANT, OWNER, "knowledge/topics/user-note.md")
    assert isinstance(state, PresentPath)

    with pytest.raises(DocumentConflictError):
        store.adopt(
            TENANT,
            OWNER,
            "knowledge/topics/user-note.md",
            expected_raw_sha256="f" * 64,
        )
    assert store.read_raw(TENANT, OWNER, relative_path="knowledge/topics/user-note.md") == original

    adopted = store.adopt(
        TENANT,
        OWNER,
        "knowledge/topics/user-note.md",
        expected_raw_sha256=state.raw_sha256,
    )

    assert adopted.raw_bytes.endswith(original)
    assert adopted.document_id.startswith("memdoc_")
    assert store.read_raw(TENANT, OWNER, document_id=adopted.document_id) == adopted.raw_bytes


def test_create_rejects_duplicate_id_and_owner_scopes_document_lookup(tmp_path: Path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    raw = render_new_document(DOCUMENT_ID, "one")
    store.create(TENANT, OWNER, "knowledge/topics/one.md", raw, expected=ABSENT)

    with pytest.raises(DocumentConflictError, match="already registered"):
        store.create(TENANT, OWNER, "knowledge/topics/two.md", raw, expected=ABSENT)
    with pytest.raises(DocumentNotFoundError):
        store.read_raw(TENANT, "bob", document_id=DOCUMENT_ID)
    with pytest.raises(DocumentNotFoundError):
        store.read_raw("tenant-b", OWNER, document_id=DOCUMENT_ID)


def test_create_and_adopt_reject_unsafe_markdown_without_writing(tmp_path: Path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    with pytest.raises(DocumentUnsafeError, match="BOM"):
        store.create(
            TENANT,
            OWNER,
            "profile.md",
            b"\xef\xbb\xbf---\nmemoryos_schema: 1\ndocument_id: memdoc_0123456789ABCDEF\n---\n",
            expected=ABSENT,
        )
    assert store.read_state(TENANT, OWNER, "profile.md") == ABSENT

    invalid = b"---\nmemoryos_schema: 1\ndocument_id: invalid\n---\nbody"
    path = _external_write(tmp_path, "preferences.md", invalid)
    digest = hashlib.sha256(invalid).hexdigest()
    with pytest.raises((DocumentConflictError, DocumentUnsafeError)):
        store.adopt(TENANT, OWNER, "preferences.md", expected_raw_sha256=digest)
    assert path.read_bytes() == invalid


def test_store_rejects_runtime_root_with_symlink_ancestor(tmp_path: Path) -> None:
    real = tmp_path / "real"
    runtime = real / "runtime"
    runtime.mkdir(parents=True)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(DocumentUnsafeError, match="symbolic link"):
        FileSystemMemoryDocumentStore(linked / "runtime")


def test_absent_owner_root_has_no_publishable_identity(tmp_path: Path) -> None:
    scan = FileSystemMemoryDocumentStore(tmp_path).full_scan(TENANT, OWNER)

    assert scan.complete is True
    assert scan.root_identity == ""
    assert scan.registrations == ()


def test_complete_absence_clears_old_path_id_for_a_later_create(tmp_path: Path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    relative = "knowledge/topics/reused.md"
    original = render_new_document(DOCUMENT_ID, "old")
    replacement = render_new_document(OTHER_DOCUMENT_ID, "new")
    store.create(TENANT, OWNER, relative, original, expected=ABSENT)
    assert isinstance(store.full_scan(TENANT, OWNER).registrations[0], ManagedDocument)

    path = user_memory_root(tmp_path, TENANT, OWNER) / relative
    path.unlink()
    assert store.full_scan(TENANT, OWNER).registrations == ()
    path.write_bytes(replacement)

    registration = store.full_scan(TENANT, OWNER).registrations[0]
    assert isinstance(registration, ManagedDocument)
    assert registration.document_id == OTHER_DOCUMENT_ID


def test_full_scan_never_follows_an_ancestor_swapped_to_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    relative = "knowledge/topics/race.md"
    inside = render_new_document(DOCUMENT_ID, "inside")
    outside = render_new_document(OTHER_DOCUMENT_ID, "OUTSIDE_SECRET")
    store.create(TENANT, OWNER, relative, inside, expected=ABSENT)
    topics = user_memory_root(tmp_path, TENANT, OWNER) / "knowledge" / "topics"
    held = topics.with_name("topics-held")
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    (outside_root / "race.md").write_bytes(outside)
    original_open = os.open
    swapped = False

    def swap_before_open(path, flags, mode=0o777, *, dir_fd=None):  # noqa: ANN001, ANN202
        nonlocal swapped
        if path == "topics" and dir_fd is not None and not swapped:
            topics.rename(held)
            topics.symlink_to(outside_root, target_is_directory=True)
            swapped = True
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "open", swap_before_open)
        scan = store.full_scan(TENANT, OWNER)
    if swapped:
        topics.unlink()
        held.rename(topics)

    assert scan.managed == ()
    assert any(item.relative_path == "knowledge/topics" for item in scan.unsafe_paths)
    assert all(getattr(item, "raw_sha256", "") != hashlib.sha256(outside).hexdigest() for item in scan.registrations)


def test_full_scan_rejects_file_that_grows_past_limit_while_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FileSystemMemoryDocumentStore(
        tmp_path,
        max_file_bytes=128,
        max_front_matter_bytes=32,
    )
    raw = render_new_document(DOCUMENT_ID, "small")
    path = _external_write(tmp_path, "knowledge/topics/growing.md", raw)
    original_read = os.read
    grew = False

    def grow_after_first_read(descriptor: int, size: int) -> bytes:
        nonlocal grew
        data = original_read(descriptor, size)
        if not grew:
            with path.open("ab") as stream:
                stream.write(b"x" * 256)
            grew = True
        return data

    monkeypatch.setattr(os, "read", grow_after_first_read)
    scan = store.full_scan(TENANT, OWNER)

    assert scan.managed == ()
    assert any("byte limit" in item.reason for item in scan.unsafe_paths)


def test_full_scan_bounds_non_file_entry_enumeration(tmp_path: Path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path, max_scan_files=2)
    memory_root = _memory_root(tmp_path)
    for index in range(8):
        (memory_root / f"unexpected-{index}").mkdir()

    scan = store.full_scan(TENANT, OWNER)

    assert scan.complete is False
    assert "memory scan entry limit exceeded" in scan.errors


def test_create_rejects_casefold_collision_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    raw = render_new_document(DOCUMENT_ID, "collision")
    original_listdir = os.listdir

    def colliding_listdir(path):  # noqa: ANN001, ANN202
        if isinstance(path, int):
            return ["STRASSE.md"]
        return original_listdir(path)

    monkeypatch.setattr(os, "listdir", colliding_listdir)
    with pytest.raises(DocumentConflictError, match="casefold"):
        store.create(
            TENANT,
            OWNER,
            "knowledge/topics/straße.md",
            raw,
            expected=ABSENT,
        )
    assert store.read_state(TENANT, OWNER, "knowledge/topics/straße.md") == ABSENT


def test_operation_bound_temp_is_durably_cleaned_after_crash(tmp_path: Path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    relative = "knowledge/topics/temp-crash.md"
    raw = render_new_document(DOCUMENT_ID, "temp crash")

    def crash(stage: str) -> None:
        if stage == "temp_file_fsynced":
            raise RuntimeError("simulated process crash")

    with pytest.raises(RuntimeError, match="simulated"):
        store.create(
            TENANT,
            OWNER,
            relative,
            raw,
            expected=ABSENT,
            operation_id="mdintent_test_operation",
            fault_hook=crash,
        )
    assert len(tuple(user_memory_root(tmp_path, TENANT, OWNER).rglob("*.tmp"))) == 1

    assert (
        store.cleanup_operation_temps(
            TENANT,
            OWNER,
            {relative: hashlib.sha256(raw).hexdigest()},
            "mdintent_test_operation",
        )
        == 1
    )
    assert tuple(user_memory_root(tmp_path, TENANT, OWNER).rglob("*.tmp")) == ()


def test_operation_temp_cleanup_preserves_digest_mismatch(tmp_path: Path) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    relative = "knowledge/topics/temp-collision.md"
    raw = render_new_document(DOCUMENT_ID, "prepared bytes")

    def crash(stage: str) -> None:
        if stage == "temp_file_fsynced":
            raise RuntimeError("simulated process crash")

    with pytest.raises(RuntimeError, match="simulated"):
        store.create(
            TENANT,
            OWNER,
            relative,
            raw,
            expected=ABSENT,
            operation_id="mdintent_collision",
            fault_hook=crash,
        )
    temporary = next(user_memory_root(tmp_path, TENANT, OWNER).rglob("*.tmp"))
    temporary.write_bytes(b"externally modified temp")

    with pytest.raises(DocumentConflictError, match="prepared digest"):
        store.cleanup_operation_temps(
            TENANT,
            OWNER,
            {relative: hashlib.sha256(raw).hexdigest()},
            "mdintent_collision",
        )

    assert temporary.read_bytes() == b"externally modified temp"


@pytest.mark.parametrize(
    "fault_stage",
    ["temp_file_fsynced", "atomic_installed", "parent_fsynced"],
)
def test_adopt_operation_id_closes_store_crash_windows(
    tmp_path: Path,
    fault_stage: str,
) -> None:
    store = FileSystemMemoryDocumentStore(tmp_path)
    relative = "knowledge/topics/adopt-crash.md"
    original = b"# Adopt crash\n\nPreserve these exact user bytes.\n"
    path = _external_write(tmp_path, relative, original)
    expected_digest = hashlib.sha256(original).hexdigest()
    operation_id = "mdadopt_durable_receipt"
    expected_after = adopt_raw_document(
        original,
        DOCUMENT_ID,
        max_header_bytes=store.max_front_matter_bytes,
        max_depth=store.max_front_matter_depth,
    )

    def crash(stage: str) -> None:
        if stage == fault_stage:
            raise RuntimeError(f"simulated adopt crash at {stage}")

    with pytest.raises(RuntimeError, match="simulated adopt crash"):
        store.adopt(
            TENANT,
            OWNER,
            relative,
            expected_raw_sha256=expected_digest,
            assigned_document_id=DOCUMENT_ID,
            operation_id=operation_id,
            fault_hook=crash,
        )

    restarted = FileSystemMemoryDocumentStore(tmp_path)
    if fault_stage == "temp_file_fsynced":
        assert path.read_bytes() == original
        assert len(tuple(user_memory_root(tmp_path, TENANT, OWNER).rglob("*.tmp"))) == 1
        adopted = restarted.adopt(
            TENANT,
            OWNER,
            relative,
            expected_raw_sha256=expected_digest,
            assigned_document_id=DOCUMENT_ID,
            operation_id=operation_id,
        )
        assert adopted.raw_bytes == expected_after
    else:
        assert path.read_bytes() == expected_after
        scan = restarted.full_scan(TENANT, OWNER)
        assert len(scan.managed) == 1
        assert scan.managed[0].document_id == DOCUMENT_ID
        assert scan.managed[0].raw_sha256 == hashlib.sha256(expected_after).hexdigest()
    assert tuple(user_memory_root(tmp_path, TENANT, OWNER).rglob("*.tmp")) == ()
