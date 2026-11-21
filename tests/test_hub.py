import hashlib

import pytest

from superslim import hub


def test_download_model_verifies_checksum(tmp_path, monkeypatch):
    src = tmp_path / "src.npz"
    src.write_bytes(b"model-bytes")
    spec = {"url": src.as_uri(), "sha256": hashlib.sha256(b"model-bytes").hexdigest(), "filename": "m.npz", "bytes": 11, "license": "test"}
    monkeypatch.setitem(hub.MODELS, "test", spec)
    out = hub.download_model("test", dest=str(tmp_path / "dl"))
    assert out.read_bytes() == b"model-bytes"
    assert hub.download_model("test", dest=str(tmp_path / "dl")) == out  # already present and verified


def test_download_model_rejects_wrong_checksum(tmp_path, monkeypatch):
    src = tmp_path / "src.npz"
    src.write_bytes(b"tampered")
    monkeypatch.setitem(hub.MODELS, "bad", {"url": src.as_uri(), "sha256": "0" * 64, "filename": "b.npz", "bytes": 8, "license": "test"})
    with pytest.raises(RuntimeError):
        hub.download_model("bad", dest=str(tmp_path / "dl"))
    assert not list((tmp_path / "dl").glob("*"))  # nothing left behind


def test_unknown_model_name():
    with pytest.raises(ValueError):
        hub.download_model("nope")
