import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(*values):
    return hashlib.sha256(canonical(values).encode("utf-8")).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_text(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            for line in lines:
                stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, value):
    atomic_text(path, [canonical(value) + "\n"])


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path, rows):
    atomic_text(path, (canonical(row) + "\n" for row in rows))


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def environment():
    packages = ["torch", "transformers", "tokenizers", "huggingface-hub",
                "pyarrow", "numpy", "matplotlib", "PyYAML"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions}


def code_hash():
    return stable_id(*(file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))))
