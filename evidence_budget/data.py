import json
import re
import shutil
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from .common import file_hash, read_json, read_rows, stable_id, write_json, write_rows


MLQA_URL = "https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip"
MLQA_SHA = "246e8089933d13007fe80684d5c5c0713d6834cf8b3b4a0ec7c66f0a0d2baac8"
LANGUAGES = {
    "eng_Latn": "English", "ben_Beng": "Bangla", "hin_Deva": "Hindi",
    "urd_Arab": "Urdu", "npi_Deva": "Nepali", "sin_Sinh": "Sinhala",
    "pbt_Arab": "Southern Pashto", "tam_Taml": "Tamil"
}
REGISTRY = {
    "belebele": {"source": "facebook/belebele", "split": "test", "license": "CC-BY-SA-4.0",
                 "provenance": "English questions and choices authored then professionally translated; FLORES-200 parallel passages",
                 "alignment": "link + question_number"},
    "mlqa": {"source": MLQA_URL, "split": "dev and test", "license": "CC-BY-SA-3.0",
             "version": "MLQA_V1", "provenance": "English questions translated, with aligned Wikipedia contexts and annotated local spans",
             "alignment": "shared question ID, English/Hindi intersection"},
    "tydi": {"source": "google-research-datasets/tydiqa", "split": "secondary_task validation",
             "license": "Apache-2.0 (release mirror metadata); underlying Wikipedia text has separate terms",
             "version": "GoldP v1.1",
             "provenance": "independently authored information-seeking questions and native Wikipedia contexts",
             "alignment": "none across languages"}
}
SCRIPTS = {
    "Beng": r"[\u0980-\u09ff]", "Deva": r"[\u0900-\u097f]",
    "Arab": r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]",
    "Sinh": r"[\u0d80-\u0dff]", "Taml": r"[\u0b80-\u0bff]",
    "Latn": r"[A-Za-z]"
}


def download(url, path):
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle, urllib.request.urlopen(url, timeout=120) as response:
            shutil.copyfileobj(response, handle)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def hub_url(repository, revision, filename):
    return f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{filename}"


def belebele_rows(config, raw):
    indexes = {}
    paths = {}
    for language in config["languages"]:
        url = hub_url("facebook/belebele", config["belebele_revision"], f"data/{language}.jsonl")
        path = download(url, raw / "belebele" / config["belebele_revision"] / f"{language}.jsonl")
        paths[language] = path
        index = {}
        passages = {}
        for row in read_rows(path):
            link = row["link"]
            text = row["flores_passage"]
            key = (link, int(row["question_number"]))
            if (key in index or row["dialect"] != language or
                    (link in passages and passages[link] != text)):
                raise ValueError(f"Inconsistent Belebele passage or question: {language} {key}")
            if not re.search(SCRIPTS[language.split("_")[1]], text):
                raise ValueError(f"Passage has no expected script characters: {language} {key}")
            passages[link] = text
            options = [row[f"mc_answer{number}"] for number in range(1, 5)]
            answer = int(row["correct_answer_num"]) - 1
            if answer not in range(4) or not row["question"] or any(not option for option in options):
                raise ValueError(f"Invalid Belebele question: {language} {key}")
            index[key] = {"text": text, "question": row["question"], "options": options,
                          "label": "ABCD"[answer], "source_id": link, "release_date": row["ds"]}
        if len(index) != 900:
            raise ValueError(f"{language}: expected 900 rows, got {len(index)}")
        indexes[language] = index
    reference = indexes["eng_Latn"]
    for language, index in indexes.items():
        if reference.keys() != index.keys():
            raise ValueError(f"Missing or extra alignment IDs in {language}")
        if any(index[key]["label"] != reference[key]["label"] for key in reference):
            raise ValueError(f"Answer index mismatch in {language}")
    rows = [{"dataset": "belebele", "split": "test", "id": stable_id("belebele", key),
             "source_key": list(key), "cluster": key[0],
             "variants": {language: index[key] for language, index in indexes.items()}}
            for key in sorted(reference)]
    return rows, {language: file_hash(path) for language, path in paths.items()}


def squad_rows(document, language):
    index = {}
    for entry in document["data"]:
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                spans = [{"start": int(answer["answer_start"]), "text": answer["text"]}
                         for answer in qa["answers"]]
                for span in spans:
                    start = span["start"]
                    if start < 0 or not span["text"] or context[start:start + len(span["text"])] != span["text"]:
                        raise ValueError(f"Invalid answer offset in {language}/{qa['id']}: {span!r}")
                if qa["id"] in index or not spans:
                    raise ValueError(f"Duplicate or unanswerable question {qa['id']}")
                index[qa["id"]] = {"text": context, "question": qa["question"],
                                   "spans": spans, "source_id": entry["title"]}
    return index


def components(pairs):
    parents = {}

    def root(node):
        while parents.get(node, node) != node:
            node = parents[node]
        return node

    for first, second in pairs:
        a, b = sorted((root(first), root(second)))
        if a != b:
            parents[b] = a
    return {node: root(node) for pair in pairs for node in pair}


def mlqa_rows(raw):
    path = download(MLQA_URL, raw / "mlqa" / "MLQA_V1.zip")
    if file_hash(path) != MLQA_SHA:
        raise ValueError(f"Unexpected MLQA_V1 archive checksum: {path}")
    rows = []
    coverage = {}
    with zipfile.ZipFile(path) as archive:
        for split in ("dev", "test"):
            en, hi = [squad_rows(json.loads(archive.read(
                f"MLQA_V1/{split}/{split}-context-{language}-question-{language}.json"
            )), language) for language in ("en", "hi")]
            common = sorted(en.keys() & hi.keys())
            coverage[split] = {"english": len(en), "hindi": len(hi), "paired": len(common)}
            if not common:
                raise ValueError(f"No paired English and Hindi examples in {split}")
            pairs = [(("en", en[key]["source_id"]), ("hi", hi[key]["source_id"])) for key in common]
            groups = components(pairs)
            rows.extend({"dataset": "mlqa", "split": split, "id": stable_id("mlqa", split, key),
                         "source_key": key, "cluster": stable_id("mlqa", groups[pair[0]]),
                         "variants": {"eng_Latn": en[key], "hin_Deva": hi[key]}}
                        for key, pair in zip(common, pairs))
    return rows, {"zip_sha256": file_hash(path), "coverage": coverage}


def align_span(context, start, text):
    if not text or start < 0:
        return None
    if context[start:start + len(text)] == text:
        return {"start": start, "text": text, "original_start": start}
    nearby = [i for i in range(max(0, start - 8), min(len(context), start + 9))
              if context.startswith(text, i)]
    if len(nearby) != 1:
        return None
    return {"start": nearby[0], "text": text, "original_start": start}


def tydi_rows(config, raw):
    import pyarrow.parquet as pq

    repository = "google-research-datasets/tydiqa"
    url = hub_url(repository, config["tydi_revision"], "secondary_task/validation-00000-of-00001.parquet")
    path = download(url, raw / "tydi" / config["tydi_revision"] / "goldp_validation.parquet")
    rows = []
    seen = set()
    repairs = []
    excluded = []
    languages = {"bengali": "ben_Beng", "english": "eng_Latn"}
    for row in pq.read_table(path).to_pylist():
        key = row["id"]
        language = languages.get(key.split("-", 1)[0].lower())
        if language is None:
            continue
        if key in seen:
            raise ValueError(f"Duplicate TyDi question {key}")
        seen.add(key)
        spans = []
        for index, (start, text) in enumerate(zip(row["answers"]["answer_start"],
                                                  row["answers"]["text"], strict=True)):
            aligned = align_span(row["context"], start, text)
            if aligned is None:
                excluded.append({"id": key, "index": index, "released_start": start})
                continue
            if aligned["start"] != start:
                repairs.append({"id": key, "index": index, "released_start": start,
                                "verified_start": aligned["start"]})
            spans.append(aligned)
        if not spans:
            excluded.append({"id": key, "reason": "no_verified_span"})
            continue
        rows.append({"dataset": "tydi", "split": "validation", "id": stable_id("tydi", key),
                     "source_key": key, "cluster": stable_id("tydi", language, row["title"]),
                     "variants": {language: {"text": row["context"], "question": row["question"],
                                              "spans": spans, "source_id": row["title"]}}})
    if {language for row in rows for language in row["variants"]} != set(languages.values()):
        raise ValueError("Missing TyDi English or Bangla native data")
    return rows, {"parquet_sha256": file_hash(path), "offset_repairs": repairs,
                  "annotation_exclusions": excluded}


def prepare(config, root):
    raw = root / "raw"
    prepared = root / "prepared"
    manifest = {}
    for name, loader in (("belebele", lambda: belebele_rows(config, raw)),
                         ("mlqa", lambda: mlqa_rows(raw)),
                         ("tydi", lambda: tydi_rows(config, raw))):
        rows, source = loader()
        path = prepared / f"{name}.jsonl"
        write_rows(path, rows)
        coverage = defaultdict(int)
        for row in rows:
            for language in row["variants"]:
                coverage[f'{row["split"]}/{language}'] += 1
        manifest[name] = {**REGISTRY[name], "integrity": source, "rows": len(rows),
                          "coverage": dict(coverage), "prepared_sha256": file_hash(path)}
    manifest["revisions"] = {key: config[key] for key in ("belebele_revision", "tydi_revision")}
    write_json(prepared / "manifest.json", manifest)
    return manifest


def prepared_data(root):
    manifest = read_json(root / "prepared" / "manifest.json")
    for dataset in REGISTRY:
        path = root / "prepared" / f"{dataset}.jsonl"
        if manifest[dataset]["prepared_sha256"] != file_hash(path):
            raise ValueError(f"Prepared data checksum mismatch: {path}")
        yield dataset, list(read_rows(path))


def sample_rows(rows, dataset, profile, config):
    pilot = profile in {"pilot", "smoke"}
    if dataset == "mlqa":
        dev_clusters = {row["cluster"] for row in rows if row["split"] == "dev"}
        eligible = [row for row in rows if row["split"] == ("dev" if pilot else "test")
                    and (pilot or row["cluster"] not in dev_clusters)]
    else:
        eligible = [row for row in rows if
                    (int(stable_id(config["seed"], row["cluster"])[:8], 16) % 5 == 0) == pilot]
    groups = defaultdict(list)
    for row in eligible:
        group = next(iter(row["variants"])) if dataset == "tydi" else "paired"
        groups[group].append(row)
    limit = config["profiles"][profile][dataset]
    chosen = []
    for group in sorted(groups):
        ranked = sorted(groups[group], key=lambda row: stable_id(config["seed"], "sample", row["id"]))
        if not ranked:
            raise ValueError(f"No eligible {dataset} rows for {profile}/{group}")
        chosen.extend(ranked[:limit])
    if not chosen:
        raise ValueError(f"No eligible {dataset} rows for {profile}")
    return chosen, {"source_rows": len(rows), "eligible": len(eligible), "selected": len(chosen),
                    "requested_per_group": limit, "eligible_by_group": {k: len(v) for k, v in groups.items()}}
