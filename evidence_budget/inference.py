import contextlib
import re
import sqlite3
import time
from collections import defaultdict

from .common import canonical, code_hash, environment, file_hash, read_json, read_rows, stable_id, write_json
from .design import load_tokenizer


def result_rows(run):
    plan = run / "plan.jsonl"
    spec = read_json(run / "plan_manifest.json")
    if file_hash(plan) != spec["plan_sha256"]:
        raise ValueError(f"Plan checksum changed: {plan}")
    return (row for row in read_rows(plan) if row["kind"] == "evaluation")


def label_of(text):
    match = re.fullmatch(r"\s*\(?([ABCD])\)?[.)]?\s*", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def setup_database(path, identity):
    connection = sqlite3.connect(path, timeout=60)
    with connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE IF NOT EXISTS metadata (identity TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS results (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS failures (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("INSERT INTO metadata SELECT ? WHERE NOT EXISTS (SELECT 1 FROM metadata)",
                           (canonical(identity),))
        found = connection.execute("SELECT identity FROM metadata").fetchall()
        if found != [(canonical(identity),)]:
            raise ValueError(f"Execution fingerprint changed for {path}")
    return connection


class Runner:
    def __init__(self, spec, device, attention):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

        self.torch = torch
        self.spec = spec
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Evaluation requires a CUDA GPU")
        torch.cuda.set_device(self.device)
        self.tokenizer = load_tokenizer(spec)
        config = AutoConfig.from_pretrained(spec["id"], revision=spec["revision"])
        text_config = config.get_text_config()
        positional_cap = getattr(text_config, "max_position_embeddings", None)
        if positional_cap is not None and positional_cap < spec["context"]:
            raise ValueError(f"Configured context exceeds checkpoint support: {positional_cap}")
        loader = (AutoModelForImageTextToText if config.model_type == "gemma3" and
                  getattr(config, "vision_config", None) is not None else AutoModelForCausalLM)
        self.model = loader.from_pretrained(spec["id"], revision=spec["revision"],
                                            torch_dtype=getattr(torch, spec["precision"]),
                                            attn_implementation=attention, device_map=None,
                                            low_cpu_mem_usage=True).eval().to(self.device)
        self.identity = {"attention": attention, "device": torch.cuda.get_device_name(self.device),
                         "capability": torch.cuda.get_device_capability(self.device),
                         "gpu_total_bytes": torch.cuda.get_device_properties(self.device).total_memory,
                         "cuda": torch.version.cuda, "environment": environment(), "code_sha256": code_hash()}

    def run(self, row, benchmark=False):
        torch = self.torch
        request_started = time.perf_counter()
        inputs = self.tokenizer(row["prompt"], add_special_tokens=False, truncation=False,
                                return_tensors="pt").to(self.device)
        count = inputs["input_ids"].shape[1]
        if count != row["input_tokens"] or count > row["input_cap"]:
            raise ValueError(f"Prompt token count changed for {row['id']}")
        if count + row["output_limit"] > self.spec["context"]:
            raise ValueError(f"Input plus output exceeds model context: {row['id']}")
        torch.cuda.synchronize(self.device)
        if benchmark:
            torch.cuda.reset_peak_memory_stats(self.device)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if benchmark else []
        prefill_started = False
        prefill_ended = False

        def before(_module, _arguments):
            nonlocal prefill_started
            if not prefill_started:
                events[1].record()
                prefill_started = True

        def after(_module, _arguments, _result):
            nonlocal prefill_ended
            if not prefill_ended:
                events[2].record()
                prefill_ended = True

        with contextlib.ExitStack() as stack, torch.inference_mode():
            if benchmark:
                stack.enter_context(self.model.register_forward_pre_hook(before))
                stack.enter_context(self.model.register_forward_hook(after))
                events[0].record()
            started = time.perf_counter()
            generated = self.model.generate(**inputs, do_sample=False, num_beams=1, use_cache=True,
                                            max_new_tokens=row["output_limit"],
                                            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            if benchmark:
                events[3].record()
            torch.cuda.synchronize(self.device)
            generation_seconds = time.perf_counter() - started
        completion = generated[0, count:].tolist()
        eos = self.model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        text = self.tokenizer.decode(completion, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False).strip()
        result = {"id": row["id"], "output": text, "output_tokens": len(completion),
                  "input_tokens": count, "generation_seconds": generation_seconds,
                  "generation_hit_cap": len(completion) == row["output_limit"] and completion[-1] not in eos_ids,
                  "end_to_end_model_seconds": time.perf_counter() - request_started,
                  "prediction": label_of(text) if row["dataset"] == "belebele" else text}
        if benchmark:
            if not prefill_ended:
                raise RuntimeError("Could not time prefill with the selected model backend")
            result.update({"prefill_seconds": events[1].elapsed_time(events[2]) / 1000,
                           "after_prefill_seconds": events[2].elapsed_time(events[3]) / 1000,
                           "gpu_event_seconds": events[0].elapsed_time(events[3]) / 1000,
                           "allocated_peak_bytes": torch.cuda.max_memory_allocated(self.device),
                           "reserved_peak_bytes": torch.cuda.max_memory_reserved(self.device)})
        return result


def evaluate(run, device, attention, shard, shards):
    spec = read_json(run / "plan_manifest.json")
    if not 0 <= shard < shards:
        raise ValueError("Expected 0 <= shard < shards")
    row_iterator = result_rows(run)
    runner = Runner(spec["model"], device, attention)
    path = run / f"predictions-{shards:03d}-{shard:03d}.sqlite3"
    identity = {"plan_sha256": spec["plan_sha256"], "shard": shard, "shards": shards,
                **runner.identity}
    complete = failed = pending = consecutive_failures = 0
    started = time.perf_counter()
    with contextlib.closing(setup_database(path, identity)) as connection:
        present = {key for (key,) in connection.execute("SELECT id FROM results")}
        for row in row_iterator:
            key = row["id"]
            if row["status"] != "ready" or int(key[:12], 16) % shards != shard:
                continue
            if key in present:
                complete += 1
                continue
            pending += 1
            try:
                result = runner.run(row)
            except (RuntimeError, ValueError) as error:
                failed += 1
                consecutive_failures += 1
                payload = canonical({"error": str(error), "error_type": type(error).__name__})
                with connection:
                    connection.execute("INSERT OR REPLACE INTO failures SELECT ?, ? WHERE NOT EXISTS "
                                       "(SELECT 1 FROM results WHERE id = ?)", (key, payload, key))
                if isinstance(error, runner.torch.cuda.OutOfMemoryError):
                    runner.torch.cuda.empty_cache()
                if consecutive_failures >= 3:
                    break
                continue
            with connection:
                connection.execute("INSERT OR IGNORE INTO results VALUES (?, ?)", (key, canonical(result)))
                connection.execute("DELETE FROM failures WHERE id = ?", (key,))
            complete += 1
            consecutive_failures = 0
        report = {"complete": complete, "attempted_this_call": pending, "failures_this_call": failed,
                  "wall_seconds_this_call": time.perf_counter() - started, "fingerprint": identity}
        write_json(run / f"execution-{shards:03d}-{shard:03d}.json", report)
    if failed:
        raise RuntimeError(f"{failed} inference errors recorded in {path}")
    return report


def benchmark(run, device, attention, per_stratum, repeats, warmups):
    if min(per_stratum, repeats, warmups) < 1:
        raise ValueError("Benchmark count, repeats, and warmups must be positive")
    spec = read_json(run / "plan_manifest.json")
    groups = defaultdict(list)
    for row in result_rows(run):
        if row["status"] == "ready":
            key = (row["dataset"], row["input_tokens"].bit_length())
            groups[key].append(row)
    runner = Runner(spec["model"], device, attention)
    trials = []
    plan = {}
    for key, group in sorted(groups.items()):
        dataset, band = key
        ids = sorted(group, key=lambda row: stable_id("benchmark", row["id"]))[:per_stratum]
        plan[f"{dataset}/{band}"] = len(group)
        longest = max(group, key=lambda row: row["input_tokens"])
        for _ in range(warmups):
            runner.run(longest)
        for row in ids:
            for repeat in range(repeats):
                measured = runner.run(row, benchmark=True)
                trials.append({**measured, "dataset": dataset, "band": band, "repeat": repeat,
                               "condition": row["condition"], "language": row["language"],
                               "selection_seconds": row["selection_seconds"]})
    output = {"plan_sha256": spec["plan_sha256"], "hardware": runner.identity, "warmups": warmups,
              "repeats": repeats, "batch_size": 1, "planned_by_stratum": plan, "trials": trials}
    write_json(run / "benchmark.json", output)
    return output


def probe(run, device, attention):
    spec = read_json(run / "plan_manifest.json")
    records = [row for row in result_rows(run) if row["status"] == "ready"]
    if not records:
        raise ValueError("Probe requires ready evaluation rows")
    shortest = min(records, key=lambda row: row["input_tokens"])
    longest = max(records, key=lambda row: row["input_tokens"])
    mc_rows = [row for row in records if row["dataset"] == "belebele" and row["budget"] is not None]
    budget_edge = max(mc_rows, key=lambda row: row["input_tokens"], default=longest)
    selection = {row["id"]: row for row in (shortest, budget_edge, longest)}
    runner = Runner(spec["model"], device, attention)
    trials = []
    for row in selection.values():
        trials.append({**runner.run(row, benchmark=True), "condition": row["condition"],
                       "language": row["language"], "budget": row["budget"]})
    output = {"plan_sha256": spec["plan_sha256"], "hardware": runner.identity, "trials": trials}
    write_json(run / "probe.json", output)
    return output
