import time
from collections import Counter, defaultdict

import numpy as np

from .common import code_hash, environment, file_hash, read_json, stable_id, write_json, write_rows
from .data import prepared_data, sample_rows
from .prompt import MARKER, MC, QA, Prompt, head_tail, pack_ranked, prefix, span_window, whole
from .selection import Candidates, Retriever


def positioned(distractors, passage, location):
    index = {"early": 0, "middle": len(distractors) // 2, "late": len(distractors)}[location]
    return distractors[:index] + [passage] + distractors[index:]


def shared_fit(prompts, pool, ids, gold, limit):
    low, high = 0, len(ids) + 1
    while low + 1 < high:
        size = (low + high) // 2
        layout = positioned(ids[:size], gold, "middle")
        if all(prompt.count([whole(key, pool[key][language]) for key in layout]) <= limit
               for language, prompt in prompts.items()):
            low = size
        else:
            high = size
    return ids[:low]


def diagnostic_scaffold(prompts, pool, ids, gold, budget):
    selected = ids[:2]

    def lengths(layout):
        return {language: prompt.count([whole(key, pool[key][language]) for key in layout])
                for language, prompt in prompts.items()}

    evidence_lengths = lengths(positioned(selected, gold, "middle"))
    for place in ("early", "middle", "late"):
        if max(lengths(positioned(selected, gold, place)).values()) > budget:
            return None, None
    eligible = []
    for candidate in ids[2:]:
        replacement_lengths = lengths(positioned(selected, candidate, "middle"))
        if max(replacement_lengths.values()) <= budget:
            gap = max(abs(replacement_lengths[lang] - tokens) / tokens
                      for lang, tokens in evidence_lengths.items())
            eligible.append((gap, candidate))
    return selected, min(eligible)[1] if eligible else None


def measurement(row, language, prompt, pieces, condition, budget, seed, model, config,
                candidates, gold, selection_seconds=0, extra=None):
    item = row["variants"][language]
    cap = model["context"] - config["output_tokens"][row["dataset"]]
    maximum = min(budget, cap) if budget is not None else cap
    result = {"id": stable_id(row["id"], language, condition, budget, seed),
              "example_id": row["id"], "cluster": row["cluster"], "source_key": row["source_key"],
              "source_id": item["source_id"], "dataset": row["dataset"], "split": row["split"],
              "language": language, "condition": condition, "budget": budget, "order_seed": seed,
              "candidate_ids": candidates, "relevant_id": gold,
              "selection_seconds": selection_seconds, "answer": item.get("label"),
              "spans": item.get("spans", []), "output_limit": config["output_tokens"][row["dataset"]],
              "prompt_overhead": prompt.count([]), "input_cap": maximum, **(extra or {})}
    if pieces is None:
        return {**result, "status": "infeasible", "reason": "required content or scaffold does not fit"}
    details = prompt.describe(pieces, gold, result["spans"])
    if details["input_tokens"] > maximum:
        if condition in {"relevant_only", "same_complete", "no_relevant"}:
            return {**result, "status": "infeasible", "reason": "model context limit",
                    "required_input_tokens": details["input_tokens"]}
        raise ValueError(f"Budget violation in {result['id']}: {details['input_tokens']} > {maximum}")
    return {**result, **details, "status": "ready"}


def controlled(row, tokenizer, candidates, model, config, profile, diagnostic, replicate_order, retriever):
    prompts = {language: Prompt(tokenizer, item) for language, item in row["variants"].items()}
    gold = row["cluster"]
    all_budgets = config["budgets"]["belebele"]
    diagnostic_budget = config["diagnostic_budget"]
    matched_budget = config["matched_budget"]
    cap = model["context"] - config["output_tokens"]["belebele"]
    orders = config["profiles"][profile]["orders"]
    for order in (orders if replicate_order else orders[:1]):
        draw_seed = orders[0] if order in orders[:2] else order
        proposed, rejected = candidates.select(row, config["max_candidates"], config["seed"], draw_seed)
        proposed.sort(key=lambda key: stable_id("passage_order", order, row["id"], key))
        pool = candidates.pool
        distractors = shared_fit(prompts, pool, proposed, gold, cap)
        if not distractors:
            raise ValueError(f"No constructed context can fit all languages for {row['id']}")
        layout = positioned(distractors, gold, "middle")
        scaffold, replacement = (diagnostic_scaffold(prompts, pool, proposed, gold, min(cap, matched_budget))
                                 if diagnostic else (None, None))
        if diagnostic:
            high_position_ids = proposed[:config["position_distractors"]]
            high_limit = min(cap, config["position_budget"])
            high_position_fits = all(
                prompt.count([whole(key, pool[key][lang]) for key in positioned(high_position_ids, gold, place)])
                <= high_limit for lang, prompt in prompts.items() for place in ("early", "middle", "late"))
            shared_candidates = []
            for key in layout:
                trial = [*shared_candidates, key]
                if not all(prompt.count([whole(unit, pool[unit][lang]) for unit in trial]) <= min(cap, matched_budget)
                           for lang, prompt in prompts.items()):
                    break
                shared_candidates = trial
        for language, prompt in prompts.items():
            def materialize(ids):
                return [whole(key, pool[key][language]) for key in ids]

            def measure(condition, pieces, budget=None, seconds=0):
                return measurement(row, language, prompt, pieces, condition, budget, order,
                                   model, config, layout, gold, seconds, extra)

            pieces = materialize(layout)
            extra = {"distractor_rejections": rejected, "candidate_count_requested": len(proposed),
                     "unpacked_candidate_ids": [key for key in proposed if key not in distractors],
                     "unbudgeted_input_tokens": prompt.count(pieces), "replacement_id": replacement,
                     "distractor_draw_seed": draw_seed, "screened_candidate_ids": proposed}
            if order == orders[0]:
                yield measure("relevant_only", materialize([gold]))
                yield measure("question_only", [])
                yield measure("same_complete", pieces)
                for budget in all_budgets:
                    limit = min(cap, budget)
                    started = time.perf_counter()
                    kept = prefix(prompt, pieces, limit) if prompt.count([]) <= limit else None
                    yield measure("prefix", kept, budget, time.perf_counter() - started)
            if not diagnostic:
                continue
            budget = diagnostic_budget
            limit = min(cap, budget)
            if order != orders[0]:
                for repeat_budget in sorted({diagnostic_budget, matched_budget}):
                    repeat_cap = min(cap, repeat_budget)
                    started = time.perf_counter()
                    kept = prefix(prompt, pieces, repeat_cap) if prompt.count([]) <= repeat_cap else None
                    yield measure("prefix", kept, repeat_budget, time.perf_counter() - started)
            started = time.perf_counter()
            kept = head_tail(prompt, pieces, limit) if prompt.count([]) <= limit else None
            yield measure("head_tail", kept, budget, time.perf_counter() - started)
            started = time.perf_counter()
            ranks = sorted(layout, key=lambda key: stable_id(config["seed"], order, row["id"], "random", key))
            kept = pack_ranked(prompt, pieces, ranks, limit) if prompt.count([]) <= limit else None
            yield measure("random", kept, budget, time.perf_counter() - started)
            if retriever is not None:
                ranks, seconds = retriever.rank(language, row["variants"][language]["question"], layout)
                started = time.perf_counter()
                kept = pack_ranked(prompt, pieces, ranks, limit) if prompt.count([]) <= limit else None
                yield measure("retrieval", kept, budget, seconds + time.perf_counter() - started)
            ranks = [gold] + [key for key in layout if key != gold]
            kept = pack_ranked(prompt, pieces, ranks, limit)
            yield measure("oracle", kept if any(piece.passage_id == gold for piece in kept) else None, budget)
            for location in ("early", "middle", "late"):
                matched = materialize(positioned(scaffold, gold, location)) if scaffold else None
                yield measure(f"matched_{location}", matched, matched_budget)
                high = materialize(positioned(high_position_ids, gold, location)) if high_position_fits else None
                yield measure(f"matched_{location}", high, config["position_budget"])
            absent = materialize(positioned(scaffold, replacement, "middle")) if replacement else None
            yield measure("matched_no_evidence", absent, matched_budget)
            yield measure("matched_prefix", materialize(shared_candidates), matched_budget)
            yield measure("no_relevant", materialize(distractors))


def natural(row, tokenizer, model, config, profile):
    dataset = row["dataset"]
    cap = model["context"] - config["output_tokens"][dataset]
    for language, item in row["variants"].items():
        prompt = Prompt(tokenizer, item)
        piece = whole(row["id"], item["text"])
        extra = {"unbudgeted_input_tokens": prompt.count([piece])}

        def measure(condition, pieces, budget=None, seconds=0):
            return measurement(row, language, prompt, pieces, condition, budget, config["seed"],
                               model, config, [row["id"]], row["id"], seconds, extra)

        yield measure("same_complete", [piece])
        yield measure("question_only", [])
        for budget in config["budgets"][dataset]:
            limit = min(budget, cap)
            started = time.perf_counter()
            retained = prefix(prompt, [piece], limit) if prompt.count([]) <= limit else None
            yield measure("prefix", retained, budget, time.perf_counter() - started)
            if profile != "expanded" and budget != config["natural_diagnostic_budget"][dataset]:
                continue
            started = time.perf_counter()
            retained = head_tail(prompt, [piece], limit) if prompt.count([]) <= limit else None
            yield measure("head_tail", retained, budget, time.perf_counter() - started)
            retained = span_window(prompt, piece, item["spans"], limit)
            yield measure("oracle_span_window", retained, budget)


def load_tokenizer(model):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["tokenizer_revision"], use_fast=True)
    if not tokenizer.is_fast or not tokenizer.chat_template:
        raise ValueError(f"{model['id']} requires a fast tokenizer and a chat template")
    return tokenizer


def make_plan(config, profile, key, data_root, out, use_retriever, device):
    model = config["models"][key]
    tokenizer = load_tokenizer(model)
    metadata = {"profile": profile, "model": model, "model_key": key,
                "config": config, "dataset_manifest": read_json(data_root / "prepared" / "manifest.json"),
                "prompt_specification": {"multiple_choice": MC, "extractive": QA,
                                         "context_slot": MARKER, "chat_template": tokenizer.chat_template},
                "environment": environment(), "code_sha256": code_hash(), "samples": {},
                "retrieval_enabled": use_retriever, "retrieval_device": device if use_retriever else None}
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    lengths = defaultdict(list)
    screening = set()
    skipped = defaultdict(list)
    path = out / "plan.jsonl"
    manifest_path = out / "plan_manifest.json"
    if path.exists() or manifest_path.exists():
        raise FileExistsError(f"Plan already exists: {out}. Use a new run directory.")

    def records():
        for dataset, rows in prepared_data(data_root):
            sample, stats = sample_rows(rows, dataset, profile, config)
            metadata["samples"][dataset] = stats
            candidates = None
            retriever = None
            diagnostics = set()
            replications = set()
            if dataset == "belebele":
                pilot = profile in {"pilot", "smoke"}
                pool = [row for row in rows
                        if (int(stable_id(config["seed"], row["cluster"])[:8], 16) % 5 == 0) == pilot]
                candidates = Candidates(pool)
                diagnostics = {row["id"] for row in sample[:config["profiles"][profile]["diagnostic_limit"]]}
                replications = {row["id"] for row in sample[:config["profiles"][profile]["order_replication_limit"]]}
                if use_retriever:
                    spec = config["retriever"]
                    retriever = Retriever(spec["id"], spec["revision"], device)
                    metadata["retrieval"] = retriever.build(candidates)
            for row in sample:
                if dataset == "belebele":
                    generated = controlled(row, tokenizer, candidates, model, config, profile,
                                           row["id"] in diagnostics, row["id"] in replications, retriever)
                else:
                    generated = natural(row, tokenizer, model, config, profile)
                for condition in generated:
                    label = "/".join(str(condition[field]) for field in
                                     ("dataset", "language", "condition", "budget", "order_seed", "status"))
                    counts[label] += 1
                    if condition["condition"] == "same_complete":
                        lengths[f'{dataset}/{condition["language"]}'].append(condition["unbudgeted_input_tokens"])
                    if condition["status"] == "infeasible":
                        skipped[label].append(condition["id"])
                    if dataset == "belebele":
                        screen_key = condition["example_id"], condition["order_seed"]
                        if screen_key not in screening:
                            yield {"kind": "construction", "example_id": screen_key[0], "order_seed": screen_key[1],
                                   "candidate_ids": condition["candidate_ids"],
                                   "distractor_draw_seed": condition["distractor_draw_seed"],
                                   "screened_candidate_ids": condition["screened_candidate_ids"],
                                   "unpacked_candidate_ids": condition["unpacked_candidate_ids"],
                                   "distractor_rejections": condition["distractor_rejections"]}
                            screening.add(screen_key)
                    yield {"kind": "evaluation", **condition}

    write_rows(path, records())
    metadata["row_counts"] = dict(counts)
    metadata["infeasible_ids"] = dict(skipped)
    metadata["length_quantiles"] = {key: dict(zip(("min", "p25", "p50", "p75", "p90", "max"),
                                                np.quantile(values, [0, .25, .5, .75, .9, 1]).tolist()))
                                    for key, values in lengths.items()}
    metadata["plan_sha256"] = file_hash(path)
    write_json(manifest_path, metadata)
    return metadata


def audit(config, key, data_root, out):
    model = config["models"][key]
    tokenizer = load_tokenizer(model)
    rows = []
    groups = defaultdict(list)
    paired = defaultdict(list)
    answer_lengths = defaultdict(list)
    for dataset, examples in prepared_data(data_root):
        for item in examples:
            item_lengths = {}
            for language, variant in item["variants"].items():
                prompt = Prompt(tokenizer, variant)
                piece = whole(item["cluster"], variant["text"])
                length = prompt.count([piece])
                context_tokens = len(prompt.encode(piece.text)["input_ids"])
                groups[(dataset, item["split"], language)].append((length, context_tokens))
                item_lengths[language] = context_tokens
                answer_max = (max(len(prompt.encode(span["text"])["input_ids"]) for span in variant["spans"])
                              if "spans" in variant else None)
                if answer_max is not None:
                    answer_lengths[(dataset, item["split"], language)].append(answer_max)
                rows.append({"dataset": dataset, "id": item["id"], "split": item["split"],
                             "source_id": variant["source_id"], "language": language,
                             "chars": len(piece.text), "whitespace_words": len(piece.text.split()),
                             "context_tokens": context_tokens, "template_only_tokens": prompt.count([]),
                             "full_tokens": length, "max_reference_answer_tokens": answer_max})
            if len(item_lengths) > 1 and item_lengths.get("eng_Latn"):
                for language, length in item_lengths.items():
                    paired[(dataset, item["split"], language)].append(length / item_lengths["eng_Latn"])
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "lengths.jsonl", rows)
    summary = {}
    for (dataset, split, language), values in groups.items():
        lengths = np.array(values)
        summary["/".join((dataset, split, language))] = {
            "n": len(values), "prompt_quantiles": np.quantile(lengths[:, 0], [0, .1, .25, .5, .75, .9, 1]).tolist(),
            "context_quantiles": np.quantile(lengths[:, 1], [0, .1, .25, .5, .75, .9, 1]).tolist(),
            "fraction_over_candidate_budgets": {budget: float(np.mean(lengths[:, 0] > budget))
                                                 for budget in config["budgets"][dataset]}}
    ratios = {"/".join(key): {"n": len(values),
                             "p10_p50_p90": np.quantile(values, [.1, .5, .9]).tolist()}
              for key, values in paired.items()}
    for (dataset, split, language), values in answer_lengths.items():
        summary["/".join((dataset, split, language))]["fraction_items_with_reference_over_output_cap"] = float(
            np.mean(np.array(values) > config["output_tokens"][dataset]))
    write_json(out / "audit.json", {"model": model, "environment": environment(), "summary": summary,
                                   "paired_context_token_ratio_vs_english": ratios})
    return summary
