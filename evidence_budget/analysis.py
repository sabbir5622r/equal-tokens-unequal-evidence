import csv
import io
import json
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from contextlib import closing

import numpy as np

from .common import atomic_text, read_json, stable_id, write_json, write_rows
from .inference import result_rows


def answer_tokens(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(character for character in text if not unicodedata.category(character).startswith("P"))
    return text.split()


def scores(row, prediction):
    if row["dataset"] == "belebele":
        return {"accuracy": float(prediction == row["answer"])}
    guessed = answer_tokens(prediction or "")
    gold = [answer_tokens(span["text"]) for span in row["spans"]]
    exact = [float(guessed == reference) for reference in gold]
    f1 = []
    for reference in gold:
        common = sum((Counter(guessed) & Counter(reference)).values())
        f1.append(2 * common / (len(guessed) + len(reference)) if common else float(guessed == reference))
    return {"exact_match": max(exact), "f1": max(f1)}


def read_predictions(run, fingerprint):
    predictions = {}
    failures = {}
    shard_counts = set()
    for path in sorted(run.glob("predictions-*.sqlite3")):
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            metadata = connection.execute("SELECT identity FROM metadata").fetchall()
            if len(metadata) != 1:
                raise ValueError(f"Invalid checkpoint metadata: {path}")
            identity = json.loads(metadata[0][0])
            if identity["plan_sha256"] != fingerprint:
                raise ValueError(f"Prediction checkpoint belongs to a different plan: {path}")
            shard_counts.add(identity["shards"])
            for key, value in connection.execute("SELECT id, payload FROM results"):
                if key in predictions:
                    raise ValueError(f"Duplicate predictions across checkpoints: {key}")
                predictions[key] = json.loads(value)
            for key, value in connection.execute("SELECT id, payload FROM failures"):
                failures[key] = json.loads(value)
    if len(shard_counts) > 1:
        raise ValueError(f"Mixed shard counts in {run}")
    return predictions, failures


def availability(row):
    if row["dataset"] == "belebele":
        return row["passage_state"] == "complete"
    return row["answer_span_retained"]


def load_outcomes(run, allow_partial):
    manifest = read_json(run / "plan_manifest.json")
    predictions, failures = read_predictions(run, manifest["plan_sha256"])
    ready = set()
    missing = []
    outcomes = []
    infeasible = Counter()
    for row in result_rows(run):
        if row["status"] != "ready":
            infeasible["/".join(str(row[f]) for f in ("dataset", "language", "condition", "budget"))] += 1
            continue
        if row["id"] in ready:
            raise ValueError(f"Duplicate evaluation ID {row['id']}")
        ready.add(row["id"])
        prediction = predictions.get(row["id"])
        if prediction is None:
            missing.append(row["id"])
            continue
        if prediction["input_tokens"] != row["input_tokens"]:
            raise ValueError(f"Prediction input lengths differ for {row['id']}")
        score = scores(row, prediction["prediction"])
        outcomes.append({key: row[key] for key in (
            "id", "example_id", "cluster", "dataset", "language", "condition", "budget",
            "order_seed", "passage_state", "first_evidence_token", "answer_span_retained",
            "input_tokens", "prompt_overhead", "unbudgeted_input_tokens", "candidate_ids",
            "complete_passage_ids", "retained", "selection_seconds"
        )} | {"scores": score, "available": availability(row), "generation_seconds": prediction["generation_seconds"],
              "output_tokens": prediction["output_tokens"], "generation_hit_cap": prediction["generation_hit_cap"],
              "format_failure":
              prediction["prediction"] is None if row["dataset"] == "belebele" else not prediction["prediction"]})
    if set(predictions) - ready or set(failures) - ready:
        raise ValueError("Checkpoint contains IDs absent from the plan")
    status = {"profile": manifest["profile"], "plan_sha256": manifest["plan_sha256"],
              "ready": len(ready), "complete": len(outcomes), "missing_ids": missing,
              "failure_details": failures, "infeasible_counts": dict(infeasible),
              "retrieval_enabled": manifest["retrieval_enabled"],
              "complete_evaluation": not missing and not failures,
              "publication_status": "unreviewed"}
    if (missing or failures) and not allow_partial:
        raise ValueError(f"{len(missing)} missing and {len(failures)} failed ready rows in {run}. "
                         "Resume evaluation or explicitly request provisional analysis.")
    return manifest, outcomes, status


def interval(values, draws, seed):
    clusters = defaultdict(list)
    for cluster, value in values:
        clusters[cluster].append(value)
    n = len(values)
    if not n:
        return {"n": 0, "clusters": 0, "mean": None, "lo": None, "hi": None}
    mean = sum(value for _, value in values) / n
    if len(clusters) < 2:
        return {"n": n, "clusters": len(clusters), "mean": mean, "lo": None, "hi": None}
    sums = np.array([sum(clusters[key]) for key in sorted(clusters)])
    counts = np.array([len(clusters[key]) for key in sorted(clusters)])
    rng = np.random.default_rng(seed)
    results = []
    for offset in range(0, draws, 128):
        chosen = rng.integers(len(clusters), size=(min(128, draws - offset), len(clusters)))
        results.extend((sums[chosen].sum(axis=1) / counts[chosen].sum(axis=1)).tolist())
    lo, hi = np.quantile(results, [.025, .975])
    return {"n": n, "clusters": len(clusters), "mean": mean, "lo": float(lo), "hi": float(hi)}


def index_outcomes(outcomes):
    return {(r["dataset"], r["language"], r["condition"], r["budget"],
             r["order_seed"], r["example_id"]): r for r in outcomes}


def primary_measure(row):
    return row["scores"]["accuracy" if row["dataset"] == "belebele" else "exact_match"]


def summary(outcomes, draws, seed):
    groups = defaultdict(list)
    for row in outcomes:
        key = (row["dataset"], row["language"], row["condition"], row["budget"], row["order_seed"])
        groups[key].append(row)
    output = []
    labels = ("dataset", "language", "condition", "budget", "order_seed")
    for key, rows in groups.items():
        prefix = dict(zip(labels, key, strict=True))
        measures = {
            "available": lambda row: row["available"],
            "complete_passage": lambda row: row["passage_state"] == "complete",
            "partial_passage": lambda row: row["passage_state"] == "partial",
            "excluded_passage": lambda row: row["passage_state"] == "absent",
            "format_failure": lambda row: row["format_failure"],
            "generation_hit_cap": lambda row: row["generation_hit_cap"],
            "input_tokens": lambda row: row["input_tokens"],
            "format_overhead_tokens": lambda row: row["prompt_overhead"],
            "first_evidence_token_when_present": lambda row: row["first_evidence_token"],
            "retained_complete_units": lambda row: len(row["complete_passage_ids"]),
            "retained_units": lambda row: len({piece["passage_id"] for piece in row["retained"]}),
            "retained_context_chars": lambda row: sum(piece["source_end"] - piece["source_start"] for piece in row["retained"]),
            "correct_and_unavailable": lambda row: primary_measure(row) * (not row["available"]),
            "incorrect_and_unavailable": lambda row: (1 - primary_measure(row)) * (not row["available"]),
            "incorrect_and_available": lambda row: (1 - primary_measure(row)) * row["available"],
            "correct_and_available": lambda row: primary_measure(row) * row["available"],
        }
        for metric in rows[0]["scores"]:
            measures[metric] = lambda row, name=metric: row["scores"][name]
        for metric, function in measures.items():
            values = [(row["cluster"], value) for row in rows if (value := function(row)) is not None]
            output.append({**prefix, "metric": metric, **interval(values, draws, seed)})
    return output


def comparisons(outcomes, manifest, draws, seed):
    by_key = index_outcomes(outcomes)
    values = defaultdict(list)
    config = manifest["config"]
    primary_order = config["profiles"][manifest["profile"]]["orders"][0]
    matched_budget = config["matched_budget"]
    pairings = {
        "prefix": ("same_complete", "question_only", "relevant_only", "matched_middle"),
        "matched_prefix": ("prefix", "matched_middle"),
        "matched_middle": ("matched_no_evidence", "relevant_only", "matched_early", "matched_late"),
        "oracle": ("prefix", "relevant_only"),
        "retrieval": ("prefix", "head_tail", "random", "oracle"),
        "head_tail": ("prefix",),
        "random": ("prefix",),
        "oracle_span_window": ("prefix", "same_complete"),
    }

    def lookup(row, condition, language=None):
        budget = row["budget"]
        order = row["order_seed"]
        if condition in {"same_complete", "question_only", "relevant_only"}:
            budget, order = None, primary_order if row["dataset"] == "belebele" else config["seed"]
        if condition.startswith("matched") and row["condition"] in {"prefix", "matched_prefix"}:
            budget = matched_budget
        if condition == "matched_no_evidence" and budget != matched_budget:
            return None
        return by_key.get((row["dataset"], language or row["language"], condition, budget, order, row["example_id"]))

    for row in outcomes:
        dataset, language, condition, budget, order = (
            row[key] for key in ("dataset", "language", "condition", "budget", "order_seed")
        )
        score = primary_measure(row)
        for reference in pairings.get(condition, ()):
            if condition == "prefix" and reference == "matched_middle" and budget != matched_budget:
                continue
            other = lookup(row, reference)
            if other is not None:
                difference = score - primary_measure(other)
                label = f"{condition}-{reference}"
                values[(dataset, language, budget, order, label)].append((row["cluster"], difference))
                if dataset != "belebele":
                    f1diff = row["scores"]["f1"] - other["scores"]["f1"]
                    values[(dataset, language, budget, order, f"{label}/f1")].append((row["cluster"], f1diff))
                if reference == "matched_no_evidence" and abs(row["input_tokens"] - other["input_tokens"]) <= .10 * row["input_tokens"]:
                    values[(dataset, language, budget, order, f"{label}/within_10pct_length")].append(
                        (row["cluster"], difference))
                if reference == "relevant_only" and row["available"]:
                    available_label = f"{label}/available_and_baseline_correct"
                    if primary_measure(other):
                        values[(dataset, language, budget, order, available_label)].append((row["cluster"], difference))
                availability_difference = int(row["available"]) - int(other["available"])
                values[(dataset, language, budget, order, f"{label}/availability")].append(
                    (row["cluster"], availability_difference))
        if dataset not in {"belebele", "mlqa"} or (dataset == "mlqa" and condition != "prefix"):
            continue
        if condition not in {"prefix", "matched_middle", "matched_prefix"}:
            continue
        languages = config["languages"] if dataset == "belebele" else ["eng_Latn", "hin_Deva"]
        all_rows = [lookup(row, condition, lang) for lang in languages]
        if any(other is None for other in all_rows):
            continue
        baseline_name = "relevant_only" if dataset == "belebele" else "same_complete"
        baseline = [lookup(row, baseline_name, lang) for lang in languages]
        english = all_rows[0]
        values[(dataset, language, budget, order, f"{condition}/language_gap_vs_english")].append(
            (row["cluster"], score - primary_measure(english)))
        values[(dataset, language, budget, order, f"{condition}/availability_gap_vs_english")].append(
            (row["cluster"], int(row["available"]) - int(english["available"])))
        values[(dataset, language, budget, order, f"{condition}/complete_units_gap_vs_english")].append(
            (row["cluster"], len(row["complete_passage_ids"]) - len(english["complete_passage_ids"])))
        if not all(other["available"] for other in all_rows):
            continue
        values[(dataset, language, budget, order, f"{condition}/common_available_score")].append(
            (row["cluster"], score))
        if any(other is None for other in baseline):
            continue
        base_english = primary_measure(baseline[0])
        base_language = primary_measure(baseline[languages.index(language)])
        values[(dataset, language, budget, order, f"{condition}/common_available_vs_baseline")].append(
            (row["cluster"], score - base_language))
        gap = (score - base_language) - (primary_measure(english) - base_english)
        values[(dataset, language, budget, order, f"{condition}/common_available_baseline_adjusted_gap")].append(
            (row["cluster"], gap))
        if all(primary_measure(other) for other in baseline):
            values[(dataset, language, budget, order, f"{condition}/common_available_all_baseline_correct")].append(
                (row["cluster"], score))
    return [{"dataset": dataset, "language": language, "budget": budget, "order_seed": order,
             "contrast": contrast, **interval(group, draws, seed)}
            for (dataset, language, budget, order, contrast), group in values.items()]


def budget_auc(outcomes, config, draws, seed):
    groups = defaultdict(dict)
    for row in outcomes:
        if row["condition"] == "prefix":
            key = tuple(row[name] for name in ("dataset", "language", "order_seed", "cluster", "example_id"))
            groups[key][row["budget"]] = row
    summaries = defaultdict(list)
    for (dataset, language, order, cluster, _), budget_rows in groups.items():
        budgets = config["budgets"][dataset]
        if sorted(budget_rows) != budgets:
            continue
        for name, measure in [
            ("score", primary_measure), ("availability", availability),
            ("complete_units", lambda row: len(row["complete_passage_ids"]))
        ]:
            samples = [measure(budget_rows[budget]) for budget in budgets]
            auc = np.trapezoid(samples, x=budgets) / (budgets[-1] - budgets[0])
            summaries[(dataset, language, order, name)].append((cluster, float(auc)))
    return [{"dataset": dataset, "language": language, "order_seed": order, "metric": metric,
             "budgets": " ".join(map(str, config["budgets"][dataset])), **interval(values, draws, seed)}
            for (dataset, language, order, metric), values in summaries.items()]


def position_contrasts(outcomes, config, draws, seed):
    index = index_outcomes(outcomes)
    budgets = (config["matched_budget"], config["position_budget"])
    languages = config["languages"]
    groups = defaultdict(list)
    for row in outcomes:
        if row["dataset"] != "belebele" or row["condition"] != "matched_middle" or row["budget"] != budgets[0]:
            continue
        def pair_gap(language, budget, endpoint):
            key = ("belebele", language)
            suffix = (budget, row["order_seed"], row["example_id"])
            middle = index.get((*key, "matched_middle", *suffix))
            edge = index.get((*key, f"matched_{endpoint}", *suffix))
            if middle is None or edge is None:
                return None
            return primary_measure(edge) - primary_measure(middle)

        for endpoint in ("early", "late"):
            gaps = {(language, budget): pair_gap(language, budget, endpoint)
                    for language in languages for budget in budgets}
            if any(value is None for value in gaps.values()):
                continue
            language = row["language"]
            low, high = (gaps[(language, budget)] for budget in budgets)
            en_low, en_high = (gaps[("eng_Latn", budget)] for budget in budgets)
            for metric, value in (("short_edge_minus_middle", low), ("long_edge_minus_middle", high),
                                  ("length_regime_interaction", high - low),
                                  ("short_language_interaction", low - en_low),
                                  ("long_language_interaction", high - en_high)):
                groups[(language, row["order_seed"], endpoint, metric)].append((row["cluster"], value))
    return [{"language": language, "order_seed": order, "position": endpoint, "contrast": name,
             **interval(values, draws, seed)}
            for (language, order, endpoint, name), values in groups.items()]


def order_contrasts(outcomes, profile, config, draws, seed):
    orders = config["profiles"][profile]["orders"]
    first = orders[0]
    index = index_outcomes(outcomes)
    groups = defaultdict(list)
    for row in outcomes:
        if row["dataset"] != "belebele" or row["condition"] != "prefix" or row["order_seed"] == first:
            continue
        key = ("belebele", row["language"], "prefix", row["budget"], first, row["example_id"])
        base = index.get(key)
        if base is None:
            continue
        if row["order_seed"] in orders[:2] and set(row["candidate_ids"]) != set(base["candidate_ids"]):
            continue
        kind = "same_candidate_set" if row["order_seed"] in orders[:2] else "new_distractor_draw"
        for metric, value in (("accuracy", primary_measure(row) - primary_measure(base)),
                              ("availability", int(row["available"]) - int(base["available"]))):
            label = (row["language"], row["budget"], row["order_seed"], kind, metric)
            groups[label].append((row["cluster"], value))
    return [{"language": language, "budget": budget, "order_seed": order, "comparison": kind, "metric": metric,
             **interval(values, draws, seed)}
            for (language, budget, order, kind, metric), values in groups.items()]


def write_csv(path, rows):
    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_text(path, [buffer.getvalue()])


def figures(table, out, provisional):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .data import LANGUAGES

    groups = defaultdict(list)
    for row in table:
        if row["condition"] == "prefix" and row["metric"] in {"accuracy", "exact_match", "available"}:
            groups[(row["dataset"], row["metric"], row["order_seed"])].append(row)
    for (dataset, metric, order), rows in groups.items():
        figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
        languages = list(dict.fromkeys(row["language"] for row in rows))
        for language in languages:
            selected = sorted((row for row in rows if row["language"] == language),
                              key=lambda row: row["budget"])
            x = [row["budget"] for row in selected]
            axis.plot(x, [row["mean"] for row in selected], marker="o", label=LANGUAGES[language])
            if all(row["lo"] is not None for row in selected):
                axis.fill_between(x, [row["lo"] for row in selected],
                                  [row["hi"] for row in selected], alpha=.10)
        axis.set(xlabel="Input budget (tokens)", ylabel=metric.replace("_", " ").capitalize(), ylim=(0, 1.02))
        axis.set_title(f"{dataset.upper()} · prefix · order {order}" + (" · provisional" if provisional else ""))
        axis.legend(frameon=False, ncols=2)
        figure.savefig(out / f"{dataset}-{metric}-{order}.pdf")
        plt.close(figure)


def effects_figure(rows, out, budget, order, provisional):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .data import LANGUAGES

    contrasts = {
        "matched_middle-matched_no_evidence": "Relevant passage vs. same-slot distractor",
        "matched_middle-relevant_only": "Matched context vs. relevant passage only"
    }
    selected = {(row["language"], row["contrast"]): row for row in rows
                if row["dataset"] == "belebele" and row["budget"] == budget and
                row["order_seed"] == order and row["contrast"] in contrasts}
    if not selected:
        return
    labels = [language for language in LANGUAGES if any((language, name) in selected for name in contrasts)]
    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    for (contrast, label), shift in zip(contrasts.items(), [-.16, .16], strict=True):
        plotted = False
        for index, language in enumerate(labels):
            row = selected.get((language, contrast))
            if row is None:
                continue
            errors = None if row["lo"] is None else [[max(0, row["mean"] - row["lo"]) * 100],
                                                       [max(0, row["hi"] - row["mean"]) * 100]]
            axis.errorbar(row["mean"] * 100, index + shift, xerr=errors, fmt="o",
                          color="#174c75" if shift < 0 else "#a24a19", capsize=2,
                          label=label if not plotted else None)
            plotted = True
    axis.axvline(0, color=".6", linewidth=1)
    axis.set_yticks(range(len(labels)), [LANGUAGES[language] for language in labels])
    axis.invert_yaxis()
    axis.set_xlabel("Paired accuracy difference (percentage points)")
    if provisional:
        axis.set_title("Development or provisional results")
    axis.legend(frameon=False, loc="best")
    figure.savefig(out / "paired-evidence-effects.pdf")
    plt.close(figure)


def analyze(run, allow_partial):
    manifest, outcomes, status = load_outcomes(run, allow_partial)
    base = run / ("analysis" if status["complete_evaluation"] else "provisional")
    base.mkdir(exist_ok=True)
    write_json(base / "status.json", status)
    write_rows(base / "outcomes.jsonl", outcomes)
    config = manifest["config"]
    draws, seed = config["bootstrap_draws"], config["seed"]
    table = summary(outcomes, draws, seed)
    write_csv(base / "conditions.csv", table)
    compares = comparisons(outcomes, manifest, draws, seed)
    write_csv(base / "paired_contrasts.csv", compares)
    write_csv(base / "budget_auc.csv", budget_auc(outcomes, config, draws, seed))
    position = position_contrasts(outcomes, config, draws, seed)
    write_csv(base / "position_contrasts.csv", position)
    write_csv(base / "order_contrasts.csv", order_contrasts(outcomes, manifest["profile"], config, draws, seed))
    provisional = not status["complete_evaluation"] or manifest["profile"] in {"smoke", "pilot"}
    effects_figure(compares, base, config["matched_budget"],
                   config["profiles"][manifest["profile"]]["orders"][0], provisional)
    figures(table, base, provisional)
    return status


def compare_models(runs, out, allow_partial):
    manifests, outputs, statuses = zip(*(load_outcomes(run, allow_partial) for run in runs), strict=True)
    first = manifests[0]
    if len({m["model_key"] for m in manifests}) != len(runs):
        raise ValueError("Provide distinct model runs")
    if any(m["config"] != first["config"] or m["profile"] != first["profile"] or
           m["dataset_manifest"] != first["dataset_manifest"] for m in manifests[1:]):
        raise ValueError("Cross-model analysis requires the same protocol, profile, and data")
    budget = first["config"]["matched_budget"]
    order = first["config"]["profiles"][first["profile"]]["orders"][0]
    indexes = [{(r["example_id"], r["language"], r["condition"]): r for r in rows
                if r["dataset"] == "belebele" and r["budget"] == budget and r["order_seed"] == order
                and r["condition"] in {"matched_middle", "prefix"}} for rows in outputs]
    eligible = set.intersection(*(set((example, language) for example, language, _ in index)
                                  for index in indexes))
    by_example = defaultdict(list)
    for example, language in sorted(eligible):
        prefix_keys = (example, language, "prefix")
        matched_keys = (example, language, "matched_middle")
        if not all(prefix_keys in index and matched_keys in index for index in indexes):
            continue
        fixed = [index[prefix_keys] for index in indexes]
        matched = [index[matched_keys] for index in indexes]
        layouts = {tuple((p["passage_id"], p["source_start"], p["source_end"]) for p in row["retained"])
                   for row in matched}
        candidates = {tuple(row["candidate_ids"]) for row in fixed}
        if len(layouts) == 1 and len(candidates) == 1:
            by_example[example].append((language, fixed, matched))
    groups = defaultdict(list)
    configured_languages = set(first["config"]["languages"])
    for values in by_example.values():
        if {language for language, _, _ in values} != configured_languages:
            continue
        for language, prefix_rows, matched_rows in values:
            for i, manifest in enumerate(manifests):
                for j in range(i + 1, len(manifests)):
                    fixed_gap = primary_measure(prefix_rows[i]) - primary_measure(prefix_rows[j])
                    matched_gap = primary_measure(matched_rows[i]) - primary_measure(matched_rows[j])
                    for metric, value in (("prefix_gap", fixed_gap), ("matched_gap", matched_gap),
                                          ("gap_change", matched_gap - fixed_gap)):
                        label = (manifest["model_key"], manifests[j]["model_key"], language, metric)
                        groups[label].append((prefix_rows[i]["cluster"], value))
    out.mkdir(parents=True, exist_ok=True)
    if not groups:
        raise ValueError("No complete shared examples with identical candidate and retained layouts")
    table = [{"first_model": a, "second_model": b, "language": language, "metric": metric,
              **interval(values, first["config"]["bootstrap_draws"], first["config"]["seed"])}
             for (a, b, language, metric), values in groups.items()]
    write_csv(out / "model_contrasts.csv", table)
    write_json(out / "status.json", {"runs": list(statuses), "matched_budget": budget, "order_seed": order})
    return table


def estimate_cost(run, hourly, gpu_count, source, observed_on, setup_minutes):
    from datetime import date

    date.fromisoformat(observed_on)
    if hourly < 0 or gpu_count < 1 or setup_minutes < 0 or not source.startswith("https://"):
        raise ValueError("Provide a nonnegative hourly quote, a GPU count, and an HTTPS rate source")
    path = run / "benchmark.json"
    data = read_json(path)
    spec = read_json(run / "plan_manifest.json")
    if data["plan_sha256"] != spec["plan_sha256"]:
        raise ValueError("Benchmark is bound to a different evaluation plan")
    by_id = defaultdict(list)
    for row in data["trials"]:
        by_id[(row["dataset"], row["band"], row["id"])].append(row)
    samples = defaultdict(list)
    for (dataset, band, _), rows in by_id.items():
        samples[f"{dataset}/{band}"].append(np.mean([item["end_to_end_model_seconds"] for item in rows]))
    seconds = sum(length * float(np.mean(samples[key])) for key, length in data["planned_by_stratum"].items())
    aggregate_gpu_hours = seconds / 3600
    total_gpu_hours = aggregate_gpu_hours + gpu_count * setup_minutes / 60
    bench_seconds = sum(row["end_to_end_model_seconds"] for row in data["trials"])
    report = {"source": source, "rate_date": observed_on, "hourly_usd_per_gpu": hourly, "gpu_count": gpu_count,
              "projected_parallel_runtime_hours_ideal": aggregate_gpu_hours / gpu_count + setup_minutes / 60,
              "projected_inference_gpu_hours": aggregate_gpu_hours, "allocated_gpu_hours_with_setup": total_gpu_hours,
              "projected_usd_with_setup": total_gpu_hours * hourly,
              "benchmark_measured_examples_per_second": len(data["trials"]) / bench_seconds,
              "benchmark_measured_input_tokens_per_second": sum(r["input_tokens"] for r in data["trials"]) / bench_seconds,
              "benchmark_measured_output_tokens_per_second": sum(r["output_tokens"] for r in data["trials"]) / bench_seconds,
              "max_allocated_bytes_in_measured_benchmark": max(r["allocated_peak_bytes"] for r in data["trials"]),
              "max_reserved_bytes_in_measured_benchmark": max(r["reserved_peak_bytes"] for r in data["trials"]),
              "retrieval_index_seconds_measured_separately": spec.get("retrieval", {}).get("index_seconds"),
              "retrieval_device": spec.get("retrieval_device"),
              "rate_excludes_storage_and_network": True, "benchmark_full_job": False}
    write_json(run / "cost.json", report)
    return report
