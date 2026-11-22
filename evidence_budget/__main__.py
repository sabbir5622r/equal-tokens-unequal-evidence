import argparse
import json
from pathlib import Path

import yaml

from .analysis import analyze, compare_models, estimate_cost
from .common import code_hash, read_json, stable_id, write_json
from .data import LANGUAGES, prepare
from .design import audit, make_plan
from .inference import benchmark, evaluate, probe


def parser():
    base = argparse.ArgumentParser(prog="python -m evidence_budget")
    base.add_argument("--config", type=Path, default=Path("config.yaml"))
    base.add_argument("--data-root", type=Path, default=Path("data"))
    commands = base.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    audits = commands.add_parser("audit")
    plans = commands.add_parser("plan")
    for command in (audits, plans):
        command.add_argument("--model", required=True)
        command.add_argument("--out", type=Path, required=True)
    plans.add_argument("--profile", required=True)
    plans.add_argument("--no-retrieval", action="store_true")
    plans.add_argument("--retrieval-device", default="cpu")
    plans.add_argument("--protocol", type=Path)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--pilot", type=Path, action="append", required=True)
    freeze.add_argument("--out", type=Path, required=True)
    for name in ("evaluate", "probe", "benchmark", "analyze", "cost"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        if name in {"evaluate", "probe", "benchmark"}:
            command.add_argument("--device", default="cuda:0")
            command.add_argument("--attention", choices=["eager", "sdpa"], default="sdpa")
        if name == "evaluate":
            command.add_argument("--shard", type=int, default=0)
            command.add_argument("--shards", type=int, default=1)
        if name == "benchmark":
            command.add_argument("--per-stratum", type=int, default=5)
            command.add_argument("--repeats", type=int, default=3)
            command.add_argument("--warmups", type=int, default=2)
        if name == "analyze":
            command.add_argument("--allow-partial", action="store_true")
        if name == "cost":
            command.add_argument("--usd-per-gpu-hour", type=float, required=True)
            command.add_argument("--gpu-count", type=int, default=1)
            command.add_argument("--source", required=True)
            command.add_argument("--observed-on", required=True)
            command.add_argument("--setup-minutes", type=float, default=0)
    compare = commands.add_parser("compare-models")
    compare.add_argument("--runs", type=Path, nargs="+", required=True)
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--allow-partial", action="store_true")
    return base


def load_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["languages"] != list(LANGUAGES):
        raise ValueError("The main protocol requires English and all seven planned South Asian language variants")
    budgets = config["budgets"]
    if config["diagnostic_budget"] not in budgets["belebele"]:
        raise ValueError("The diagnostic budget must also be a primary Belebele budget")
    if (config["matched_budget"] not in budgets["belebele"] or
            config["matched_budget"] < config["diagnostic_budget"] or
            config["position_budget"] <= config["matched_budget"] or
            not 2 < config["position_distractors"] <= config["max_candidates"]):
        raise ValueError("The longer position intervention requires a higher budget and more distractors")
    input_cap = min(model["context"] for model in config["models"].values())
    if config["position_budget"] + config["output_tokens"]["belebele"] > input_cap:
        raise ValueError("Position budget exceeds the shortest model context")
    for dataset, values in budgets.items():
        if (len(values) < 2 or values != sorted(set(values)) or min(values) < 1 or
                max(values) + config["output_tokens"][dataset] > input_cap):
            raise ValueError(f"Invalid or unsupported candidate budgets for {dataset}")
    if any(budget not in budgets[dataset] for dataset, budget in config["natural_diagnostic_budget"].items()):
        raise ValueError("Natural diagnostic budgets must be included in the prefix grid")
    if any(profile["order_replication_limit"] > profile["diagnostic_limit"] for profile in config["profiles"].values()):
        raise ValueError("Order replications must be drawn from the diagnostic subset")
    if config["bootstrap_draws"] < 1 or config["max_candidates"] < 3:
        raise ValueError("Positive bootstrap draws and at least three distractors are required")
    return config


def freeze_protocol(config, pilots, out, data_root):
    if out.exists():
        raise FileExistsError(out)
    data_manifest = read_json(data_root / "prepared" / "manifest.json")
    pilot_hashes = {}
    for pilot in pilots:
        manifest = read_json(pilot / "plan_manifest.json")
        status = read_json(pilot / "analysis" / "status.json")
        if (manifest["profile"] != "pilot" or not status["complete_evaluation"] or
                not manifest["retrieval_enabled"] or
                status["plan_sha256"] != manifest["plan_sha256"] or
                manifest["dataset_manifest"] != data_manifest):
            raise ValueError(f"Expected a complete development pilot on the current source data: {pilot}")
        pilot_hashes[manifest["model_key"]] = manifest["plan_sha256"]
    lock = {"config": config, "sha256": stable_id(config), "development_pilots": pilot_hashes,
            "code_sha256": code_hash(), "dataset_manifest": data_manifest}
    write_json(out, lock)
    return lock


def main():
    options = parser().parse_args()
    command = options.command
    if command in {"prepare", "audit", "plan", "freeze"}:
        config = load_config(options.config)
    if command == "prepare":
        result = prepare(config, options.data_root)
    elif command == "audit":
        result = audit(config, options.model, options.data_root, options.out)
    elif command == "plan":
        if options.profile in {"minimum", "expanded"} and (options.no_retrieval or not options.protocol):
            raise ValueError("Primary evaluation needs the frozen protocol and all planned policies")
        if options.protocol:
            lock = read_json(options.protocol)
            if (lock["config"] != config or lock["sha256"] != stable_id(config) or
                    lock["code_sha256"] != code_hash() or
                    lock["dataset_manifest"] != read_json(options.data_root / "prepared" / "manifest.json")):
                raise ValueError("Current configuration, data, or source code differs from the frozen protocol")
        manifest = make_plan(config, options.profile, options.model, options.data_root, options.out,
                             not options.no_retrieval, options.retrieval_device)
        result = {"model": manifest["model_key"], "profile": manifest["profile"],
                  "plan_sha256": manifest["plan_sha256"], "samples": manifest["samples"]}
    elif command == "freeze":
        result = freeze_protocol(config, options.pilot, options.out, options.data_root)
    elif command == "evaluate":
        result = evaluate(options.run, options.device, options.attention, options.shard, options.shards)
    elif command == "probe":
        result = probe(options.run, options.device, options.attention)
    elif command == "benchmark":
        measured = benchmark(options.run, options.device, options.attention, options.per_stratum,
                             options.repeats, options.warmups)
        result = {"trials": len(measured["trials"]), "hardware": measured["hardware"]}
    elif command == "analyze":
        result = analyze(options.run, options.allow_partial)
    elif command == "compare-models":
        if len(options.runs) < 2:
            raise ValueError("Provide at least two model runs")
        result = {"contrasts": len(compare_models(options.runs, options.out, options.allow_partial))}
    else:
        result = estimate_cost(options.run, options.usd_per_gpu_hour, options.gpu_count, options.source,
                               options.observed_on, options.setup_minutes)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
