import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", choices=["smoke", "pilot", "minimum", "expanded"], required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--retrieval-device", default="cpu")
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--stage", choices=["prepare", "plan", "probe", "evaluate", "benchmark", "analyze"],
                        action="append", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    base = [sys.executable, "-m", "evidence_budget", "--config", str(args.config.resolve()),
            "--data-root", str(args.data_root.resolve())]
    run = ["--run", str(args.run.resolve())]
    hardware = ["--device", args.device, "--attention", args.attention]
    stages = {
        "prepare": [],
        "plan": ["--out", str(args.run.resolve()), "--model", args.model, "--profile", args.profile,
                 "--retrieval-device", args.retrieval_device],
        "probe": run + hardware,
        "evaluate": run + hardware + ["--shard", str(args.shard), "--shards", str(args.shards)],
        "benchmark": run + hardware,
        "analyze": run,
    }
    if args.protocol:
        stages["plan"] += ["--protocol", str(args.protocol.resolve())]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    for stage in args.stage:
        subprocess.run([*base, stage, *stages[stage]], cwd=source, env=env, check=True)


if __name__ == "__main__":
    main()
