#!/usr/bin/env python3
"""External Python proposal worker for MueLu-owned GPTune tuning.

The protocol is newline-delimited JSON on stdin/stdout.  Stdout is reserved for
protocol messages; diagnostics go to stderr.  This worker keeps the C++ side in
control of candidate evaluation: it proposes one candidate, waits for a matching
`tell`, records the objective, and then proposes the next candidate.
"""

import json
import random
import sys


def log(message):
    print(message, file=sys.stderr, flush=True)


def emit(message):
    print(json.dumps(message, separators=(",", ":")), flush=True)


def try_import_gptune():
    try:
        import GPTune  # noqa: F401

        log("Imported GPTune; using protocol-safe deterministic proposal mode for this initial worker.")
        return True
    except Exception as exc:  # GPTune remains an optional runtime dependency.
        log(f"GPTune import failed; falling back to deterministic proposals: {exc}")
        return False


def values_for_param(param, rng):
    ptype = param["type"]
    if "initial_value" in param:
        initial = [param["initial_value"]]
    else:
        initial = []

    if ptype == "categorical":
        values = list(param.get("values", []))
    elif ptype == "integer":
        lo = int(param["min"])
        hi = int(param["max"])
        mid = (lo + hi) // 2
        values = [lo, mid, hi, rng.randint(lo, hi)]
    else:
        lo = float(param["min"])
        hi = float(param["max"])
        values = [lo, 0.5 * (lo + hi), hi, rng.uniform(lo, hi)]

    out = []
    for value in initial + values:
        if value not in out:
            out.append(value)
    return out


def make_candidates(spec):
    rng = random.Random(int(spec.get("seed", 0)))
    search_space = spec.get("search_space", [])
    max_evaluations = int(spec.get("max_evaluations", 1))

    value_sets = [(param["name"], values_for_param(param, rng)) for param in search_space]
    baseline = {name: values[0] for name, values in value_sets if values}

    candidates = []

    def add(candidate):
        if candidate not in candidates and len(candidates) < max_evaluations:
            candidates.append(dict(candidate))

    add(baseline)
    for name, values in value_sets:
        if len(candidates) >= max_evaluations:
            break
        for value in values:
            candidate = dict(baseline)
            candidate[name] = value
            add(candidate)
            if len(candidates) >= max_evaluations:
                break

    while candidates and len(candidates) < max_evaluations:
        candidate = {}
        for name, values in value_sets:
            candidate[name] = rng.choice(values)
        add(candidate)
        if len(candidates) == 1 and max_evaluations > 1:
            break

    return candidates


def main():
    init_line = sys.stdin.readline()
    if not init_line:
        return 1
    spec = json.loads(init_line)
    if spec.get("type") != "init":
        raise RuntimeError("expected init message")
    if spec.get("protocol") != "muelu-gptune-worker":
        raise RuntimeError("unsupported protocol")

    try_import_gptune()
    candidates = make_candidates(spec)
    best_candidate = {}
    best_objective = None
    evaluations = 0

    emit({"type": "ready", "protocol": "muelu-gptune-worker", "version": 1})

    for idx, candidate in enumerate(candidates):
        emit({"type": "candidate", "candidate_id": str(idx), "candidate": candidate})
        tell_line = sys.stdin.readline()
        if not tell_line:
            return 1
        tell = json.loads(tell_line)
        if tell.get("type") == "shutdown":
            return 0
        if tell.get("type") != "tell":
            raise RuntimeError("expected tell message")
        evaluations += 1
        if tell.get("status", "success") == "success":
            objective = float(tell.get("objective", 0.0))
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_candidate = candidate

    emit(
        {
            "type": "best",
            "best_candidate": best_candidate,
            "best_objective": 0.0 if best_objective is None else best_objective,
            "evaluations_performed": evaluations,
            "status": "success" if best_objective is not None else "failed",
            "message": "Python GPTune worker selected the best evaluated candidate.",
        }
    )

    for line in sys.stdin:
        message = json.loads(line)
        if message.get("type") == "shutdown":
            return 0
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"muelu_gptune_worker.py failed: {exc}")
        raise
