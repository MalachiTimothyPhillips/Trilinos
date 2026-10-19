#!/usr/bin/env python3
"""Deterministic newline-delimited JSON worker for MueLu GPTune tests."""

import json
import sys


def log(message):
    print(message, file=sys.stderr, flush=True)


def emit(message):
    print(json.dumps(message, separators=(",", ":")), flush=True)


def candidate_values(param):
    if param["type"] == "categorical":
        return list(param.get("values", []))
    if param["type"] == "integer":
        lo = int(param["min"])
        hi = int(param["max"])
        values = [lo]
        if hi != lo:
            values.append(hi)
        return values
    lo = float(param["min"])
    hi = float(param["max"])
    values = [lo]
    mid = 0.5 * (lo + hi)
    if mid != lo and mid != hi:
        values.append(mid)
    if hi != lo:
        values.append(hi)
    return values


def make_candidates(spec):
    search_space = spec.get("search_space", [])
    max_evaluations = int(spec.get("max_evaluations", 1))

    baseline = {}
    for param in search_space:
        name = param["name"]
        if "initial_value" in param:
            baseline[name] = param["initial_value"]
        else:
            values = candidate_values(param)
            if values:
                baseline[name] = values[0]

    candidates = []

    def add(candidate):
        if candidate not in candidates and len(candidates) < max_evaluations:
            candidates.append(dict(candidate))

    add(baseline)
    for param in search_space:
        if len(candidates) >= max_evaluations:
            break
        for value in candidate_values(param):
            candidate = dict(baseline)
            candidate[param["name"]] = value
            add(candidate)
            if len(candidates) >= max_evaluations:
                break

    return candidates


def main():
    line = sys.stdin.readline()
    if not line:
        return 1
    init = json.loads(line)
    if init.get("type") != "init":
        raise RuntimeError("expected init message")

    candidates = make_candidates(init)
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
        objective = float(tell.get("objective", 0.0))
        if tell.get("status", "success") == "success" and (best_objective is None or objective < best_objective):
            best_objective = objective
            best_candidate = candidate

    emit(
        {
            "type": "best",
            "best_candidate": best_candidate,
            "best_objective": 0.0 if best_objective is None else best_objective,
            "evaluations_performed": evaluations,
            "status": "success" if best_objective is not None else "failed",
            "message": "Fake GPTune worker selected the best evaluated candidate.",
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
    except Exception as exc:  # pragma: no cover - defensive process diagnostics
        log(f"fake_muelu_gptune_worker.py failed: {exc}")
        raise
