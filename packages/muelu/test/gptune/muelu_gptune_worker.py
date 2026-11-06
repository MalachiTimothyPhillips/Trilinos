#!/usr/bin/env python3
"""External GPTune proposal worker for MueLu-owned tuning.

The protocol is newline-delimited JSON on stdin/stdout.  MueLu evaluates every
candidate; this worker runs GPTune's own tuning loop and turns each GPTune
objective call into a `candidate` message followed by a blocking wait for the
matching `tell`.  Stdout is reserved for protocol messages, so GPTune's own
output is redirected to stderr.
"""

import json
import math
import os
import sys

# Keep a private handle on the real stdout for protocol messages and send
# everything else (including output from C extensions) to stderr.
PROTOCOL_OUT = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
sys.stdout = sys.stderr


def log(message):
    print(message, file=sys.stderr, flush=True)


def emit(message):
    PROTOCOL_OUT.write(json.dumps(message, separators=(",", ":")) + "\n")
    PROTOCOL_OUT.flush()


class Shutdown(Exception):
    pass


def read_message():
    line = sys.stdin.readline()
    if not line:
        raise Shutdown()
    message = json.loads(line)
    if message.get("type") == "shutdown":
        raise Shutdown()
    return message


def build_space(search_space, Integer, Real, Categoricalnorm):
    """Map MueLu search parameters onto GPTune dimensions named p0, p1, ..."""
    dims = []
    for i, param in enumerate(search_space):
        name = f"p{i}"
        ptype = param["type"]
        if ptype == "categorical":
            dims.append(Categoricalnorm([str(v) for v in param["values"]], transform="onehot", name=name))
        elif ptype == "integer":
            # Upper bound is padded so the top value is reachable after rounding.
            dims.append(Integer(int(param["min"]), int(param["max"]) + 1, transform="normalize", name=name))
        else:
            lo, hi = float(param["min"]), float(param["max"])
            if param.get("transform") == "log":
                lo, hi = math.log(lo), math.log(hi)
            dims.append(Real(lo, hi, transform="normalize", name=name))
    return dims


def to_candidate(search_space, point):
    candidate = {}
    for i, param in enumerate(search_space):
        value = point[f"p{i}"]
        ptype = param["type"]
        if ptype == "categorical":
            candidate[param["name"]] = str(value)
        elif ptype == "integer":
            candidate[param["name"]] = min(int(param["max"]), max(int(param["min"]), int(round(float(value)))))
        else:
            value = float(value)
            if param.get("transform") == "log":
                value = math.exp(value)
            candidate[param["name"]] = min(float(param["max"]), max(float(param["min"]), value))
    return candidate


def main():
    init = json.loads(sys.stdin.readline() or "{}")
    if init.get("type") != "init" or init.get("protocol") != "muelu-gptune-worker":
        raise RuntimeError("expected muelu-gptune-worker init message")

    from autotune.problem import TuningProblem
    from autotune.space import Integer, Real, Space
    from GPTune.computer import Computer
    from GPTune.data import Categoricalnorm, Data
    from GPTune.gptune import GPTune
    from GPTune.options import Options

    search_space = init["search_space"]
    max_evaluations = int(init.get("max_evaluations", 1))
    seed = int(init.get("seed", 0))

    state = {"count": 0, "best": None, "best_objective": None, "worst": None}

    def objectives(point):
        candidate = to_candidate(search_space, point)
        candidate_id = str(state["count"])
        state["count"] += 1
        emit({"type": "candidate", "candidate_id": candidate_id, "candidate": candidate})

        tell = read_message()
        if tell.get("type") != "tell":
            raise RuntimeError("expected tell message")
        objective = float(tell.get("objective", 0.0))
        log(f"MUELU_GPTUNE_EVAL id={candidate_id} status={tell.get('status')} objective={objective} candidate={candidate}")
        if tell.get("status", "success") == "success" and math.isfinite(objective):
            if state["best_objective"] is None or objective < state["best_objective"]:
                state["best_objective"] = objective
                state["best"] = candidate
            state["worst"] = objective if state["worst"] is None else max(state["worst"], objective)
            return [objective]
        # A huge failure penalty would wreck the GP fit; report a value just
        # above the worst successful objective instead.
        return [2.0 * state["worst"] if state["worst"] else 1.0]

    input_space = Space([Integer(0, 1, transform="normalize", name="task")])
    parameter_space = Space(build_space(search_space, Integer, Real, Categoricalnorm))
    output_space = Space([Real(float("-Inf"), float("Inf"), name="objective")])
    problem = TuningProblem(input_space, parameter_space, output_space, objectives, {}, None)
    computer = Computer(nodes=1, cores=2)

    options = Options()
    options["distributed_memory_parallelism"] = False
    options["shared_memory_parallelism"] = False
    options["objective_evaluation_parallelism"] = False
    options["model_processes"] = 1
    options["model_class"] = "Model_GPy_LCM"
    options["model_restarts"] = 1
    options["sample_class"] = "SampleNUMPY"
    options["sample_random_seed"] = seed
    options["search_class"] = "SearchSciPy"
    options["search_algo"] = "l-bfgs-b"
    options["search_more_samples"] = 1
    options["search_random_seed"] = seed
    options["model_random_seed"] = seed
    options["verbose"] = False
    options.validate(computer=computer)

    emit({"type": "ready", "protocol": "muelu-gptune-worker", "version": 1})

    gt = GPTune(problem, computer=computer, data=Data(problem), options=options, historydb=False)
    ns1 = max(1, min(max_evaluations - 1, max_evaluations // 2))
    try:
        gt.MLA(NS=max_evaluations, NS1=ns1, NI=1, Tgiven=[[0]])
    except Shutdown:
        return 0

    emit(
        {
            "type": "best",
            "best_candidate": state["best"] or {},
            "best_objective": 0.0 if state["best_objective"] is None else state["best_objective"],
            "evaluations_performed": state["count"],
            "status": "success" if state["best"] is not None else "failed",
            "message": "GPTune selected the best evaluated candidate.",
        }
    )

    try:
        while True:
            read_message()
    except Shutdown:
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"muelu_gptune_worker.py failed: {exc}")
        raise
