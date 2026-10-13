#!/usr/bin/env python3
"""Optional GPTune smoke demo for a small MueLu diffusion problem.

This script is intentionally a runtime-only demo: it imports GPTune if the
Python stack is available, lets GPTune choose MueLu parameters, and evaluates
those candidates by launching the existing MueLu scaling driver.  It does not
exercise the placeholder C++ RunGPTuneBridge path used by a `gptune` sublist in
CreatePreconditioner; wiring that bridge to a real backend is separate work.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


OBJECTIVE_RE = re.compile(r"Number of iterations performed for this solve:\s*([0-9]+)")
PENALTY = 1.0e9


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--muelu-driver", required=True, help="Path to the MueLu scaling driver executable")
    parser.add_argument("--work-dir", default="GPTunePythonDemoWork", help="Directory for generated XML files")
    parser.add_argument("--gptune-root", default=os.environ.get("GPTUNE_ROOT", ""), help="Optional GPTune checkout to prepend to sys.path")
    parser.add_argument("--nx", type=int, default=24, help="Laplace2D x dimension")
    parser.add_argument("--ny", type=int, default=24, help="Laplace2D y dimension")
    parser.add_argument("--nrun", type=int, default=4, help="Number of GPTune evaluations")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout in seconds for each MueLu driver run")
    parser.add_argument("--require-gptune", action="store_true", help="Fail instead of skipping if GPTune imports fail")
    return parser.parse_args()


def skip_or_fail(message, require_gptune):
    prefix = "GPTUNE_DEMO_FAIL" if require_gptune else "GPTUNE_DEMO_SKIP"
    print(f"{prefix} {message}")
    return 1 if require_gptune else 0


def import_gptune(args):
    if args.gptune_root:
        sys.path.insert(0, args.gptune_root)

    try:
        from autotune.space import Space, Integer, Real
        from autotune.problem import TuningProblem
        from GPTune.gptune import GPTune
        from GPTune.computer import Computer
        from GPTune.data import Data
        from GPTune.options import Options
    except Exception as exc:  # pragma: no cover - depends on optional runtime stack
        return None, exc

    return {
        "Space": Space,
        "Integer": Integer,
        "Real": Real,
        "TuningProblem": TuningProblem,
        "GPTune": GPTune,
        "Computer": Computer,
        "Data": Data,
        "Options": Options,
    }, None


def write_muelu_xml(path, max_levels, relaxation_sweeps, damping_factor, coarse_max_size):
    path.write_text(
        f"""<ParameterList name=\"MueLu\">
  <Parameter name=\"verbosity\" type=\"string\" value=\"none\"/>
  <Parameter name=\"coarse: max size\" type=\"int\" value=\"{coarse_max_size}\"/>
  <Parameter name=\"cycle type\" type=\"string\" value=\"V\"/>
  <Parameter name=\"multigrid algorithm\" type=\"string\" value=\"sa\"/>
  <Parameter name=\"transpose: use implicit\" type=\"bool\" value=\"true\"/>
  <Parameter name=\"max levels\" type=\"int\" value=\"{max_levels}\"/>
  <Parameter name=\"number of equations\" type=\"int\" value=\"1\"/>
  <Parameter name=\"sa: use filtered matrix\" type=\"bool\" value=\"true\"/>
  <Parameter name=\"aggregation: type\" type=\"string\" value=\"uncoupled\"/>
  <Parameter name=\"smoother: type\" type=\"string\" value=\"RELAXATION\"/>
  <ParameterList name=\"smoother: params\">
    <Parameter name=\"relaxation: type\" type=\"string\" value=\"Jacobi\"/>
    <Parameter name=\"relaxation: sweeps\" type=\"int\" value=\"{relaxation_sweeps}\"/>
    <Parameter name=\"relaxation: damping factor\" type=\"double\" value=\"{damping_factor}\"/>
  </ParameterList>
</ParameterList>
""",
        encoding="utf-8",
    )


def parse_iterations(output):
    match = OBJECTIVE_RE.search(output)
    if match is None:
        return None
    return float(match.group(1))


def compact_output(text, max_lines=30):
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines] + ["... output truncated ..."])


def as_int(point, name):
    return int(round(float(point[name])))


def run_driver(args, work_dir, eval_id, point):
    n = as_int(point, "n")
    max_levels = as_int(point, "max_levels")
    relaxation_sweeps = as_int(point, "relaxation_sweeps")
    damping_factor = float(point["relaxation_damping_factor"])
    coarse_max_size = as_int(point, "coarse_max_size")

    xml_path = work_dir / f"muelu_gptune_eval_{eval_id}.xml"
    write_muelu_xml(xml_path, max_levels, relaxation_sweeps, damping_factor, coarse_max_size)

    command = [
        args.muelu_driver,
        "--linAlgebra=Tpetra",
        "--matrixType=Laplace2D",
        f"--nx={n}",
        f"--ny={max(1, int(args.ny))}",
        f"--xml={xml_path}",
        "--solver=belos",
        "--belosType=cg",
        "--its=100",
        "--tol=1e-8",
        "--notimings",
        "--no-stacked-timer",
        "--no-sacrificial-setup",
        "--no-sacrificial-solve",
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    except Exception as exc:
        print(f"GPTUNE_DEMO_EVAL_FAIL eval={eval_id} error={exc}")
        return PENALTY

    combined = completed.stdout + "\n" + completed.stderr
    iterations = parse_iterations(combined)
    if completed.returncode != 0 or iterations is None:
        print(f"GPTUNE_DEMO_EVAL_FAIL eval={eval_id} returncode={completed.returncode}")
        print(compact_output(combined))
        return PENALTY

    print(
        "GPTUNE_DEMO_EVAL "
        f"eval={eval_id} n={n} max_levels={max_levels} "
        f"relaxation_sweeps={relaxation_sweeps} "
        f"relaxation_damping_factor={damping_factor:.6g} "
        f"coarse_max_size={coarse_max_size} objective={iterations:.12g}",
        flush=True,
    )
    return iterations


def best_from_data(data, parameter_names):
    best_index = 0
    best_value = float("inf")
    for i, raw_value in enumerate(data.O[0]):
        value = float(raw_value[0] if hasattr(raw_value, "__len__") else raw_value)
        if value < best_value:
            best_value = value
            best_index = i

    raw_params = data.P[0][best_index]
    if isinstance(raw_params, dict):
        params = raw_params
    else:
        params = {name: raw_params[i] for i, name in enumerate(parameter_names)}

    return params, best_value


def main():
    args = parse_args()
    if args.nrun < 2:
        return skip_or_fail("--nrun must be at least 2", True)

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    modules, import_error = import_gptune(args)
    if import_error is not None:
        return skip_or_fail(f"GPTune Python stack is unavailable: {import_error}", args.require_gptune)
    print("GPTUNE_DEMO_IMPORT_OK", flush=True)

    driver = Path(args.muelu_driver)
    if not driver.exists():
        print(f"GPTUNE_DEMO_FAIL MueLu driver does not exist: {driver}")
        return 1

    Space = modules["Space"]
    Integer = modules["Integer"]
    Real = modules["Real"]
    TuningProblem = modules["TuningProblem"]
    GPTune = modules["GPTune"]
    Computer = modules["Computer"]
    Data = modules["Data"]
    Options = modules["Options"]

    eval_counter = {"value": 0}

    def objectives(point):
        eval_counter["value"] += 1
        value = run_driver(args, work_dir, eval_counter["value"], point)
        return [value]

    parameter_names = [
        "max_levels",
        "relaxation_sweeps",
        "relaxation_damping_factor",
        "coarse_max_size",
    ]

    input_space = Space([Integer(2, max(2, args.nx), transform="normalize", name="n")])
    parameter_space = Space([
        Integer(2, 4, transform="normalize", name="max_levels"),
        Integer(1, 3, transform="normalize", name="relaxation_sweeps"),
        Real(0.5, 1.0, transform="normalize", name="relaxation_damping_factor"),
        Integer(20, 200, transform="normalize", name="coarse_max_size"),
    ])
    output_space = Space([Real(float("-Inf"), float("Inf"), name="iterations")])

    problem = TuningProblem(input_space, parameter_space, output_space, objectives, {}, None)
    computer = Computer(nodes=1, cores=2)

    options = Options()
    options["distributed_memory_parallelism"] = False
    options["shared_memory_parallelism"] = False
    options["objective_evaluation_parallelism"] = False
    options["model_processes"] = 1
    options["model_class"] = "Model_GPy_LCM"
    options["model_restarts"] = 1
    options["model_max_iters"] = 10
    options["sample_class"] = "SampleNUMPY"
    options["sample_random_seed"] = 0
    options["search_class"] = "SearchSciPy"
    options["search_algo"] = "l-bfgs-b"
    options["search_more_samples"] = 1
    options["search_random_seed"] = 0
    options["verbose"] = False
    options.validate(computer=computer)

    data = Data(problem)
    gt = GPTune(problem, computer=computer, data=data, options=options, historydb=False, driverabspath=os.path.abspath(__file__))
    ns1 = max(1, min(args.nrun - 1, args.nrun // 2))
    data, _modeler, _stats = gt.MLA(NS=args.nrun, NS1=ns1, NI=1, Tgiven=[[int(args.nx)]])

    best_params, best_value = best_from_data(data, parameter_names)
    print(
        "GPTUNE_DEMO_BEST "
        f"max_levels={as_int(best_params, 'max_levels')} "
        f"relaxation_sweeps={as_int(best_params, 'relaxation_sweeps')} "
        f"relaxation_damping_factor={float(best_params['relaxation_damping_factor']):.6g} "
        f"coarse_max_size={as_int(best_params, 'coarse_max_size')} "
        f"objective={best_value:.12g}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
