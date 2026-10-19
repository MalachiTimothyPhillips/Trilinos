# Optional MueLu/GPTune Python integration test

This directory contains two optional MueLu/GPTune paths:

- `diffusion_gptune_demo.py`: a Python-driver integration test that runs the
  GPTune Python stack against a small MueLu diffusion problem.
- `muelu_gptune_worker.py`: an external worker for the MueLu-owned tuning path,
  where MueLu applies/evaluates candidates and Python only proposes them through
  a newline-delimited JSON ask/tell protocol.

Both paths are off by default and intended for environments that can provide the
needed Python runtime dependencies.

The integration test does the following:

1. imports GPTune through Python,
2. asks GPTune to tune a few MueLu parameters for a small `Laplace2D` problem,
3. evaluates each GPTune candidate by launching the existing MueLu scaling
   driver, and
4. reports the best parameter set with a `GPTUNE_DEMO_BEST ...` line.

The Python-driver integration path remains useful as an end-to-end GPTune smoke
test.  The C++ `gptune` sublist now also has a MueLu-owned startup tuning loop
with a fixed deterministic backend and an optional external Python-worker backend.
That path records typed candidate history, supports solve-oriented objectives,
and can return the best evaluated hierarchy without a duplicate final setup.  The
worker backend keeps Python/GPTune out of the Trilinos process: C++ sends an
`init` JSON message, reads `candidate` messages, evaluates each candidate inside
MueLu, sends `tell` messages with objectives, and receives a final `best`
message.

For a standalone setup guide suitable for sharing with other users, see
[`SETUP.md`](SETUP.md).

## CMake options

Configure Trilinos with MueLu and enable the optional test:

```bash
cmake -S /path/to/Trilinos -B /path/to/build \
  -DTrilinos_ENABLE_MueLu=ON \
  -DMueLu_ENABLE_GPTUNE_PYTHON_DEMO=ON \
  -DMueLu_GPTUNE_PYTHONPATH=/path/to/GPTune \
  -DPython3_EXECUTABLE=/path/to/gptune-python
```

`MueLu_GPTUNE_PYTHONPATH` should point at a GPTune checkout whose parent path
lets Python import `GPTune`.  When this variable is set, the CTest environment
uses it directly instead of appending the ambient `PYTHONPATH`; this avoids
mixing the requested Python executable with incompatible site-packages from the
Trilinos build environment.

To opt into the MueLu-owned external worker backend, also configure with:

```bash
-DMueLu_ENABLE_GPTUNE_PYTHON_WORKER=ON
```

Then use `gptune.backend = python` and provide `gptune.worker.argv` in the MueLu
parameter list, for example an argv array equivalent to:

```text
python3 /path/to/muelu_gptune_worker.py
```

The deterministic `fake_muelu_gptune_worker.py` script uses the same protocol and
is intended for process/protocol tests that should not import GPTune.

A known-good local configuration used during development was:

```bash
-DMueLu_ENABLE_GPTUNE_PYTHON_DEMO=ON
-DMueLu_GPTUNE_PYTHONPATH=/scratch/malphil/tuner-ldrd/GPTune
-DPython3_EXECUTABLE=/scratch/malphil/tuner-ldrd/gptune-venv-313-test/bin/python
```

## Verify the Python environment

Before configuring Trilinos, check that the selected Python can import the
packages the integration test needs:

```bash
PYTHONPATH=/path/to/GPTune /path/to/gptune-python - <<'PY'
from autotune.space import Space, Integer, Real
from autotune.problem import TuningProblem
from GPTune.gptune import GPTune
from GPTune.computer import Computer
from GPTune.data import Data
from GPTune.options import Options
print("GPTUNE_IMPORT_OK")
PY
```

The PyPI package named `autotune` is not sufficient for GPTune; the required
module is the GPTune/ytopt `autotune` package that provides
`autotune.space` and `autotune.problem`.

## Run the integration test

After configuring and building, run the integration test with verbose output:

```bash
ctest --test-dir /path/to/build -R GPTunePythonIntegration -V
```

A real GPTune run prints lines like:

```text
GPTUNE_DEMO_IMPORT_OK
GPTUNE_DEMO_EVAL eval=1 ... objective=...
GPTUNE_DEMO_BEST max_levels=... relaxation_sweeps=... objective=...
```

If the GPTune Python stack is unavailable, the test prints:

```text
GPTUNE_DEMO_SKIP GPTune Python stack is unavailable: ...
```

and passes.  This skip behavior keeps GPTune as an optional runtime dependency
for MueLu.  To turn missing GPTune imports into a hard failure while debugging,
run `diffusion_gptune_demo.py` manually with `--require-gptune`.

## Dependency notes

GPTune's Python dependencies include packages such as `GPy`, `autotune`,
`scikit-optimize`, `cGP`, and `hybridMinimization`.  Some GPTune requirement
files refer to GitHub-hosted dependencies, so offline or restricted containers
may need those repositories pre-fetched or mirrored.

When using older system Python installations, make sure the Python development
headers are available.  Without them, compiled dependencies such as `GPy` or
`george` can fail while building with errors such as `Python.h: No such file or
directory`.
