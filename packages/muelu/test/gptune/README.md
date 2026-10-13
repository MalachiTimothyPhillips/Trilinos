# Optional MueLu/GPTune Python integration test

This directory contains an optional integration test that runs the GPTune Python
stack against a small MueLu diffusion problem.  The test is off by default and
is intended for environments that can provide GPTune's Python dependencies at
runtime.

The integration test does the following:

1. imports GPTune through Python,
2. asks GPTune to tune a few MueLu parameters for a small `Laplace2D` problem,
3. evaluates each GPTune candidate by launching the existing MueLu scaling
   driver, and
4. reports the best parameter set with a `GPTUNE_DEMO_BEST ...` line.

This test is a Python-driver integration path.  It does **not** mean that the
C++ `gptune` sublist in `CreatePreconditioner` invokes GPTune yet; that path
still goes through the current C++ bridge implementation.

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
