# Setting up the optional MueLu/GPTune Python integration test

This note describes how to set up and run the optional MueLu/GPTune Python
integration test. The test is off by default and is intended for environments
that can provide GPTune's Python dependencies at runtime.

## What the test does

The integration test lives in:

```text
packages/muelu/test/gptune
```

When enabled, it:

1. imports GPTune through Python,
2. asks GPTune to tune a few MueLu parameters for a small `Laplace2D` diffusion
   problem,
3. evaluates each candidate by launching the existing MueLu scaling driver, and
4. reports the best candidate with a `GPTUNE_DEMO_BEST ...` line.

This is a Python-driver integration path. It does **not** mean that the C++
`gptune` sublist in `CreatePreconditioner` invokes real GPTune yet; that path
still goes through the current C++ bridge implementation.

## Configure Trilinos

Configure Trilinos with MueLu and enable the optional test:

```bash
cmake -S /path/to/Trilinos -B /path/to/build \
  -DTrilinos_ENABLE_MueLu=ON \
  -DMueLu_ENABLE_GPTUNE_PYTHON_DEMO=ON \
  -DMueLu_GPTUNE_PYTHONPATH=/path/to/GPTune \
  -DPython3_EXECUTABLE=/path/to/gptune-python
```

`MueLu_GPTUNE_PYTHONPATH` should point at the GPTune checkout path that lets
Python import `GPTune`.

When `MueLu_GPTUNE_PYTHONPATH` is set, the CTest environment uses that path
directly instead of appending the ambient `PYTHONPATH`. This is intentional: the
ambient Trilinos build environment may contain site-packages from a different
Python version, and mixing those with the selected `Python3_EXECUTABLE` can make
GPTune imports fail.

A known-good local configuration used during development was:

```bash
-DMueLu_ENABLE_GPTUNE_PYTHON_DEMO=ON
-DMueLu_GPTUNE_PYTHONPATH=/scratch/malphil/tuner-ldrd/GPTune
-DPython3_EXECUTABLE=/scratch/malphil/tuner-ldrd/gptune-venv-313-test/bin/python
```

## Verify the Python environment

Before configuring Trilinos, check that the selected Python can import GPTune and
the modules the integration test needs:

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

If this fails, the CTest integration test will skip by default instead of failing
the MueLu test suite.

## Build and run the integration test

After configuring and building Trilinos, run the integration test with verbose
output:

```bash
ctest --test-dir /path/to/build -R GPTunePythonIntegration -V
```

A real GPTune run should include output like:

```text
GPTUNE_DEMO_IMPORT_OK
GPTUNE_DEMO_EVAL eval=1 ... objective=...
GPTUNE_DEMO_BEST max_levels=... relaxation_sweeps=... relaxation_damping_factor=... coarse_max_size=... objective=...
```

If GPTune is unavailable, the test prints:

```text
GPTUNE_DEMO_SKIP GPTune Python stack is unavailable: ...
```

and passes. This keeps GPTune as an optional runtime dependency for MueLu.

For strict manual debugging, run `diffusion_gptune_demo.py` directly with
`--require-gptune`; that converts missing GPTune imports into a hard failure.

## Dependency notes

GPTune's Python dependencies include packages such as:

- `GPy`
- `autotune`
- `scikit-optimize`
- `cGP`
- `hybridMinimization`
- `george`

A few setup pitfalls are worth calling out:

- The PyPI package named `autotune` is not sufficient for GPTune. GPTune needs
  the GPTune/ytopt `autotune` package that provides `autotune.space` and
  `autotune.problem`.
- Some GPTune requirement files refer to GitHub-hosted dependencies, so offline
  or restricted containers may need those repositories pre-fetched or mirrored.
- Compiled dependencies such as `GPy` and `george` may require Python development
  headers. Older system Python installations without `Python.h` can fail during
  install.
- Avoid mixing a Python executable with site-packages from a different Python
  version. Use a clean virtual environment and set `MueLu_GPTUNE_PYTHONPATH` to
  the GPTune checkout explicitly.

## Example local command sequence

Using the known-good local environment from development:

```bash
cmake -S /scratch/malphil/tuner-ldrd/Trilinos \
  -B /scratch/malphil/tuner-ldrd/Trilinos/pr_build \
  -DTrilinos_ENABLE_MueLu=ON \
  -DMueLu_ENABLE_GPTUNE_PYTHON_DEMO=ON \
  -DMueLu_GPTUNE_PYTHONPATH=/scratch/malphil/tuner-ldrd/GPTune \
  -DPython3_EXECUTABLE=/scratch/malphil/tuner-ldrd/gptune-venv-313-test/bin/python

cmake --build /scratch/malphil/tuner-ldrd/Trilinos/pr_build

ctest --test-dir /scratch/malphil/tuner-ldrd/Trilinos/pr_build \
  -R GPTunePythonIntegration -V
```
