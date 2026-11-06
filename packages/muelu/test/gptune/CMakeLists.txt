ASSERT_DEFINED(Python3_EXECUTABLE)

TRIBITS_COPY_FILES_TO_BINARY_DIR(
  MueLuGPTunePythonDemo_cp
  SOURCE_FILES
    diffusion_gptune_demo.py
    fake_muelu_gptune_worker.py
    muelu_gptune_worker.py
)

IF(Python3_EXECUTABLE)
  SET(MUELU_GPTUNE_DEMO_PYTHONPATH "$ENV{PYTHONPATH}")
  IF(NOT "${${PACKAGE_NAME}_GPTUNE_PYTHONPATH}" STREQUAL "")
    # Use the GPTune path by itself when provided.  Appending the ambient
    # PYTHONPATH can mix this test's Python with incompatible site-packages
    # from the Trilinos build environment.
    SET(MUELU_GPTUNE_DEMO_PYTHONPATH "${${PACKAGE_NAME}_GPTUNE_PYTHONPATH}")
  ENDIF()

  IF(${PACKAGE_NAME}_ENABLE_GPTUNE_PYTHON_DEMO)
  TRIBITS_ADD_TEST(
    ${Python3_EXECUTABLE}
    NOEXEPREFIX
    NOEXESUFFIX
    NAME GPTunePythonIntegration
    ARGS "diffusion_gptune_demo.py --muelu-driver $<TARGET_FILE:${PACKAGE_NAME}_Driver> --work-dir GPTunePythonIntegrationWork --nx 24 --ny 24 --nrun 4"
    NUM_MPI_PROCS 1
    COMM serial mpi
    CATEGORIES BASIC
    PASS_REGULAR_EXPRESSION "GPTUNE_DEMO_(BEST|SKIP)"
    ENVIRONMENT "PYTHONPATH=${MUELU_GPTUNE_DEMO_PYTHONPATH}"
  )
  ENDIF()

  IF(${PACKAGE_NAME}_ENABLE_GPTUNE_PYTHON_WORKER)
    # MueLu-owned tuning driven purely by the gptune sublist of an XML input.
    CONFIGURE_FILE(gptune_worker_poisson.xml.in
                   ${CMAKE_CURRENT_BINARY_DIR}/gptune_worker_poisson.xml @ONLY)

    TRIBITS_ADD_TEST(
      Driver
      NAME GPTuneWorkerXMLPoisson
      DIRECTORY ../scaling
      ARGS "--xml=gptune_worker_poisson.xml --matrixType=Laplace2D --nx=60 --ny=60"
      NUM_MPI_PROCS 2
      COMM serial mpi
      PASS_REGULAR_EXPRESSION "Belos converged"
      ENVIRONMENT "PYTHONPATH=${MUELU_GPTUNE_DEMO_PYTHONPATH}"
    )
  ENDIF()
ENDIF()
