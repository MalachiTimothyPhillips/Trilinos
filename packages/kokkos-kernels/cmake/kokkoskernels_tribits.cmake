INCLUDE(CMakeParseArguments)
INCLUDE(CTest)

IF (KOKKOSKERNELS_HAS_TRILINOS)
INCLUDE(TribitsETISupport)
ENDIF()

FUNCTION(VERIFY_EMPTY CONTEXT)
  IF(${ARGN})
    MESSAGE(FATAL_ERROR "Kokkos does not support all of Tribits. Unhandled arguments in ${CONTEXT}:\n${ARGN}")
  ENDIF()
ENDFUNCTION()

#MESSAGE(STATUS "The project name is: ${PROJECT_NAME}")

MACRO(KOKKOSKERNELS_PACKAGE_POSTPROCESS)
  IF (KOKKOSKERNELS_HAS_TRILINOS)
    TRIBITS_PACKAGE_POSTPROCESS()
  ELSE()
    INCLUDE(CMakePackageConfigHelpers)
    configure_package_config_file(cmake/KokkosKernelsConfig.cmake.in
         "${KokkosKernels_BINARY_DIR}/KokkosKernelsConfig.cmake"
         INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosKernels)
    write_basic_package_version_file("${KokkosKernels_BINARY_DIR}/KokkosKernelsConfigVersion.cmake"
            VERSION "${KokkosKernels_VERSION_MAJOR}.${KokkosKernels_VERSION_MINOR}.${KokkosKernels_VERSION_PATCH}"
            COMPATIBILITY AnyNewerVersion)

    INSTALL(FILES
      "${KokkosKernels_BINARY_DIR}/KokkosKernelsConfig.cmake"
      "${KokkosKernels_BINARY_DIR}/KokkosKernelsConfigVersion.cmake"
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosKernels)

    INSTALL(EXPORT KokkosKernelsTargets
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosKernels
            NAMESPACE Kokkos::)
  ENDIF()
ENDMACRO(KOKKOSKERNELS_PACKAGE_POSTPROCESS)

MACRO(KOKKOSKERNELS_SUBPACKAGE NAME)
IF (KOKKOSKERNELS_HAS_TRILINOS)
  TRIBITS_SUBPACKAGE(${NAME})
ELSE()
  SET(PACKAGE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  SET(PARENT_PACKAGE_NAME ${PACKAGE_NAME})
  SET(PACKAGE_NAME ${PACKAGE_NAME}${NAME})
  STRING(TOUPPER ${PACKAGE_NAME} PACKAGE_NAME_UC)
  SET(${PACKAGE_NAME}_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
ENDIF()
ENDMACRO(KOKKOSKERNELS_SUBPACKAGE)

MACRO(KOKKOSKERNELS_SUBPACKAGE_POSTPROCESS)
IF (KOKKOSKERNELS_HAS_TRILINOS)
  TRIBITS_SUBPACKAGE_POSTPROCESS()
ELSE()
ENDIF()
ENDMACRO(KOKKOSKERNELS_SUBPACKAGE_POSTPROCESS)

MACRO(KOKKOSKERNELS_PROCESS_SUBPACKAGES)
IF (KOKKOSKERNELS_HAS_TRILINOS)
  TRIBITS_PROCESS_SUBPACKAGES()
ENDIF()
ENDMACRO(KOKKOSKERNELS_PROCESS_SUBPACKAGES)

MACRO(KOKKOSKERNELS_PACKAGE)
IF (KOKKOSKERNELS_HAS_TRILINOS)
  TRIBITS_PACKAGE(KokkosKernels)
ELSE()
  SET(PACKAGE_NAME KokkosKernels)
  SET(PACKAGE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  STRING(TOUPPER ${PACKAGE_NAME} PACKAGE_NAME_UC)
  SET(${PACKAGE_NAME}_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
ENDIF()
ENDMACRO(KOKKOSKERNELS_PACKAGE)

FUNCTION(KOKKOSKERNELS_INTERNAL_ADD_LIBRARY LIBRARY_NAME)
CMAKE_PARSE_ARGUMENTS(PARSE 
  "STATIC;SHARED"
  ""
  "HEADERS;SOURCES"
  ${ARGN})

IF(PARSE_HEADERS)
  LIST(REMOVE_DUPLICATES PARSE_HEADERS)
ENDIF()
IF(PARSE_SOURCES)
  LIST(REMOVE_DUPLICATES PARSE_SOURCES)
ENDIF()

ADD_LIBRARY(
  ${LIBRARY_NAME}
  ${PARSE_HEADERS}
  ${PARSE_SOURCES}
)
ADD_LIBRARY(Kokkos::${LIBRARY_NAME} ALIAS ${LIBRARY_NAME})

INSTALL(
  TARGETS ${LIBRARY_NAME}
  EXPORT ${PROJECT_NAME}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT ${PACKAGE_NAME}
)

INSTALL(
  TARGETS ${LIBRARY_NAME}
  EXPORT KokkosKernelsTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

INSTALL(
  FILES  ${PARSE_HEADERS}
  DESTINATION ${KOKKOSKERNELS_HEADER_INSTALL_DIR}
  COMPONENT ${PACKAGE_NAME}
)

INSTALL(
  FILES  ${PARSE_HEADERS}
  DESTINATION ${KOKKOSKERNELS_HEADER_INSTALL_DIR}
)

ENDFUNCTION(KOKKOSKERNELS_INTERNAL_ADD_LIBRARY LIBRARY_NAME)

FUNCTION(KOKKOSKERNELS_ADD_LIBRARY LIBRARY_NAME)
IF (KOKKOSKERNELS_HAS_TRILINOS)
  TRIBITS_ADD_LIBRARY(${LIBRARY_NAME} ${ARGN})
ELSE()
  KOKKOSKERNELS_INTERNAL_ADD_LIBRARY(
    ${LIBRARY_NAME} ${ARGN})
ENDIF()
ENDFUNCTION()

FUNCTION(KOKKOSKERNELS_ADD_EXECUTABLE EXE_NAME)
CMAKE_PARSE_ARGUMENTS(PARSE
  ""
  ""
  "SOURCES;COMPONENTS;TESTONLYLIBS"
  ${ARGN})
VERIFY_EMPTY(KOKKOSKERNELS_ADD_EXECUTABLE ${PARSE_UNPARSED_ARGUMENTS})

KOKKOSKERNELS_IS_ENABLED(
  COMPONENTS ${PARSE_COMPONENTS}
  OUTPUT_VARIABLE IS_ENABLED
)

IF (IS_ENABLED)
  IF (KOKKOSKERNELS_HAS_TRILINOS)
    TRIBITS_ADD_EXECUTABLE(${EXE_NAME}
      SOURCES ${PARSE_SOURCES}
      TESTONLYLIBS ${PARSE_TESTONLYLIBS})
  ELSE()
    ADD_EXECUTABLE(${EXE_NAME} ${PARSE_SOURCES})
    #AJP, BMK altered:
    IF(KOKKOSKERNELS_ENABLE_TESTS_AND_PERFSUITE)
      TARGET_LINK_LIBRARIES(${EXE_NAME} PRIVATE common ${PARSE_TESTONLYLIBS})
    ENDIF()

    IF (PARSE_TESTONLYLIBS)
      TARGET_LINK_LIBRARIES(${EXE_NAME} PRIVATE Kokkos::kokkoskernels ${PARSE_TESTONLYLIBS})
    ELSE ()
      TARGET_LINK_LIBRARIES(${EXE_NAME} PRIVATE Kokkos::kokkoskernels)
    ENDIF()
  ENDIF()
ELSE()
  MESSAGE(STATUS "Skipping executable ${EXE_NAME} because not all necessary components enabled")
ENDIF()
ENDFUNCTION()

FUNCTION(KOKKOSKERNELS_ADD_UNIT_TEST ROOT_NAME)
  KOKKOSKERNELS_ADD_EXECUTABLE_AND_TEST(
    ${ROOT_NAME}
    TESTONLYLIBS kokkoskernels_gtest
    ${ARGN}
  )
ENDFUNCTION()

FUNCTION(KOKKOSKERNELS_IS_ENABLED)
  CMAKE_PARSE_ARGUMENTS(PARSE
    ""
    "OUTPUT_VARIABLE"
    "COMPONENTS"
    ${ARGN})

  IF (KOKKOSKERNELS_ENABLED_COMPONENTS STREQUAL "ALL")
    SET(${PARSE_OUTPUT_VARIABLE} TRUE PARENT_SCOPE)
  ELSEIF(PARSE_COMPONENTS)
    SET(ENABLED TRUE)
    FOREACH(comp ${PARSE_COMPONENTS})
      STRING(TOUPPER ${comp} COMP_UC)
      # make sure this is in the list of enabled components
      IF(NOT "${COMP_UC}" IN_LIST KOKKOSKERNELS_ENABLED_COMPONENTS)
        # if not in the list, one or more components is missing
        SET(ENABLED FALSE)
      ENDIF()
    ENDFOREACH()
    SET(${PARSE_OUTPUT_VARIABLE} ${ENABLED} PARENT_SCOPE)
  ELSE()
    # we did not enable all components and no components
    # were given as part of this - we consider this enabled
    SET(${PARSE_OUTPUT_VARIABLE} TRUE PARENT_SCOPE)
  ENDIF()
ENDFUNCTION()

FUNCTION(KOKKOSKERNELS_ADD_EXECUTABLE_AND_TEST ROOT_NAME)

CMAKE_PARSE_ARGUMENTS(PARSE
  ""
  ""
  "SOURCES;CATEGORIES;COMPONENTS;TESTONLYLIBS"
  ${ARGN})
VERIFY_EMPTY(KOKKOSKERNELS_ADD_EXECUTABLE_AND_RUN_VERIFY ${PARSE_UNPARSED_ARGUMENTS})

KOKKOSKERNELS_IS_ENABLED(
  COMPONENTS ${PARSE_COMPONENTS}
  OUTPUT_VARIABLE IS_ENABLED
)

IF (IS_ENABLED)
  IF (KOKKOSKERNELS_HAS_TRILINOS)
    TRIBITS_ADD_EXECUTABLE_AND_TEST(
      ${ROOT_NAME}
      SOURCES ${PARSE_SOURCES}
      CATEGORIES ${PARSE_CATEGORIES}
      TESTONLYLIBS ${PARSE_TESTONLYLIBS}
      NUM_MPI_PROCS 1
      COMM serial mpi
    )
  ELSE()
    SET(EXE_NAME ${PACKAGE_NAME}_${ROOT_NAME})
    KOKKOSKERNELS_ADD_EXECUTABLE(${EXE_NAME}
      SOURCES ${PARSE_SOURCES}
    )
    IF (PARSE_TESTONLYLIBS)
      TARGET_LINK_LIBRARIES(${EXE_NAME} PRIVATE ${PARSE_TESTONLYLIBS})
    ENDIF()
    KOKKOSKERNELS_ADD_TEST(NAME ${ROOT_NAME}
      EXE ${EXE_NAME}
    )
  ENDIF()
ELSE()
  MESSAGE(STATUS "Skipping executable/test ${ROOT_NAME} because not all necessary components enabled")
ENDIF()

ENDFUNCTION()

MACRO(ADD_COMPONENT_SUBDIRECTORY SUBDIR)
  KOKKOSKERNELS_IS_ENABLED(
    COMPONENTS ${SUBDIR}
    OUTPUT_VARIABLE COMP_SUBDIR_ENABLED
  )
  IF (COMP_SUBDIR_ENABLED)
    ADD_SUBDIRECTORY(${SUBDIR})
  ELSE()
    MESSAGE(STATUS "Skipping subdirectory ${SUBDIR} because component is not enabled")
  ENDIF()
  UNSET(COMP_SUBDIR_ENABLED)
ENDMACRO()
