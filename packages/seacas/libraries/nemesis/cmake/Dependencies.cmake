TRIBITS_PACKAGE_DEFINE_DEPENDENCIES(
  LIB_REQUIRED_PACKAGES SEACASExodus
  LIB_REQUIRED_TPLS Netcdf
  LIB_OPTIONAL_TPLS Pthread HDF5 Pnetcdf MPI
  TEST_REQUIRED_TPLS Netcdf
)
