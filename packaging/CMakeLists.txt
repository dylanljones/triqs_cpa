# Configure the version of packaging files
configure_file(conda/meta.yaml.in conda/meta.yaml)
configure_file(conda/build.sh conda/build.sh COPYONLY)
configure_file(conda/conda_build_config.yaml conda/conda_build_config.yaml COPYONLY)
configure_file(pypi/MANIFEST.in pypi/MANIFEST.in COPYONLY)
configure_file(pypi/pyproject.toml.in pypi/pyproject.toml)
configure_file(TRIQS-triqs_cpa-foss-2021b.eb.in TRIQS-triqs_cpa-foss-2021b.eb)


