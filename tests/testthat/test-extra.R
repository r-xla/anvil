# we don't want to include torch in Suggests just for the tests, as it's a relatively
# heavy dependency
# We have a CI job that installs torch

extra_tests <- list.files(system.file("extra-tests", package = "anvil"), full.names = TRUE)

lapply(extra_tests, source)
