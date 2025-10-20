# we don't want to include torch in Suggests just for the tests, as it's a relatively
# heavy dependency
# We have a CI job that installs torch

skip_if_not_installed("torch")
source(system.file("extra-tests", "test-jit-torch.R", package = "anvil"))
