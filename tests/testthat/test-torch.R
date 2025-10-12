skip_if(!Sys.getenv("NVL_PULLBACK") == "true")

# we don't want to include torch in Suggests just for the tests, as it's a relatively
# heavy dependency

testthat::test_dir(system.file("extra-tests", "test-pullback-torch.R", package = "anvil"))
