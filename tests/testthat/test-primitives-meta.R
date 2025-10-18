#
#
#test_that("Primitives test 'jit'", {
#  nms <- names(asNamespace("anvil"))
#  prims <- nms[startsWith(nms, "p_")]
#
#  # read jit test file
#  test_file <- readLines(system.file("tests", "testthat", "test-primitives-jit.R", package = "anvil"))
#
#  lines <- test_file[grepl("test_that\\(\"p_", test_file)]
#  # remove all characters in each line that is not a primitive name
#  # how do I match anything?
#  *
#    prim <- "p_add"
#  # regex for anything followed by prim
#  test_file <- sapply(test_file, function(line) {
#    gsub(".*(?=p_)", "", line)
#  })
#
#
#  s
#
#  # get this primitives from prims that are matched by "p_<name>"
#
#  sapply(prims, function(prim) p
#
#  for (prim %in% prims) {
#    if (any(grepl(prim, lines, fixed = TRUE))) {
#      expect_true(TRUE, info = paste("missing jit test for", prim))
#    }
#  }
#})
#
#test_that("Differentiable primitives test 'pullback'", {
## TODO:
#
#})
#
