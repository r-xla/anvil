test_that("stablehlo rule is tested", {
  nms <- names(asNamespace("anvil"))
  primitive_names <- nms[grepl("^p_", nms)]

  tests_dir <- testthat::test_path()
  candidate_files <- c(
    system.file("extra-tests", "test-primitives-stablehlo-torch.R", package = "anvil"),
    file.path(testthat::test_path(), "test-primitives-stablehlo.R")
  )

  content <- paste(
    vapply(candidate_files, function(file) paste(readLines(file, warn = FALSE), collapse = "\n"), character(1L)),
    collapse = "\n"
  )
  missing <- Filter(
    function(nm) {
      !grepl(paste0('(test_that|describe)\\("', nm), content)
    },
    primitive_names
  )

  expect_true(length(missing) == 0L, info = paste(missing, collapse = ", "), label = "stablehlo rule is tested")
})


test_that("backward rule is tested", {
  nms <- names(asNamespace("anvil"))
  primitive_names <- nms[grepl("^p_", nms)]

  primitive_names <- Filter(
    function(nm) {
      !is.null(getFromNamespace(nm, "anvil")$rules[["backward"]])
    },
    primitive_names
  )

  candidate_files <- c(
    system.file("extra-tests", "test-primitives-backward-torch.R", package = "anvil"),
    file.path(testthat::test_path(), "test-primitives-backward.R")
  )

  content <- do.call(c, lapply(candidate_files, readLines))
  content <- content[grepl("(test_that|describe)\\(", content)]
  missing <- Filter(function(nm) !any(grepl(nm, content, fixed = TRUE)), primitive_names)

  expect_true(length(missing) == 0L, info = paste(missing, collapse = ", "), label = "Backward rule is tested")
})
