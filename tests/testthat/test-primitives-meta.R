test_that("Jit rule is tested", {
  nms <- names(asNamespace("anvil"))
  primitive_names <- nms[grepl("^p_", nms)]

  tests_dir <- testthat::test_path()
  candidate_files <- c(
    system.file("extra-tests", "test-primitives-jit-torch.R", package = "anvil"),
    system.file("tests", "testthat", "test-primitives-jit.R", package = "anvil")
  )

  content <- paste(vapply(candidate_files, function(file) paste(readLines(file, warn = FALSE), collapse = "\n"), character(1L)), collapse = "\n")
  missing <- Filter(function(nm) !grepl(paste0('test_that("', nm), content, fixed = TRUE), primitive_names)

  expect_true(length(missing) == 0L, info = paste(missing, collapse = ", "), label = "Jit rule is tested")
})


test_that("Pullback rule is tested", {
  nms <- names(asNamespace("anvil"))
  primitive_names <- nms[grepl("^p_", nms)]

  primitive_names <- Filter(
    function(nm) {
      !is.null(getFromNamespace(nm, "anvil")@rules[["pullback"]])
    },
    primitive_names
  )

  tests_dir <- testthat::test_path()
  candidate_files <- c(
    system.file("extra-tests", "test-primitives-pullback-torch.R", package = "anvil"),
    system.file("tests", "testthat", "test-primitives-pullback.R", package = "anvil")
  )

  content <- paste(vapply(candidate_files, function(file) paste(readLines(file, warn = FALSE), collapse = "\n"), character(1L)), collapse = "\n")
  missing <- Filter(function(nm) !grepl(paste0('test_that("', nm), content, fixed = TRUE), primitive_names)
  expect_true(length(missing) == 0L, info = paste(missing, collapse = ", "), label = "Pullback rule is tested")
})
