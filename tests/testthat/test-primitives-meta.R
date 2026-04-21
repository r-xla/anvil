test_that("stablehlo rule is tested", {
  nms <- names(asNamespace("anvil"))
  primitive_names <- setdiff(nms[grepl("^prim_", nms)], "prim_dict")
  # Deduplicate primitives that share a registered name via aliases (e.g., prim_select <- prim_ifelse).
  seen <- character()
  primitive_names <- Filter(function(nm) {
    obj <- getFromNamespace(nm, "anvil")
    p <- if (inherits(obj, "JitPrimitive")) attr(obj, "primitive") else obj
    if (is.null(p) || !inherits(p, "AnvilPrimitive")) return(FALSE)
    key <- p$name
    keep <- !(key %in% seen)
    if (keep) seen <<- c(seen, key)
    keep
  }, primitive_names)

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


test_that("reverse rule is tested", {
  nms <- names(asNamespace("anvil"))
  primitive_names <- setdiff(nms[grepl("^prim_", nms)], "prim_dict")
  # Deduplicate primitives that share a registered name via aliases (e.g., prim_select <- prim_ifelse).
  seen <- character()
  primitive_names <- Filter(function(nm) {
    obj <- getFromNamespace(nm, "anvil")
    p <- if (inherits(obj, "JitPrimitive")) attr(obj, "primitive") else obj
    if (is.null(p) || !inherits(p, "AnvilPrimitive")) return(FALSE)
    key <- p$name
    keep <- !(key %in% seen)
    if (keep) seen <<- c(seen, key)
    keep
  }, primitive_names)

  primitive_names <- Filter(
    function(nm) {
      !is.null(getFromNamespace(nm, "anvil")[["reverse"]])
    },
    primitive_names
  )

  candidate_files <- c(
    system.file("extra-tests", "test-primitives-reverse-torch.R", package = "anvil"),
    file.path(testthat::test_path(), "test-primitives-reverse.R")
  )

  content <- do.call(c, lapply(candidate_files, readLines))
  content <- content[grepl("(test_that|describe|it)\\(", content)]
  missing <- Filter(function(nm) !any(grepl(nm, content, fixed = TRUE)), primitive_names)

  expect_true(length(missing) == 0L, info = paste(missing, collapse = ", "), label = "Reverse rule is tested")
})
