test_that("graph_to_quickr_function handles long scalar chain (stress)", {
  testthat::skip_if_not_installed("quickr")

  decay <- nv_scalar(0.999, dtype = "f64")
  shift <- nv_scalar(0.001, dtype = "f64")

  # Stress the graph-to-R lowering with a long linear chain, but avoid
  # compiling it with {quickr} here because compilation time grows quickly
  # with the number of statements.
  n_steps <- 150L
  chain_fn <- function(x) {
    for (i in seq_len(n_steps)) {
      x <- x * decay + shift
    }
    x
  }

  graph <- trace_fn(chain_fn, list(x = nv_scalar(0.0, dtype = "f64")))
  expect_gt(length(graph@calls), 250L)

  f_r <- anvil:::graph_to_quickr_r_function(graph, include_declare = FALSE, pack_output = FALSE)
  const_args <- lapply(graph@constants, function(node) {
    as_array(node@aval@data)
  })

  x <- 0.123
  out_r <- do.call(f_r, c(list(x), const_args))

  expected <- x
  for (i in seq_len(n_steps)) {
    expected <- expected * 0.999 + 0.001
  }

  expect_identical(out_r, expected)
})
