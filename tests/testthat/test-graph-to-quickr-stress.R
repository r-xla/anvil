test_that("graph_to_quickr_function handles long scalar chain (stress)", {
  testthat::skip_if_not_installed("quickr")

  decay <- nv_scalar(0.999, dtype = "f64")
  shift <- nv_scalar(0.001, dtype = "f64")

  # Stress the graph-to-quickr lowering with a longer linear chain, while
  # keeping {quickr} compilation time reasonable for the test suite.
  n_steps <- 8L
  chain_fn <- function(x) {
    for (i in seq_len(n_steps)) {
      x <- x * decay + shift
    }
    x
  }

  graph <- trace_fn(chain_fn, list(x = nv_scalar(0.0, dtype = "f64")))
  expect_gt(length(graph$calls), 12L)

  x <- 0.123
  f_quick <- graph_to_quickr_function(graph)
  out_quick <- f_quick(x)

  expected <- x
  for (i in seq_len(n_steps)) {
    expected <- expected * 0.999 + 0.001
  }

  expect_equal(out_quick, expected, tolerance = 1e-12)
})
