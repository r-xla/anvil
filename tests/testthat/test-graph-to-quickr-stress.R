test_that("graph_to_quickr_function handles long scalar chain (stress)", {
  skip_if_no_quickr_or_pjrt()

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

  x <- 0.123
  templates <- list(x = nv_scalar(0.0, dtype = "f64"))
  run <- list(args = list(x = x), info = "run")
  expect_quickr_matches_pjrt_fn(chain_fn, templates, list(run), tolerance = 1e-12)
})
