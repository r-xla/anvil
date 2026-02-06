test_that("graph_to_quickr_function matches R for reduce_max/reduce_min (PJRT currently differs)", {
  testthat::skip_if_not_installed("quickr")

  withr::local_seed(44)

  x <- matrix(rnorm(6, sd = 0.2), nrow = 2, ncol = 3)
  templ <- list(x = nv_tensor(x, dtype = "f64", shape = dim(x)))

  graph <- trace_fn(function(x) nvl_reduce_max(x, dims = 2L, drop = TRUE), templ)
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(as.vector(f_quick(x)), apply(x, 1L, max))

  graph <- trace_fn(function(x) nvl_reduce_min(x, dims = 1L, drop = TRUE), templ)
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(as.vector(f_quick(x)), apply(x, 2L, min))
})
