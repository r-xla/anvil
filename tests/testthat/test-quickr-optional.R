test_that("graph_to_r_function include_declare stays runnable", {
  graph <- trace_fn(
    function(x) x + x,
    list(x = nv_scalar(1.0, dtype = "f32"))
  )

  f <- graph_to_r_function(graph, include_declare = TRUE)
  expect_equal(f(2), 4)

  if (!requireNamespace("quickr", quietly = TRUE)) {
    expect_error(graph_to_quickr_function(graph), "quickr", fixed = FALSE)
  }
})
