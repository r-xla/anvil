test_that("graph_to_quickr_function requires {quickr}", {
  graph <- trace_fn(
    function(x) x + x,
    list(x = nv_scalar(1.0, dtype = "f32"))
  )

  if (!requireNamespace("quickr", quietly = TRUE)) {
    expect_error(graph_to_quickr_function(graph), "quickr", fixed = FALSE)
    return()
  }

  f <- graph_to_quickr_function(graph)
  expect_equal(f(2), 4)
})
