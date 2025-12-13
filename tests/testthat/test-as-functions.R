test_that("as_r_function() matches graph_to_r_function()", {
  graph <- trace_fn(
    function(x, y) x + y,
    list(
      x = nv_scalar(1.0, dtype = "f32"),
      y = nv_scalar(2.0, dtype = "f32")
    )
  )

  f1 <- as_r_function(graph)
  f2 <- graph_to_r_function(graph, include_declare = TRUE)
  expect_equal(f1(3, 4), f2(3, 4))
})

test_that("as_quickr_function() matches graph_to_quickr_function()", {
  testthat::skip_if_not_installed("quickr")

  graph <- trace_fn(
    function(x, y) x * y + x,
    list(
      x = nv_scalar(1.0, dtype = "f32"),
      y = nv_scalar(2.0, dtype = "f32")
    )
  )

  f1 <- as_quickr_function(graph)
  f2 <- graph_to_quickr_function(graph)

  expect_equal(f1(3, 4), f2(3, 4), tolerance = 1e-6)
})
