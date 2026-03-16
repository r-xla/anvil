test_that("jit: quickr backend compiles simple function", {
  skip_if_not_installed("quickr")

  f <- jit(function(x, y) x + y, backend = "quickr")

  expect_equal(f(1, 2), 3)
})

test_that("jit: default backend can be configured via option", {
  skip_if_not_installed("quickr")

  withr::local_options(list(anvil.default_backend = "quickr"))

  f <- jit(function(x, y) x + y)

  expect_equal(f(1, 2), 3)
})

test_that("jit: quickr backend does not support donate or device", {
  expect_error(
    jit(function(x) x, backend = "quickr", donate = "x"),
    "donate",
    fixed = TRUE
  )
  expect_error(
    jit(function(x) x, backend = "quickr", device = "cpu"),
    "device",
    fixed = TRUE
  )
})

test_that("graph_to_r_function lowers a graph to a plain R function", {
  skip_if_not_installed("quickr")

  graph <- trace_fn(
    function(x) x + 1,
    list(x = nv_scalar(1.0, dtype = "f64"))
  )

  f <- graph_to_r_function(graph)

  expect_equal(f(2), 3)
})
