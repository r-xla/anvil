expect_graph_to_quickr_error <- function(fn, templates, pattern) {
  graph <- trace_fn(fn, templates)
  testthat::expect_error(graph_to_quickr_function(graph), pattern, fixed = FALSE)
}

test_that("graph_to_quickr_function errors on unsupported primitives", {
  testthat::skip_if_not_installed("quickr")

  unsupported_fns <- list(
    function(x) nv_popcnt(x),
    function(x, y) nv_shift_left(x, y),
    function(x, y) nv_shift_right_logical(x, y),
    function(x, y) nv_shift_right_arithmetic(x, y)
  )

  templates_1 <- list(x = nv_tensor(1L, dtype = "i32", shape = 1L))
  templates_2 <- list(
    x = nv_tensor(1L, dtype = "i32", shape = 1L),
    y = nv_tensor(1L, dtype = "i32", shape = 1L)
  )

  for (fn in unsupported_fns) {
    templates <- if (length(formals(fn)) == 1L) templates_1 else templates_2
    expect_graph_to_quickr_error(fn, templates, "does not support these primitives")
  }
})

test_that("graph_to_quickr_function errors on unsupported ranks", {
  testthat::skip_if_not_installed("quickr")

  # input rank > 5 is rejected during quickr argument declaration
  x6 <- array(0, dim = rep(1L, 6L))
  expect_graph_to_quickr_error(
    function(x) x,
    list(x = nv_tensor(x6, dtype = "f64", shape = dim(x6))),
    "supports tensors up to rank 5"
  )

  # output rank > 5 in fill is rejected in the emitter
  expect_graph_to_quickr_error(
    function() nv_fill(1.0, shape = rep(1L, 6L), dtype = "f64"),
    list(),
    "only tensors up to rank 5"
  )
})

test_that("graph_to_quickr_function rejects unsupported dtypes", {
  testthat::skip_if_not_installed("quickr")

  graph <- trace_fn(function(x) x + x, list(x = nv_aten("i64", c())))
  testthat::expect_error(graph_to_quickr_function(graph), "Unsupported dtype.*i64", fixed = FALSE)
})

test_that("graph_to_quickr_function rejects transpose ranks other than 2", {
  testthat::skip_if_not_installed("quickr")

  x3 <- array(1:8, dim = c(2L, 2L, 2L))
  graph <- trace_fn(
    function(x) nvl_transpose(x, permutation = c(2L, 1L, 3L)),
    list(x = nv_tensor(x3, dtype = "i32", shape = dim(x3)))
  )
  testthat::expect_error(graph_to_quickr_function(graph), "transpose: only rank-2", fixed = FALSE)
})

test_that("graph_to_quickr_function rejects reshape ranks > 5", {
  testthat::skip_if_not_installed("quickr")

  x5 <- array(1, dim = rep(1L, 5L))
  graph <- trace_fn(
    function(x) nvl_reshape(x, shape = rep(1L, 6L)),
    list(x = nv_tensor(x5, dtype = "i32", shape = dim(x5)))
  )
  testthat::expect_error(graph_to_quickr_function(graph), "reshape: only tensors up to rank 5", fixed = FALSE)
})

test_that("graph_to_quickr_function rejects broadcast_in_dim ranks > 5", {
  testthat::skip_if_not_installed("quickr")

  graph <- trace_fn(
    function(x) nvl_broadcast_in_dim(x, shape = rep(1L, 6L), broadcast_dimensions = 6L),
    list(x = nv_tensor(1L, dtype = "i32", shape = 1L))
  )
  expect_error(graph_to_quickr_function(graph), "broadcast_in_dim: only tensors up to rank 5", fixed = FALSE)
})

test_that("graph_to_quickr_function rejects reductions over empty dimensions", {
  skip_if_not_installed("quickr")

  templ <- list(x = nv_aten("f64", c(2L, 0L)))
  graph <- trace_fn(function(x) nvl_reduce_max(x, dims = 2L, drop = TRUE), templ)
  expect_error(graph_to_quickr_function(graph), "empty dimensions", fixed = FALSE)
})

test_that("graph_to_quickr_function rejects unsupported reduce_sum variants", {
  skip_if_not_installed("quickr")

  graph <- trace_fn(
    function(x) nvl_reduce_sum(x, dims = 1L, drop = TRUE),
    list(x = nv_scalar(0.0, dtype = "f64"))
  )
  expect_error(graph_to_quickr_function(graph), "sum: scalar reduction dims must be empty", fixed = FALSE)

  graph <- trace_fn(
    function(x) nvl_reduce_sum(x, dims = 2L, drop = TRUE),
    list(x = nv_tensor(1:4, dtype = "i32", shape = 4L))
  )
  expect_error(graph_to_quickr_function(graph), "sum: unsupported reduction dims for rank-1 tensor", fixed = FALSE)

  graph <- trace_fn(
    function(x) nvl_reduce_sum(x, dims = 3L, drop = TRUE),
    list(x = nv_tensor(matrix(1:6, nrow = 2, ncol = 3), dtype = "i32", shape = c(2L, 3L)))
  )
  expect_error(graph_to_quickr_function(graph), "sum: unsupported reduction dims for rank-2 tensor", fixed = FALSE)

  graph <- trace_fn(
    function(x) nvl_reduce_sum(x, dims = 2L, drop = TRUE),
    list(x = nv_tensor(array(1:8, dim = c(2L, 2L, 2L)), dtype = "i32", shape = c(2L, 2L, 2L)))
  )
  expect_error(graph_to_quickr_function(graph), "for rank > 2, only full reductions", fixed = FALSE)
})
