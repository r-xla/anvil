test_that("graph_to_quickr_function matches PJRT for fill (including pred)", {
  testthat::skip_if_not_installed("quickr")

  graph <- trace_fn(function() nv_fill(3.25, shape = integer(), dtype = "f64"), list())
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(), eval_graph_pjrt(graph))

  graph <- trace_fn(function() nv_fill(2L, shape = c(4L), dtype = "i32"), list())
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(), eval_graph_pjrt(graph))

  graph <- trace_fn(function() nv_fill(TRUE, shape = c(2L, 3L), dtype = "pred"), list())
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(), eval_graph_pjrt(graph))
})

test_that("graph_to_quickr_function matches PJRT for transpose permutations", {
  testthat::skip_if_not_installed("quickr")

  x <- matrix(1:6, nrow = 2, ncol = 3)
  templ <- list(x = nv_tensor(x, dtype = "i32", shape = dim(x)))

  graph <- trace_fn(function(x) nvl_transpose(x, permutation = c(1L, 2L)), templ)
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(x), eval_graph_pjrt(graph, x))

  graph <- trace_fn(function(x) nvl_transpose(x, permutation = c(2L, 1L)), templ)
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(x), eval_graph_pjrt(graph, x))
})

test_that("graph_to_quickr_function matches PJRT for reshape edge cases", {
  testthat::skip_if_not_installed("quickr")

  # size-1 reshape hits the nflat==1 fast path in the emitter
  graph <- trace_fn(
    function(x) nvl_reshape(x, shape = c(1L, 1L)),
    list(x = nv_scalar(2.5, dtype = "f64"))
  )
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(2.5), eval_graph_pjrt(graph, 2.5))

  x <- array(seq_len(12), dim = c(2L, 2L, 3L))
  graph <- trace_fn(
    function(x) nvl_reshape(x, shape = c(4L, 3L)),
    list(x = nv_tensor(x, dtype = "i32", shape = dim(x)))
  )
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(x), eval_graph_pjrt(graph, x))
})

test_that("graph_to_quickr_function matches PJRT for broadcast_in_dim", {
  testthat::skip_if_not_installed("quickr")

  # scalar -> matrix
  graph <- trace_fn(
    function(x) nvl_broadcast_in_dim(x, shape = c(2L, 3L), broadcast_dimensions = integer()),
    list(x = nv_scalar(1.0, dtype = "f64"))
  )
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(2.5), eval_graph_pjrt(graph, 2.5))

  # singleton dim -> expanded dim
  x <- matrix(1:2, nrow = 2, ncol = 1)
  graph <- trace_fn(
    function(x) nvl_broadcast_in_dim(x, shape = c(2L, 3L), broadcast_dimensions = c(1L, 2L)),
    list(x = nv_tensor(x, dtype = "i32", shape = dim(x)))
  )
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(x), eval_graph_pjrt(graph, x))

  # higher-rank broadcast
  x <- array(1:2, dim = c(2L, 1L, 1L))
  graph <- trace_fn(
    function(x) nvl_broadcast_in_dim(x, shape = c(2L, 3L, 4L), broadcast_dimensions = c(1L, 2L, 3L)),
    list(x = nv_tensor(x, dtype = "i32", shape = dim(x)))
  )
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(x), eval_graph_pjrt(graph, x))
})

test_that("graph_to_quickr_function matches PJRT for reduce_sum variants", {
  testthat::skip_if_not_installed("quickr")

  # rank-1, drop TRUE/FALSE
  x <- 1:4
  graph <- trace_fn(
    function(x) nvl_reduce_sum(x, dims = 1L, drop = TRUE),
    list(x = nv_tensor(x, dtype = "i32", shape = c(length(x))))
  )
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(x), eval_graph_pjrt(graph, x))

  graph <- trace_fn(
    function(x) nvl_reduce_sum(x, dims = 1L, drop = FALSE),
    list(x = nv_tensor(x, dtype = "i32", shape = c(length(x))))
  )
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(x), eval_graph_pjrt(graph, x))

  # rank-2 reductions along each axis + full reduction
  x <- matrix(1:6, nrow = 2, ncol = 3)
  templ <- list(x = nv_tensor(x, dtype = "i32", shape = dim(x)))

  graph <- trace_fn(function(x) nvl_reduce_sum(x, dims = 1L, drop = TRUE), templ)
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(x), eval_graph_pjrt(graph, x))

  graph <- trace_fn(function(x) nvl_reduce_sum(x, dims = 2L, drop = TRUE), templ)
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(x), eval_graph_pjrt(graph, x))

  graph <- trace_fn(function(x) nvl_reduce_sum(x, dims = c(1L, 2L), drop = FALSE), templ)
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(x), eval_graph_pjrt(graph, x))

  # rank > 2: full reduction and no-op reduction
  x <- array(seq_len(24), dim = c(2L, 3L, 4L))
  templ <- list(x = nv_tensor(x, dtype = "i32", shape = dim(x)))

  graph <- trace_fn(function(x) nvl_reduce_sum(x, dims = 1:3, drop = TRUE), templ)
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(x), eval_graph_pjrt(graph, x))

  graph <- trace_fn(function(x) nvl_reduce_sum(x, dims = integer(), drop = TRUE), templ)
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(x), eval_graph_pjrt(graph, x))
})

test_that("graph_to_quickr_function matches PJRT for dot_general variants", {
  testthat::skip_if_not_installed("quickr")

  # No contracting dims: scalar * scalar
  graph <- trace_fn(
    function(a, b) {
      nvl_dot_general(
        a,
        b,
        contracting_dims = list(integer(), integer()),
        batching_dims = list(integer(), integer())
      )
    },
    list(
      a = nv_scalar(0.0, dtype = "f64"),
      b = nv_scalar(0.0, dtype = "f64")
    )
  )
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(2, 3), eval_graph_pjrt(graph, 2, 3))

  # No contracting dims: outer product
  a <- 1:2
  b <- c(10L, 20L, 30L)
  graph <- trace_fn(
    function(a, b) {
      nvl_dot_general(
        a,
        b,
        contracting_dims = list(integer(), integer()),
        batching_dims = list(integer(), integer())
      )
    },
    list(
      a = nv_tensor(a, dtype = "i32", shape = c(length(a))),
      b = nv_tensor(b, dtype = "i32", shape = c(length(b)))
    )
  )
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(a, b), eval_graph_pjrt(graph, a, b))

  # Contracting dims: dot product -> scalar
  a <- 1:3
  b <- 4:6
  graph <- trace_fn(
    function(a, b) {
      nvl_dot_general(
        a,
        b,
        contracting_dims = list(1L, 1L),
        batching_dims = list(integer(), integer())
      )
    },
    list(
      a = nv_tensor(a, dtype = "i32", shape = c(length(a))),
      b = nv_tensor(b, dtype = "i32", shape = c(length(b)))
    )
  )
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(a, b), eval_graph_pjrt(graph, a, b))
})

test_that("graph_to_quickr_function matches PJRT for convert scalar and rank-3", {
  testthat::skip_if_not_installed("quickr")

  graph <- trace_fn(
    function(x) nvl_convert(x, dtype = "i32"),
    list(x = nv_scalar(0.0, dtype = "f64"))
  )
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(2.25), eval_graph_pjrt(graph, 2.25))

  x <- array(sample.int(10, 24, replace = TRUE), dim = c(2L, 3L, 4L))
  graph <- trace_fn(
    function(x) nvl_convert(x, dtype = "f64"),
    list(x = nv_tensor(x, dtype = "i32", shape = dim(x)))
  )
  f_quick <- graph_to_quickr_function(graph)
  expect_equal(f_quick(x), eval_graph_pjrt(graph, x))
})
