test_that("graph_to_r_function: simple add", {
  graph <- trace_fn(
    function(x, y) {
      nvl_add(x, y)
    },
    list(x = nv_scalar(1.0, dtype = "f32"), y = nv_scalar(2.0, dtype = "f32"))
  )

  f <- graph_to_r_function(graph)
  expect_equal(f(1, 2), 3)
})

test_that("graph_to_r_function: closed-over constants can be inlined", {
  x <- nv_scalar(1.0, dtype = "f32")
  graph <- trace_fn(
    function(y) {
      nvl_add(x, y)
    },
    list(y = nv_scalar(2.0, dtype = "f32"))
  )

  f <- graph_to_r_function(graph, constants = "inline")
  expect_equal(f(2), 3)
})

test_that("graph_to_r_function: closed-over constants can be arguments", {
  x <- nv_scalar(1.0, dtype = "f32")
  graph <- trace_fn(
    function(y) {
      nvl_add(x, y)
    },
    list(y = nv_scalar(2.0, dtype = "f32"))
  )

  f <- graph_to_r_function(graph, constants = "args")
  expect_equal(f(2, 1), 3)
})

test_that("graph_to_r_function: constant primitive is supported", {
  graph <- trace_fn(
    function(x) {
      nvl_add(x, nv_full(1.0, shape = integer(), dtype = "f32"))
    },
    list(x = nv_scalar(2.0, dtype = "f32"))
  )

  f <- graph_to_r_function(graph)
  expect_equal(f(2), 3)
})

test_that("graph_to_r_function: scalar broadcast_in_dim is supported", {
  graph <- trace_fn(
    function(x) {
      nv_add(x, nv_full(1.0, shape = c(3L), dtype = "f32"))
    },
    list(x = nv_scalar(2.0, dtype = "f32"))
  )

  f <- graph_to_r_function(graph)
  out <- f(2)
  expect_equal(length(out), 3L)
  expect_equal(as.vector(out), rep(3, 3))
})
