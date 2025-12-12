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

test_that("graph_to_r_function: maximum/minimum are supported", {
  X <- matrix(c(1, 5, 3, 4), nrow = 2, ncol = 2)
  Y <- matrix(c(2, 4, 9, 0), nrow = 2, ncol = 2)

  g_max <- trace_fn(
    function(X, Y) nv_max(X, Y),
    list(
      X = nv_tensor(X, dtype = "f32", shape = dim(X)),
      Y = nv_tensor(Y, dtype = "f32", shape = dim(Y))
    )
  )
  g_min <- trace_fn(
    function(X, Y) nv_min(X, Y),
    list(
      X = nv_tensor(X, dtype = "f32", shape = dim(X)),
      Y = nv_tensor(Y, dtype = "f32", shape = dim(Y))
    )
  )

  f_max <- graph_to_r_function(g_max)
  f_min <- graph_to_r_function(g_min)

  expect_equal(f_max(X, Y), pmax(X, Y))
  expect_equal(f_min(X, Y), pmin(X, Y))
})

test_that("graph_to_r_function: transpose and reshape are supported", {
  X <- matrix(1:6, nrow = 2, ncol = 3, byrow = TRUE)

  g_t <- trace_fn(
    function(X) nv_transpose(X, permutation = c(2L, 1L)),
    list(X = nv_tensor(X, dtype = "f32", shape = dim(X)))
  )
  f_t <- graph_to_r_function(g_t)
  expect_equal(f_t(X), t(X))

  g_r <- trace_fn(
    function(X) nv_reshape(X, shape = c(6L)),
    list(X = nv_tensor(X, dtype = "f32", shape = dim(X)))
  )
  f_r <- graph_to_r_function(g_r)
  expect_equal(as.vector(f_r(X)), as.numeric(1:6))
})

test_that("graph_to_r_function: convert is supported", {
  graph <- trace_fn(
    function(x) nv_convert(x, "i32"),
    list(x = nv_scalar(2.0, dtype = "f32"))
  )

  f <- graph_to_r_function(graph)
  expect_equal(f(2.0), 2L)
})

test_that("graph_to_r_function: fuses broadcast_in_dim mul for rank-4", {
  shape_x <- c(3L, 4L, 5L, 6L)
  x <- array(runif(prod(shape_x)), dim = shape_x)
  y <- as.numeric(seq_len(shape_x[[3L]]))

  graph <- trace_fn(
    function(x, y) {
      anvil:::nvl_mul(
        x,
        anvil:::nvl_broadcast_in_dim(y, shape_out = shape_x, broadcast_dimensions = 3L)
      )
    },
    list(
      x = nv_tensor(x, dtype = "f32", shape = shape_x),
      y = nv_tensor(y, dtype = "f32", shape = c(length(y)))
    )
  )

  f <- graph_to_r_function(graph)
  got <- f(x, y)

  expect <- x
  for (k in seq_len(shape_x[[3L]])) {
    expect[, , k, ] <- expect[, , k, ] * y[[k]]
  }

  expect_equal(got, expect, tolerance = 1e-6)

  body_txt <- paste(deparse(body(f)), collapse = "\n")
  expect_match(body_txt, "for \\(")
  expect_match(body_txt, "\\[,\\s*,\\s*[^,]")
})

test_that("graph_to_r_function errors on unsupported higher-order primitives", {
  pred <- nv_scalar(TRUE, dtype = "pred")
  graph <- trace_fn(
    function(pred) {
      anvil:::nvl_if(pred, nv_scalar(1.0, dtype = "f32"), nv_scalar(2.0, dtype = "f32"))
    },
    list(pred = pred)
  )

  expect_error(graph_to_r_function(graph), "does not support.*\\bif\\b")
})
