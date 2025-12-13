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

test_that("graph_to_r_function: dot_general supports nonstandard contracting dims (rank-2)", {
  A <- matrix(rnorm(6 * 5), nrow = 6, ncol = 5)
  B <- matrix(rnorm(12 * 5), nrow = 12, ncol = 5)

  g_22 <- trace_fn(
    function(A, B) {
      anvil:::nvl_dot_general(A, B, contracting_dims = list(2L, 2L), batching_dims = list(integer(), integer()))
    },
    list(
      A = nv_tensor(A, dtype = "f32", shape = dim(A)),
      B = nv_tensor(B, dtype = "f32", shape = dim(B))
    )
  )
  f_22 <- graph_to_r_function(g_22)
  expect_equal(f_22(A, B), A %*% t(B), tolerance = 1e-5)

  C <- matrix(rnorm(6 * 12), nrow = 6, ncol = 12)
  D <- matrix(rnorm(6 * 8), nrow = 6, ncol = 8)
  g_11 <- trace_fn(
    function(C, D) {
      anvil:::nvl_dot_general(C, D, contracting_dims = list(1L, 1L), batching_dims = list(integer(), integer()))
    },
    list(
      C = nv_tensor(C, dtype = "f32", shape = dim(C)),
      D = nv_tensor(D, dtype = "f32", shape = dim(D))
    )
  )
  f_11 <- graph_to_r_function(g_11)
  expect_equal(f_11(C, D), t(C) %*% D, tolerance = 1e-5)
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

test_that("graph_to_r_function: fuses broadcast_in_dim mul for rank-3 axis=1", {
  shape_x <- c(3L, 4L, 5L)
  x <- array(runif(prod(shape_x)), dim = shape_x)
  y <- as.numeric(seq_len(shape_x[[1L]]))

  graph <- trace_fn(
    function(x, y) {
      anvil:::nvl_mul(
        x,
        anvil:::nvl_broadcast_in_dim(y, shape_out = shape_x, broadcast_dimensions = 1L)
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
  for (k in seq_len(shape_x[[1L]])) {
    expect[k, , ] <- expect[k, , ] * y[[k]]
  }

  expect_equal(got, expect, tolerance = 1e-6)

  body_txt <- paste(deparse(body(f)), collapse = "\n")
  expect_match(body_txt, "seq_len\\(")
  expect_match(body_txt, "y\\[k_")
})

test_that("graph_to_r_function: fuses when broadcasted vector is lhs of mul", {
  shape_x <- c(2L, 3L, 4L)
  x <- array(runif(prod(shape_x)), dim = shape_x)
  y <- as.numeric(seq_len(shape_x[[2L]]))

  graph <- trace_fn(
    function(x, y) {
      anvil:::nvl_mul(
        anvil:::nvl_broadcast_in_dim(y, shape_out = shape_x, broadcast_dimensions = 2L),
        x
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
  for (k in seq_len(shape_x[[2L]])) {
    expect[, k, ] <- expect[, k, ] * y[[k]]
  }

  expect_equal(got, expect, tolerance = 1e-6)
})

test_that("graph_to_r_function: does not fuse broadcast_in_dim if used twice", {
  shape_x <- c(3L, 4L, 5L, 6L)
  x <- array(runif(prod(shape_x)), dim = shape_x)
  y <- as.numeric(seq_len(shape_x[[3L]]))

  graph <- trace_fn(
    function(x, y) {
      b <- anvil:::nvl_broadcast_in_dim(y, shape_out = shape_x, broadcast_dimensions = 3L)
      anvil:::nvl_add(anvil:::nvl_mul(x, b), anvil:::nvl_mul(x, b))
    },
    list(
      x = nv_tensor(x, dtype = "f32", shape = shape_x),
      y = nv_tensor(y, dtype = "f32", shape = c(length(y)))
    )
  )

  expect_error(
    graph_to_r_function(graph),
    "broadcast_in_dim: only rank-1/2 outputs are currently supported"
  )
})

test_that("graph_to_r_function: supports if primitive (scalar output)", {
  graph <- trace_fn(
    function(pred) {
      anvil:::nvl_if(
        pred,
        nv_scalar(1.25, dtype = "f32"),
        nv_scalar(-2.0, dtype = "f32")
      )
    },
    list(pred = nv_scalar(TRUE, dtype = "pred"))
  )

  f <- graph_to_r_function(graph)
  expect_equal(f(TRUE), 1.25)
  expect_equal(f(FALSE), -2.0)
})

test_that("graph_to_r_function: if can use inputs in branches", {
  graph <- trace_fn(
    function(pred, x) {
      anvil:::nvl_if(
        pred,
        x + nv_scalar(1.0, dtype = "f32"),
        x - nv_scalar(1.0, dtype = "f32")
      )
    },
    list(
      pred = nv_scalar(TRUE, dtype = "pred"),
      x = nv_scalar(0.0, dtype = "f32")
    )
  )

  f <- graph_to_r_function(graph)
  expect_equal(f(TRUE, 3), 4)
  expect_equal(f(FALSE, 3), 2)
})

test_that("graph_to_r_function: if can use computed predicate", {
  graph <- trace_fn(
    function(x) {
      pred <- x > nv_scalar(0.0, dtype = "f32")
      anvil:::nvl_if(
        pred,
        x + nv_scalar(1.0, dtype = "f32"),
        x - nv_scalar(1.0, dtype = "f32")
      )
    },
    list(x = nv_scalar(0.0, dtype = "f32"))
  )

  f <- graph_to_r_function(graph)
  expect_equal(f(2), 3)
  expect_equal(f(-2), -3)
})

test_that("graph_to_r_function: supports if primitive (tensor output)", {
  graph <- trace_fn(
    function(pred) {
      anvil:::nvl_if(
        pred,
        nv_full(1.0, shape = c(2L, 3L), dtype = "f32"),
        nv_full(2.0, shape = c(2L, 3L), dtype = "f32")
      )
    },
    list(pred = nv_scalar(TRUE, dtype = "pred"))
  )

  f <- graph_to_r_function(graph)
  expect_equal(f(TRUE), matrix(1.0, nrow = 2, ncol = 3))
  expect_equal(f(FALSE), matrix(2.0, nrow = 2, ncol = 3))
})

test_that("graph_to_r_function: supports if primitive (rank-3 tensor output)", {
  shape_out <- c(2L, 3L, 4L)
  graph <- trace_fn(
    function(pred) {
      anvil:::nvl_if(
        pred,
        nv_full(1.0, shape = shape_out, dtype = "f32"),
        nv_full(2.0, shape = shape_out, dtype = "f32")
      )
    },
    list(pred = nv_scalar(TRUE, dtype = "pred"))
  )

  f <- graph_to_r_function(graph)
  got_t <- f(TRUE)
  got_f <- f(FALSE)
  expect_equal(dim(got_t), shape_out)
  expect_equal(dim(got_f), shape_out)
  expect_equal(as.numeric(got_t), rep(1.0, prod(shape_out)))
  expect_equal(as.numeric(got_f), rep(2.0, prod(shape_out)))
})

test_that("graph_to_r_function: supports nested if", {
  graph <- trace_fn(
    function(p1, p2) {
      anvil:::nvl_if(
        p1,
        anvil:::nvl_if(p2, nv_scalar(1.0, dtype = "f32"), nv_scalar(2.0, dtype = "f32")),
        nv_scalar(3.0, dtype = "f32")
      )
    },
    list(
      p1 = nv_scalar(TRUE, dtype = "pred"),
      p2 = nv_scalar(TRUE, dtype = "pred")
    )
  )

  f <- graph_to_r_function(graph)
  expect_equal(f(TRUE, TRUE), 1.0)
  expect_equal(f(TRUE, FALSE), 2.0)
  expect_equal(f(FALSE, TRUE), 3.0)

  body_txt <- paste(deparse(body(f)), collapse = "\n")
  expect_match(body_txt, "\\bif \\(")
})

test_that("graph_to_r_function: supports if with broadcast fusion inside branch", {
  shape_x <- c(3L, 4L, 5L, 6L)
  x <- array(runif(prod(shape_x)), dim = shape_x)
  y <- as.numeric(seq_len(shape_x[[3L]]))

  graph <- trace_fn(
    function(pred, x, y) {
      anvil:::nvl_if(
        pred,
        anvil:::nvl_mul(
          x,
          anvil:::nvl_broadcast_in_dim(y, shape_out = shape_x, broadcast_dimensions = 3L)
        ),
        x
      )
    },
    list(
      pred = nv_scalar(TRUE, dtype = "pred"),
      x = nv_tensor(x, dtype = "f32", shape = shape_x),
      y = nv_tensor(y, dtype = "f32", shape = c(length(y)))
    )
  )

  f <- graph_to_r_function(graph)
  got_t <- f(TRUE, x, y)
  got_f <- f(FALSE, x, y)

  expect_t <- x
  for (k in seq_len(shape_x[[3L]])) {
    expect_t[, , k, ] <- expect_t[, , k, ] * y[[k]]
  }
  expect_equal(got_t, expect_t, tolerance = 1e-6)
  expect_equal(got_f, x, tolerance = 1e-6)
})

test_that("graph_to_r_function: if predicate must be scalar", {
  graph <- trace_fn(
    function(pred) {
      anvil:::nvl_if(
        pred,
        nv_scalar(1.0, dtype = "f32"),
        nv_scalar(2.0, dtype = "f32")
      )
    },
    list(pred = nv_tensor(matrix(c(TRUE, FALSE), nrow = 1, ncol = 2), dtype = "pred", shape = c(1, 2)))
  )

  expect_error(graph_to_r_function(graph), "predicate must be a scalar")
})

test_that("graph_to_r_function errors on unsupported higher-order primitives", {
  graph <- trace_fn(
    function(x) {
      anvil:::nvl_while(
        init = list(x = x),
        cond = function(x) x < nv_scalar(10.0, dtype = "f32"),
        body = function(x) list(x = x + nv_scalar(1.0, dtype = "f32"))
      )$x
    },
    list(x = nv_scalar(0.0, dtype = "f32"))
  )

  expect_error(graph_to_r_function(graph), "does not support.*\\bwhile\\b")
})
