test_that("graph_to_quickr_function matches PJRT for scalar add", {
  testthat::skip_if_not_installed("quickr")

  graph <- trace_fn(
    function(x1, x2) {
      x1 + x2
    },
    list(
      x1 = nv_scalar(1.25, dtype = "f32"),
      x2 = nv_scalar(-0.5, dtype = "f32")
    )
  )

  f_quick <- graph_to_quickr_function(graph)

  x1 <- 3.5
  x2 <- -2.25
  out_quick <- f_quick(x1, x2)
  out_pjrt <- eval_graph_pjrt(graph, x1, x2)
  expect_equal(out_quick, out_pjrt, tolerance = 1e-6)
})

test_that("graph_to_quickr_function matches PJRT for scalar + vector constant", {
  testthat::skip_if_not_installed("quickr")

  graph <- trace_fn(
    function(x) {
      x + nv_full(1.0, shape = c(3L), dtype = "f32")
    },
    list(x = nv_scalar(2.0, dtype = "f32"))
  )

  f_quick <- graph_to_quickr_function(graph)

  x <- 2.25
  out_quick <- f_quick(x)
  out_pjrt <- eval_graph_pjrt(graph, x)

  expect_equal(as.vector(out_quick), as.vector(out_pjrt), tolerance = 1e-6)
})

test_that("graph_to_quickr_function matches PJRT for matmul", {
  testthat::skip_if_not_installed("quickr")

  X <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3)
  B <- matrix(c(1, 0, 0, 1, 1, 1), nrow = 3, ncol = 2)

  graph <- trace_fn(
    function(X, B) {
      nv_matmul(X, B)
    },
    list(
      X = nv_tensor(X, dtype = "f32", shape = dim(X)),
      B = nv_tensor(B, dtype = "f32", shape = dim(B))
    )
  )

  f_quick <- graph_to_quickr_function(graph)

  out_quick <- f_quick(X, B)
  out_pjrt <- eval_graph_pjrt(graph, X, B)

  expect_equal(out_quick, out_pjrt, tolerance = 1e-5)
})

test_that("graph_to_quickr_function matches PJRT for sum reduction", {
  testthat::skip_if_not_installed("quickr")

  x <- c(1, 2, 3, 4, 5)

  graph <- trace_fn(
    function(x) {
      sum(x)
    },
    list(x = nv_tensor(x, dtype = "f32", shape = c(length(x))))
  )

  f_quick <- graph_to_quickr_function(graph)

  out_quick <- f_quick(x)
  out_pjrt <- eval_graph_pjrt(graph, x)

  expect_equal(out_quick, out_pjrt, tolerance = 1e-6)
})

test_that("graph_to_quickr_function matches PJRT for prod reduction", {
  skip_if_not_installed("quickr")

  f <- function(x) {
    nv_reduce_prod(x, dims = 1, drop = TRUE)
  }
  x <- nv_tensor(runif(7), dtype = "f32", shape = c(7))
  graph <- trace_fn(f, list(x = x))

  got_pjrt <- eval_graph_pjrt(graph, x)
  f_quick <- graph_to_quickr_function(graph)
  got_quick <- f_quick(as_array(x))

  expect_equal(as.numeric(got_quick), as.numeric(got_pjrt), tolerance = 1e-4)
})

test_that("graph_to_quickr_function matches PJRT for min reduction along rows/cols", {
  skip_if_not_installed("quickr")

  x <- nv_tensor(matrix(runif(12), nrow = 3, ncol = 4), dtype = "f32", shape = c(3, 4))

  # quickr can't return list, so test separately
  graph_rows <- trace_fn(function(x) nv_reduce_min(x, dims = 2, drop = TRUE), list(x = x))
  graph_cols <- trace_fn(function(x) nv_reduce_min(x, dims = 1, drop = TRUE), list(x = x))

  got_rows_pjrt <- eval_graph_pjrt(graph_rows, x)
  got_cols_pjrt <- eval_graph_pjrt(graph_cols, x)

  f_rows <- graph_to_quickr_function(graph_rows)
  f_cols <- graph_to_quickr_function(graph_cols)

  got_rows_quick <- f_rows(as_array(x))
  got_cols_quick <- f_cols(as_array(x))

  expect_equal(as.numeric(got_rows_quick), as.numeric(got_rows_pjrt), tolerance = 1e-4)
  expect_equal(as.numeric(got_cols_quick), as.numeric(got_cols_pjrt), tolerance = 1e-4)
})

test_that("graph_to_quickr_function matches PJRT for any/all reductions", {
  skip_if_not_installed("quickr")

  f_any <- function(x) nv_reduce_any(x, dims = 2, drop = FALSE)
  f_all <- function(x) nv_reduce_all(x, dims = 1, drop = FALSE)

  x <- nv_tensor(matrix(sample(c(TRUE, FALSE), 15, replace = TRUE), nrow = 3, ncol = 5), dtype = "pred", shape = c(3, 5))
  g_any <- trace_fn(f_any, list(x = x))
  g_all <- trace_fn(f_all, list(x = x))

  got_any_pjrt <- eval_graph_pjrt(g_any, x)
  got_all_pjrt <- eval_graph_pjrt(g_all, x)

  f_any_quick <- graph_to_quickr_function(g_any)
  f_all_quick <- graph_to_quickr_function(g_all)

  got_any_quick <- f_any_quick(as_array(x))
  got_all_quick <- f_all_quick(as_array(x))

  expect_equal(as.logical(got_any_quick), as.logical(got_any_pjrt))
  expect_equal(as.logical(got_all_quick), as.logical(got_all_pjrt))
})

test_that("graph_to_quickr_function matches PJRT for select", {
  skip_if_not_installed("quickr")

  f <- function(pred, x, y) {
    anvil:::nvl_select(pred, x, y)
  }
  pred <- nv_tensor(matrix(sample(c(TRUE, FALSE), 12, replace = TRUE), nrow = 3, ncol = 4), dtype = "pred", shape = c(3, 4))
  x <- nv_tensor(matrix(runif(12), nrow = 3, ncol = 4), dtype = "f32", shape = c(3, 4))
  y <- nv_tensor(matrix(runif(12), nrow = 3, ncol = 4), dtype = "f32", shape = c(3, 4))

  graph <- trace_fn(f, list(pred = pred, x = x, y = y))
  got_pjrt <- eval_graph_pjrt(graph, pred, x, y)

  f_quick <- graph_to_quickr_function(graph)
  got_quick <- f_quick(as_array(pred), as_array(x), as_array(y))

  expect_equal(as.numeric(got_quick), as.numeric(got_pjrt), tolerance = 1e-4)
})

test_that("graph_to_quickr_function matches PJRT for closed-over scalar constant (inline)", {
  testthat::skip_if_not_installed("quickr")

  x <- nv_scalar(1.0, dtype = "f32")
  graph <- trace_fn(
    function(y) {
      x + y
    },
    list(y = nv_scalar(2.0, dtype = "f32"))
  )

  f_quick <- graph_to_quickr_function(graph, constants = "inline")

  y <- 2.25
  out_quick <- f_quick(y)
  out_pjrt <- eval_graph_pjrt(graph, y)
  expect_equal(out_quick, out_pjrt, tolerance = 1e-6)
})

test_that("graph_to_quickr_function matches PJRT for closed-over scalar constant (args)", {
  testthat::skip_if_not_installed("quickr")

  x <- nv_scalar(1.0, dtype = "f32")
  graph <- trace_fn(
    function(y) {
      x + y
    },
    list(y = nv_scalar(2.0, dtype = "f32"))
  )

  f_quick <- graph_to_quickr_function(graph, constants = "args")

  y <- 2.25
  out_quick <- f_quick(y, 1.0)
  out_pjrt <- eval_graph_pjrt(graph, y)
  expect_equal(out_quick, out_pjrt, tolerance = 1e-6)
})

test_that("graph_to_quickr_function matches PJRT for mnist-like MLP loss", {
  testthat::skip_if_not_installed("quickr")

  batch <- 4L
  nin <- 3L
  hidden <- 5L
  nout <- 2L

  set.seed(123)
  X <- matrix(rnorm(batch * nin), nrow = batch, ncol = nin)
  W1 <- matrix(rnorm(nin * hidden), nrow = nin, ncol = hidden)
  b1 <- matrix(0, nrow = 1, ncol = hidden)
  W2 <- matrix(rnorm(hidden * nout), nrow = hidden, ncol = nout)
  b2 <- matrix(0, nrow = 1, ncol = nout)
  y_idx <- sample.int(nout, size = batch, replace = TRUE)
  y <- matrix(0, nrow = batch, ncol = nout)
  for (i in seq_len(batch)) {
    y[i, y_idx[[i]]] <- 1
  }

  loss <- function(X, y, W1, b1, W2, b2) {
    h <- nv_matmul(X, W1)
    h <- h + nv_broadcast_to(b1, shape = shape(h))
    h <- nv_max(h, nv_scalar(0, dtype = "f32"))

    logits <- nv_matmul(h, W2)
    logits <- logits + nv_broadcast_to(b2, shape = shape(logits))

    z_max <- nv_reduce_max(logits, dims = 2, drop = FALSE)
    logits_shift <- logits - nv_broadcast_to(z_max, shape = shape(logits))
    logsumexp <- z_max + nv_log(nv_reduce_sum(nv_exp(logits_shift), dims = 2, drop = FALSE))
    correct_logit <- nv_reduce_sum(y * logits, dims = 2, drop = FALSE)
    nv_reduce_mean(-correct_logit + logsumexp, dims = c(1, 2))
  }

  graph <- trace_fn(
    loss,
    list(
      X = nv_tensor(X, dtype = "f32", shape = dim(X)),
      y = nv_tensor(y, dtype = "f32", shape = dim(y)),
      W1 = nv_tensor(W1, dtype = "f32", shape = dim(W1)),
      b1 = nv_tensor(b1, dtype = "f32", shape = dim(b1)),
      W2 = nv_tensor(W2, dtype = "f32", shape = dim(W2)),
      b2 = nv_tensor(b2, dtype = "f32", shape = dim(b2))
    )
  )

  f_quick <- graph_to_quickr_function(graph)

  out_quick <- f_quick(X, y, W1, b1, W2, b2)
  out_pjrt <- eval_graph_pjrt(graph, X, y, W1, b1, W2, b2)

  expect_equal(as.numeric(out_quick), as.numeric(out_pjrt), tolerance = 1e-4)
})

test_that("graph_to_quickr_function matches PJRT for broadcasted vector mul along axis", {
  testthat::skip_if_not_installed("quickr")

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

  f_quick <- graph_to_quickr_function(graph)

  out_quick <- f_quick(x, y)
  out_pjrt <- eval_graph_pjrt(graph, x, y)

  expect_equal(out_quick, out_pjrt, tolerance = 1e-4)
})
