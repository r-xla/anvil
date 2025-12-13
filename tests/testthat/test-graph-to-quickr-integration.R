test_that("integration: dense NN softmax loss (forward)", {
  testthat::skip_if_not_installed("quickr")
  testthat::skip_if_not_installed("pjrt")
  testthat::skip_if_not_installed("stablehlo")

  batch <- 8L
  nin <- 16L
  hidden <- 32L
  nout <- 10L

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

test_that("integration: dense NN (2-layer linear) MSE gradient (dW1)", {
  testthat::skip_if_not_installed("quickr")
  testthat::skip_if_not_installed("pjrt")
  testthat::skip_if_not_installed("stablehlo")

  batch <- 6L
  nin <- 8L
  hidden <- 12L
  nout <- 5L

  set.seed(321)
  X <- matrix(rnorm(batch * nin), nrow = batch, ncol = nin)
  W1 <- matrix(rnorm(nin * hidden), nrow = nin, ncol = hidden)
  b1 <- matrix(0, nrow = 1, ncol = hidden)
  W2 <- matrix(rnorm(hidden * nout), nrow = hidden, ncol = nout)
  b2 <- matrix(0, nrow = 1, ncol = nout)
  y <- matrix(rnorm(batch * nout), nrow = batch, ncol = nout)

  loss <- function(X, y, W1, b1, W2, b2) {
    h <- nv_matmul(X, W1)
    h <- h + nv_broadcast_to(b1, shape = shape(h))
    out <- nv_matmul(h, W2)
    out <- out + nv_broadcast_to(b2, shape = shape(out))
    resid <- out - y
    mse <- nv_reduce_sum(resid * resid, dims = c(1, 2), drop = TRUE) / nv_scalar(batch * nout, dtype = "f32")
    mse
  }

  gW1 <- gradient(loss, wrt = "W1")
  dW1 <- function(X, y, W1, b1, W2, b2) {
    gW1(X, y, W1, b1, W2, b2)[[1L]]
  }

  graph <- trace_fn(
    dW1,
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
  expect_equal(out_quick, out_pjrt, tolerance = 1e-4)
})

test_that("integration: Bayesian logistic regression log posterior", {
  testthat::skip_if_not_installed("quickr")
  testthat::skip_if_not_installed("pjrt")
  testthat::skip_if_not_installed("stablehlo")

  n <- 32L
  d <- 8L

  set.seed(999)
  X <- matrix(rnorm(n * d), nrow = n, ncol = d)
  beta <- matrix(rnorm(d), nrow = d, ncol = 1L)
  y <- matrix(sample(c(0, 1), n, replace = TRUE), nrow = n, ncol = 1L)
  tau2 <- 2.0

  logpost <- function(X, y, beta) {
    one <- nv_scalar(1.0, dtype = "f32")
    minus_half <- nv_scalar(-0.5, dtype = "f32")
    eta <- nv_matmul(X, beta) # (n, 1)
    p <- one / (one + nv_exp(-eta))
    ll <- nv_reduce_sum(
      y * nv_log(p) + (one - y) * nv_log(one - p),
      dims = c(1, 2),
      drop = TRUE
    )
    prior <- minus_half * nv_reduce_sum(beta * beta, dims = c(1, 2), drop = TRUE) / nv_scalar(tau2, dtype = "f32")
    ll + prior
  }

  graph <- trace_fn(
    logpost,
    list(
      X = nv_tensor(X, dtype = "f32", shape = dim(X)),
      y = nv_tensor(y, dtype = "f32", shape = dim(y)),
      beta = nv_tensor(beta, dtype = "f32", shape = dim(beta))
    )
  )

  f_quick <- graph_to_quickr_function(graph)
  out_quick <- f_quick(X, y, beta)
  out_pjrt <- eval_graph_pjrt(graph, X, y, beta)
  expect_equal(as.numeric(out_quick), as.numeric(out_pjrt), tolerance = 1e-4)
})

test_that("integration: Gaussian linear regression log posterior (LA-heavy)", {
  testthat::skip_if_not_installed("quickr")
  testthat::skip_if_not_installed("pjrt")
  testthat::skip_if_not_installed("stablehlo")

  n <- 24L
  d <- 6L

  set.seed(2024)
  X <- matrix(rnorm(n * d), nrow = n, ncol = d)
  beta <- matrix(rnorm(d), nrow = d, ncol = 1L)
  y <- matrix(rnorm(n), nrow = n, ncol = 1L)
  sigma2 <- 0.5
  tau2 <- 1.5

  logpost <- function(X, y, beta) {
    minus_half <- nv_scalar(-0.5, dtype = "f32")
    # loglik (up to constant): -0.5/sigma2 * ||y - X beta||^2
    resid <- y - nv_matmul(X, beta)
    rss <- nv_reduce_sum(resid * resid, dims = c(1, 2), drop = TRUE)
    ll <- minus_half * rss / nv_scalar(sigma2, dtype = "f32")

    # prior (up to constant): -0.5/tau2 * beta' beta
    XtX <- nv_matmul(nv_transpose(X, permutation = c(2L, 1L)), X)
    quad <- nv_matmul(nv_transpose(beta, permutation = c(2L, 1L)), nv_matmul(XtX, beta))
    quad_s <- nv_reduce_sum(quad, dims = c(1, 2), drop = TRUE)
    prior <- minus_half * quad_s / nv_scalar(tau2, dtype = "f32")
    ll + prior
  }

  graph <- trace_fn(
    logpost,
    list(
      X = nv_tensor(X, dtype = "f32", shape = dim(X)),
      y = nv_tensor(y, dtype = "f32", shape = dim(y)),
      beta = nv_tensor(beta, dtype = "f32", shape = dim(beta))
    )
  )

  f_quick <- graph_to_quickr_function(graph)
  out_quick <- f_quick(X, y, beta)
  out_pjrt <- eval_graph_pjrt(graph, X, y, beta)
  expect_equal(as.numeric(out_quick), as.numeric(out_pjrt), tolerance = 1e-4)
})
