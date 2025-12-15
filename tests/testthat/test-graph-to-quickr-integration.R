test_that("integration: training loop via quickr-compiled loss + grad", {
  testthat::skip_if_not_installed("quickr")

  set.seed(1)

  n <- 12L
  d <- 3L
  h <- 4L

  X <- matrix(rnorm(n * d, sd = 0.5), nrow = n, ncol = d)
  y <- matrix(rnorm(n, sd = 0.25), nrow = n, ncol = 1L)

  W1 <- matrix(rnorm(d * h, sd = 0.1), nrow = d, ncol = h)
  b1 <- rnorm(h, sd = 0.1)
  W2 <- matrix(rnorm(h, sd = 0.1), nrow = h, ncol = 1L)
  b2 <- 0.0

  scale <- nv_scalar(1.0 / n, dtype = "f32")

  loss_fn <- function(X, y, W1, b1, W2, b2) {
    hidden <- nv_matmul(X, W1) + nv_broadcast_to(b1, shape = c(n, h))
    act <- hidden * hidden
    pred <- nv_matmul(act, W2) + b2
    resid <- pred - y
    sum(resid * resid) * scale
  }

  loss_and_grad <- function(X, y, W1, b1, W2, b2) {
    g <- gradient(loss_fn, wrt = c("W1", "b1", "W2", "b2"))
    list(
      loss = loss_fn(X, y, W1, b1, W2, b2),
      grad = g(X, y, W1, b1, W2, b2)
    )
  }

  graph <- trace_fn(
    loss_and_grad,
    list(
      X = nv_tensor(X, dtype = "f32", shape = dim(X)),
      y = nv_tensor(y, dtype = "f32", shape = dim(y)),
      W1 = nv_tensor(W1, dtype = "f32", shape = dim(W1)),
      b1 = nv_tensor(b1, dtype = "f32", shape = c(h)),
      W2 = nv_tensor(W2, dtype = "f32", shape = dim(W2)),
      b2 = nv_scalar(b2, dtype = "f32")
    )
  )

  f_quick <- graph_to_quickr_function(graph)

  out_quick0 <- f_quick(X, y, W1, b1, W2, b2)
  out_pjrt0 <- eval_graph_pjrt(graph, X, y, W1, b1, W2, b2)
  expect_equal(out_quick0, out_pjrt0, tolerance = 1e-4)

  lr <- 0.05
  losses <- numeric(5)
  for (iter in seq_along(losses)) {
    out <- f_quick(X, y, W1, b1, W2, b2)
    losses[[iter]] <- as.numeric(out$loss)
    W1 <- W1 - lr * out$grad$W1
    b1 <- b1 - lr * out$grad$b1
    W2 <- W2 - lr * out$grad$W2
    b2 <- b2 - lr * as.numeric(out$grad$b2)
  }
  expect_lt(losses[[length(losses)]], losses[[1L]])
})

test_that("integration: tfp/greta-like log_prob + grad workflow via quickr", {
  testthat::skip_if_not_installed("quickr")

  set.seed(2)

  n <- 20L
  x <- as.numeric(scale(seq_len(n)))
  y <- 1.5 * x - 0.3 + rnorm(n, sd = 0.15)

  x_nv <- nv_tensor(x, dtype = "f32", shape = c(n))
  y_nv <- nv_tensor(y, dtype = "f32", shape = c(n))

  half <- nv_scalar(0.5, dtype = "f32")

  log_joint <- function(w, b) {
    mu <- w * x_nv + b
    resid <- y_nv - mu
    ll <- -half * sum(resid * resid)
    prior <- -half * (w * w + b * b)
    ll + prior
  }

  logp_and_grad <- function(w, b) {
    g <- gradient(log_joint, wrt = c("w", "b"))
    list(
      log_prob = log_joint(w, b),
      grad = g(w, b)
    )
  }

  graph <- trace_fn(
    logp_and_grad,
    list(
      w = nv_scalar(0.0, dtype = "f32"),
      b = nv_scalar(0.0, dtype = "f32")
    )
  )

  f_quick <- graph_to_quickr_function(graph)

  out_quick0 <- f_quick(0.1, -0.2)
  out_pjrt0 <- eval_graph_pjrt(graph, 0.1, -0.2)
  expect_equal(out_quick0, out_pjrt0, tolerance = 1e-4)

  # A few steps of gradient ascent (MAP for this quadratic objective).
  w <- 0.0
  b <- 0.0
  lp0 <- as.numeric(f_quick(w, b)$log_prob)

  step <- 0.01
  for (iter in seq_len(10)) {
    out <- f_quick(w, b)
    w <- w + step * as.numeric(out$grad$w)
    b <- b + step * as.numeric(out$grad$b)
  }

  lp1 <- as.numeric(f_quick(w, b)$log_prob)
  expect_gt(lp1, lp0)
})
