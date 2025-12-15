test_that("integration: MNIST-shaped classifier training from rank-5 image batch via quickr loss + grad", {
  testthat::skip_if_not_installed("quickr")

  set.seed(1)

  n <- 16L
  h <- 28L
  w <- 28L
  c <- 1L
  t <- 1L
  d <- h * w * c * t
  k <- 10L

  X <- array(rnorm(n * h * w * c * t, sd = 0.5), dim = c(n, h, w, c, t))
  labels <- sample.int(k, size = n, replace = TRUE)
  Y <- matrix(0, nrow = n, ncol = k)
  Y[cbind(seq_len(n), labels)] <- 1

  W <- matrix(rnorm(d * k, sd = 0.05), nrow = d, ncol = k)
  b <- rnorm(k, sd = 0.05)

  scale <- nv_scalar(1.0 / (n * k), dtype = "f32")

  loss_fn <- function(X, Y, W, b) {
    X2 <- nv_reshape(X, shape = c(n, d))
    logits <- nv_matmul(X2, W) + nv_broadcast_to(b, shape = c(n, k))
    resid <- logits - Y
    sum(resid * resid) * scale
  }

  loss_and_grad <- function(X, Y, W, b) {
    g <- gradient(loss_fn, wrt = c("W", "b", "X"))
    list(
      loss = loss_fn(X, Y, W, b),
      grad = g(X, Y, W, b)
    )
  }

  graph <- trace_fn(
    loss_and_grad,
    list(
      X = nv_tensor(X, dtype = "f32", shape = dim(X)),
      Y = nv_tensor(Y, dtype = "f32", shape = dim(Y)),
      W = nv_tensor(W, dtype = "f32", shape = dim(W)),
      b = nv_tensor(b, dtype = "f32", shape = c(k))
    )
  )

  f_quick <- graph_to_quickr_function(graph)

  out_quick0 <- f_quick(X, Y, W, b)
  out_pjrt0 <- eval_graph_pjrt(graph, X, Y, W, b)
  expect_equal(out_quick0, out_pjrt0, tolerance = 1e-4)

  lr <- 0.1
  losses <- numeric(5)
  for (iter in seq_along(losses)) {
    out <- f_quick(X, Y, W, b)
    losses[[iter]] <- as.numeric(out$loss)
    W <- W - lr * out$grad$W
    b <- b - lr * out$grad$b
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
