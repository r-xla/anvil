test_that("greta-like: attitude simple linear regression log joint (forward)", {
  testthat::skip_if_not_installed("quickr")
  testthat::skip_if_not_installed("pjrt")
  testthat::skip_if_not_installed("stablehlo")

  data(attitude, package = "datasets")
  x <- matrix(as.numeric(attitude$complaints), ncol = 1L)
  y <- matrix(as.numeric(attitude$rating), ncol = 1L)
  n <- nrow(x)

  log_joint <- function(intercept, coef, sd, x, y) {
    s <- function(val) nv_scalar(val, dtype = "f32")

    sd_pos <- nv_abs(sd) + s(1e-3)
    mu <- intercept + coef * x
    resid <- (y - mu) / sd_pos

    ll <- -s(0.5) * nv_reduce_sum(resid * resid, dims = c(1, 2), drop = TRUE) -
      s(n) * nv_log(sd_pos)

    z_intercept <- intercept / s(10)
    z_coef <- coef / s(10)
    z_sd <- sd_pos / s(3)

    lp_intercept <- -s(0.5) * (z_intercept * z_intercept)
    lp_coef <- -s(0.5) * (z_coef * z_coef)
    lp_sd <- -nv_log(s(1) + (z_sd * z_sd))

    ll + lp_intercept + lp_coef + lp_sd
  }

  graph <- trace_fn(
    log_joint,
    list(
      intercept = nv_scalar(0.1, dtype = "f32"),
      coef = nv_scalar(0.01, dtype = "f32"),
      sd = nv_scalar(1.0, dtype = "f32"),
      x = nv_tensor(x, dtype = "f32", shape = dim(x)),
      y = nv_tensor(y, dtype = "f32", shape = dim(y))
    )
  )

  f_quick <- graph_to_quickr_function(graph)
  out_quick <- f_quick(0.3, -0.02, 2.0, x, y)
  out_pjrt <- eval_graph_pjrt(graph, 0.3, -0.02, 2.0, x, y)
  expect_equal(as.numeric(out_quick), as.numeric(out_pjrt), tolerance = 1e-4)
})

test_that("greta-like: attitude multiple regression log joint (dcoefs)", {
  testthat::skip_if_not_installed("quickr")
  testthat::skip_if_not_installed("pjrt")
  testthat::skip_if_not_installed("stablehlo")

  data(attitude, package = "datasets")
  X <- as.matrix(attitude[, 2:7])
  y <- matrix(as.numeric(attitude$rating), ncol = 1L)
  n <- nrow(X)
  p <- ncol(X)

  log_joint <- function(intercept, coefs, sd, X, y) {
    s <- function(val) nv_scalar(val, dtype = "f32")

    sd_pos <- nv_abs(sd) + s(1e-3)
    mu <- intercept + nv_matmul(X, coefs)
    resid <- (y - mu) / sd_pos

    ll <- -s(0.5) * nv_reduce_sum(resid * resid, dims = c(1, 2), drop = TRUE) -
      s(n) * nv_log(sd_pos)

    z_intercept <- intercept / s(10)
    z_coefs <- coefs / s(10)
    z_sd <- sd_pos / s(3)

    lp_intercept <- -s(0.5) * (z_intercept * z_intercept)
    lp_coefs <- -s(0.5) * nv_reduce_sum(
      z_coefs * z_coefs,
      dims = c(1, 2),
      drop = TRUE
    )
    lp_sd <- -nv_log(s(1) + (z_sd * z_sd))

    ll + lp_intercept + lp_coefs + lp_sd
  }

  dcoefs <- gradient(log_joint, wrt = "coefs")
  dcoefs_fn <- function(intercept, coefs, sd, X, y) {
    dcoefs(intercept, coefs, sd, X, y)[[1L]]
  }

  graph <- trace_fn(
    dcoefs_fn,
    list(
      intercept = nv_scalar(0.1, dtype = "f32"),
      coefs = nv_tensor(matrix(rnorm(p), nrow = p, ncol = 1L), dtype = "f32", shape = c(p, 1L)),
      sd = nv_scalar(1.0, dtype = "f32"),
      X = nv_tensor(X, dtype = "f32", shape = dim(X)),
      y = nv_tensor(y, dtype = "f32", shape = dim(y))
    )
  )

  f_quick <- graph_to_quickr_function(graph)

  intercept0 <- 0.2
  coefs0 <- matrix(runif(p, -0.1, 0.1), nrow = p, ncol = 1L)
  sd0 <- 1.3
  out_quick <- f_quick(intercept0, coefs0, sd0, X, y)
  out_pjrt <- eval_graph_pjrt(graph, intercept0, coefs0, sd0, X, y)
  expect_equal(out_quick, out_pjrt, tolerance = 1e-4)
})

test_that("greta-like: warpbreaks Poisson regression log joint (forward)", {
  testthat::skip_if_not_installed("quickr")
  testthat::skip_if_not_installed("pjrt")
  testthat::skip_if_not_installed("stablehlo")

  data(warpbreaks, package = "datasets")
  X <- model.matrix(breaks ~ wool + tension, data = warpbreaks)
  y <- matrix(as.numeric(warpbreaks$breaks), ncol = 1L)
  n <- nrow(X)
  p <- ncol(X)

  log_joint <- function(beta, X, y) {
    s <- function(val) nv_scalar(val, dtype = "f32")

    eta <- nv_matmul(X, beta) # (n, 1)
    # unnormalized Poisson loglik: sum(y * eta - exp(eta))
    ll <- nv_reduce_sum(y * eta - nv_exp(eta), dims = c(1, 2), drop = TRUE)
    lp <- -s(0.5) * nv_reduce_sum(beta * beta, dims = c(1, 2), drop = TRUE) / s(25)
    ll + lp
  }

  graph <- trace_fn(
    log_joint,
    list(
      beta = nv_tensor(matrix(rnorm(p), nrow = p, ncol = 1L), dtype = "f32", shape = c(p, 1L)),
      X = nv_tensor(X, dtype = "f32", shape = dim(X)),
      y = nv_tensor(y, dtype = "f32", shape = dim(y))
    )
  )

  f_quick <- graph_to_quickr_function(graph)
  beta0 <- matrix(runif(p, -0.2, 0.2), nrow = p, ncol = 1L)
  out_quick <- f_quick(beta0, X, y)
  out_pjrt <- eval_graph_pjrt(graph, beta0, X, y)
  expect_equal(as.numeric(out_quick), as.numeric(out_pjrt), tolerance = 1e-4)
})

test_that("greta-like: iris hierarchical regression via one-hot random intercepts", {
  testthat::skip_if_not_installed("quickr")
  testthat::skip_if_not_installed("pjrt")
  testthat::skip_if_not_installed("stablehlo")

  data(iris, package = "datasets")
  iris <- iris[seq_len(120), , drop = FALSE]
  n <- nrow(iris)

  x <- matrix(as.numeric(iris$Sepal.Width), ncol = 1L)
  y <- matrix(as.numeric(iris$Sepal.Length), ncol = 1L)

  sp <- as.integer(iris$Species)
  K <- max(sp)
  onehot <- matrix(0, nrow = n, ncol = K)
  for (i in seq_len(n)) {
    onehot[i, sp[[i]]] <- 1
  }

  log_joint <- function(intercept, coef, sd, eff, x, y, onehot) {
    s <- function(val) nv_scalar(val, dtype = "f32")

    sd_pos <- nv_abs(sd) + s(1e-3)
    species <- nv_matmul(onehot, eff) # (n,1)
    mu <- intercept + coef * x + species
    resid <- (y - mu) / sd_pos
    ll <- -s(0.5) * nv_reduce_sum(resid * resid, dims = c(1, 2), drop = TRUE) -
      s(n) * nv_log(sd_pos)

    z_intercept <- intercept / s(10)
    z_coef <- coef / s(10)

    lp_intercept <- -s(0.5) * (z_intercept * z_intercept)
    lp_coef <- -s(0.5) * (z_coef * z_coef)
    lp_eff <- -s(0.5) * nv_reduce_sum(eff * eff, dims = c(1, 2), drop = TRUE)
    ll + lp_intercept + lp_coef + lp_eff
  }

  graph <- trace_fn(
    log_joint,
    list(
      intercept = nv_scalar(0.1, dtype = "f32"),
      coef = nv_scalar(0.01, dtype = "f32"),
      sd = nv_scalar(1.0, dtype = "f32"),
      eff = nv_tensor(matrix(rnorm(K), nrow = K, ncol = 1L), dtype = "f32", shape = c(K, 1L)),
      x = nv_tensor(x, dtype = "f32", shape = dim(x)),
      y = nv_tensor(y, dtype = "f32", shape = dim(y)),
      onehot = nv_tensor(onehot, dtype = "f32", shape = dim(onehot))
    )
  )

  f_quick <- graph_to_quickr_function(graph)
  eff0 <- matrix(runif(K, -0.2, 0.2), nrow = K, ncol = 1L)
  out_quick <- f_quick(0.2, -0.1, 1.2, eff0, x, y, onehot)
  out_pjrt <- eval_graph_pjrt(graph, 0.2, -0.1, 1.2, eff0, x, y, onehot)
  expect_equal(as.numeric(out_quick), as.numeric(out_pjrt), tolerance = 1e-4)
})
