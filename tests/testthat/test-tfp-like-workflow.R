test_that("tfp-like: logistic model HMC via quickr-compiled log_prob + grad", {
  testthat::skip_if_not_installed("quickr")
  testthat::skip_if_not_installed("pjrt")
  testthat::skip_if_not_installed("stablehlo")

  n <- 25L
  temperature_raw <- seq(30, 80, length.out = n)
  temperature <- as.numeric(scale(temperature_raw))

  # Generate synthetic "Challenger-like" data: lower temperature => higher risk.
  alpha_true <- -1.0
  beta_true <- 2.5
  p_true <- 1.0 / (1.0 + exp(beta_true * temperature + alpha_true))
  set.seed(1)
  D <- rbinom(n, size = 1L, prob = p_true)

  temperature_nv <- nv_tensor(temperature, dtype = "f32", shape = c(n))
  D_nv <- nv_tensor(D, dtype = "f32", shape = c(n))

  s <- function(val) nv_scalar(val, dtype = "f32")

  normal_log_prob <- function(x, loc = 0, scale = 10) {
    z <- (x - s(loc)) / s(scale)
    -s(0.5) * (z * z) - nv_log(s(scale)) - s(0.5) * nv_log(s(2 * pi))
  }

  log_joint <- function(alpha, beta) {
    eps <- 1e-6
    one <- s(1.0)

    logits <- beta * temperature_nv + alpha
    p <- one / (one + nv_exp(logits))
    p <- p * s(1.0 - 2.0 * eps) + s(eps)

    ll <- nv_reduce_sum(
      D_nv * nv_log(p) + (one - D_nv) * nv_log(one - p),
      dims = 1,
      drop = TRUE
    )

    ll + normal_log_prob(alpha) + normal_log_prob(beta)
  }

  logp_and_grad <- function(alpha, beta) {
    g <- gradient(log_joint, wrt = c("alpha", "beta"))
    list(
      log_prob = log_joint(alpha, beta),
      grad = g(alpha = alpha, beta = beta)
    )
  }

  graph <- trace_fn(
    logp_and_grad,
    list(
      alpha = nv_scalar(0.0, dtype = "f32"),
      beta = nv_scalar(0.0, dtype = "f32")
    )
  )

  f_quick <- graph_to_quickr_function(graph)

  out_quick0 <- f_quick(0.1, 0.2)
  out_pjrt0 <- eval_graph_pjrt(
    graph,
    nv_scalar(0.1, dtype = "f32"),
    nv_scalar(0.2, dtype = "f32")
  )
  expect_equal(out_quick0, out_pjrt0, tolerance = 1e-4)

  eval_lp_grad <- function(q) {
    out <- f_quick(q[[1L]], q[[2L]])
    list(
      lp = as.numeric(out$log_prob),
      grad = c(
        alpha = as.numeric(out$grad$alpha),
        beta = as.numeric(out$grad$beta)
      )
    )
  }

  hmc_run <- function(init, step_size, n_leapfrog, burnin, n_samples) {
    stopifnot(length(init) == 2L)

    q <- as.numeric(init)
    cur <- eval_lp_grad(q)

    accepted <- logical(burnin + n_samples)
    samples <- matrix(NA_real_, nrow = n_samples, ncol = 2L)
    colnames(samples) <- c("alpha", "beta")

    for (iter in seq_len(burnin + n_samples)) {
      q0 <- q
      lp0 <- cur$lp
      grad0 <- unname(cur$grad)

      p0 <- rnorm(2L)
      p <- p0 + 0.5 * step_size * grad0

      q_prop <- q0
      lp_prop <- lp0
      grad_prop <- grad0

      for (lf in seq_len(n_leapfrog)) {
        q_prop <- q_prop + step_size * p
        ev <- eval_lp_grad(q_prop)
        lp_prop <- ev$lp
        grad_prop <- unname(ev$grad)
        if (lf != n_leapfrog) {
          p <- p + step_size * grad_prop
        } else {
          p <- p + 0.5 * step_size * grad_prop
        }
      }

      p <- -p

      H0 <- -lp0 + 0.5 * sum(p0 * p0)
      H1 <- -lp_prop + 0.5 * sum(p * p)
      accept <- is.finite(lp_prop) && log(stats::runif(1L)) < (H0 - H1)

      accepted[[iter]] <- accept
      if (accept) {
        q <- q_prop
        cur <- list(lp = lp_prop, grad = grad_prop)
      }

      if (iter > burnin) {
        samples[iter - burnin, ] <- q
      }
    }

    list(
      samples = samples,
      accept_rate = mean(accepted[(burnin + 1L):(burnin + n_samples)])
    )
  }

  set.seed(2)
  res <- hmc_run(
    init = c(0, 0),
    step_size = 0.12,
    n_leapfrog = 6L,
    burnin = 50L,
    n_samples = 100L
  )

  expect_gt(res$accept_rate, 0.05)
  expect_gt(mean(res$samples[, "beta"]), 0)
  expect_lt(mean(res$samples[, "alpha"]), 0)
})
