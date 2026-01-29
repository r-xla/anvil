# Benchmark: Async execution vs nv_while for logistic regression
#
# This benchmark compares three approaches:
# 1. nv_while: Single compiled function with XLA while loop
# 2. R loop + async=TRUE: R loop calling async jit function
# 3. R loop + async=FALSE: R loop calling sync jit function

devtools::load_all(".")

# --- Data Setup ---
set.seed(42)

# Create synthetic data for logistic regression
n <- 1000L
p <- 10L

X <- matrix(rnorm(n * p), nrow = n, ncol = p)
beta_true <- rnorm(p)
alpha_true <- 0.5
probs <- 1 / (1 + exp(-(X %*% beta_true + alpha_true)))
y <- rbinom(n, 1, probs)

X_tensor <- nv_tensor(X, dtype = "f32")
y_tensor <- nv_tensor(y, dtype = "f32", shape = c(n, 1L))

# --- Model Functions ---
predict_proba <- function(X, beta, alpha) {
  logits <- X %*% beta + alpha
  nv_logistic(logits)
}

binary_cross_entropy <- function(y_true, y_pred) {
  eps <- 1e-7
  y_pred_clipped <- nv_clamp(eps, y_pred, 1 - eps)
  loss <- -(y_true * log(y_pred_clipped) + (1 - y_true) * log(1 - y_pred_clipped))
  mean(loss)
}

model_loss <- function(X, y, beta, alpha) {
  y_pred <- predict_proba(X, beta, alpha)
  binary_cross_entropy(y, y_pred)
}

model_loss_grad <- gradient(model_loss, wrt = c("beta", "alpha"))

# --- Approach 1: nv_while (single compiled function) ---
fit_logreg_while <- jit(function(X, y, beta, alpha, n_epochs, lr) {
  output <- nv_while(
    list(beta = beta, alpha = alpha, epoch = nv_scalar(0L)),
    \(beta, alpha, epoch) epoch < n_epochs,
    \(beta, alpha, epoch) {
      grads <- model_loss_grad(X, y, beta, alpha)
      list(
        beta = beta - lr * grads$beta,
        alpha = alpha - lr * grads$alpha,
        epoch = epoch + 1L
      )
    }
  )
  list(beta = output$beta, alpha = output$alpha)
})

# --- Approach 2 & 3: R loop with jit step function ---
make_fit_logreg_rloop <- function(async) {
  step_fn <- jit(function(X, y, beta, alpha, lr) {
    grads <- model_loss_grad(X, y, beta, alpha)
    list(
      beta = beta - lr * grads$beta,
      alpha = alpha - lr * grads$alpha
    )
  }, async = async, donate = c("beta", "alpha"))

  function(X, y, beta, alpha, n_epochs, lr) {
    for (i in seq_len(n_epochs)) {
      result <- step_fn(X, y, beta, alpha, lr)
      beta <- result$beta
      alpha <- result$alpha
    }
    list(beta = beta, alpha = alpha)
  }
}

fit_logreg_rloop_async <- make_fit_logreg_rloop(async = TRUE)
fit_logreg_rloop_sync <- make_fit_logreg_rloop(async = FALSE)

# --- Benchmark ---
run_benchmark <- function(n_epochs, n_reps = 5) {
  # Helper to create fresh initial parameters (needed because donate invalidates buffers)
  make_init <- function() {
    list(
      beta = nv_tensor(rnorm(p), dtype = "f32", shape = c(p, 1L)),
      alpha = nv_scalar(0, dtype = "f32"),
      lr = nv_scalar(0.1),
      n_epochs_tensor = nv_scalar(as.integer(n_epochs))
    )
  }

  # Warmup runs
  cat("Warming up...\n")
  init <- make_init()
  invisible(fit_logreg_while(X_tensor, y_tensor, init$beta, init$alpha, init$n_epochs_tensor, init$lr))
  init <- make_init()
  invisible(fit_logreg_rloop_async(X_tensor, y_tensor, init$beta, init$alpha, n_epochs, init$lr))
  init <- make_init()
  invisible(fit_logreg_rloop_sync(X_tensor, y_tensor, init$beta, init$alpha, n_epochs, init$lr))

  cat(sprintf("\nBenchmarking with %d epochs, %d repetitions...\n\n", n_epochs, n_reps))

  # Benchmark nv_while
  times_while <- numeric(n_reps)
  for (i in seq_len(n_reps)) {
    init <- make_init()
    t0 <- Sys.time()
    result <- fit_logreg_while(X_tensor, y_tensor, init$beta, init$alpha, init$n_epochs_tensor, init$lr)
    # Force evaluation
    invisible(as_array(result$beta))
    times_while[i] <- as.numeric(Sys.time() - t0, units = "secs")
  }

  # Benchmark R loop + async
  times_async <- numeric(n_reps)
  for (i in seq_len(n_reps)) {
    init <- make_init()
    t0 <- Sys.time()
    result <- fit_logreg_rloop_async(X_tensor, y_tensor, init$beta, init$alpha, n_epochs, init$lr)
    # Force evaluation
    invisible(as_array(result$beta))
    times_async[i] <- as.numeric(Sys.time() - t0, units = "secs")
  }

  # Benchmark R loop + sync
  times_sync <- numeric(n_reps)
  for (i in seq_len(n_reps)) {
    init <- make_init()
    t0 <- Sys.time()
    result <- fit_logreg_rloop_sync(X_tensor, y_tensor, init$beta, init$alpha, n_epochs, init$lr)
    # Force evaluation
    invisible(as_array(result$beta))
    times_sync[i] <- as.numeric(Sys.time() - t0, units = "secs")
  }

  # Results
  cat("Results (seconds):\n")
  cat(sprintf("  nv_while:         %.4f (sd: %.4f)\n", mean(times_while), sd(times_while)))
  cat(sprintf("  R loop + async:   %.4f (sd: %.4f)\n", mean(times_async), sd(times_async)))
  cat(sprintf("  R loop + sync:    %.4f (sd: %.4f)\n", mean(times_sync), sd(times_sync)))

  cat("\nSpeedup ratios:\n")
  cat(sprintf("  nv_while vs R loop + sync:  %.2fx\n", mean(times_sync) / mean(times_while)))
  cat(sprintf("  nv_while vs R loop + async:  %.2fx\n", mean(times_async) / mean(times_while)))
  cat(sprintf("  async vs sync:              %.2fx\n", mean(times_sync) / mean(times_async)))

  invisible(list(
    nv_while = times_while,
    async = times_async,
    sync = times_sync
  ))
}

# Run benchmarks with different epoch counts
cat("=== Small iteration count (100 epochs) ===\n")
run_benchmark(n_epochs = 100)

cat("\n=== Medium iteration count (500 epochs) ===\n")
run_benchmark(n_epochs = 500)

cat("\n=== Large iteration count (1000 epochs) ===\n")
run_benchmark(n_epochs = 1000)
