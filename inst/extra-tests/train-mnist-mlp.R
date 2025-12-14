run_mnist_mlp_training <- function(
  train_n = 60000L,
  test_n = 10000L,
  hidden = 64L,
  epochs = 5L,
  batch_size = 256L,
  lr = 0.1,
  seed = 123
) {
  if (!requireNamespace("pjrt", quietly = TRUE)) {
    cli::cli_abort("{.pkg pjrt} must be installed to train with {.pkg anvil}")
  }

  set.seed(seed)

  if (!requireNamespace("anvil", quietly = TRUE)) {
    cli::cli_abort("{.pkg anvil} must be installed to run MNIST training")
  }

  mnist <- anvil:::.load_mnist()
  if (is.null(mnist)) {
    cli::cli_abort("MNIST not available; provide mnist.rds or install mnist")
  }

  data <- anvil:::.prepare_mnist_mlp_data(mnist, train_n = train_n, test_n = test_n)
  x_train <- data$train$x
  y_train <- data$train$y
  y_train_oh <- data$train$y_one_hot
  x_test <- data$test$x
  y_test <- data$test$y
  y_test_oh <- data$test$y_one_hot
  train_n <- dim(x_train)[[1L]]
  test_n <- dim(x_test)[[1L]]

  relu <- function(x) {
    nv_convert(x > nv_scalar(0, dtype = "f32"), "f32") * x
  }

  loss <- function(X, y, W1, b1, W2, b2) {
    h <- nv_matmul(X, W1)
    h <- h + nv_broadcast_to(b1, shape = shape(h))
    h <- relu(h)

    logits <- nv_matmul(h, W2)
    logits <- logits + nv_broadcast_to(b2, shape = shape(logits))

    z_max <- nv_reduce_max(logits, dims = 2, drop = FALSE)
    logits_shift <- logits - nv_broadcast_to(z_max, shape = shape(logits))
    logsumexp <- z_max + nv_log(nv_reduce_sum(nv_exp(logits_shift), dims = 2, drop = FALSE))
    log_probs <- logits - nv_broadcast_to(logsumexp, shape = shape(logits))

    nll <- -nv_reduce_sum(y * log_probs, dims = 2, drop = FALSE)
    nv_reduce_mean(nll, dims = c(1, 2))
  }

  g <- gradient(loss, wrt = c("W1", "b1", "W2", "b2"))
  lr_nv <- nv_scalar(lr, dtype = "f32")

  step <- jit(
    function(X, y, W1, b1, W2, b2) {
      grads <- g(X, y, W1, b1, W2, b2)
      list(
        loss(X, y, W1, b1, W2, b2),
        W1 - lr_nv * grads[[1L]],
        b1 - lr_nv * grads[[2L]],
        W2 - lr_nv * grads[[3L]],
        b2 - lr_nv * grads[[4L]]
      )
    },
    donate = c("W1", "b1", "W2", "b2")
  )

  predict_logits <- jit(function(X, W1, b1, W2, b2) {
    h <- relu(nv_matmul(X, W1) + nv_broadcast_to(b1, shape = c(shape(X)[[1L]], hidden)))
    nv_matmul(h, W2) + nv_broadcast_to(b2, shape = c(shape(X)[[1L]], 10L))
  })

  init <- function() {
    W1 <- nv_tensor(matrix(rnorm(784L * hidden, sd = 0.05), nrow = 784L, ncol = hidden), dtype = "f32", shape = c(784L, hidden))
    b1 <- nv_tensor(matrix(0, nrow = 1L, ncol = hidden), dtype = "f32", shape = c(1L, hidden))
    W2 <- nv_tensor(matrix(rnorm(hidden * 10L, sd = 0.05), nrow = hidden, ncol = 10L), dtype = "f32", shape = c(hidden, 10L))
    b2 <- nv_tensor(matrix(0, nrow = 1L, ncol = 10L), dtype = "f32", shape = c(1L, 10L))
    list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)
  }

  params <- init()

  batch_idxs <- function(n, bs) {
    n_full <- (n %/% bs) * bs
    idx <- sample.int(n_full)
    split(idx, rep(seq_len(n_full / bs), each = bs))
  }

  cat(sprintf("Training: n=%d, epochs=%d, batch=%d, hidden=%d, lr=%g\n", train_n, epochs, batch_size, hidden, lr))
  for (epoch in seq_len(epochs)) {
    losses <- numeric()
    idxs <- batch_idxs(train_n, batch_size)
    for (idx in idxs) {
      Xb <- nv_tensor(x_train[idx, , drop = FALSE], dtype = "f32", shape = c(length(idx), 784L))
      yb <- nv_tensor(y_train_oh[idx, , drop = FALSE], dtype = "f32", shape = c(length(idx), 10L))
      out <- step(Xb, yb, params$W1, params$b1, params$W2, params$b2)
      losses <- c(losses, as.numeric(as_array(out[[1L]])))
      params$W1 <- out[[2L]]
      params$b1 <- out[[3L]]
      params$W2 <- out[[4L]]
      params$b2 <- out[[5L]]
    }

    # eval on test (host-side argmax)
    logits <- predict_logits(nv_tensor(x_test, dtype = "f32", shape = c(test_n, 784L)), params$W1, params$b1, params$W2, params$b2)
    pred <- max.col(as_array(logits), ties.method = "first") - 1L
    acc <- mean(pred == as.integer(y_test))
    cat(sprintf("epoch=%d loss=%.4f acc=%.3f\n", epoch, mean(losses), acc))
  }

  invisible(params)
}
