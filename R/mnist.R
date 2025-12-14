.is_mnist_dataset <- function(mnist) {
  is.list(mnist) &&
    !is.null(mnist$train) &&
    !is.null(mnist$test) &&
    !is.null(mnist$train$x) &&
    !is.null(mnist$train$y) &&
    !is.null(mnist$test$x) &&
    !is.null(mnist$test$y)
}

.load_mnist <- function(
  rds_path = Sys.getenv("ANVIL_MNIST_RDS", "mnist.rds")
) {
  if (nzchar(rds_path) && file.exists(rds_path)) {
    mnist <- readRDS(rds_path)
    if (.is_mnist_dataset(mnist)) {
      return(mnist)
    }
  }

  if (!requireNamespace("mnist", quietly = TRUE)) {
    return(NULL)
  }

  mnist <- mnist::mnist
  out <- list(
    train = list(x = mnist$train_images, y = mnist$train_labels),
    test = list(x = mnist$test_images, y = mnist$test_labels)
  )
  if (.is_mnist_dataset(out)) {
    return(out)
  }

  NULL
}

.mnist_one_hot <- function(y, nclass = 10L) {
  y <- as.integer(y)
  if (min(y) == 0L) {
    y <- y + 1L
  }

  out <- matrix(0, nrow = length(y), ncol = nclass)
  out[cbind(seq_along(y), y)] <- 1
  out
}

.prepare_mnist_mlp_data <- function(mnist, train_n, test_n, nclass = 10L) {
  slice_first <- function(x, n) {
    d <- dim(x)
    if (is.null(d)) {
      cli::cli_abort("MNIST images must have a dim attribute")
    }
    if (length(d) == 2L) {
      return(x[seq_len(n), , drop = FALSE])
    }
    if (length(d) == 3L) {
      return(x[seq_len(n), , , drop = FALSE])
    }
    if (length(d) == 4L) {
      return(x[seq_len(n), , , , drop = FALSE])
    }
    cli::cli_abort("Unsupported MNIST image rank: {length(d)}")
  }

  x_train <- mnist$train$x
  y_train <- mnist$train$y
  x_test <- mnist$test$x
  y_test <- mnist$test$y

  train_n <- min(as.integer(train_n), dim(x_train)[[1L]])
  test_n <- min(as.integer(test_n), dim(x_test)[[1L]])

  x_train <- slice_first(x_train, train_n)
  y_train <- y_train[seq_len(train_n)]
  x_test <- slice_first(x_test, test_n)
  y_test <- y_test[seq_len(test_n)]

  x_train <- array(as.numeric(x_train) / 255, dim = c(train_n, prod(dim(x_train)) %/% train_n))
  x_test <- array(as.numeric(x_test) / 255, dim = c(test_n, prod(dim(x_test)) %/% test_n))

  list(
    train = list(
      x = x_train,
      y = y_train,
      y_one_hot = .mnist_one_hot(y_train, nclass = nclass)
    ),
    test = list(
      x = x_test,
      y = y_test,
      y_one_hot = .mnist_one_hot(y_test, nclass = nclass)
    )
  )
}
