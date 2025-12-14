test_that("internal MNIST helpers: load and preprocess", {
  mnist <- list(
    train = list(
      x = array(sample(0:255, 5L * 28L * 28L, replace = TRUE), dim = c(5L, 28L, 28L)),
      y = as.integer(sample(0:9, 5L, replace = TRUE))
    ),
    test = list(
      x = array(sample(0:255, 4L * 28L * 28L, replace = TRUE), dim = c(4L, 28L, 28L)),
      y = as.integer(sample(0:9, 4L, replace = TRUE))
    )
  )

  tmp <- tempfile(fileext = ".rds")
  saveRDS(mnist, tmp)

  loaded <- anvil:::.load_mnist(rds_path = tmp, pkgs = character())
  expect_true(is.list(loaded))
  expect_true(is.list(loaded$train))
  expect_true(is.list(loaded$test))

  data <- anvil:::.prepare_mnist_mlp_data(loaded, train_n = 3L, test_n = 2L)
  expect_equal(dim(data$train$x), c(3L, 784L))
  expect_equal(dim(data$test$x), c(2L, 784L))
  expect_equal(dim(data$train$y_one_hot), c(3L, 10L))
  expect_equal(dim(data$test$y_one_hot), c(2L, 10L))
  expect_true(all(data$train$x >= 0))
  expect_true(all(data$train$x <= 1))

  y <- as.integer(0:3)
  y_one_hot <- anvil:::.mnist_one_hot(y, nclass = 10L)
  expect_equal(dim(y_one_hot), c(4L, 10L))
  expect_equal(rowSums(y_one_hot), rep(1, 4L))
  expect_equal(max.col(y_one_hot, ties.method = "first") - 1L, y)
})

test_that("internal MNIST helpers: invalid rds returns NULL", {
  tmp <- tempfile(fileext = ".rds")
  saveRDS(list(not_mnist = TRUE), tmp)
  expect_null(anvil:::.load_mnist(rds_path = tmp, pkgs = character()))
})

test_that("internal MNIST helpers: one_hot does not shift 1-based labels", {
  y <- as.integer(1:4)
  y_one_hot <- anvil:::.mnist_one_hot(y, nclass = 10L)
  expect_equal(dim(y_one_hot), c(4L, 10L))
  expect_equal(rowSums(y_one_hot), rep(1, 4L))
  expect_equal(max.col(y_one_hot, ties.method = "first"), y)
})

test_that("internal MNIST helpers: preprocess respects train_n/test_n", {
  mnist <- list(
    train = list(
      x = array(255, dim = c(5L, 28L, 28L)),
      y = as.integer(0:4)
    ),
    test = list(
      x = array(0, dim = c(4L, 28L, 28L)),
      y = as.integer(0:3)
    )
  )

  data <- anvil:::.prepare_mnist_mlp_data(mnist, train_n = 100L, test_n = 2L)
  expect_equal(dim(data$train$x), c(5L, 784L))
  expect_equal(dim(data$test$x), c(2L, 784L))
  expect_equal(as.numeric(range(data$train$x)), c(1, 1))
  expect_equal(as.numeric(range(data$test$x)), c(0, 0))
})
