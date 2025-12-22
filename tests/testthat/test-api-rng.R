test_that("nv_rnorm", {
  # statistical validity checks are in inst/random
  f <- function() {
    nv_rnorm(nv_tensor(c(1, 2), dtype = "ui64"), dtype = "f32", shape = c(2, 3))
  }
  g <- jit(f)
  out <- g()
  expect_equal(c(as_array(out[[1]])), c(1L, 6L))
  expect_true(inherits(dtype.AnvilTensor(out[[2]]), FloatType))
  expect_equal(dtype.AnvilTensor(out[[2]])@value, 32L)
  expect_equal(shape(out[[2]]), c(2L, 3L))

  # test with uneven total number of RVs
  f <- function() {
    nv_rnorm(nv_tensor(c(1, 2), dtype = "ui64"), dtype = "f32", shape = c(3, 3))
  }
  g <- jit(f)
  out <- g()
  expect_equal(c(as_array(out[[1]])), c(1L, 8L))
  expect_equal(shape(out[[2]]), c(3L, 3L))

  # test mu/sigma parameters with small sample
  f <- function() {
    nv_rnorm(
      nv_tensor(c(3, 83), dtype = "ui64"),
      dtype = "f64",
      shape = c(2L, 3L),
      mu = 10,
      sigma = 9
    )
  }
  g <- jit(f)
  out <- g()
  expect_equal(dtype(out[[1]]), as_dtype("ui64"))
  expect_equal(shape(out[[1]]), 2L)
  expect_equal(shape(out[[2]]), c(2L, 3L))
  expect_equal(dtype(out[[2]]), as_dtype("f64"))
})

test_that("nv_runif", {
  # statistical validity checks are in inst/random
  f <- function() {
    nv_runif(
      nv_tensor(c(1, 2), dtype = "ui64"),
      dtype = "f32",
      shape = c(3, 4),
      lower = -1,
      upper = 1
    )
  }
  g <- jit(f)
  out <- g()

  expect_equal(dtype(out[[1]]), as_dtype("ui64"))
  expect_equal(shape(out[[1]]), 2L)
  expect_equal(shape(out[[2]]), c(3L, 4L))
  expect_equal(dtype(out[[2]]), as_dtype("f32"))
})

test_that("nv_rbinom", {
  # statistical validity checks are in inst/random
  f <- function() {
    nv_rbinom(nv_tensor(c(1, 2), dtype = "ui64"), dtype = "i32", shape = c(2, 5))
  }
  g <- jit(f)
  out <- g()

  expect_equal(shape(out[[1]]), 2L)
  expect_equal(shape(out[[2]]), c(2L, 5L))
  expect_equal(dtype(out[[2]]), as_dtype("i32"))

  # All values should be 0 or 1
  values <- c(as_array(out[[2]]))
  expect_true(all(values %in% c(0L, 1L)))

  # Test with different dtype
  f2 <- function() {
    nv_rbinom(nv_tensor(c(1, 2), dtype = "ui64"), dtype = "f32", shape = 10L)
  }
  g2 <- jit(f2)
  out2 <- g2()
  expect_equal(dtype(out2[[2]]), as_dtype("f32"))
  expect_equal(shape(out2[[2]]), 10L)

  # Test with non-multiple-of-8 shape (tests slicing)
  f3 <- function() {
    nv_rbinom(nv_tensor(c(1, 2), dtype = "ui64"), dtype = "i32", shape = c(3, 3))
  }
  g3 <- jit(f3)
  out3 <- g3()
  expect_equal(shape(out3[[2]]), c(3L, 3L))
})

test_that("nv_rdiscrete", {
  # statistical validity checks are in inst/random

  # Test with equal probabilities (replace = TRUE)
  f1 <- function() {
    nv_rdiscrete(n = 6L, shape = 10L, initial_state = nv_tensor(c(1, 2), dtype = "ui64"))
  }
  g1 <- jit(f1)
  out1 <- g1()

  expect_equal(shape(out1[[1]]), 2L)
  expect_equal(shape(out1[[2]]), 10L)
  expect_equal(dtype(out1[[2]]), as_dtype("i32"))

  # All values should be in 1:6
  values1 <- c(as_array(out1[[2]]))
  expect_true(all(values1 >= 1L & values1 <= 6L))

  # Test with custom probabilities (replace = TRUE)
  f2 <- function(p) {
    nv_rdiscrete(n = 2L, shape = 20L, prob = p, initial_state = nv_tensor(c(1, 2), dtype = "ui64"))
  }
  g2 <- jit(f2)
  prob <- nv_tensor(c(0.8, 0.2), dtype = "f64")
  out2 <- g2(prob)

  values2 <- c(as_array(out2[[2]]))
  expect_true(all(values2 %in% c(1L, 2L)))

  # Test 2D output shape
  f3 <- function() {
    nv_rdiscrete(n = 4L, shape = c(2L, 3L), initial_state = nv_tensor(c(1, 2), dtype = "ui64"))
  }
  g3 <- jit(f3)
  out3 <- g3()
  expect_equal(shape(out3[[2]]), c(2L, 3L))

  # Test replace = FALSE with equal probabilities
  f4 <- function() {
    nv_rdiscrete(n = 10L, shape = 5L, replace = FALSE, initial_state = nv_tensor(c(1, 2), dtype = "ui64"))
  }
  g4 <- jit(f4)
  out4 <- g4()

  expect_equal(shape(out4[[2]]), 5L)
  expect_equal(dtype(out4[[2]]), as_dtype("i32"))
  values4 <- c(as_array(out4[[2]]))
  expect_true(all(values4 >= 1L & values4 <= 10L))
  # All values should be unique (no replacement)
  expect_equal(length(unique(values4)), 5L)

  # Test replace = FALSE with custom probabilities
  f5 <- function(p) {
    nv_rdiscrete(n = 5L, shape = 3L, replace = FALSE, prob = p, initial_state = nv_tensor(c(1, 2), dtype = "ui64"))
  }
  g5 <- jit(f5)
  prob5 <- nv_tensor(c(0.1, 0.1, 0.3, 0.3, 0.2), dtype = "f64")
  out5 <- g5(prob5)

  values5 <- c(as_array(out5[[2]]))
  expect_true(all(values5 >= 1L & values5 <= 5L))
  expect_equal(length(unique(values5)), 3L)

  # Test sampling all elements (should be a permutation)
  f6 <- function() {
    nv_rdiscrete(n = 4L, shape = 4L, replace = FALSE, initial_state = nv_tensor(c(1, 2), dtype = "ui64"))
  }
  g6 <- jit(f6)
  out6 <- g6()

  values6 <- c(as_array(out6[[2]]))
  expect_equal(sort(values6), 1:4)
})
