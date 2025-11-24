test_that("p_sine", {
  x <- nv_tensor(c(0, pi / 2, pi, 3 / 2 * pi), dtype = "f64")
  out <- as_array(jit(nvl_sine)(x))
  expect_equal(c(out), c(0, 1, 0, -1), tolerance = 1e-15)
})

test_that("p_cosine", {
  x <- nv_tensor(c(0, pi / 2, pi, 3 / 2 * pi), dtype = "f64")
  out <- as_array(jit(nvl_cosine)(x))
  expect_equal(c(out), c(1, 0, -1, 0), tolerance = 1e-15)
})

test_that("p_rng_bit_generator", {
  f <- function() {
    nv_rng_bit_generator(nv_tensor(c(1, 2), dtype = "ui64"), "THREE_FRY", "i64", c(2, 2))
  }
  g <- jit(f)
  out <- g()
  expect_equal(c(as_array(out[[1]])), c(1L, 6L))
  expect_equal(as_array(out[[2]]), array(c(43444564L, 1672743891L, -315321645L, 2109414752L), c(2, 2)))
})

test_that("p_slice", {
  f <- function() {
    nv_slice(
      nv_tensor(c(1:6), dtype = "ui64", shape = c(2, 3)),
      start_indices = c(0, 0),
      limit_indices = c(2, 2),
      strides = c(1, 1)
    )
  }
  g <- jit(f)
  out <- g()
  expect_equal(as_array(out), matrix(c(1:4), nrow = 2))
})

test_that("p_runif", {
  f <- function() {
    nv_runif(nv_tensor(c(1, 2), dtype = "ui64"), dtype = "f32", shape_out = c(2, 3), lower = -1, upper = 1)
  }
  g <- jit(f)
  out <- g()
  expect_equal(c(as_array(out[[1]])), c(1L, 5L))
  expect_equal(
    as_array(out[[2]]),
    array(c(-0.9798, 0.0015, 0.8532, 0.7442, -0.2211, 0.9746), c(2, 3)),
    tolerance = 1e-4
  )
})

test_that("p_rnorm", {
  f <- function() {
    nv_rnorm(nv_tensor(c(1, 2), dtype = "ui64"), dtype = "f32", shape_out = c(2, 3))
  }
  g <- jit(f)
  out <- g()
  expect_equal(c(as_array(out[[1]])), c(1L, 6L))
  expect_equal(
    as_array(out[[2]]),
    array(c(1.9399, -2.3290, -0.0936, 1.1724, 0.0217, -0.3899), c(2, 3)),
    tolerance = 1e-4
  )
})

test_that("p_concatenate", {
  f <- function() {
    nv_concatenate(
      nv_tensor(c(1:6), dtype = "ui64", shape = c(2, 3)),
      nv_tensor(c(7:10), dtype = "ui64", shape = c(2, 2)),
      dimension = 2L
    )
  }
  g <- jit(f)
  out <- g()
  expect_equal(dim(as_array(out)), c(2, 5))
})

test_that("p_shift_left", {
  x <- nv_tensor(as.integer(c(1L, 2L, 3L, 8L)), dtype = "i32")
  y <- nv_tensor(as.integer(c(0L, 1L, 2L, 3L)), dtype = "i32")
  out <- as.integer(as_array(jit(nvl_shift_left)(x, y)))
  expect_equal(out, as.integer(c(1L, 4L, 12L, 64L)))
})

test_that("p_shift_right_logical", {
  x <- nv_tensor(as.integer(c(16L, 8L, 7L, 1L)), dtype = "i32")
  y <- nv_tensor(as.integer(c(0L, 1L, 2L, 0L)), dtype = "i32")
  out <- as.integer(as_array(jit(nvl_shift_right_logical)(x, y)))
  expect_equal(out, as.integer(c(16L, 4L, 1L, 1L)))
})

test_that("p_shift_right_arithmetic", {
  x <- nv_tensor(as.integer(c(-8L, -1L, 8L, -17L)), dtype = "i32")
  y <- nv_tensor(as.integer(c(1L, 3L, 2L, 4L)), dtype = "i32")
  out <- as.integer(as_array(jit(nvl_shift_right_arithmetic)(x, y)))
  expect_equal(out, as.integer(c(-4L, -1L, 2L, -2L)))
})

# Reduction ops (simplified hardcoded examples, no torch comparisons)

test_that("p_reduce_sum", {
  x <- array(1:6, c(2, 3))
  f <- jit(function(a) nvl_reduce_sum(a, dims = 2L, drop = TRUE))
  out <- as_array(f(nv_tensor(x, dtype = "f32")))
  expect_equal(out, array(c(9, 12)))
})

test_that("p_reduce_prod", {
  x <- array(1:6, c(2, 3))
  f <- jit(function(a) nvl_reduce_prod(a, dims = 1L, drop = FALSE))
  out <- as_array(f(nv_tensor(x, dtype = "f32")))
  expect_equal(out, array(c(2, 12, 30), c(1, 3)))
})

test_that("p_reduce_max", {
  x <- array(c(-1, 4, 0, 2), c(2, 2))
  f <- jit(function(a) nvl_reduce_max(a, dims = 2L, drop = TRUE))
  out <- as_array(f(nv_tensor(x, dtype = "f32")))
  expect_equal(out, array(c(0, 4)))
})

test_that("p_reduce_min", {
  x <- array(c(-1, 4, 0, 2), c(2, 2))
  f <- jit(function(a) nvl_reduce_min(a, dims = 2L, drop = TRUE))
  out <- as_array(f(nv_tensor(x, dtype = "f32")))
  expect_equal(out, array(c(-1, 2)))
})

test_that("p_reduce_any", {
  x <- array(c(TRUE, FALSE, TRUE, FALSE, FALSE, FALSE), c(2, 3))
  f <- jit(function(a) nvl_reduce_any(a, dims = 2L, drop = TRUE))
  out <- as_array(f(nv_tensor(x, dtype = "pred")))
  expect_equal(out, array(c(TRUE, FALSE)))
})

test_that("p_reduce_all", {
  x <- array(c(TRUE, FALSE, TRUE, FALSE, FALSE, FALSE), c(2, 3))
  f <- jit(function(a) nvl_reduce_all(a, dims = 1L, drop = FALSE))
  out <- as_array(f(nv_tensor(x, dtype = "pred")))
  expect_equal(out, array(rep(FALSE, 3), c(1, 3)))
})

test_that("p_broadcast_in_dim", {
  x <- 1L
  f <- jit(nvl_broadcast_in_dim, static = c("shape_out", "broadcast_dimensions"))
  expect_equal(
    f(nv_scalar(1L), shape_out <- c(1, 2), integer()),
    nv_tensor(1L, shape = c(1, 2)),
    tolerance = 1e-5
  )
})

test_that("p_reshape", {
  f <- jit(nvl_reshape, static = "shape")
  x <- array(1:6, c(3, 2))
  expect_equal(
    f(nv_tensor(x), shape = 6),
    nv_tensor(as.integer(c(1, 4, 2, 5, 3, 6)), "i32")
  )
})

test_that("p_transpose", {
  x <- array(1:4, c(2, 2))
  f <- jit(\(x) nvl_transpose(x, c(2, 1)))
  expect_equal(
    t(x),
    as_array(f(nv_tensor(x)))
  )
})

# we don't want to include torch in Suggests just for the tests, as it's a relatively
# heavy dependency
# We have a CI job that installs torch, so it's at least tested once

if (nzchar(system.file(package = "torch"))) {
  source(system.file("extra-tests", "test-primitives-jit-torch.R", package = "anvil"))
}
