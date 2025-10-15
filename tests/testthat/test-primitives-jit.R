# binary ops

test_that("nvl_add", {
  expect_jit_binary(nvl_add, `+`, 1.2, -0.7)
})
test_that("nvl_sub", {
  expect_jit_binary(nvl_sub, `-`, 1.2, -0.7)
})

test_that("nvl_mul", {
  expect_jit_binary(nvl_mul, `*`, 1.2, -0.7)
})

test_that("nvl_neg", {
  expect_jit_unary(nvl_neg, \(x) -x, 1.7)
})

test_that("nvl_div", {
  expect_jit_binary(nvl_div, `/`, 1.2, 0.2)
})

test_that("nvl_pow", {
  expect_jit_binary(nvl_pow, `^`, 1, 0.3)
})


test_that("nvl_transpose", {
  # just use nv_transpose for the default here
  x <- array(1:4, c(2, 2))
  expect_jit_unary(nv_transpose, t, x)
  x2 <- array(1:8, c(2, 2, 2))
  expect_jit_unary(
    \(x) nv_transpose(x, c(1, 3, 2)),
    \(x) aperm(x, c(1, 3, 2)),
    x2
  )
})

test_that("matmul", {
  # simple
  A <- array(1:9, dim = c(3, 3))
  B <- array(1:9, dim = c(3, 3))
  expect_equal(
    as_array(jit(nv_matmul)(nv_tensor(A), nv_tensor(B))),
    A %*% B
  )

  # broadcasting
  x <- nv_tensor(A, shape = c(1, 1, 3, 3))
  y <- nv_tensor(B, shape = c(1, 3, 3))
  out <- jit(nv_matmul)(x, y)
  expect_equal(
    as_array(out),
    array(A %*% B, dim = c(1, 1, 3, 3))
  )
})

test_that("nv_reduce_sum", {
  x <- nv_tensor(1:3, dtype = "f32")
  f <- jit(
    \(x, drop) {
      nvl_reduce_sum(x, dims = 1L, drop = drop)
    },
    static = "drop"
  )
  expect_equal(f(x, TRUE), nv_scalar(6))
  expect_equal(f(x, FALSE), nv_tensor(6))
})

test_that("nvl_reshape", {
  x <- nv_tensor(1:3, dtype = "f32")
  expect_equal(
    jit(nvl_reshape, static = "shape")(x, c(3, 1)),
    nv_tensor(1:3, dtype = "f32", shape = c(3, 1))
  )
})

# comparisons
test_that("nvl_eq", {
  expect_jit_binary(nvl_eq, `==`, sample(1:10, 1), sample(1:10, 1))
  expect_jit_binary(nvl_eq, `==`, rnorm(1), rnorm(1))
  # unsigned
  f <- jit(`==`)
  expect_equal(
    f(nv_scalar(1L, dtype = "ui32"), nv_scalar(1L, dtype = "ui32")),
    nv_scalar(TRUE, dtype = "pred")
  )
  expect_equal(
    f(nv_scalar(2L, dtype = "ui32"), nv_scalar(1L, dtype = "ui32")),
    nv_scalar(FALSE, dtype = "pred")
  )
})

test_that("nvl_ne", {
  expect_jit_binary(nvl_ne, `!=`, sample(-10:10, 1), sample(-10:10, 1))
  expect_jit_binary(nvl_ne, `!=`, rnorm(1), rnorm(1))
})

test_that("nvl_gt", {
  expect_jit_binary(nvl_gt, `>`, sample(-10:10, 1), sample(-10:10, 1))
  expect_jit_binary(nvl_gt, `>`, rnorm(1), rnorm(1))
})

test_that("nvl_ge", {
  expect_jit_binary(nvl_ge, `>=`, sample(-10:10, 1), sample(-10:10, 1))
  expect_jit_binary(nvl_ge, `>=`, rnorm(1), rnorm(1))
})

test_that("nvl_lt", {
  expect_jit_binary(nvl_lt, `<`, sample(-10:10, 1), sample(-10:10, 1))
  expect_jit_binary(nvl_lt, `<`, rnorm(1), rnorm(1))
})

test_that("nvl_le", {
  expect_jit_binary(nvl_le, `<=`, sample(-10:10, 1), sample(-10:10, 1))
  expect_jit_binary(nvl_le, `<=`, rnorm(1), rnorm(1))
})
