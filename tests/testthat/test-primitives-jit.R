# binary ops

test_that("p_add", {
  expect_jit_binary(nvl_add, `+`, 1.2, -0.7)
})
test_that("p_sub", {
  expect_jit_binary(nvl_sub, `-`, 1.2, -0.7)
})

test_that("p_mul", {
  expect_jit_binary(nvl_mul, `*`, 1.2, -0.7)
})

test_that("p_neg", {
  expect_jit_unary(nvl_neg, \(x) -x, 1.7)
})

test_that("p_div", {
  expect_jit_binary(nvl_div, `/`, 1.2, 0.2)
})

test_that("p_pow", {
  expect_jit_binary(nvl_pow, `^`, 1, 0.3)
})


test_that("p_transpose", {
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

test_that("p_reshape", {
  x <- nv_tensor(1:3, dtype = "f32")
  expect_equal(
    jit(nvl_reshape, static = "shape")(x, c(3, 1)),
    nv_tensor(1:3, dtype = "f32", shape = c(3, 1))
  )
})

# comparisons
test_that("p_eq", {
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

test_that("p_ne", {
  expect_jit_binary(nvl_ne, `!=`, sample(-10:10, 1), sample(-10:10, 1))
  expect_jit_binary(nvl_ne, `!=`, rnorm(1), rnorm(1))
})

test_that("p_gt", {
  expect_jit_binary(nvl_gt, `>`, sample(-10:10, 1), sample(-10:10, 1))
  expect_jit_binary(nvl_gt, `>`, rnorm(1), rnorm(1))
})

test_that("p_ge", {
  expect_jit_binary(nvl_ge, `>=`, sample(-10:10, 1), sample(-10:10, 1))
  expect_jit_binary(nvl_ge, `>=`, rnorm(1), rnorm(1))
})

test_that("p_lt", {
  expect_jit_binary(nvl_lt, `<`, sample(-10:10, 1), sample(-10:10, 1))
  expect_jit_binary(nvl_lt, `<`, rnorm(1), rnorm(1))
})

test_that("p_le", {
  expect_jit_binary(nvl_le, `<=`, sample(-10:10, 1), sample(-10:10, 1))
  expect_jit_binary(nvl_le, `<=`, rnorm(1), rnorm(1))
})

# additional simple binary ops
test_that("p_max", {
  expect_jit_binary(nvl_max, pmax, rnorm(1), rnorm(1))
})

test_that("p_min", {
  expect_jit_binary(nvl_min, pmin, rnorm(1), rnorm(1))
})

test_that("p_remainder", {
  expect_jit_binary(nvl_remainder, `%%`, sample(1:10, 1), sample(1:10, 1))
})

test_that("p_and", {
  expect_jit_binary(nvl_and, `&`, sample(c(TRUE, FALSE), 1), sample(c(TRUE, FALSE), 1))
})

test_that("p_or", {
  expect_jit_binary(nvl_or, `|`, sample(c(TRUE, FALSE), 1), sample(c(TRUE, FALSE), 1))
})

test_that("p_xor", {
  f <- jit(function(x, y) nvl_xor(x, y))
  # use integer bitwise XOR for determinism
  x <- nv_scalar(5L, dtype = "i32")
  y <- nv_scalar(3L, dtype = "i32")
  expect_equal(as_array(f(x, y)), bitwXor(5L, 3L))
})

test_that("p_shift_left/right", {
  f_sl <- jit(function(x, y) nvl_shift_left(x, y))
  f_srl <- jit(function(x, y) nvl_shift_right_logical(x, y))
  f_sra <- jit(function(x, y) nvl_shift_right_arithmetic(x, y))
  x <- nv_scalar(8L, dtype = "i32")
  y <- nv_scalar(1L, dtype = "i32")
  expect_equal(as_array(f_sl(x, y)), bitwShiftL(8L, 1L))
  expect_equal(as_array(f_srl(x, y)), bitwShiftR(8L, 1L))
  expect_equal(as_array(f_sra(nv_scalar(-8L, dtype = "i32"), y)), as.integer(-8L %/% 2L))
})

test_that("p_atan2", {
  expect_jit_binary(nvl_atan2, atan2, rnorm(1), rnorm(1))
})
