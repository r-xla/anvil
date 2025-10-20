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

# reduction ops

test_that("empty reduce", {
  x <- nv_empty("f32", shape = c(0, 1))
  f <- jit(function(x) {
    nv_reduce_sum(x, dims = 1:2)
  })
  expect_equal(f(x), nv_scalar(0))
})

test_that("drop reduce", {
  x <- nv_tensor(1:3, dtype = "f32")
  f <- jit(
    function(x, drop) {
      nv_reduce_sum(x, dims = 1L, drop = drop)
    },
    static = "drop"
  )
  expect_equal(f(x, TRUE), nv_scalar(6))
  expect_equal(f(x, FALSE), nv_tensor(6))
})

test_that("p_reduce_sum", {
  f <- jit(
    function(x) {
      nvl_reduce_sum(x, dims = 1L)
    }
  )
  expect_equal(f(nv_tensor(1:3, dtype = "f32")), nv_scalar(6))
})

test_that("p_reduce_prod", {
  f <- jit(
    function(x) {
      nvl_reduce_prod(x, dims = 1L)
    }
  )
  expect_equal(f(nv_tensor(1:4, dtype = "f32")), nv_scalar(24))
})

test_that("p_reduce_max", {
  f <- jit(
    function(x) {
      nvl_reduce_max(x, dims = 1L)
    }
  )
  expect_equal(f(nv_tensor(c(FALSE, FALSE))), nv_scalar(FALSE))
  expect_equal(f(nv_tensor(c(FALSE, TRUE))), nv_scalar(TRUE))
  expect_equal(f(nv_tensor(c(1, 2, 3))), nv_scalar(3))
})

test_that("p_reduce_min", {
  f <- jit(
    function(x) {
      nvl_reduce_min(x, dims = 1L)
    }
  )
  expect_equal(f(nv_tensor(c(FALSE, TRUE))), nv_scalar(FALSE))
  expect_equal(f(nv_tensor(c(1, 2, 3), dtype = "f32")), nv_scalar(1))
})

test_that("p_reduce_any", {
  f <- jit(function(x) {
    nvl_reduce_any(x, dims = 1L)
  })
  expect_equal(f(nv_tensor(c(FALSE, FALSE))), nv_scalar(FALSE))
  expect_equal(f(nv_tensor(c(TRUE, FALSE))), nv_scalar(TRUE))
})

test_that("p_reduce_all", {
  f <- jit(function(x) {
    nvl_reduce_all(x, dims = 1L)
  })
  expect_equal(f(nv_tensor(c(TRUE, FALSE))), nv_scalar(FALSE))
  expect_equal(f(nv_tensor(c(TRUE, TRUE))), nv_scalar(TRUE))
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

# additional jit rule coverage required by meta-tests ---------------------------------

test_that("p_abs", {
  expect_jit_unary(nvl_abs, abs, 1.7)
})

test_that("p_sqrt", {
  expect_jit_unary(nvl_sqrt, sqrt, 1.7)
})

test_that("p_rsqrt", {
  expect_jit_unary(nvl_rsqrt, function(x) 1 / sqrt(x), 1.7)
})

test_that("p_log", {
  expect_jit_unary(nvl_log, log, 1.7)
})

test_that("p_tanh", {
  expect_jit_unary(nvl_tanh, tanh, 0.5)
})

test_that("p_tan", {
  expect_jit_unary(nvl_tan, tan, 0.5)
})

test_that("p_floor", {
  expect_jit_unary(nvl_floor, floor, 1.7)
})

test_that("p_ceil", {
  expect_jit_unary(nvl_ceil, ceiling, 1.7)
})

test_that("p_sign", {
  expect_jit_unary(nvl_sign, sign, -1.7)
})

test_that("p_exp", {
  expect_jit_unary(nvl_exp, exp, 0.5)
})

test_that("p_round", {
  x <- nv_tensor(c(-2.5, -1.5, -0.5, 0.5, 1.5), dtype = "f32")
  f_even <- jit(function(a) nvl_round(a, method = "nearest_even"))
  f_afz <- jit(function(a) nvl_round(a, method = "afz"))
  expect_equal(as_array(f_even(x)), round(as_array(x)))
  # away-from-zero
  expect_equal(as_array(f_afz(x)), sign(as_array(x)) * floor(abs(as_array(x)) + 0.5))
})

test_that("p_convert", {
  x <- nv_tensor(rnorm(4), dtype = "f32")
  f <- jit(function(a) nvl_convert(a, "f64"))
  expect_equal(as_array(f(x)), as_array(x))
})

test_that("p_shift_right_logical", {
  f <- jit(function(x, y) nvl_shift_right_logical(x, y))
  x <- nv_scalar(8L, dtype = "i32")
  y <- nv_scalar(1L, dtype = "i32")
  expect_equal(as_array(f(x, y)), bitwShiftR(8L, 1L))
})

test_that("p_shift_right_arithmetic", {
  f <- jit(function(x, y) nvl_shift_right_arithmetic(x, y))
  expect_equal(as_array(f(nv_scalar(-8L, dtype = "i32"), nv_scalar(1L, dtype = "i32"))), as.integer(-8L %/% 2L))
})

test_that("p_select", {
  f <- jit(function(p, a, b) nvl_select(p, a, b))
  p <- nv_tensor(c(TRUE, FALSE, TRUE), dtype = "pred")
  a <- nv_tensor(as.integer(c(1, 2, 3)), dtype = "i32")
  b <- nv_tensor(as.integer(c(10, 20, 30)), dtype = "i32")
  expect_equal(as.integer(as_array(f(p, a, b))), as.integer(c(1, 20, 3)))
})

test_that("p_broadcast_in_dim", {
  x <- nv_tensor(1:6, dtype = "i32", shape = c(2, 3))
  shape_out <- c(4L, 2L, 3L)
  bdims <- c(2L, 3L)
  f <- jit(function(a) nvl_broadcast_in_dim(a, shape_out, bdims))
  out <- f(x)
  expect_equal(dim(as_array(out)), shape_out)
  expect_equal(as_array(out)[1, , ], matrix(1:6, nrow = 2))
})

test_that("p_dot_general", {
  # vector dot product
  x <- nv_tensor(rnorm(4), dtype = "f32")
  y <- nv_tensor(rnorm(4), dtype = "f32")
  f <- jit(function(a, b) {
    nvl_dot_general(a, b, contracting_dims = list(c(1L), c(1L)), batching_dims = list(integer(), integer()))
  })
  expect_equal(as_array(f(x, y)), sum(as_array(x) * as_array(y)), tolerance = 1e-6)
})
