test_that("p_add", {
  expect_jit_torch_binary(nvl_add, torch::torch_add, c(2, 3), c(2, 3))
})

test_that("p_sub", {
  expect_jit_torch_binary(nvl_sub, torch::torch_sub, c(2, 3), c(2, 3))
})

test_that("p_mul", {
  expect_jit_torch_binary(nvl_mul, torch::torch_mul, c(2, 3), c(2, 3))
})

test_that("p_neg", {
  expect_jit_torch_unary(nvl_neg, torch::torch_neg, c(2, 3))
})

test_that("p_div", {
  expect_jit_torch_binary(nvl_div, torch::torch_div, c(2, 3), c(2, 3))
})

test_that("p_pow", {
  expect_jit_torch_binary(
    nvl_pow,
    torch::torch_pow,
    c(2, 3),
    c(2, 3),
    non_negative = list(TRUE, FALSE)
  )
})


## Comparisons

test_that("p_eq", {
  expect_jit_torch_binary(nvl_eq, torch::torch_eq, c(2, 3), c(2, 3))
})

test_that("p_ne", {
  expect_jit_torch_binary(nvl_ne, torch::torch_ne, c(2, 3), c(2, 3))
})

test_that("p_gt", {
  expect_jit_torch_binary(nvl_gt, torch::torch_gt, c(2, 3), c(2, 3))
})

test_that("p_ge", {
  expect_jit_torch_binary(nvl_ge, torch::torch_ge, c(2, 3), c(2, 3))
})

test_that("p_lt", {
  expect_jit_torch_binary(nvl_lt, torch::torch_lt, c(2, 3), c(2, 3))
})

test_that("p_le", {
  expect_jit_torch_binary(nvl_le, torch::torch_le, c(2, 3), c(2, 3))
})

test_that("p_max", {
  expect_jit_torch_binary(nvl_max, torch::torch_maximum, c(2, 3), c(2, 3))
})

test_that("p_min", {
  expect_jit_torch_binary(nvl_min, torch::torch_minimum, c(2, 3), c(2, 3))
})

test_that("p_remainder", {
  pos_int_nz <- function(shp, dtype) {
    nelts <- if (!length(shp)) 1L else prod(shp)
    vals <- sample(10, size = nelts, replace = TRUE)
    if (!length(shp)) vals else array(vals, shp)
  }
  expect_jit_torch_binary(
    nvl_remainder,
    torch::torch_remainder,
    c(2, 3),
    c(2, 3),
    dtype = "i32",
    gen_x = pos_int_nz,
    gen_y = pos_int_nz
  )
})

test_that("p_and", {
  expect_jit_torch_binary(nvl_and, torch::torch_logical_and, c(2, 3), c(2, 3), dtype = "pred")
})

test_that("p_not", {
  expect_jit_torch_unary(nvl_not, \(x) !x, c(2, 3), dtype = "pred")
})

test_that("p_or", {
  expect_jit_torch_binary(nvl_or, torch::torch_logical_or, c(2, 3), c(2, 3), dtype = "pred")
})

test_that("p_xor", {
  expect_jit_torch_binary(nvl_xor, torch::torch_logical_xor, c(2, 3), c(2, 3), dtype = "pred")
})


test_that("p_atan2", {
  expect_jit_torch_binary(nvl_atan2, torch::torch_atan2, c(2, 3), c(2, 3))
})

# Unary math

test_that("p_abs", {
  expect_jit_torch_unary(nvl_abs, torch::torch_abs, c(2, 3))
})

test_that("p_sqrt", {
  expect_jit_torch_unary(
    nvl_sqrt,
    torch::torch_sqrt,
    c(2, 3),
    non_negative = TRUE
  )
})

test_that("p_rsqrt", {
  expect_jit_torch_unary(
    nvl_rsqrt,
    torch::torch_rsqrt,
    c(2, 3),
    non_negative = TRUE
  )
})

test_that("p_log", {
  expect_jit_torch_unary(
    nvl_log,
    torch::torch_log,
    c(2, 3),
    non_negative = TRUE
  )
})

test_that("p_tanh", {
  expect_jit_torch_unary(nvl_tanh, torch::torch_tanh, c(2, 3))
})

test_that("p_tan", {
  expect_jit_torch_unary(nvl_tan, torch::torch_tan, c(2, 3))
})

test_that("p_floor", {
  expect_jit_torch_unary(nvl_floor, torch::torch_floor, c(2, 3))
})

test_that("p_ceil", {
  expect_jit_torch_unary(nvl_ceil, torch::torch_ceil, c(2, 3))
})

test_that("p_sign", {
  expect_jit_torch_unary(nvl_sign, torch::torch_sign, c(2, 3))
})

test_that("p_exp", {
  expect_jit_torch_unary(nvl_exp, torch::torch_exp, c(2, 3))
})

test_that("p_round", {
  nv_even <- function(a) nvl_round(a, method = "nearest_even")
  th_even <- function(a) torch::torch_round(a)
  expect_jit_torch_unary(nv_even, th_even, c(2, 3))

  nv_afz <- function(a) nvl_round(a, method = "afz")
  th_afz <- function(a) torch::torch_sign(a) * torch::torch_floor(torch::torch_abs(a) + 0.5)
  expect_jit_torch_unary(nv_afz, th_afz, c(2, 3))
})

test_that("p_convert", {
  nv_fun <- function(a) nvl_convert(a, "f64")
  th_fun <- function(a) a$to(dtype = torch::torch_float64())
  expect_jit_torch_unary(nv_fun, th_fun, c(2, 3))
})

test_that("p_broadcast_in_dim", {
  input_shape <- c(2L, 3L)
  target_shape <- c(4L, 2L, 3L)
  bdims <- c(2L, 3L)
  x <- array(generate_test_data(input_shape, dtype = "f32"), input_shape)
  f <- jit(function(a) nvl_broadcast_in_dim(a, target_shape, bdims))
  out_nv <- f(nv_tensor(x))
  out_th <- torch::torch_tensor(x)$unsqueeze(1)$expand(target_shape)
  testthat::expect_equal(sum(as_array(out_nv)), as.numeric(torch::as_array(out_th$sum())), tolerance = 1e-5)
})

test_that("p_select", {
  p <- nv_tensor(c(TRUE, FALSE, TRUE, FALSE), dtype = "pred")
  a <- nv_tensor(as.integer(c(1, 2, 3, 4)), dtype = "i32")
  b <- nv_tensor(as.integer(c(10, 20, 30, 40)), dtype = "i32")
  out <- jit(nvl_select)(p, a, b)
  pt <- torch::torch_tensor(as_array(p), dtype = torch::torch_bool())
  at <- torch::torch_tensor(as_array(a), dtype = torch::torch_int32())
  bt <- torch::torch_tensor(as_array(b), dtype = torch::torch_int32())
  expect_equal(as_array(out), as_array_torch(torch::torch_where(pt, at, bt)))
})

test_that("p_dot_general", {
  # vector dot
  x <- nv_tensor(rnorm(4), dtype = "f32")
  y <- nv_tensor(rnorm(4), dtype = "f32")
  out <- jit(function(a, b) {
    nvl_dot_general(a, b, contracting_dims = list(c(1L), c(1L)), batching_dims = list(integer(), integer()))
  })(x, y)
  tx <- torch::torch_tensor(as_array(x))
  ty <- torch::torch_tensor(as_array(y))
  expect_equal(as_array(out), as.numeric(torch::torch_sum(tx * ty)), tolerance = 1e-5)

  # matrix-vector -> vector
  A <- nv_tensor(matrix(rnorm(6), 3, 2), dtype = "f32")
  v <- nv_tensor(rnorm(2), dtype = "f32")
  out2 <- jit(function(a, b) {
    nvl_dot_general(a, b, contracting_dims = list(c(2L), c(1L)), batching_dims = list(integer(), integer()))
  })(A, v)
  tA <- torch::torch_tensor(as_array(A))
  tv <- torch::torch_tensor(as_array(v))
  expect_equal(as_array(out2), as_array_torch(tA$matmul(tv)), tolerance = 1e-5)

  # batched matmul
  X <- nv_tensor(array(rnorm(2 * 3 * 4), c(2, 3, 4)), dtype = "f32")
  Y <- nv_tensor(array(rnorm(2 * 4 * 5), c(2, 4, 5)), dtype = "f32")
  out3 <- jit(function(a, b) {
    nvl_dot_general(a, b, contracting_dims = list(c(3L), c(2L)), batching_dims = list(c(1L), c(1L)))
  })(X, Y)
  tX <- torch::torch_tensor(as_array(X))
  tY <- torch::torch_tensor(as_array(Y))
  expect_equal(as_array(out3), as_array_torch(tX$matmul(tY)), tolerance = 1e-5)
})

# New primitives ---------------------------------------------------------------

test_that("p_cbrt", {
  expect_jit_torch_unary(
    nvl_cbrt,
    \(x) torch::torch_pow(x, 1 / 3),
    c(2, 3),
    non_negative = TRUE
  )
})

test_that("p_expm1", {
  expect_jit_torch_unary(nvl_expm1, torch::torch_expm1, c(2, 3))
})

test_that("p_log1p", {
  expect_jit_torch_unary(
    nvl_log1p,
    torch::torch_log1p,
    c(2, 3),
    non_negative = TRUE
  )
})

test_that("p_logistic", {
  expect_jit_torch_unary(nvl_logistic, torch::torch_sigmoid, c(2, 3))
})

test_that("p_is_finite", {
  # Test with normal values
  expect_jit_torch_unary(nvl_is_finite, torch::torch_isfinite, c(2, 3))

  # Test with special values
  special_vals <- c(1.0, Inf, -Inf, NaN, 0.0, -1.0)
  x_nv <- nv_tensor(special_vals, dtype = "f32")
  x_th <- torch::torch_tensor(special_vals, dtype = torch::torch_float32())

  out_nv <- jit(nvl_is_finite)(x_nv)
  out_th <- torch::torch_isfinite(x_th)

  expect_equal(as_array(out_nv), as_array_torch(out_th))
})

test_that("p_clamp", {
  nv_clamp_fun <- function(x) {
    min_val <- nv_broadcast_to(nv_scalar(-1.0, "f32"), shape(x))
    max_val <- nv_broadcast_to(nv_scalar(1.0, "f32"), shape(x))
    nvl_clamp(min_val, x, max_val)
  }
  th_clamp_fun <- function(x) torch::torch_clamp(x, min = -1.0, max = 1.0)
  expect_jit_torch_unary(nv_clamp_fun, th_clamp_fun, c(2, 3))
})

test_that("p_reverse", {
  x_arr <- array(1:24, c(2, 3, 4))
  x_nv <- nv_tensor(x_arr, dtype = "f32")
  x_th <- torch::torch_tensor(x_arr, dtype = torch::torch_float32())

  # Reverse along dimension 1
  out_nv1 <- jit(function(a) nvl_reverse(a, 1L))(x_nv)
  out_th1 <- torch::torch_flip(x_th, 1L) # torch uses 1-indexed for flip
  expect_equal(as_array(out_nv1), as_array_torch(out_th1))

  # Reverse along dimension 2
  out_nv2 <- jit(function(a) nvl_reverse(a, 2L))(x_nv)
  out_th2 <- torch::torch_flip(x_th, 2L)
  expect_equal(as_array(out_nv2), as_array_torch(out_th2))

  # Reverse along multiple dimensions
  out_nv3 <- jit(function(a) nvl_reverse(a, c(1L, 3L)))(x_nv)
  out_th3 <- torch::torch_flip(x_th, c(1L, 3L))
  expect_equal(as_array(out_nv3), as_array_torch(out_th3))
})

test_that("p_iota", {
  # Simple 1D iota
  out_nv <- jit(function() nvl_iota(1L, "i32", 5L))()
  out_th <- torch::torch_arange(0, 4, dtype = torch::torch_int32())
  expect_equal(c(as_array(out_nv)), c(as_array_torch(out_th)))

  # 2D iota along first dimension
  out_nv2 <- jit(function() nvl_iota(1L, "i32", c(3L, 4L)))()
  # Creates a 3x4 matrix where each row has the same row index
  # Row 0: [0,0,0,0], Row 1: [1,1,1,1], Row 2: [2,2,2,2]
  expected2 <- matrix(rep(0:2, times = 4), nrow = 3, ncol = 4)
  expect_equal(as_array(out_nv2), expected2)

  # 2D iota along second dimension
  out_nv3 <- jit(function() nvl_iota(2L, "i32", c(3L, 4L)))()
  # Creates a 3x4 matrix where each column has the same column index
  # [[0,1,2,3], [0,1,2,3], [0,1,2,3]]
  expected3 <- matrix(rep(0:3, each = 3), nrow = 3, ncol = 4)
  expect_equal(as_array(out_nv3), expected3)
})

test_that("p_pad", {
  x_arr <- array(1:6, c(2, 3))
  x_nv <- nv_tensor(x_arr, dtype = "f32")
  x_th <- torch::torch_tensor(x_arr, dtype = torch::torch_float32())

  # Simple edge padding
  out_nv <- jit(function(a) {
    nvl_pad(a, nv_scalar(0.0, "f32"), c(1L, 1L), c(1L, 1L), c(0L, 0L))
  })(x_nv)
  # torch.nn.functional.pad uses (left, right, top, bottom) for 2D
  out_th <- torch::nnf_pad(x_th, c(1L, 1L, 1L, 1L), value = 0.0)
  expect_equal(as_array(out_nv), as_array_torch(out_th))
})

test_that("p_popcnt", {
  # Population count for integers
  vals <- as.integer(c(0, 1, 2, 3, 7, 15, 255))
  x_nv <- nv_tensor(vals, dtype = "i32")

  out_nv <- jit(nvl_popcnt)(x_nv)
  # Expected: number of 1 bits in each value
  expected <- c(0L, 1L, 1L, 2L, 3L, 4L, 8L)
  expect_equal(c(as_array(out_nv)), expected)
})
