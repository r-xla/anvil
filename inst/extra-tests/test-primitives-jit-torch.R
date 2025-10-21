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
    vals <- sample(1:10, size = nelts, replace = TRUE)
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
  skip_if(is_metal())
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
  testthat::expect_equal(sum(as_array(out_nv)), as.numeric(torch::as_array(out_th$sum())), tolerance = 1e-6)
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
