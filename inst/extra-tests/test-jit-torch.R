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


# Reductions

test_that("p_reduce_sum", {
  shp <- c(2, 3, 4)
  dims <- sample(seq_along(shp), 2)
  drop <- sample(c(TRUE, FALSE), 1)
  f <- jit(function(x) nvl_reduce_sum(x, dims = dims, drop = drop))
  x <- array(generate_test_data(shp, dtype = "f32"), shp)
  out_nv <- f(nv_tensor(x))
  out_th <- torch::torch_sum(torch::torch_tensor(x), dim = dims, keepdim = !drop)
  testthat::expect_equal(as_array(out_nv), as_array_torch(out_th), tolerance = 1e-6)
})

test_that("p_reduce_prod", {
  shp <- c(2, 3, 2)
  dims <- 2L
  drop <- TRUE
  f <- jit(function(x) nvl_reduce_prod(x, dims = dims, drop = drop))
  x <- array(generate_test_data(shp, dtype = "f32"), shp)
  out_nv <- f(nv_tensor(x))
  out_th <- torch::torch_prod(torch::torch_tensor(x), dim = dims, keepdim = !drop)
  testthat::expect_equal(as_array(out_nv), as_array_torch(out_th), tolerance = 1e-6)
})

test_that("p_reduce_max", {
  shp <- c(2, 3, 2)
  dims <- 1L
  drop <- FALSE
  x <- array(generate_test_data(shp, dtype = "f32"), shp)
  fmax <- jit(function(x) nvl_reduce_max(x, dims = dims, drop = drop))
  out_nv_max <- fmax(nv_tensor(x))
  out_th_max <- torch::torch_amax(torch::torch_tensor(x), dim = dims, keepdim = !drop)
  testthat::expect_equal(as_array(out_nv_max), as_array_torch(out_th_max), tolerance = 1e-6)
})

test_that("p_reduce_min", {
  shp <- c(2, 3, 2)
  dims <- 1L
  drop <- FALSE
  x <- array(generate_test_data(shp, dtype = "f32"), shp)
  fmin <- jit(function(x) nvl_reduce_min(x, dims = dims, drop = drop))
  out_nv_min <- fmin(nv_tensor(x))
  out_th_min <- torch::torch_amin(torch::torch_tensor(x), dim = dims, keepdim = !drop)
  testthat::expect_equal(as_array(out_nv_min), as_array_torch(out_th_min), tolerance = 1e-6)
})

test_that("p_reduce_any", {
  shp <- c(2, 3, 2)
  dims <- 3L
  drop <- TRUE
  x <- array(generate_test_data(shp, dtype = "pred"), shp)
  fany <- jit(function(x) nvl_reduce_any(x, dims = dims, drop = drop))
  out_nv_any <- fany(nv_tensor(x, dtype = "pred"))
  xt <- torch::torch_tensor(x, dtype = torch::torch_bool())
  out_th_any <- xt$any(dim = dims, keepdim = !drop)
  testthat::expect_equal(as_array(out_nv_any), as_array_torch(out_th_any))
})

test_that("p_reduce_all", {
  shp <- c(2, 3, 2)
  dims <- 3L
  drop <- TRUE
  x <- array(generate_test_data(shp, dtype = "pred"), shp)
  fall <- jit(function(x) nvl_reduce_all(x, dims = dims, drop = drop))
  out_nv_all <- fall(nv_tensor(x, dtype = "pred"))
  xt <- torch::torch_tensor(x, dtype = torch::torch_bool())
  out_th_all <- xt$all(dim = dims, keepdim = !drop)
  testthat::expect_equal(as_array(out_nv_all), as_array_torch(out_th_all))
})

# Shape ops

test_that("p_transpose", {
  shp <- c(2, 3, 4)
  perm <- sample(1:3)
  f <- jit(function(x) nvl_transpose(x, permutation = perm))
  x <- array(generate_test_data(shp, dtype = "f32"), shp)
  out_nv <- f(nv_tensor(x))
  out_th <- torch::torch_tensor(x)$permute(perm)
  testthat::expect_equal(as_array(out_nv), as_array_torch(out_th))
})

test_that("p_reshape", {
  x <- nv_tensor(1:6, dtype = "f32")
  f <- jit(nvl_reshape, static = "shape")
  out_nv <- f(x, c(2, 3))
  out_th <- torch::torch_tensor(1:6, dtype = torch::torch_float32())$reshape(c(2, 3))
  testthat::expect_equal(as_array(out_nv), as_array_torch(out_th))
})

test_that("p_broadcast_to", {
  input_shape <- c(2L, 1L, 3L)
  target_shape <- c(4L, 2L, 5L, 3L)
  x <- array(generate_test_data(input_shape, dtype = "f32"), input_shape)
  f <- jit(function(operand, shape) nv_broadcast_to(operand, shape), static = "shape")
  out_nv <- f(nv_tensor(x), target_shape)
  out_th <- torch::torch_tensor(x)$broadcast_to(target_shape)
  # compare a reduction to avoid huge arrays in print
  testthat::expect_equal(
    as_array(out_nv),
    as_array_torch(out_th),
    tolerance = 1e-6
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

test_that("p_or", {
  expect_jit_torch_binary(nvl_or, torch::torch_logical_or, c(2, 3), c(2, 3), dtype = "pred")
})

test_that("p_xor", {
  expect_jit_torch_binary(nvl_xor, torch::torch_logical_xor, c(2, 3), c(2, 3), dtype = "pred")
})

test_that("p_shift_left", {
  # use signed 32-bit for both logical and arithmetic tests to satisfy backend
  x <- nv_tensor(sample(0:65535, 6, TRUE), dtype = "i32")
  y <- nv_tensor(sample(0:7, 6, TRUE), dtype = "i32")
  x_r <- as.integer(as_array(x))
  y_r <- as.integer(as_array(y))
  exp_l <- bitwShiftL(x_r, y_r)
  exp_r <- bitwShiftR(x_r, y_r)
  expect_equal(as.integer(as_array(jit(nvl_shift_left)(x, y))), as.integer(exp_l))
})

test_that("p_shift_right_logical", {
  x <- nv_tensor(sample(0:65535, 6, TRUE), dtype = "i32")
  y <- nv_tensor(sample(0:7, 6, TRUE), dtype = "i32")
  x_r <- as.integer(as_array(x))
  y_r <- as.integer(as_array(y))
  exp_r <- bitwShiftR(x_r, y_r)
  expect_equal(as.integer(as_array(jit(nvl_shift_right_logical)(x, y))), as.integer(exp_r))
})

test_that("p_shift_right_arithmetic", {
  # Arithmetic right shift should preserve sign; simulate expected with floor division by 2^k
  x <- nv_tensor(sample(-32768:32767, 6, TRUE), dtype = "i32")
  y <- nv_tensor(sample(0:7, 6, TRUE), dtype = "i32")
  x_r <- as.integer(as_array(x))
  y_r <- as.integer(as_array(y))
  exp_r <- vapply(
    seq_along(x_r),
    function(i) {
      k <- y_r[i]
      if (k == 0L) {
        return(x_r[i])
      }
      if (x_r[i] >= 0L) {
        bitwShiftR(x_r[i], k)
      } else {
        two_pow_k <- bitwShiftL(1L, k)
        neg_x <- as.integer(-x_r[i])
        neg_div <- bitwShiftR(as.integer(neg_x + (two_pow_k - 1L)), k)
        -neg_div
      }
    },
    integer(1)
  )
  expect_equal(as.integer(as_array(jit(nvl_shift_right_arithmetic)(x, y))), as.integer(exp_r))
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
  # compare against torch expand after unsqueeze on the missing leading dim
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
