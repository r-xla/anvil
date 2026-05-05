test_that("prim_add", {
  expect_jit_torch_binary(prim_add, torch::torch_add, c(2, 3), c(2, 3))
})

test_that("prim_sub", {
  expect_jit_torch_binary(prim_sub, torch::torch_sub, c(2, 3), c(2, 3))
})

test_that("prim_mul", {
  expect_jit_torch_binary(prim_mul, torch::torch_mul, c(2, 3), c(2, 3))
})

test_that("prim_negate", {
  expect_jit_torch_unary(prim_negate, torch::torch_neg, c(2, 3))
})

test_that("prim_div", {
  expect_jit_torch_binary(prim_div, torch::torch_div, c(2, 3), c(2, 3))
})

test_that("prim_pow", {
  expect_jit_torch_binary(
    prim_pow,
    torch::torch_pow,
    c(2, 3),
    c(2, 3),
    non_negative = list(TRUE, FALSE)
  )
})


## Comparisons

test_that("prim_eq", {
  expect_jit_torch_binary(prim_eq, torch::torch_eq, c(2, 3), c(2, 3))
})

test_that("prim_ne", {
  expect_jit_torch_binary(prim_ne, torch::torch_ne, c(2, 3), c(2, 3))
})

test_that("prim_gt", {
  expect_jit_torch_binary(prim_gt, torch::torch_gt, c(2, 3), c(2, 3))
})

test_that("prim_ge", {
  expect_jit_torch_binary(prim_ge, torch::torch_ge, c(2, 3), c(2, 3))
})

test_that("prim_lt", {
  expect_jit_torch_binary(prim_lt, torch::torch_lt, c(2, 3), c(2, 3))
})

test_that("prim_le", {
  expect_jit_torch_binary(prim_le, torch::torch_le, c(2, 3), c(2, 3))
})

test_that("prim_max", {
  expect_jit_torch_binary(prim_max, torch::torch_maximum, c(2, 3), c(2, 3))
})

test_that("prim_min", {
  expect_jit_torch_binary(prim_min, torch::torch_minimum, c(2, 3), c(2, 3))
})

test_that("prim_remainder", {
  # prim_remainder is IEEE-754 truncating (sign of dividend), so compare against
  # torch_fmod, not torch_remainder. Use mixed-sign nonzero divisors to exercise
  # the case where flooring would silently disagree.
  signed_int_nz <- function(shp, dtype) {
    nelts <- if (!length(shp)) 1L else prod(shp)
    mag <- sample(1:10, size = nelts, replace = TRUE)
    sgn <- sample(c(-1L, 1L), size = nelts, replace = TRUE)
    vals <- as.integer(mag * sgn)
    if (!length(shp)) vals else array(vals, shp)
  }
  expect_jit_torch_binary(
    prim_remainder,
    torch::torch_fmod,
    c(2, 3),
    c(2, 3),
    dtype = "i32",
    gen_x = signed_int_nz,
    gen_y = signed_int_nz
  )
})

test_that("prim_and", {
  expect_jit_torch_binary(prim_and, torch::torch_logical_and, c(2, 3), c(2, 3), dtype = "bool")
})

test_that("prim_not", {
  expect_jit_torch_unary(prim_not, \(x) !x, c(2, 3), dtype = "bool")
})

test_that("prim_or", {
  expect_jit_torch_binary(prim_or, torch::torch_logical_or, c(2, 3), c(2, 3), dtype = "bool")
})

test_that("prim_xor", {
  expect_jit_torch_binary(prim_xor, torch::torch_logical_xor, c(2, 3), c(2, 3), dtype = "bool")
})


test_that("prim_atan2", {
  expect_jit_torch_binary(prim_atan2, torch::torch_atan2, c(2, 3), c(2, 3))
})

# Unary math

test_that("prim_abs", {
  expect_jit_torch_unary(prim_abs, torch::torch_abs, c(2, 3))
})

test_that("prim_sqrt", {
  expect_jit_torch_unary(
    prim_sqrt,
    torch::torch_sqrt,
    c(2, 3),
    non_negative = TRUE
  )
})

test_that("prim_rsqrt", {
  expect_jit_torch_unary(
    prim_rsqrt,
    torch::torch_rsqrt,
    c(2, 3),
    non_negative = TRUE
  )
})

test_that("prim_log", {
  expect_jit_torch_unary(
    prim_log,
    torch::torch_log,
    c(2, 3),
    non_negative = TRUE
  )
})

test_that("prim_tanh", {
  expect_jit_torch_unary(prim_tanh, torch::torch_tanh, c(2, 3))
})

test_that("prim_tan", {
  expect_jit_torch_unary(prim_tan, torch::torch_tan, c(2, 3))
})

test_that("prim_floor", {
  expect_jit_torch_unary(prim_floor, torch::torch_floor, c(2, 3))
})

test_that("prim_ceil", {
  expect_jit_torch_unary(prim_ceil, torch::torch_ceil, c(2, 3))
})

test_that("prim_sign", {
  expect_jit_torch_unary(prim_sign, torch::torch_sign, c(2, 3))
})

test_that("prim_exp", {
  expect_jit_torch_unary(prim_exp, torch::torch_exp, c(2, 3))
})

test_that("prim_round", {
  nv_even <- function(a) prim_round(a, method = "nearest_even")
  th_even <- function(a) torch::torch_round(a)
  expect_jit_torch_unary(nv_even, th_even, c(2, 3))

  nv_afz <- function(a) prim_round(a, method = "afz")
  th_afz <- function(a) torch::torch_sign(a) * torch::torch_floor(torch::torch_abs(a) + 0.5)
  expect_jit_torch_unary(nv_afz, th_afz, c(2, 3))
})

test_that("prim_convert", {
  nv_fun <- function(a) prim_convert(a, "f64")
  th_fun <- function(a) a$to(dtype = torch::torch_float64())
  expect_jit_torch_unary(nv_fun, th_fun, c(2, 3))
})

test_that("prim_broadcast_in_dim", {
  input_shape <- c(2L, 3L)
  target_shape <- c(4L, 2L, 3L)
  bdims <- c(2L, 3L)
  x <- generate_test_data(input_shape, dtype = "f32")
  f <- jit(function(a) prim_broadcast_in_dim(a, target_shape, bdims))
  out_nv <- f(nv_array(x))
  out_th <- torch::torch_tensor(x)$unsqueeze(1)$expand(target_shape)
  testthat::expect_equal(sum(as_array(out_nv)), as.numeric(torch::as_array(out_th$sum())), tolerance = 1e-5)
})

test_that("prim_ifelse", {
  p <- nv_array(c(TRUE, FALSE, TRUE, FALSE), dtype = "bool")
  a <- nv_array(as.integer(c(1, 2, 3, 4)), dtype = "i32")
  b <- nv_array(as.integer(c(10, 20, 30, 40)), dtype = "i32")
  out <- jit(prim_ifelse)(p, a, b)
  pt <- torch::torch_tensor(as_array(p), dtype = torch::torch_bool())
  at <- torch::torch_tensor(as_array(a), dtype = torch::torch_int32())
  bt <- torch::torch_tensor(as_array(b), dtype = torch::torch_int32())
  expect_equal(as_array(out), as_array_torch(torch::torch_where(pt, at, bt)))
})

test_that("prim_dot_general", {
  # vector dot
  x <- nv_array(rnorm(4), dtype = "f32")
  y <- nv_array(rnorm(4), dtype = "f32")
  out <- jit(function(a, b) {
    prim_dot_general(a, b, contracting_dims = list(1L, 1L), batching_dims = list(integer(), integer()))
  })(x, y)
  tx <- torch::torch_tensor(as_array(x))
  ty <- torch::torch_tensor(as_array(y))
  expect_equal(as_array(out), as.numeric(torch::torch_sum(tx * ty)), tolerance = 1e-5)

  # matrix-vector -> vector
  A <- nv_array(matrix(rnorm(6), 3, 2), dtype = "f32")
  v <- nv_array(rnorm(2), dtype = "f32")
  out2 <- jit(function(a, b) {
    prim_dot_general(a, b, contracting_dims = list(2L, 1L), batching_dims = list(integer(), integer()))
  })(A, v)
  tA <- torch::torch_tensor(as_array(A))
  tv <- torch::torch_tensor(as_array(v))
  expect_equal(as_array(out2), as_array_torch(tA$matmul(tv)), tolerance = 1e-5)

  # batched matmul
  X <- nv_array(array(rnorm(2 * 3 * 4), c(2, 3, 4)), dtype = "f32")
  Y <- nv_array(array(rnorm(2 * 4 * 5), c(2, 4, 5)), dtype = "f32")
  out3 <- jit(function(a, b) {
    prim_dot_general(a, b, contracting_dims = list(3L, 2L), batching_dims = list(1L, 1L))
  })(X, Y)
  tX <- torch::torch_tensor(as_array(X))
  tY <- torch::torch_tensor(as_array(Y))
  expect_equal(as_array(out3), as_array_torch(tX$matmul(tY)), tolerance = 1e-5)
})

test_that("prim_cbrt", {
  expect_jit_torch_unary(
    prim_cbrt,
    \(x) torch::torch_pow(x, 1 / 3),
    c(2, 3),
    non_negative = TRUE
  )
})

test_that("prim_expm1", {
  expect_jit_torch_unary(prim_expm1, torch::torch_expm1, c(2, 3))
})

test_that("prim_log1p", {
  expect_jit_torch_unary(
    prim_log1p,
    torch::torch_log1p,
    c(2, 3),
    non_negative = TRUE
  )
})

test_that("prim_logistic", {
  expect_jit_torch_unary(prim_logistic, torch::torch_sigmoid, c(2, 3))
})

describe("prim_chol", {
  it("lower = TRUE", {
    A <- crossprod(matrix(rnorm(9), 3, 3)) + diag(3)
    out_nv <- as_array(jit(function(a) prim_chol(a, lower = TRUE))(nv_array(A, dtype = "f64")))
    out_th <- as_array_torch(torch::linalg_cholesky(torch::torch_tensor(A, dtype = torch::torch_float64())))
    expect_equal(out_nv, out_th, tolerance = 1e-6)
  })

  it("lower = FALSE", {
    A <- crossprod(matrix(rnorm(9), 3, 3)) + diag(3)
    out_nv <- as_array(jit(function(a) prim_chol(a, lower = FALSE))(nv_array(A, dtype = "f64")))
    out_th <- as_array_torch(torch::linalg_cholesky(torch::torch_tensor(A, dtype = torch::torch_float64()))$t())
    expect_equal(out_nv, out_th, tolerance = 1e-6)
  })
})

describe("prim_triangular_solve", {
  it("left_side, lower", {
    L <- matrix(c(3, 1, 0, 2), nrow = 2)
    b <- matrix(c(6, 5), nrow = 2)
    out_nv <- as_array(jit(function(a, b) {
      prim_triangular_solve(a, b, left_side = TRUE, lower = TRUE, unit_diagonal = FALSE, transpose_a = "NO_TRANSPOSE")
    })(nv_array(L, dtype = "f64"), nv_array(b, dtype = "f64")))
    out_th <- as_array_torch(torch::linalg_solve_triangular(
      torch::torch_tensor(L, dtype = torch::torch_float64()),
      torch::torch_tensor(b, dtype = torch::torch_float64()),
      upper = FALSE,
      left = TRUE
    ))
    expect_equal(out_nv, out_th, tolerance = 1e-6)
  })

  it("right_side, upper", {
    U <- matrix(c(3, 0, 1, 2), nrow = 2)
    b <- matrix(c(6, 5, 4, 3), nrow = 2)
    out_nv <- as_array(jit(function(a, b) {
      prim_triangular_solve(a, b, left_side = FALSE, lower = FALSE, unit_diagonal = FALSE, transpose_a = "NO_TRANSPOSE")
    })(nv_array(U, dtype = "f64"), nv_array(b, dtype = "f64")))
    out_th <- as_array_torch(torch::linalg_solve_triangular(
      torch::torch_tensor(U, dtype = torch::torch_float64()),
      torch::torch_tensor(b, dtype = torch::torch_float64()),
      upper = TRUE,
      left = FALSE
    ))
    expect_equal(out_nv, out_th, tolerance = 1e-6)
  })
})

test_that("prim_pad", {
  x_arr <- array(1:6, c(2, 3))
  x_nv <- nv_array(x_arr, dtype = "f32")
  x_th <- torch::torch_tensor(x_arr, dtype = torch::torch_float32())

  # Simple edge padding
  out_nv <- jit(function(a) {
    prim_pad(a, nv_scalar(0.0, "f32"), c(1L, 1L), c(1L, 1L), c(0L, 0L))
  })(x_nv)
  # torch.nn.functional.pad uses (left, right, top, bottom) for 2D
  out_th <- torch::nnf_pad(x_th, c(1L, 1L, 1L, 1L), value = 0.0)
  expect_equal(as_array(out_nv), as_array_torch(out_th))
})

# R torch is 1-based for both dim args and returned indices, matching
# anvl's convention. Both break ties on the smallest index, so direct
# equality holds.
.argmax_argmin_compare <- function(arr, dtype, dim_anvl) {
  arr_nv <- nv_array(arr, dtype = dtype)
  arr_th <- torch::torch_tensor(arr, dtype = str_to_torch_dtype(dtype))
  for (op in list(
    list(anvl = prim_argmax, torch = torch::torch_argmax),
    list(anvl = prim_argmin, torch = torch::torch_argmin)
  )) {
    out_nv <- op$anvl(arr_nv, dim = dim_anvl)
    out_th <- op$torch(arr_th, dim = dim_anvl)
    expect_equal(as_array(out_nv), as_array_torch(out_th))
  }
}

describe("prim_argmax / prim_argmin", {
  it("(forward) match torch on 1D", {
    .argmax_argmin_compare(c(3, 1, 4, 1.5, 5), "f32", dim_anvl = 1L)
  })

  it("tie-break: smallest index wins", {
    # All-equal: both anvl and torch should pick index 1.
    .argmax_argmin_compare(c(7, 7, 7, 7), "f32", dim_anvl = 1L)
    # Duplicate max in middle.
    .argmax_argmin_compare(c(1, 5, 5, 3, 5), "f32", dim_anvl = 1L)
    # Duplicate min at multiple positions.
    .argmax_argmin_compare(c(2, 1, 1, 4, 1), "f32", dim_anvl = 1L)
  })

  it("match torch on 2D inputs", {
    m <- matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    .argmax_argmin_compare(m, "f32", dim_anvl = 2L)
    .argmax_argmin_compare(m, "f32", dim_anvl = 1L)

    # 2D with duplicates per row/column.
    m2 <- matrix(c(1, 5, 5, 3, 2, 2), nrow = 2, byrow = TRUE)
    .argmax_argmin_compare(m2, "f32", dim_anvl = 2L)
  })

  it("match torch on integer inputs", {
    .argmax_argmin_compare(c(5L, 2L, 8L, 1L, 8L), "i32", dim_anvl = 1L)
  })
})
