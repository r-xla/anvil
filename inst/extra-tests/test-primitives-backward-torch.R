build_extra_args <- function(args_f, shp, dtype) {
  if (is.null(args_f)) {
    return(list(list(), list()))
  }
  args_f(shp, dtype)
}

wrap_uni_anvil <- function(.f, args_anvil, shp) {
  if (identical(shp, integer())) {
    return(\(operand) {
      do.call(.f, c(list(operand), args_anvil))
    })
  }

  \(operand) {
    x <- do.call(.f, c(list(operand), args_anvil))
    nv_reduce_sum(x, dims = seq_along(shape(x)), drop = TRUE)
  }
}

wrap_uni_torch <- function(.g, args_torch, shp) {
  if (identical(shp, integer())) {
    return(\(operand) {
      do.call(.g, c(list(operand), args_torch))
    })
  }

  \(operand) {
    x <- do.call(.g, c(list(operand), args_torch))
    torch::torch_sum(x, dim = seq_along(x$shape), keepdim = FALSE)
  }
}

wrap_biv_anvil <- function(.f, args_anvil, shp) {
  if (identical(shp, integer())) {
    return(\(lhs, rhs) {
      do.call(.f, c(list(lhs, rhs), args_anvil))
    })
  }

  \(lhs, rhs) {
    x <- do.call(.f, c(list(lhs, rhs), args_anvil))
    nv_reduce_sum(x, dims = seq_along(shape(x)), drop = TRUE)
  }
}

wrap_biv_torch <- function(.g, args_torch, shp) {
  \(lhs, rhs) {
    x <- do.call(.g, c(list(lhs, rhs), args_torch))
    num_remaining <- length(shp)
    if (num_remaining > 0L) torch::torch_sum(x, dim = seq_len(num_remaining)) else x
  }
}

verify_grad_uni_scalar <- function(
  .f,
  .g,
  ndims = 0L,
  dtypes = "f32",
  args_f = NULL,
  tol = 0,
  non_negative = FALSE,
  gen = NULL
) {
  dtype <- sample(dtypes, 1L)
  shp <- integer()

  if (is.null(gen)) {
    operand <- generate_test_data(integer(), dtype, non_negative = non_negative)
  } else {
    operand <- gen(shp, dtype)
  }

  operand_anvil <- nv_scalar(operand, dtype = dtype)

  # I think there is a bug in torch, so we can't use torch_scalar_tensor
  operand_torch <- torch::torch_scalar_tensor(operand, requires_grad = TRUE, dtype = str_to_torch_dtype(dtype))
  operand_torch$retain_grad()

  args <- build_extra_args(args_f, shp, dtype)
  args_anvil <- args[[1L]]
  args_torch <- args[[2L]]

  .f_anvil <- \(operand) {
    do.call(.f, c(list(operand), args_anvil))
  }
  .g_torch <- \(operand) {
    do.call(.g, c(list(operand), args_torch))
  }

  grads_anvil <- jit(gradient(.f_anvil))(operand_anvil)
  out <- .g_torch(operand_torch)
  out$backward(retrain_graph = TRUE)

  testthat::expect_equal(
    tengen::as_array(grads_anvil[[1L]]),
    as_array_torch(operand_torch$grad),
    tolerance = tol
  )
}

verify_grad_uni_tensor <- function(
  .f,
  .g,
  ndims = sample(1:3, 1L),
  dtypes = "f32",
  args_f = NULL,
  shape = NULL,
  tol = 0,
  non_negative = FALSE,
  gen = NULL
) {
  shp <- if (is.null(shape)) sample(1:3, ndims, replace = TRUE) else shape
  dtype <- sample(dtypes, 1L)

  if (is.null(gen)) {
    operand <- array(
      generate_test_data(shp, dtype = dtype, non_negative = non_negative),
      shp
    )
  } else {
    operand <- gen(shp, dtype)
  }

  operand_anvil <- nv_tensor(operand, dtype = dtype)

  operand_torch <- torch::torch_tensor(
    operand,
    requires_grad = TRUE,
    dtype = str_to_torch_dtype(dtype)
  )

  args <- build_extra_args(args_f, shp, dtype)
  args_anvil <- args[[1L]]
  args_torch <- args[[2L]]

  .f_anvil <- wrap_uni_anvil(.f, args_anvil, shp)
  .g_torch <- wrap_uni_torch(.g, args_torch, shp)

  grads_anvil <- jit(gradient(.f_anvil))(operand_anvil)
  .g_torch(operand_torch)$backward()

  testthat::expect_equal(
    tengen::as_array(grads_anvil[[1L]]),
    as_array_torch(operand_torch$grad),
    tolerance = tol
  )
}

verify_grad_biv_scalar <- function(
  .f,
  .g,
  ndims = 0L,
  dtypes = "f32",
  args_f = NULL,
  tol = 1e-5,
  non_negative = list(FALSE, FALSE),
  gen_lhs = NULL,
  gen_rhs = NULL
) {
  dtype <- sample(dtypes, 1L)
  shp <- integer()

  if (length(non_negative) < 2) {
    non_negative <- rep(non_negative, 2)
  }

  if (is.null(gen_lhs)) {
    lhs <- generate_test_data(integer(), dtype, non_negative = non_negative[[1]])
  } else {
    lhs <- gen_lhs(shp, dtype)
  }
  if (is.null(gen_rhs)) {
    rhs <- generate_test_data(integer(), dtype, non_negative = non_negative[[2]])
  } else {
    rhs <- gen_rhs(shp, dtype)
  }

  lhs_anvil <- nv_scalar(lhs, dtype = dtype)
  rhs_anvil <- nv_scalar(rhs, dtype = dtype)

  # I think there is a bug in torch, so we can't use torch_scalar_tensor
  lhs_torch <- torch::torch_scalar_tensor(lhs, requires_grad = TRUE, dtype = str_to_torch_dtype(dtype))
  lhs_torch$retain_grad()
  rhs_torch <- torch::torch_scalar_tensor(rhs, requires_grad = TRUE, dtype = str_to_torch_dtype(dtype))
  rhs_torch$retain_grad()

  args <- build_extra_args(args_f, shp, dtype)
  args_anvil <- args[[1L]]
  args_torch <- args[[2L]]

  .f_anvil <- \(lhs, rhs) {
    do.call(.f, c(list(lhs, rhs), args_anvil))
  }
  .g_torch <- \(lhs, rhs) {
    do.call(.g, c(list(lhs, rhs), args_torch))
  }

  grads_anvil <- jit(gradient(.f_anvil))(lhs_anvil, rhs_anvil)
  out <- .g_torch(lhs_torch, rhs_torch)
  out$backward(retrain_graph = TRUE)

  testthat::expect_equal(
    tengen::as_array(grads_anvil[[1L]]),
    as_array_torch(lhs_torch$grad), # nolint
    tolerance = tol
  )

  testthat::expect_equal(
    tengen::as_array(grads_anvil[[2L]]),
    as_array_torch(rhs_torch$grad), # nolint
    tolerance = tol
  )
}

verify_grad_biv_tensor <- function(
  .f,
  .g,
  ndims = sample(1:3, 1L),
  dtypes = "f32",
  args_f = NULL,
  shape = NULL,
  tol = 0,
  non_negative = list(FALSE, FALSE),
  gen_lhs = NULL,
  gen_rhs = NULL
) {
  # Prefer shapes without size-0 or size-1 axes to avoid backend broadcast edge-cases
  shp <- if (is.null(shape)) sample(1:3, ndims, replace = TRUE) else shape
  dtype <- sample(dtypes, 1)

  if (length(non_negative) < 2) {
    non_negative <- rep(non_negative, 2)
  }

  if (is.null(gen_lhs)) {
    lhs <- array(
      generate_test_data(shp, dtype = dtype, non_negative = non_negative[[1]]), # nolint
      shp
    )
  } else {
    lhs <- gen_lhs(shp, dtype)
  }
  if (is.null(gen_rhs)) {
    rhs <- array(
      generate_test_data(shp, dtype = dtype, non_negative = non_negative[[2]]), # nolint
      shp
    )
  } else {
    rhs <- gen_rhs(shp, dtype)
  }

  lhs_anvil <- nv_tensor(lhs)
  rhs_anvil <- nv_tensor(rhs)

  lhs_torch <- torch::torch_tensor(lhs, requires_grad = TRUE)
  rhs_torch <- torch::torch_tensor(rhs, requires_grad = TRUE)

  args <- build_extra_args(args_f, shp, dtype)
  args_anvil <- args[[1L]]
  args_torch <- args[[2L]]

  .f_anvil <- wrap_biv_anvil(.f, args_anvil, shp)
  .g_torch <- wrap_biv_torch(.g, args_torch, shp)

  grads_anvil <- jit(gradient(.f_anvil))(lhs_anvil, rhs_anvil)
  .g_torch(lhs_torch, rhs_torch)$backward()

  testthat::expect_equal(
    tengen::as_array(grads_anvil[[1L]]),
    as_array_torch(lhs_torch$grad), # nolint
    tolerance = tol # nolint
  )

  testthat::expect_equal(
    tengen::as_array(grads_anvil[[2L]]),
    as_array_torch(rhs_torch$grad),
    tolerance = tol
  )
}

verify_grad_biv <- function(
  f,
  g,
  ndims = sample(1:3, 1L),
  dtypes = "f32",
  args_f = NULL,
  tol = 0,
  non_negative = list(FALSE, FALSE),
  gen_lhs = NULL,
  gen_rhs = NULL
) {
  verify_grad_biv_scalar(
    f, g,
    ndims = 0L,
    dtypes = dtypes,
    args_f = args_f,
    tol = tol,
    non_negative = non_negative,
    gen_lhs = gen_lhs,
    gen_rhs = gen_rhs
  )
  verify_grad_biv_tensor(
    f,
    g,
    ndims = ndims,
    dtypes = dtypes,
    args_f = args_f,
    tol = tol,
    non_negative = non_negative,
    gen_lhs = gen_lhs,
    gen_rhs = gen_rhs
  )
}

verify_grad_uni <- function(
  f,
  g,
  ndims = sample(1:3, 1L),
  dtypes = "f32",
  args_f = NULL,
  tol = 0,
  non_negative = FALSE,
  skip_scalar = FALSE,
  gen = NULL
) {
  if (!skip_scalar) {
    verify_grad_uni_scalar(
      f, g,
      ndims = 0L,
      dtypes = dtypes,
      args_f = args_f,
      tol = tol,
      non_negative = non_negative,
      gen = gen
    )
  }
  verify_grad_uni_tensor(
    f,
    g,
    ndims = ndims,
    dtypes = dtypes,
    args_f = args_f,
    tol = tol,
    non_negative = non_negative,
    gen = gen
  )
}

test_that("p_add", {
  verify_grad_biv(nvl_add, torch::torch_add)
})

test_that("p_sub", {
  verify_grad_biv(nvl_sub, torch::torch_sub)
})

test_that("p_mul", {
  verify_grad_biv(nvl_mul, torch::torch_mul)
})

test_that("p_neg", {
  verify_grad_uni(nvl_neg, torch::torch_neg)
})

test_that("p_exp", {
  withr::local_seed(12)
  verify_grad_uni(nvl_exp, torch::torch_exp)
})

test_that("p_log", {
  verify_grad_uni(nvl_log, torch::torch_log)
})

test_that("p_div", {
  # TODO:
  # Need to determine what to do with non-differentiable values:
  # https://docs.pytorch.org/docs/stable/notes/autograd.html#gradients-for-non-differentiable-functions
  verify_grad_biv(nvl_div, torch::torch_div)
})

test_that("p_pow", {
  # TODO:
  # Need to determine what to do with non-differentiable values:
  # https://docs.pytorch.org/docs/stable/notes/autograd.html#gradients-for-non-differentiable-functions
  verify_grad_biv(nvl_pow, torch::torch_pow, non_negative = list(TRUE, FALSE), tol = 1e-5)
  verify_grad_biv(nvl_pow, torch::torch_pow, non_negative = list(FALSE, TRUE), tol = 1e-5)
})

test_that("p_reduce_sum", {
  x_arr <- array(1:6, c(2, 3))
  x <- nv_tensor(x_arr, dtype = "f32")
  f <- function(a) {
    y <- nvl_reduce_sum(a, dims = 2L, drop = TRUE)
    nvl_reduce_sum(y, dims = 1L, drop = TRUE)
  }
  grads <- jit(gradient(f))(x)
  expect_equal(tengen::as_array(grads[[1L]]), array(1, dim = c(2, 3)))
  # TODO: Also test with drop = FALSE
  f <- function(a) {
    y <- nvl_reduce_sum(a, dims = 2L, drop = FALSE)
    nvl_reduce_sum(y, dims = 1:2, drop = TRUE)
  }
  grads <- jit(gradient(f))(x)
  expect_equal(tengen::as_array(grads[[1L]]), array(1, dim = c(2, 3)))
})

test_that("p_transpose", {
  verify_grad_uni_tensor(nvl_transpose, \(x, permutation) x$permute(permutation), ndims = 3L, args_f = \(shp, dtype) {
    dims <- sample(seq_along(shp))
    list(
      list(permutation = dims),
      list(permutation = dims)
    )
  })
})

test_that("p_broadcast_in_dim", {
  input_shape <- c(2L, 1L, 3L)
  target_shape <- c(4L, 2L, 5L, 3L)

  f <- function(operand, shape) {
    x <- nv_broadcast_to(operand, shape)
    nv_reduce_sum(x, dims = seq_along(shape), drop = TRUE)
  }

  verify_grad_uni_tensor(
    nv_broadcast_to,
    \(x, shape) x$broadcast_to(shape),
    shape = input_shape,
    args_f = \(shp, dtype) {
      list(
        list(shape = target_shape),
        list(shape = target_shape)
      )
    }
  )
})

test_that("p_select", {
  shp <- c(2L, 3L)
  x_arr <- array(generate_test_data(shp, dtype = "pred"), shp)
  x_anvil <- nv_tensor(x_arr, dtype = "pred")
  x_torch <- torch::torch_tensor(x_arr, dtype = torch::torch_bool())

  a_arr <- array(generate_test_data(shp, dtype = "f32"), shp)
  b_arr <- array(generate_test_data(shp, dtype = "f32"), shp)
  a_anvil <- nv_tensor(a_arr, dtype = "f32")
  b_anvil <- nv_tensor(b_arr, dtype = "f32")
  a_torch <- torch::torch_tensor(a_arr, requires_grad = TRUE, dtype = torch::torch_float32())
  b_torch <- torch::torch_tensor(b_arr, requires_grad = TRUE, dtype = torch::torch_float32())

  f_anvil <- function(a, b) {
    out <- nvl_select(x_anvil, a, b)
    nv_reduce_sum(out, dims = 1:2, drop = TRUE)
  }
  grads <- jit(gradient(f_anvil))(a_anvil, b_anvil)

  out_t <- torch::torch_where(x_torch, a_torch, b_torch)
  torch::torch_sum(out_t)$backward()

  expect_equal(tengen::as_array(grads[[1L]]), as_array_torch(a_torch$grad), tolerance = 1e-6)
  expect_equal(tengen::as_array(grads[[2L]]), as_array_torch(b_torch$grad), tolerance = 1e-6)
})

test_that("p_reshape", {
  in_shape <- c(2L, 3L)
  out_shape <- c(3L, 2L)
  verify_grad_uni_tensor(
    nvl_reshape,
    function(x, shape) x$reshape(shape),
    shape = in_shape,
    args_f = function(shp, dtype) list(list(shape = out_shape), list(shape = out_shape))
  )
})

test_that("p_convert", {
  target_dtype <- "f64"
  verify_grad_uni_tensor(
    \(operand, dtype) nvl_convert(operand, dtype = dtype, ambiguous = FALSE),
    function(x, dtype) x$to(dtype = dtype),
    dtypes = "f32",
    args_f = function(shp, dtype) {
      list(
        list(dtype = target_dtype),
        list(dtype = torch::torch_float64())
      )
    }
  )
})

test_that("p_sqrt", {
  verify_grad_uni(nvl_sqrt, torch::torch_sqrt, non_negative = TRUE, tol = 1e-5)
})

test_that("p_rsqrt", {
  # f64 to avoid log message
  verify_grad_uni(nvl_rsqrt, torch::torch_rsqrt, non_negative = TRUE, tol = 1e-5, dtypes = "f64")
})

test_that("p_tanh", {
  withr::local_seed(12)
  verify_grad_uni(nvl_tanh, torch::torch_tanh, tol = 1e-4)
})

test_that("p_tan", {
  # values near pi/2 cause divergence -> avoid unlucky seed
  withr::local_seed(12)
  verify_grad_uni_tensor(
    nvl_tan,
    torch::torch_tan,
    tol = 1e-4
  )
})

test_that("p_sine", {
  verify_grad_uni(nvl_sine, torch::torch_sin, tol = 1e-5)
})

test_that("p_cosine", {
  verify_grad_uni(nvl_cosine, torch::torch_cos, tol = 1e-5)
})

test_that("p_abs", {
  verify_grad_uni_tensor(
    nvl_abs,
    torch::torch_abs,
    tol = 1e-5
  )
})

test_that("p_max", {
  verify_grad_biv(nvl_max, torch::torch_maximum, tol = 1e-5)
})

test_that("p_min", {
  verify_grad_biv(nvl_min, torch::torch_minimum, tol = 1e-5)
})

test_that("p_floor", {
  verify_grad_uni(nvl_floor, torch::torch_floor)
})

test_that("p_ceil", {
  verify_grad_uni(nvl_ceil, torch::torch_ceil)
})

test_that("p_sign", {
  verify_grad_uni(nvl_sign, torch::torch_sign)
})

test_that("p_round", {
  verify_grad_uni(nvl_round, torch::torch_round)
})

test_that("p_cbrt", {
  verify_grad_uni(
    nvl_cbrt,
    \(x) torch::torch_pow(x, 1 / 3),
    non_negative = TRUE,
    tol = 1e-4
  )
})

test_that("p_expm1", {
  withr::local_seed(12)
  verify_grad_uni(nvl_expm1, torch::torch_expm1, tol = 1e-5)
})

test_that("p_log1p", {
  verify_grad_uni(nvl_log1p, torch::torch_log1p, non_negative = TRUE, tol = 1e-5)
})

test_that("p_logistic", {
  verify_grad_uni(nvl_logistic, torch::torch_sigmoid, tol = 1e-5)
})

test_that("p_clamp", {
  shp <- c(2L, 3L)
  dtype <- "f32"

  x_arr <- array(sample(c(-0.6, -0.5, -0.1, 0.0, 0.1, 0.5, 0.6), prod(shp), replace = TRUE), shp)
  x_nv <- nv_tensor(x_arr, dtype = dtype)
  x_th <- torch::torch_tensor(x_arr, requires_grad = TRUE, dtype = torch::torch_float32())

  min_val <- -0.5
  max_val <- 0.5

  f_nv <- function(x) {
    y <- nvl_clamp(min_val, x, max_val)
    nv_reduce_sum(y, dims = seq_len(ndims(y)), drop = TRUE)
  }

  grads_nv <- jit(gradient(f_nv))(x_nv)

  out_th <- torch::torch_clamp(x_th, min = min_val, max = max_val)
  torch::torch_sum(out_th)$backward()

  expect_equal(
    tengen::as_array(grads_nv[[1L]]),
    as_array_torch(x_th$grad),
    tolerance = 1e-5
  )
})

test_that("p_reverse", {
  verify_grad_uni(
    nvl_reverse,
    torch::torch_flip,
    ndims = 3L,
    args_f = \(shp, dtype) {
      dims_to_reverse <- sample(seq_along(shp), size = sample.int(length(shp), 1L))
      list(
        list(dims = dims_to_reverse),
        list(dims = dims_to_reverse)
      )
    },
    tol = 1e-5,
    skip_scalar = TRUE
  )
})

test_that("p_atan2", {
  # Generator that avoids (0, 0) which is undefined
  gen_nonzero <- function(shp, dtype) {
    vals <- generate_test_data(shp, dtype = dtype)
    # Ensure we don't have both values near zero
    if (length(shp) == 0L) {
      if (abs(vals) < 0.1) vals <- vals + sign(vals + 0.1) * 0.5
    } else {
      vals[abs(vals) < 0.1] <- vals[abs(vals) < 0.1] + 0.5
    }
    if (length(shp) == 0L) vals else array(vals, shp)
  }

  verify_grad_biv(
    nvl_atan2,
    torch::torch_atan2,
    tol = 1e-5,
    gen_lhs = gen_nonzero,
    gen_rhs = gen_nonzero
  )
})

test_that("p_concatenate", {
  # Test concatenate gradient with torch_cat
  shp1 <- c(2L, 3L)
  shp2 <- c(2L, 4L)
  dtype <- "f32"

  x_arr <- array(generate_test_data(shp1, dtype = dtype), shp1)
  y_arr <- array(generate_test_data(shp2, dtype = dtype), shp2)

  x_nv <- nv_tensor(x_arr, dtype = dtype)
  y_nv <- nv_tensor(y_arr, dtype = dtype)

  x_th <- torch::torch_tensor(x_arr, requires_grad = TRUE, dtype = torch::torch_float32())
  y_th <- torch::torch_tensor(y_arr, requires_grad = TRUE, dtype = torch::torch_float32())

  f_nv <- function(x, y) {
    out <- nvl_concatenate(x, y, dimension = 2L)
    nv_reduce_sum(out, dims = seq_len(ndims(out)), drop = TRUE)
  }

  grads_nv <- jit(gradient(f_nv))(x_nv, y_nv)

  out_th <- torch::torch_cat(list(x_th, y_th), dim = 2)
  torch::torch_sum(out_th)$backward()

  expect_equal(tengen::as_array(grads_nv[[1L]]), as_array_torch(x_th$grad), tolerance = 1e-5)
  expect_equal(tengen::as_array(grads_nv[[2L]]), as_array_torch(y_th$grad), tolerance = 1e-5)
})

test_that("p_concatenate with 3 inputs", {
  shp1 <- c(2L, 2L)
  shp2 <- c(2L, 3L)
  shp3 <- c(2L, 1L)
  dtype <- "f32"

  x_arr <- array(generate_test_data(shp1, dtype = dtype), shp1)
  y_arr <- array(generate_test_data(shp2, dtype = dtype), shp2)
  z_arr <- array(generate_test_data(shp3, dtype = dtype), shp3)

  x_nv <- nv_tensor(x_arr, dtype = dtype)
  y_nv <- nv_tensor(y_arr, dtype = dtype)
  z_nv <- nv_tensor(z_arr, dtype = dtype)

  x_th <- torch::torch_tensor(x_arr, requires_grad = TRUE)
  y_th <- torch::torch_tensor(y_arr, requires_grad = TRUE)
  z_th <- torch::torch_tensor(z_arr, requires_grad = TRUE)

  f_nv <- function(x, y, z) {
    out <- nvl_concatenate(x, y, z, dimension = 2L)
    nv_reduce_sum(out, dims = seq_len(ndims(out)), drop = TRUE)
  }

  grads_nv <- jit(gradient(f_nv))(x_nv, y_nv, z_nv)

  out_th <- torch::torch_cat(list(x_th, y_th, z_th), dim = 2)
  torch::torch_sum(out_th)$backward()

  expect_equal(tengen::as_array(grads_nv[[1L]]), as_array_torch(x_th$grad), tolerance = 1e-5)
  expect_equal(tengen::as_array(grads_nv[[2L]]), as_array_torch(y_th$grad), tolerance = 1e-5)
  expect_equal(tengen::as_array(grads_nv[[3L]]), as_array_torch(z_th$grad), tolerance = 1e-5)
})

test_that("p_reduce_prod", {
  # Test with non-zero values to avoid division by zero in gradient
  gen_nonzero <- function(shp, dtype) {
    vals <- generate_test_data(shp, dtype = dtype)
    # Shift values away from zero
    if (length(shp) == 0L) {
      if (abs(vals) < 0.5) vals <- vals + sign(vals + 0.1) * 1
    } else {
      vals[abs(vals) < 0.5] <- vals[abs(vals) < 0.5] + sign(vals[abs(vals) < 0.5] + 0.1) * 1
    }
    if (length(shp) == 0L) vals else array(vals, shp)
  }

  shp <- c(2L, 3L)
  dtype <- "f32"

  x_arr <- gen_nonzero(shp, dtype)
  x_nv <- nv_tensor(x_arr, dtype = dtype)
  x_th <- torch::torch_tensor(x_arr, requires_grad = TRUE, dtype = torch::torch_float32())

  # Test reduce along one dimension
  f_nv <- function(x) {
    y <- nvl_reduce_prod(x, dims = 2L, drop = TRUE)
    nv_reduce_sum(y, dims = 1L, drop = TRUE)
  }

  grads_nv <- jit(gradient(f_nv))(x_nv)

  out_th <- torch::torch_prod(x_th, dim = 2, keepdim = FALSE)
  torch::torch_sum(out_th)$backward()

  expect_equal(tengen::as_array(grads_nv[[1L]]), as_array_torch(x_th$grad), tolerance = 1e-4)
})

test_that("p_remainder", {
  # Test remainder gradient
  # Using fmod in torch which is equivalent to stablehlo remainder
  # Only test lhs gradient since rhs gradient involves floor which has discontinuities
  shp <- c(2L, 3L)
  dtype <- "f32"

  # Generator for lhs - any values
  gen_lhs <- function(shp, dtype) {
    vals <- runif(prod(shp), min = -5, max = 5)
    if (length(shp) == 0L) vals else array(vals, shp)
  }
  # Generator for rhs - non-zero values with same sign to avoid discontinuities
  gen_rhs <- function(shp, dtype) {
    vals <- runif(prod(shp), min = 1, max = 3)
    if (length(shp) == 0L) vals else array(vals, shp)
  }

  x_arr <- gen_lhs(shp, dtype)
  y_arr <- gen_rhs(shp, dtype)

  x_nv <- nv_tensor(x_arr, dtype = dtype)
  y_nv <- nv_tensor(y_arr, dtype = dtype)

  x_th <- torch::torch_tensor(x_arr, requires_grad = TRUE, dtype = torch::torch_float32())
  y_th <- torch::torch_tensor(y_arr, requires_grad = TRUE, dtype = torch::torch_float32())

  f_nv <- function(x, y) {
    out <- nvl_remainder(x, y)
    nv_reduce_sum(out, dims = seq_len(ndims(out)), drop = TRUE)
  }

  grads_nv <- jit(gradient(f_nv))(x_nv, y_nv)

  # torch::fmod is equivalent to stablehlo remainder
  out_th <- torch::torch_fmod(x_th, y_th)
  torch::torch_sum(out_th)$backward()

  expect_equal(tengen::as_array(grads_nv[[1L]]), as_array_torch(x_th$grad), tolerance = 1e-5)
  expect_equal(tengen::as_array(grads_nv[[2L]]), as_array_torch(y_th$grad), tolerance = 1e-5)
})

test_that("p_slice", {
  # Test slice gradient using torch narrow/slice
  shp <- c(4L, 5L)
  dtype <- "f32"

  x_arr <- array(generate_test_data(shp, dtype = dtype), shp)
  x_nv <- nv_tensor(x_arr, dtype = dtype)
  x_th <- torch::torch_tensor(x_arr, requires_grad = TRUE, dtype = torch::torch_float32())

  # Slice from [2, 2] to [4, 4] with stride 1
  # anvil: start_indices is 1-based, limit_indices is the exclusive end (in the same coordinate system as stablehlo)
  # For a slice of length 2 starting at index 2 (1-based):
  # - 0-based start = 1, 0-based limit = 3 (exclusive) -> elements at 0-based indices 1, 2
  # In anvil: start_indices = 2 (1-based), limit_indices = 4 (which is 0-based limit 4, not 0-based limit 3)
  # Wait, let me recalculate. The stablehlo limit is passed through directly.
  # anvil start=2 -> 0-based start=1, anvil limit=4 -> 0-based limit=4
  # So we get elements 1,2,3 (3 elements)
  # To get 2 elements starting at index 2 (1-based), we need limit = start + length = 2 + 2 = 4
  # But 0-based that would be start=1, limit=3. So anvil limit should be 3... hmm

  # Let me check: nv_slice(x, start=1, limit=2, stride=1) on c(2,3) gives shape c(2,2)
  # So limit=2 gives 0-based indices [0, 1] for that dim, which is start=0, limit=2
  # This means anvil's limit_indices IS the 0-based exclusive end!

  # So to slice rows 2,3 (1-based) = 0-based indices 1,2 = 0-based [1:3):
  # anvil start=2 (converted to 0-based start=1), anvil limit=4 (0-based limit=4)
  # But we want 0-based limit=3 not 4. So anvil limit should be 3.

  # Actually wait - let me reread the primitive. limit_attr = r_to_constant(limit_indices, ...)
  # So limit_indices IS passed through as-is to stablehlo, which expects 0-based.
  # So if we want 0-based exclusive limit=4, we pass limit_indices=4.

  # For this test: we want rows 2,3 and cols 2,3 (1-based) = 2x2 output
  # 0-based: rows 1,2 and cols 1,2 -> 0-based range [1:3) x [1:3)
  # anvil: start_indices=c(2,2), limit_indices=c(4,4) -> 0-based start=[1,1], 0-based limit=[4,4]
  # That gives [1:4) x [1:4) = 3x3 output, not 2x2!

  # The issue is anvil's start-to-0-based conversion subtracts 1, but limit is passed through.
  # So for consistent 1-based slicing with exclusive end:
  # To get 1-based range [2:4) (exclusive) = elements 2,3, which is 2 elements:
  # We need 0-based [1:3) which is elements 1,2 (0-based).
  # anvil start=2 -> 0-based start=1. anvil limit=4 -> 0-based limit=4. That's wrong.
  # Actually anvil limit should be 4 for the 1-based exclusive end 4, but with start conversion the math doesn't work.

  # Let me just test what actually works based on the existing test:
  # nv_slice(x, c(1,1), c(2,2), c(1,1)) on c(2,3) gives c(2,2) output
  # 0-based: start=[0,0], limit=[2,2] -> [0:2) x [0:2) = 2x2. Correct!
  # So limit_indices in anvil is the 0-based exclusive end, start_indices is 1-based start.
  # This is inconsistent but it's how it is.

  # For rows 2:3 (1-based, inclusive) = 0-based 1:2 (inclusive) = 0-based [1:3) exclusive
  # anvil start=2, anvil limit=3 (since limit is already 0-based exclusive, we want 3)
  # Actually no - the slice test shows limit=c(2,2) gives 0-based limit c(2,2).

  # OK I think the confusion is: limit_indices IS the value that goes to stablehlo.
  # So to slice rows with 0-based indices 1,2 (2 elements), we need 0-based limit=3.
  # anvil limit_indices=3 -> 0-based limit=3. Start anvil=2 -> 0-based start=1.
  # Slice [1:3) gives indices 1,2, that's 2 elements. Correct!

  # So for 2 rows starting at row 2 (1-based): start=2, limit=2+2=4 NO WAIT
  # 0-based: we want indices 1,2 -> [1:3). So 0-based limit=3.
  # anvil limit=3 (this is the 0-based limit directly).
  # anvil start=2 -> 0-based start=1.
  # YES! So limit_indices = start_indices + length - 1 + 1 = start + length = 4? No...
  # Let me just compute: 0-based limit = 0-based start + length = 1 + 2 = 3.
  # Since anvil limit = 0-based limit, we have anvil limit = 3.
  # And anvil start = 1-based start = 2.

  # So for 2 elements starting at 1-based index 2:
  # anvil start = 2, anvil limit = 0-based-start + length = (2-1) + 2 = 3

  start_indices <- c(2L, 2L)
  limit_indices <- c(4L, 4L)  # 0-based limit = 0-based start + length = 1 + 3 = 4, so 3 elements
  strides <- c(1L, 1L)

  f_nv <- function(x) {
    out <- nvl_slice(x, start_indices, limit_indices, strides)
    nv_reduce_sum(out, dims = seq_len(ndims(out)), drop = TRUE)
  }

  grads_nv <- jit(gradient(f_nv))(x_nv)

  # torch narrow(dim, start, length) - start is 1-based
  # We want 3 elements starting at index 2 (1-based), so narrow(1, 2, 3)
  out_th <- x_th$narrow(1, 2, 3)$narrow(2, 2, 3)
  torch::torch_sum(out_th)$backward()

  expect_equal(tengen::as_array(grads_nv[[1L]]), as_array_torch(x_th$grad), tolerance = 1e-5)
})

test_that("p_slice with strides", {
  # Test slice gradient with non-unit strides
  shp <- c(6L, 8L)
  dtype <- "f32"

  x_arr <- array(generate_test_data(shp, dtype = dtype), shp)
  x_nv <- nv_tensor(x_arr, dtype = dtype)
  x_th <- torch::torch_tensor(x_arr, requires_grad = TRUE, dtype = torch::torch_float32())

  # Slice from start_indices with strides
  # anvil: start_indices is 1-based, limit_indices is 0-based exclusive
  # For start=1 (1-based) -> 0-based start=0
  # For limit=6 -> 0-based limit=6
  # With stride=2: 0-based indices 0, 2, 4 (3 elements)
  start_indices <- c(1L, 1L)
  limit_indices <- c(6L, 8L)  # 0-based limits
  strides <- c(2L, 2L)

  f_nv <- function(x) {
    out <- nvl_slice(x, start_indices, limit_indices, strides)
    nv_reduce_sum(out, dims = seq_len(ndims(out)), drop = TRUE)
  }

  grads_nv <- jit(gradient(f_nv))(x_nv)

  # In torch, slice with strides using index_select
  # 0-based indices with stride 2 starting at 0: [0, 2, 4] for dim 1 (6 elements, limit 6)
  # 0-based indices with stride 2 starting at 0: [0, 2, 4, 6] for dim 2 (8 elements, limit 8)
  idx1 <- torch::torch_tensor(c(1L, 3L, 5L), dtype = torch::torch_int64())  # 1-based for torch
  idx2 <- torch::torch_tensor(c(1L, 3L, 5L, 7L), dtype = torch::torch_int64())
  out_th <- x_th$index_select(1, idx1)$index_select(2, idx2)
  torch::torch_sum(out_th)$backward()

  expect_equal(tengen::as_array(grads_nv[[1L]]), as_array_torch(x_th$grad), tolerance = 1e-5)
})

# Tests for non-differentiable operations that return zero gradients
# These are bitwise/logical operations where gradients are mathematically zero
# All need to reduce to scalar output for gradient computation

# Note: For boolean operations (p_and, p_or, p_xor, p_reduce_all, p_reduce_any),
# the backward rules return zeros but cannot be directly tested since:
# 1. Boolean inputs don't support gradients (gradients must be float)
# 2. zeros_like on boolean creates boolean tensors, which can't be used as gradients
# The backward rules exist for completeness but these ops are essentially non-differentiable.

test_that("p_and backward rule exists", {
  # Verify the backward rule is defined (returns zeros for boolean ops)
  expect_true(!is.null(p_and[["backward"]]))
})

test_that("p_or backward rule exists", {
  expect_true(!is.null(p_or[["backward"]]))
})

test_that("p_xor backward rule exists", {
  expect_true(!is.null(p_xor[["backward"]]))
})

test_that("p_reduce_all backward rule exists", {
  expect_true(!is.null(p_reduce_all[["backward"]]))
})

test_that("p_reduce_any backward rule exists", {
  expect_true(!is.null(p_reduce_any[["backward"]]))
})

test_that("p_is_finite", {
  # is_finite returns zero gradients
  # Test with a mix of finite and non-finite values
  x <- nv_tensor(c(1.0, Inf, -Inf, NaN, 0.5, -0.5), dtype = "f32")

  f <- function(x) {
    out <- nvl_is_finite(x)
    out <- nv_convert(out, "f32")
    nv_reduce_sum(out, dims = 1L, drop = TRUE)
  }

  g <- jit(gradient(f))
  grads <- g(x)

  expect_equal(tengen::as_array(grads[[1L]]), array(rep(0, 6), dim = 6L))
})

test_that("p_popcnt", {
  # popcnt (population count) returns zero gradients
  x <- nv_tensor(c(0L, 1L, 3L, 7L, 15L), dtype = "i32")

  f <- function(x) {
    out <- nvl_popcnt(x)
    out <- nv_convert(out, "f32")
    nv_reduce_sum(out, dims = 1L, drop = TRUE)
  }

  g <- jit(gradient(f))
  grads <- g(x)

  expect_equal(tengen::as_array(grads[[1L]]), array(rep(0, 5), dim = 5L))
})

test_that("p_shift_left", {
  # Bit shift operations return zero gradients
  x <- nv_tensor(c(1L, 2L, 4L, 8L), dtype = "i32")
  y <- nv_tensor(c(1L, 1L, 1L, 1L), dtype = "i32")

  f <- function(x, y) {
    out <- nvl_shift_left(x, y)
    out <- nv_convert(out, "f32")
    nv_reduce_sum(out, dims = 1L, drop = TRUE)
  }

  g <- jit(gradient(f))
  grads <- g(x, y)

  expect_equal(tengen::as_array(grads[[1L]]), array(rep(0, 4), dim = 4L))
  expect_equal(tengen::as_array(grads[[2L]]), array(rep(0, 4), dim = 4L))
})

test_that("p_shift_right_arithmetic", {
  # Arithmetic right shift returns zero gradients
  x <- nv_tensor(c(8L, 16L, 32L, -8L), dtype = "i32")
  y <- nv_tensor(c(1L, 2L, 1L, 1L), dtype = "i32")

  f <- function(x, y) {
    out <- nvl_shift_right_arithmetic(x, y)
    out <- nv_convert(out, "f32")
    nv_reduce_sum(out, dims = 1L, drop = TRUE)
  }

  g <- jit(gradient(f))
  grads <- g(x, y)

  expect_equal(tengen::as_array(grads[[1L]]), array(rep(0, 4), dim = 4L))
  expect_equal(tengen::as_array(grads[[2L]]), array(rep(0, 4), dim = 4L))
})

test_that("p_shift_right_logical", {
  # Logical right shift returns zero gradients
  x <- nv_tensor(c(8L, 16L, 32L, 64L), dtype = "i32")
  y <- nv_tensor(c(1L, 2L, 1L, 2L), dtype = "i32")

  f <- function(x, y) {
    out <- nvl_shift_right_logical(x, y)
    out <- nv_convert(out, "f32")
    nv_reduce_sum(out, dims = 1L, drop = TRUE)
  }

  g <- jit(gradient(f))
  grads <- g(x, y)

  expect_equal(tengen::as_array(grads[[1L]]), array(rep(0, 4), dim = 4L))
  expect_equal(tengen::as_array(grads[[2L]]), array(rep(0, 4), dim = 4L))
})

test_that("p_bitcast_convert", {
  # bitcast_convert reinterprets bits, returns zero gradients
  x <- nv_tensor(c(1.0, 2.0, 3.0, 4.0), dtype = "f32")

  f <- function(x) {
    # bitcast from f32 to i32 and back
    out <- nvl_bitcast_convert(x, dtype = "i32")
    out <- nv_convert(out, "f32")
    nv_reduce_sum(out, dims = 1L, drop = TRUE)
  }

  g <- jit(gradient(f))
  grads <- g(x)

  expect_equal(tengen::as_array(grads[[1L]]), array(rep(0, 4), dim = 4L))
})
