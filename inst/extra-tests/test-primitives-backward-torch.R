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
    f,
    g,
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
      f,
      g,
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

verify_zero_grad_unary <- function(nvl_fn, x, f_wrapper = NULL) {
  if (is.null(f_wrapper)) {
    f <- function(x) {
      out <- nvl_fn(x)
      out <- nv_convert(out, "f32")
      nv_reduce_sum(out, dims = 1L, drop = TRUE)
    }
  } else {
    f <- f_wrapper
  }
  grads <- jit(gradient(f))(x)
  shp <- shape(x)
  testthat::expect_equal(tengen::as_array(grads[[1L]]), array(rep(0, prod(shp)), dim = shp))
}

verify_zero_grad_binary <- function(nvl_fn, x, y) {
  f <- function(x, y) {
    out <- nvl_fn(x, y)
    out <- nv_convert(out, "f32")
    nv_reduce_sum(out, dims = 1L, drop = TRUE)
  }
  grads <- jit(gradient(f))(x, y)
  shp <- shape(x)
  testthat::expect_equal(tengen::as_array(grads[[1L]]), array(rep(0, prod(shp)), dim = shp))
  testthat::expect_equal(tengen::as_array(grads[[2L]]), array(rep(0, prod(shp)), dim = shp))
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

test_that("p_negate", {
  verify_grad_uni(nvl_negate, torch::torch_neg)
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
  x_arr <- generate_test_data(shp, dtype = "pred")
  x_anvil <- nv_tensor(x_arr, dtype = "pred")
  x_torch <- torch::torch_tensor(x_arr, dtype = torch::torch_bool())

  a_arr <- generate_test_data(shp, dtype = "f32")
  b_arr <- generate_test_data(shp, dtype = "f32")
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
  verify_grad_concatenate <- function(shapes, dimension = 2L, dtype = "f32", tol = 1e-5) {
    n <- length(shapes)
    arrs <- lapply(shapes, function(shp) generate_test_data(shp, dtype = dtype))
    nvs <- lapply(arrs, function(arr) nv_tensor(arr, dtype = dtype))
    ths <- lapply(arrs, function(arr) torch::torch_tensor(arr, requires_grad = TRUE))

    f_nv <- function(...) {
      args <- list(...)
      out <- do.call(nvl_concatenate, c(args, list(dimension = dimension)))
      nv_reduce_sum(out, dims = seq_len(ndims(out)), drop = TRUE)
    }

    grads_nv <- do.call(jit(gradient(f_nv)), nvs)

    out_th <- torch::torch_cat(ths, dim = dimension)
    torch::torch_sum(out_th)$backward()

    for (i in seq_len(n)) {
      testthat::expect_equal(
        tengen::as_array(grads_nv[[i]]),
        as_array_torch(ths[[i]]$grad),
        tolerance = tol
      )
    }
  }
  verify_grad_concatenate(list(c(2L, 3L), c(2L, 4L)))
  verify_grad_concatenate(list(c(2L, 2L), c(2L, 3L), c(2L, 1L)))
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

describe("p_static_slice", {
  verify_slice_grad <- function(shp, start_indices, limit_indices, strides, torch_slice_fn) {
    dtype <- "f32"
    x_arr <- generate_test_data(shp, dtype = dtype)
    x_nv <- nv_tensor(x_arr, dtype = dtype)
    x_th <- torch::torch_tensor(x_arr, requires_grad = TRUE, dtype = torch::torch_float32())

    f_nv <- function(x) {
      out <- nvl_static_slice(x, start_indices, limit_indices, strides)
      nv_reduce_sum(out, dims = seq_len(ndims(out)), drop = TRUE)
    }

    grads_nv <- jit(gradient(f_nv))(x_nv)
    out_th <- torch_slice_fn(x_th)
    torch::torch_sum(out_th)$backward()

    testthat::expect_equal(tengen::as_array(grads_nv[[1L]]), as_array_torch(x_th$grad), tolerance = 1e-5)
  }

  it("works with unit strides", {
    verify_slice_grad(
      c(4L, 5L),
      c(2L, 2L),
      c(4L, 4L),
      c(1L, 1L),
      \(x) x$narrow(1, 2, 3)$narrow(2, 2, 3)
    )
  })

  it("works with non-unit strides", {
    verify_slice_grad(
      c(6L, 8L),
      c(1L, 1L),
      c(6L, 8L),
      c(2L, 2L),
      \(x) {
        idx1 <- torch::torch_tensor(c(1L, 3L, 5L), dtype = torch::torch_int64())
        idx2 <- torch::torch_tensor(c(1L, 3L, 5L, 7L), dtype = torch::torch_int64())
        x$index_select(1, idx1)$index_select(2, idx2)
      }
    )
  })
})

describe("boolean ops", {
  expected_zeros <- nv_tensor(rep(0, 4), dtype = "f32")

  verify_bool_binary <- function(nvl_fn) {
    x <- nv_tensor(c(TRUE, FALSE, FALSE, TRUE))
    y <- nv_tensor(c(TRUE, TRUE, FALSE, FALSE))
    f <- function(x, y) {
      out <- nvl_fn(x, y)
      nv_convert(out, "f32")
    }
    verify_zero_grad_binary(f, x, y)
  }

  verify_bool_reduce <- function(nvl_fn) {
    x <- nv_tensor(c(1.0, 1.0, 0.0, 0.0), dtype = "f32")
    f <- function(x) {
      x_pred <- nv_convert(x, "pred")
      out <- nvl_fn(x_pred, dims = 1L, drop = TRUE)
      nv_convert(out, "f32")
    }
    grads <- jit(gradient(f))(x)
    testthat::expect_equal(grads[[1L]], expected_zeros)
  }

  it("p_and returns zero gradients", {
    verify_bool_binary(nvl_and)
  })
  it("p_or returns zero gradients", {
    verify_bool_binary(nvl_or)
  })
  it("p_xor returns zero gradients", {
    verify_bool_binary(nvl_xor)
  })
  it("p_reduce_all returns zero gradients", {
    verify_bool_reduce(nvl_reduce_all)
  })
  it("p_reduce_any returns zero gradients", {
    verify_bool_reduce(nvl_reduce_any)
  })
})

test_that("p_is_finite", {
  x <- nv_tensor(c(1.0, Inf, -Inf, NaN, 0.5, -0.5), dtype = "f32")
  verify_zero_grad_unary(nvl_is_finite, x)
})

test_that("p_popcnt", {
  x <- nv_tensor(c(0L, 1L, 3L, 7L, 15L), dtype = "i32")
  verify_zero_grad_unary(nvl_popcnt, x)
})

describe("shift ops", {
  it("p_shift_left returns zero gradients", {
    x <- nv_tensor(c(1L, 2L, 4L, 8L), dtype = "i32")
    y <- nv_tensor(c(1L, 1L, 1L, 1L), dtype = "i32")
    verify_zero_grad_binary(nvl_shift_left, x, y)
  })

  it("p_shift_right_arithmetic returns zero gradients", {
    x <- nv_tensor(c(8L, 16L, 32L, -8L), dtype = "i32")
    y <- nv_tensor(c(1L, 2L, 1L, 1L), dtype = "i32")
    verify_zero_grad_binary(nvl_shift_right_arithmetic, x, y)
  })

  it("p_shift_right_logical returns zero gradients", {
    x <- nv_tensor(c(8L, 16L, 32L, 64L), dtype = "i32")
    y <- nv_tensor(c(1L, 2L, 1L, 2L), dtype = "i32")
    verify_zero_grad_binary(nvl_shift_right_logical, x, y)
  })
})

test_that("p_bitcast_convert", {
  x <- nv_tensor(c(1.0, 2.0, 3.0, 4.0), dtype = "f32")
  verify_zero_grad_unary(nvl_bitcast_convert, x, f_wrapper = function(x) {
    out <- nvl_bitcast_convert(x, dtype = "i32")
    out <- nv_convert(out, "f32")
    nv_reduce_sum(out, dims = 1L, drop = TRUE)
  })
})
