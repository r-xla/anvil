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

verify_grad_uni_scalar <- function(.f, .g, ndims = 0L, dtypes = "f32", args_f = NULL, tol = 0, non_negative = FALSE) {
  dtype <- sample(dtypes, 1L)
  shp <- integer()
  operand <- generate_test_data(integer(), dtype, non_negative = non_negative)

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
  non_negative = FALSE
) {
  shp <- if (is.null(shape)) sample(1:3, ndims, replace = TRUE) else shape
  dtype <- sample(dtypes, 1L)
  operand <- array(
    generate_test_data(shp, dtype = dtype, non_negative = non_negative),
    shp
  )

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
  non_negative = list(FALSE, FALSE)
) {
  dtype <- sample(dtypes, 1L)
  shp <- integer()

  if (length(non_negative) < 2) {
    non_negative <- rep(non_negative, 2)
  }

  lhs <- generate_test_data(integer(), dtype, non_negative = non_negative[[1]])
  rhs <- generate_test_data(integer(), dtype, non_negative = non_negative[[2]])

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
  non_negative = list(FALSE, FALSE)
) {
  # Prefer shapes without size-0 or size-1 axes to avoid backend broadcast edge-cases
  shp <- if (is.null(shape)) sample(1:3, ndims, replace = TRUE) else shape
  dtype <- sample(dtypes, 1)

  if (length(non_negative) < 2) {
    non_negative <- rep(non_negative, 2)
  }

  lhs <- array(
    generate_test_data(shp, dtype = dtype, non_negative = non_negative[[1]]), # nolint
    shp
  )
  rhs <- array(
    generate_test_data(shp, dtype = dtype, non_negative = non_negative[[2]]), # nolint
    shp
  )

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
  non_negative = list(FALSE, FALSE)
) {
  verify_grad_biv_scalar(f, g, ndims = 0L, dtypes = dtypes, args_f = args_f, tol = tol, non_negative = non_negative)
  verify_grad_biv_tensor(
    f,
    g,
    ndims = ndims,
    dtypes = dtypes,
    args_f = args_f,
    tol = tol,
    non_negative = non_negative
  )
}

verify_grad_uni <- function(
  f,
  g,
  ndims = sample(1:3, 1L),
  dtypes = "f32",
  args_f = NULL,
  tol = 0,
  non_negative = FALSE
) {
  verify_grad_uni_scalar(f, g, ndims = 0L, dtypes = dtypes, args_f = args_f, tol = tol, non_negative = non_negative)
  verify_grad_uni_tensor(
    f,
    g,
    ndims = ndims,
    dtypes = dtypes,
    args_f = args_f,
    tol = tol,
    non_negative = non_negative
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
    nvl_convert,
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
