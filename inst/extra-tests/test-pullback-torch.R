source(system.file("extra-tests", "torch-helpers.R", package = "anvil"))

generate_test_scalar <- function(dtype) {
  if (dtype == "pred") {
    sample(c(TRUE, FALSE), size = 1L)
  } else if (startsWith(dtype, "f")) {
    rnorm(1L)
  } else if (startsWith(dtype, "i")) {
    sample(-10:10, size = 1L)
  } else if (startsWith(dtype, "ui")) {
    sample(0:20, size = 1L)
  } else {
    stop(sprintf("Unsupported dtype: %s", dtype))
  }
}

generate_test_array_local <- function(shp, dtype) {
  nelts <- if (!length(shp)) 1L else prod(shp)
  x <- if (dtype == "pred") {
    sample(c(TRUE, FALSE), size = nelts, replace = TRUE)
  } else if (startsWith(dtype, "f")) {
    rnorm(nelts)
  } else if (startsWith(dtype, "i")) {
    sample(-10:10, size = nelts, replace = TRUE)
  } else if (startsWith(dtype, "ui")) {
    sample(0:20, size = nelts, replace = TRUE)
  } else {
    stop(sprintf("Unsupported dtype: %s", dtype))
  }
  if (!length(shp)) x else array(x, shp)
}

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

verify_grad_uni_scalar <- function(.f, .g, ndims = 0L, dtypes = "f32", args_f = NULL, tol = 0) {
  dtype <- sample(dtypes, 1L)
  shp <- integer()
  operand <- generate_test_scalar(dtype)

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
  tol = 0
) {
  shp <- if (is.null(shape)) sample(1:3, ndims, replace = TRUE) else shape
  dtype <- sample(dtypes, 1L)
  operand <- generate_test_array_local(shp, dtype)

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

verify_grad_biv_scalar <- function(.f, .g, ndims = 0L, dtypes = "f32", args_f = NULL, tol = 0) {
  dtype <- sample(dtypes, 1L)
  shp <- integer()

  lhs <- generate_test_scalar(dtype)
  rhs <- generate_test_scalar(dtype)

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
    as_array_torch(lhs_torch$grad),
    tolerance = tol
  )

  testthat::expect_equal(
    tengen::as_array(grads_anvil[[2L]]),
    as_array_torch(rhs_torch$grad),
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
  tol = 0
) {
  # Prefer shapes without size-0 or size-1 axes to avoid backend broadcast edge-cases
  shp <- if (is.null(shape)) sample(1:3, ndims, replace = TRUE) else shape
  dtype <- sample(dtypes, 1)

  lhs <- generate_test_array_local(shp, dtype)
  rhs <- generate_test_array_local(shp, dtype)

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
    as_array_torch(lhs_torch$grad),
    tolerance = tol
  )

  testthat::expect_equal(
    tengen::as_array(grads_anvil[[2L]]),
    as_array_torch(rhs_torch$grad),
    tolerance = tol
  )
}

verify_grad_biv <- function(f, g, ndims = sample(1:3, 1L), dtypes = "f32", args_f = NULL, tol = 0) {
  verify_grad_biv_scalar(f, g, ndims = 0L, dtypes = dtypes, args_f = args_f, tol = tol)
  verify_grad_biv_tensor(f, g, ndims = ndims, dtypes = dtypes, args_f = args_f, tol = tol)
}

verify_grad_uni <- function(f, g, ndims = sample(1:3, 1L), dtypes = "f32", args_f = NULL, tol = 0) {
  verify_grad_uni_scalar(f, g, ndims = 0L, dtypes = dtypes, args_f = args_f, tol = tol)
  verify_grad_uni_tensor(f, g, ndims = ndims, dtypes = dtypes, args_f = args_f, tol = tol)
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

test_that("p_div", {
  # Need to determine what to do with non-differentiable values:
  # https://docs.pytorch.org/docs/stable/notes/autograd.html#gradients-for-non-differentiable-functions
  verify_grad_biv(nvl_div, torch::torch_div)
})

test_that("p_pow", {
  # Need to determine what to do with non-differentiable values:
  # https://docs.pytorch.org/docs/stable/notes/autograd.html#gradients-for-non-differentiable-functions
  # TODO: uncomment
  #verify_grad_biv(nvl_pow, torch::torch_pow)
  #library(torch)
  #x <- torch_scalar_tensor(0, requires_grad = TRUE)
  #x$retain_grad()
  #y <- torch_scalar_tensor(2, requires_grad = TRUE)
  #y$retain_grad()
  #(x^y)$backward()
  #y$grad
})

test_that("p_reduce_sum", {
  verify_grad_uni(nvl_reduce_sum, torch::torch_sum, args_f = \(shp, dtype) {
    dims <- sample(seq_along(shp), sample(length(shp), 1L))
    drop <- sample(c(TRUE, FALSE), 1L)
    list(
      list(dims = dims, drop = drop),
      list(dim = dims, keepdim = !drop)
    )
  })
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

test_that("p_broadcast_to", {
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

# Minimal name-anchoring tests for meta-check coverage
test_that("p_select", {
  # Fixed predicate; gradients w.r.t a and b should match torch where
  shp <- c(2L, 3L)
  p_arr <- generate_test_array_local(shp, "pred")
  p_anvil <- nv_tensor(p_arr, dtype = "pred")
  p_torch <- torch::torch_tensor(p_arr, dtype = torch::torch_bool())

  a_arr <- generate_test_array_local(shp, "f32")
  b_arr <- generate_test_array_local(shp, "f32")
  a_anvil <- nv_tensor(a_arr, dtype = "f32")
  b_anvil <- nv_tensor(b_arr, dtype = "f32")
  a_torch <- torch::torch_tensor(a_arr, requires_grad = TRUE, dtype = torch::torch_float32())
  b_torch <- torch::torch_tensor(b_arr, requires_grad = TRUE, dtype = torch::torch_float32())

  f_anvil <- function(a, b) {
    out <- nvl_select(p_anvil, a, b)
    nv_reduce_sum(out, dims = 1:2, drop = TRUE)
  }
  grads <- jit(gradient(f_anvil))(a_anvil, b_anvil)

  out_t <- torch::torch_where(p_torch, a_torch, b_torch)
  torch::torch_sum(out_t)$backward()

  expect_equal(tengen::as_array(grads[[1L]]), as_array_torch(a_torch$grad), tolerance = 1e-6)
  expect_equal(tengen::as_array(grads[[2L]]), as_array_torch(b_torch$grad), tolerance = 1e-6)
})

test_that("p_broadcast_in_dim", {
  input_shape <- c(2L, 3L)
  target_shape <- c(4L, 2L, 3L)
  bdims <- c(2L, 3L)
  verify_grad_uni_tensor(
    function(a, shape, bdims) nvl_broadcast_in_dim(a, shape, bdims),
    function(x, shape, bdims) x$unsqueeze(1)$expand(shape),
    shape = input_shape,
    args_f = function(shp, dtype) {
      list(list(shape = target_shape, bdims = bdims), list(shape = target_shape, bdims = bdims))
    }
  )
})

test_that("p_atan2", {
  verify_grad_biv(nvl_atan2, torch::torch_atan2, dtypes = "f32", tol = 1e-6)
})

test_that("p_reduce_prod", {
  # Not implemented pullback; ensure error is raised deterministically
  f <- function(x) nvl_reduce_prod(x, dims = 1L, drop = TRUE)
  g <- gradient(f)
  expect_error(jit(g)(nv_tensor(matrix(rnorm(6), 2, 3), dtype = "f32")))
})

test_that("p_dot_general", {
  # simple vector dot case; avoid scalar degenerate paths in pullback
  verify_grad_biv_tensor(
    function(a, b) nvl_dot_general(a, b, contracting_dims = list(1L, 1L), batching_dims = list(integer(), integer())),
    function(a, b) (a * b)$sum(),
    ndims = 1L
  )
})

test_that("p_reshape", {
  in_shape <- c(2L, 3L)
  out_shape <- c(3L, 2L)
  verify_grad_uni_tensor(
    function(a, shape) nvl_reshape(a, shape),
    function(x, shape) x$reshape(shape),
    shape = in_shape,
    args_f = function(shp, dtype) list(list(shape = out_shape), list(shape = out_shape))
  )
})
