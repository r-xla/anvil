#skip_if_not_installed("torch")

str_to_torch_dtype <- function(str) {
  switch(
    str,
    "pred" = torch::torch_bool(),
    "f32" = torch::torch_float32(),
    "f64" = torch::torch_float64(),
    "i8" = torch::torch_int8(),
    "i16" = torch::torch_int16(),
    "i32" = torch::torch_int32(),
    "i64" = torch::torch_int64(),
    "ui8" = torch::torch_uint8(),
    stop(sprintf("Unsupported dtype: %s", str))
  )
}

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
    if (length(args_torch) && ("dim" %in% names(args_torch) || "dims" %in% names(args_torch))) {
      dims_t <- if ("dim" %in% names(args_torch)) args_torch$dim else args_torch$dims
      num_remaining <- max(0L, num_remaining - length(dims_t))
    }
    if (num_remaining > 0L) torch::torch_sum(x, dim = seq_len(num_remaining)) else x
  }
}

verify_grad_uni_scalar <- function(.f, .g, ndims = 0L, dtypes = "f32", args_f = NULL) {
  dtype <- sample(dtypes, 1L)
  shp <- integer()
  operand <- generate_test_scalar(dtype)

  operand_anvil <- nv_scalar(operand, dtype = dtype)

  # I think there is a bug in torch, so we can't use torch_scalar_tensor
  operand_torch <- torch::torch_tensor(operand, requires_grad = TRUE, dtype = str_to_torch_dtype(dtype))

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
  out <- .g_torch(operand_torch$squeeze())
  out$backward(retrain_graph = TRUE)

  testthat::expect_equal(
    c(as_array(grads_anvil[[1L]])),
    c(torch::as_array(operand_torch$grad))
  )
}

verify_grad_uni_tensor <- function(.f, .g, ndims = sample(1:3, 1L), dtypes = "f32", args_f = NULL) {
  shp <- sample(1:3, ndims, replace = TRUE)
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
    c(as_array(grads_anvil[[1L]])),
    c(torch::as_array(operand_torch$grad))
  )
}

verify_grad_biv_scalar <- function(.f, .g, ndims = 0L, dtypes = "f32", args_f = NULL) {
  dtype <- sample(dtypes, 1L)
  shp <- integer()

  lhs <- generate_test_scalar(dtype)
  rhs <- generate_test_scalar(dtype)

  lhs_anvil <- nv_scalar(lhs, dtype = dtype)
  rhs_anvil <- nv_scalar(rhs, dtype = dtype)

  # I think there is a bug in torch, so we can't use torch_scalar_tensor
  lhs_torch <- torch::torch_tensor(lhs, requires_grad = TRUE, dtype = str_to_torch_dtype(dtype))
  rhs_torch <- torch::torch_tensor(rhs, requires_grad = TRUE, dtype = str_to_torch_dtype(dtype))

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
  out <- .g_torch(lhs_torch$squeeze(), rhs_torch$squeeze())
  out$backward(retrain_graph = TRUE)

  testthat::expect_equal(
    c(as_array(grads_anvil[[1L]])),
    c(torch::as_array(lhs_torch$grad))
  )

  testthat::expect_equal(
    c(as_array(grads_anvil[[2L]])),
    c(torch::as_array(rhs_torch$grad))
  )
}

verify_grad_biv_tensor <- function(.f, .g, ndims = sample(1:3, 1L), dtypes = "f32", args_f = NULL) {
  # Prefer shapes without size-0 or size-1 axes to avoid backend broadcast edge-cases
  shp <- sample(1:3, ndims, replace = TRUE)
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
    c(as_array(grads_anvil[[1L]])),
    c(torch::as_array(lhs_torch$grad))
  )

  testthat::expect_equal(
    c(as_array(grads_anvil[[2L]])),
    c(torch::as_array(rhs_torch$grad))
  )
}

verify_grad_biv <- function(f, g, ndims = sample(1:3, 1L), dtypes = "f32", args_f = NULL) {
  verify_grad_biv_scalar(f, g, ndims = 0L, dtypes = dtypes, args_f = args_f)
  verify_grad_biv_tensor(f, g, ndims = ndims, dtypes = dtypes, args_f = args_f)
}

verify_grad_uni <- function(f, g, ndims = sample(1:3, 1L), dtypes = "f32", args_f = NULL) {
  verify_grad_uni_scalar(f, g, ndims = 0L, dtypes = dtypes, args_f = args_f)
  verify_grad_uni_tensor(f, g, ndims = ndims, dtypes = dtypes, args_f = args_f)
}

test_that("add", {
  verify_grad_biv(nvl_add, torch::torch_add)
})

test_that("sub", {
  verify_grad_biv(nvl_sub, torch::torch_sub)
})

test_that("mul", {
  verify_grad_biv(nvl_mul, torch::torch_mul)
})

test_that("neg", {
  verify_grad_uni(nvl_neg, torch::torch_neg)
})

test_that("div", {
  verify_grad_biv(nvl_div, torch::torch_div)
})

test_that("reduce_sum", {
  # TODO: add drop argument
  verify_grad_uni(nvl_reduce_sum, torch::torch_sum, args_f = \(shp, dtype) {
    dims <- sample(seq_along(shp), sample(length(shp), 1L))
    drop <- sample(c(TRUE, FALSE), 1L)
    list(
      list(dims = dims, drop = drop),
      list(dim = dims, keepdim = !drop)
    )
  })
})
