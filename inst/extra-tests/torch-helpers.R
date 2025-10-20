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

as_array_torch <- function(x) {
  if (!length(dim(x))) {
    torch::as_array(x)
  } else if (length(dim(x)) == 1L) {
    array(torch::as_array(x), dim = length(x))
  } else {
    torch::as_array(x)
  }
}

generate_test_array <- function(shp, dtype) {
  nelts <- if (!length(shp)) 1L else prod(shp)
  x <- if (dtype == "pred") {
    sample(c(TRUE, FALSE), size = nelts, replace = TRUE)
  } else if (startsWith(dtype, "f")) {
    rnorm(nelts)
  } else if (startsWith(dtype, "i")) {
    sample(-5:5, size = nelts, replace = TRUE)
  } else if (startsWith(dtype, "ui")) {
    sample(0:10, size = nelts, replace = TRUE)
  } else {
    stop(sprintf("Unsupported dtype: %s", dtype))
  }
  if (!length(shp)) x else array(x, shp)
}

make_nv <- function(x, dtype) {
  if (!length(dim(x))) nv_scalar(x, dtype = dtype) else nv_tensor(x, dtype = dtype)
}

make_torch <- function(x, dtype) {
  if (!length(dim(x))) {
    torch::torch_scalar_tensor(x, dtype = str_to_torch_dtype(dtype))
  } else {
    torch::torch_tensor(x, dtype = str_to_torch_dtype(dtype))
  }
}

expect_jit_torch_unary <- function(nv_fun, torch_fun, shp = integer(), dtype = "f32", args_list = list(), gen = NULL) {
  x <- if (is.null(gen)) generate_test_array(shp, dtype) else gen(shp, dtype)
  x_nv <- make_nv(x, dtype)
  x_th <- make_torch(x, dtype)

  f <- jit(function(a, ...) do.call(nv_fun, c(list(a), list(...))))
  out_nv <- do.call(f, c(list(x_nv), args_list))
  out_th <- do.call(torch_fun, c(list(x_th), args_list))

  testthat::expect_equal(as_array(out_nv), as_array_torch(out_th), tolerance = 1e-6)
}

expect_jit_torch_binary <- function(
  nv_fun,
  torch_fun,
  shp_x = integer(),
  shp_y = integer(),
  dtype = "f32",
  args_list = list(),
  gen_x = NULL,
  gen_y = NULL
) {
  x <- if (is.null(gen_x)) generate_test_array(shp_x, dtype) else gen_x(shp_x, dtype)
  y <- if (is.null(gen_y)) generate_test_array(shp_y, dtype) else gen_y(shp_y, dtype)
  x_nv <- make_nv(x, dtype)
  y_nv <- make_nv(y, dtype)
  x_th <- make_torch(x, dtype)
  y_th <- make_torch(y, dtype)

  f <- jit(function(a, b, ...) do.call(nv_fun, c(list(a, b), list(...))))
  out_nv <- do.call(f, c(list(x_nv, y_nv), args_list))
  out_th <- do.call(torch_fun, c(list(x_th, y_th), args_list))

  testthat::expect_equal(as_array(out_nv), as_array_torch(out_th), tolerance = 1e-6)
}
